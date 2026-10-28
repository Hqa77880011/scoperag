import json
import time
from dataclasses import dataclass

import numpy as np

from .domain import Domain, parsed_scope


METHODS = (
    "scoperag",
    "bm25",
    "dense",
    "latest",
    "credibility",
    "version_platform",
    "fact_nli",
    "faithful",
    "probe",
    "versionrag",
    "individual",
    "all_conditions",
    "exact_atomic",
    "no_platform",
    "no_uncertainty",
    "no_joint",
    "no_verifier",
    "human_scope",
)


def duplicate(a, b):
    common = set(a.get("values", {})) & set(b.get("values", {}))
    return sum(a["values"][k] == b["values"][k] for k in common) / max(
        len(set(a.get("values", {})) | set(b.get("values", {}))), 1
    )


def contradiction(a, b):
    return any(
        a["values"][key] != b["values"][key]
        for key in set(a.get("values", {})) & set(b.get("values", {}))
    )


@dataclass
class Bundle:
    indices: tuple
    domain: np.ndarray
    coverage: np.ndarray
    score: float
    tokens: int


def prepare_facts(query, facts, scores, domain, cfg, method, token_length):
    query_mask = domain.mask(query["scope"])
    prepared = []
    for fact, score in zip(facts, scores):
        fact = dict(fact)
        scope, opaque = parsed_scope(fact["scope"])
        if opaque:
            fact["parsed"] = False
            fact["residual"] = list(fact.get("residual", [])) + opaque
        if method == "no_platform":
            scope["allowed"] = {
                k: v for k, v in scope.get("allowed", {}).items() if k in ("version",)
            }
            scope.pop("relation", None)
            scope.pop("implications", None)
        fact_mask = domain.mask(scope)
        residual = fact.get("residual", [])
        semantic = np.asarray(score["semantic_probabilities"])
        semantic_state = (
            ("entailed", "inconsistent", "unknown")[int(semantic.argmax())]
            if semantic.max() >= cfg["head_threshold"]
            else "unknown"
        )
        if not residual:
            semantic_state = "entailed"
        applicability = domain.applicability(
            query_mask, fact_mask, semantic_state, fact.get("parsed", False)
        )
        if (
            residual
            and any(r not in query.get("known_prerequisites", []) for r in residual)
            and semantic_state != "inconsistent"
        ):
            applicability = "unknown"
        if method == "no_uncertainty" and applicability == "unknown":
            applicability = "entailed"
            fact_mask = query_mask.copy()
            fact["parsed"] = True
            fact["residual"] = []
        fact.update(
            mask=fact_mask,
            applicability=applicability,
            slot_scores=np.asarray(score["slot_scores"]),
            tokens=token_length(json.dumps({fact["id"]: evidence_payload(fact)})),
            support=float(max(score["slot_scores"], default=0)),
        )
        prepared.append(fact)
    return query_mask, prepared


def evidence_payload(fact):
    """Serialize the same source-preserving fields used in the generator prompt."""
    return {
        k: v
        for k, v in fact.items()
        if k in ("assertion", "qualifier", "source_span", "url", "values", "residual")
    }


def search(query_mask, facts, cfg, joint=True, branch_mask=None):
    mask = query_mask if branch_mask is None else query_mask & branch_mask
    if not mask.any():
        return []
    slots = len(facts[0]["slot_scores"]) if facts else 0
    beam = [Bundle((), mask, np.zeros(slots), 0.0, 0)]
    for _ in range(cfg["max_facts"]):
        proposals = {b.indices: b for b in beam}
        for bundle in beam:
            for index, fact in enumerate(facts):
                if index in bundle.indices or fact["applicability"] == "inconsistent":
                    continue
                indices = tuple(sorted(bundle.indices + (index,)))
                if indices in proposals:
                    continue
                tokens = bundle.tokens + fact["tokens"]
                if tokens > cfg["evidence_tokens"] or any(
                    contradiction(fact, facts[j]) for j in bundle.indices
                ):
                    continue
                shared = bundle.domain & fact["mask"] if joint else bundle.domain
                if not shared.any():
                    continue
                coverage = np.maximum(bundle.coverage, fact["slot_scores"])
                marginal = float((coverage - bundle.coverage).sum() / max(slots, 1))
                penalty = (
                    cfg["lambda"] * (fact["applicability"] == "unknown")
                    + cfg["mu"] * (1 - fact["support"])
                    + cfg["rho"]
                    * sum(duplicate(fact, facts[j]) for j in bundle.indices)
                )
                proposals[indices] = Bundle(
                    indices, shared, coverage, bundle.score + marginal - penalty, tokens
                )
        beam = sorted(
            proposals.values(),
            key=lambda b: (b.score, b.coverage.sum(), -b.tokens),
            reverse=True,
        )[: cfg["beam_width"]]
    return beam


def output_bundles(query, query_mask, facts, bundles, domain, cfg, method):
    chosen, covered = [], np.zeros_like(query_mask)
    used_facts, used_tokens = set(), 0
    for bundle in bundles:
        if not bundle.indices or np.any(bundle.coverage < cfg["slot_threshold"]):
            continue
        actual = query_mask.copy()
        for index in bundle.indices:
            actual &= facts[index]["mask"]
        if not actual.any():
            continue
        selected = [facts[i] for i in bundle.indices]
        opaque = [
            r
            for fact in selected
            for r in fact.get("residual", [])
            if r not in query.get("known_prerequisites", [])
        ]
        unparsed = any(not f.get("parsed", False) for f in selected)
        if opaque or unparsed:
            continue
        if np.all(~query_mask | actual) and all(
            f["applicability"] == "entailed" for f in selected
        ):
            return {
                "status": "direct",
                "branches": [
                    {
                        "id": "direct",
                        "conditions": domain.explicit(query_mask),
                        "fact_ids": [f["id"] for f in selected],
                        "values": {
                            k: v
                            for f in selected
                            for k, v in f.get("values", {}).items()
                        },
                    }
                ],
                "unanswered": [],
            }
        available = actual & ~covered
        if not available.any():
            continue
        added = set(bundle.indices) - used_facts
        extra_tokens = sum(facts[i]["tokens"] for i in added)
        if used_tokens + extra_tokens > cfg["evidence_tokens"]:
            continue
        if (
            len(domain.explicit(available)) + sum(len(r["conditions"]) for r in chosen)
            > cfg["max_cells"]
        ):
            continue
        chosen.append(
            {
                "id": "branch_" + str(len(chosen)),
                "conditions": domain.explicit(available),
                "fact_ids": [f["id"] for f in selected],
                "values": {
                    k: v for f in selected for k, v in f.get("values", {}).items()
                },
            }
        )
        covered |= available
        used_facts.update(added)
        used_tokens += extra_tokens
        if len(chosen) >= cfg["max_bundles"]:
            break
    if not chosen:
        return {
            "status": "clarification",
            "branches": [],
            "unanswered": domain.explicit(query_mask),
            "reason": "Missing prerequisites, unresolved scopes, or insufficient compatible slot support",
        }
    return {
        "status": "conditional",
        "branches": chosen,
        "unanswered": domain.explicit(query_mask & ~covered),
    }


def infer(query, candidates, scores, cfg, method, token_length):
    selection_started = time.perf_counter()
    domain = Domain(query["domain"], cfg["domain_limit"])
    scope, opaque = parsed_scope(query["scope"])
    if opaque:
        return (
            {
                "status": "clarification",
                "branches": [],
                "unanswered": [],
                "reason": "Unrecognized query restrictions: " + "; ".join(opaque),
            },
            [],
            domain,
        )
    query = dict(query, scope=scope)
    if not domain.mask(query["scope"]).any():
        return (
            {
                "status": "clarification",
                "branches": [],
                "unanswered": [],
                "reason": "Inconsistent user configuration",
            },
            [],
            domain,
        )
    query_mask, facts = prepare_facts(
        query, candidates, scores, domain, cfg, method, token_length
    )
    if method in ("no_joint", "individual"):
        bundles = search(query_mask, facts, cfg, joint=False)
        # Commit the output quota before downstream joint rejection. Rejection
        # cannot select a lower-ranked replacement from the remaining beam.
        bundles = [
            b
            for b in bundles
            if b.indices and np.all(b.coverage >= cfg["slot_threshold"])
        ][: cfg["max_bundles"]]
    elif method == "exact_atomic":
        bundles = []
        for index in np.flatnonzero(query_mask):
            atomic = np.zeros_like(query_mask)
            atomic[index] = True
            compatible = [
                dict(
                    f,
                    applicability="entailed"
                    if f["mask"][index]
                    and f.get("parsed", False)
                    and not f.get("residual")
                    else "inconsistent"
                    if not f["mask"][index]
                    else f["applicability"],
                )
                for f in facts
            ]
            bundles.extend(
                search(query_mask, compatible, cfg, joint=False, branch_mask=atomic)
            )
        bundles.sort(key=lambda b: b.score, reverse=True)
    elif method in (
        "dense",
        "bm25",
        "latest",
        "credibility",
        "versionrag",
        "version_platform",
        "fact_nli",
        "faithful",
        "probe",
        "all_conditions",
    ):
        eligible = [
            i
            for i, f in enumerate(facts)
            if f["applicability"] != "inconsistent"
            or method in ("dense", "bm25", "all_conditions")
        ]
        if method == "latest":
            from .domain import version

            eligible.sort(
                key=lambda i: version(facts[i].get("version", "0")), reverse=True
            )
        elif method == "credibility":
            eligible.sort(key=lambda i: facts[i].get("credibility", 0), reverse=True)
        elif method == "faithful":
            eligible.sort(
                key=lambda i: (
                    facts[i].get("alignment_priority", 0),
                    facts[i]["support"],
                ),
                reverse=True,
            )
        elif method in ("fact_nli", "probe"):
            eligible.sort(key=lambda i: facts[i]["support"], reverse=True)
        elif method == "versionrag":
            requested = set(query["scope"].get("allowed", {}).get("version", []))
            eligible.sort(
                key=lambda i: (
                    facts[i].get("version") in requested,
                    facts[i]["support"],
                ),
                reverse=True,
            )
        elif method == "version_platform":
            filtered = []
            for i in eligible:
                reduced = {
                    "allowed": {
                        k: v
                        for k, v in facts[i]["scope"].get("allowed", {}).items()
                        if k in ("version", "platform")
                    }
                }
                if facts[i]["scope"].get("version_range"):
                    reduced["version_range"] = facts[i]["scope"]["version_range"]
                if (query_mask & domain.mask(reduced)).any():
                    filtered.append(i)
            eligible = filtered
        indices, used = [], 0
        for i in eligible:
            if used + facts[i]["tokens"] <= cfg["evidence_tokens"] and (
                method == "all_conditions" or len(indices) < cfg["max_facts"]
            ):
                indices.append(i)
                used += facts[i]["tokens"]
        coverage = (
            np.max([facts[i]["slot_scores"] for i in indices], axis=0)
            if indices
            else np.zeros(len(query["slots"]))
        )
        bundles = [Bundle(tuple(indices), query_mask, coverage, 0.0, used)]
    else:
        bundles = search(query_mask, facts, cfg)
    conflicting = [
        (a["id"], b["id"])
        for i, a in enumerate(facts)
        for b in facts[i + 1 :]
        if a["applicability"] == b["applicability"] == "entailed"
        and contradiction(a, b)
    ]
    if conflicting:
        answer = {
            "status": "clarification",
            "branches": [],
            "unanswered": domain.explicit(query_mask),
            "reason": "Applicable sources assert opposing values for the same answer slot.",
            "conflicting_facts": conflicting,
        }
    else:
        answer = output_bundles(query, query_mask, facts, bundles, domain, cfg, method)
    answer["selection_seconds"] = time.perf_counter() - selection_started
    answer["applicability"] = {f["id"]: f["applicability"] for f in facts}
    answer["candidate_ids"] = [f["id"] for f in facts]
    return answer, facts, domain


def generate(answer, facts, llm, query, cfg):
    answer["known_prerequisites"] = query.get("known_prerequisites", [])
    if answer["status"] == "clarification":
        answer["claims"] = []
        answer["text"] = answer.get(
            "reason", "Please specify the unresolved deployment conditions."
        )
        return answer
    indexed = {f["id"]: f for f in facts}
    evidence = {
        identifier: evidence_payload(indexed[identifier])
        for b in answer["branches"]
        for identifier in b["fact_ids"]
    }
    if llm.tokens(json.dumps(evidence)) > cfg["evidence_tokens"]:
        answer.update(
            status="clarification",
            branches=[],
            claims=[],
            text="The source quotations exceed the shared evidence token budget.",
        )
        return answer
    response = llm.json(
        "Answer the technical question using atomic statements and retain every branch condition. Return {claims:[{text,branch_id,fact_ids:[...],values:{slot:value}}],text}. Only use supplied source facts. Unanswered conditions must remain explicit.\nQUESTION="
        + query["question"]
        + "\nCOMMITMENTS="
        + json.dumps(answer)
        + "\nEVIDENCE="
        + json.dumps(evidence),
        "answer",
        cfg["answer_tokens"],
    )
    answer["claims"] = response.get("claims", [])
    answer["generated_claims"] = list(answer["claims"])
    answer["text"] = response.get("text", "")
    answer["required_slots"] = query["slots"]
    return answer


class Verifier:
    def __init__(self, name="cross-encoder/nli-deberta-v3-base", device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        )
        self.device = device
        labels = self.model.config.id2label
        self.entailment = next(
            (int(i) for i, label in labels.items() if "entail" in label.lower()), None
        )
        if self.entailment is None:
            raise ValueError("Verifier checkpoint must declare its entailment label")

    def support(self, premise, hypothesis):
        import torch

        tokens = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.inference_mode():
            return float(self.model(**tokens).logits.softmax(-1)[0, self.entailment])


def verify(answer, facts, domain, verifier, threshold=0.9):
    indexed = {f["id"]: f for f in facts}
    branches = {b["id"]: b for b in answer["branches"]}
    checked, links = [], []
    for claim in answer.get("claims", []):
        branch = branches.get(claim.get("branch_id"))
        identifiers = claim.get("fact_ids", [])
        valid = (
            branch is not None
            and bool(identifiers)
            and set(identifiers) <= set(branch["fact_ids"])
        )
        for identifier in identifiers:
            if identifier not in indexed or branch is None:
                valid = False
                links.append({"fact_id": identifier, "valid": False})
                continue
            fact = indexed[identifier]
            cells = branch["conditions"]
            branch_scope = domain.mask({"relation": cells})
            scope_ok = (
                bool(np.all(~branch_scope | fact["mask"]))
                and fact.get("parsed", False)
                and set(fact.get("residual", []))
                <= set(answer.get("known_prerequisites", []))
            )
            predicate = claim["text"] + "\nUnder configurations " + json.dumps(cells)
            confidence = verifier.support(fact["source_span"], predicate)
            values_ok = all(
                key in fact.get("values", {}) and fact["values"][key] == value
                for key, value in claim.get("values", {}).items()
            )
            link_valid = bool(scope_ok and confidence >= threshold and values_ok)
            links.append(
                {
                    "fact_id": identifier,
                    "valid": link_valid,
                    "confidence": confidence,
                    "scope_valid": scope_ok,
                }
            )
            valid &= link_valid
        checked.append(dict(claim, accepted=bool(valid)))
    answer["verification"] = {"claims": checked, "links": links}
    answer["claims"] = [c for c in checked if c["accepted"]]
    retained = []
    for branch in answer["branches"]:
        values = {
            key: value
            for claim in answer["claims"]
            if claim["branch_id"] == branch["id"]
            for key, value in claim.get("values", {}).items()
        }
        if set(answer.get("required_slots", [])) <= set(values):
            retained.append(dict(branch, values=values))
        else:
            answer["unanswered"].extend(branch["conditions"])
    answer["branches"] = retained
    allowed = {b["id"] for b in retained}
    answer["claims"] = [c for c in answer["claims"] if c["branch_id"] in allowed]
    if len(answer["claims"]) != len(checked):
        answer["text"] = "\n".join(c["text"] for c in answer["claims"])
    if not answer["claims"] or not retained:
        answer["status"] = "clarification"
        answer["text"] = (
            "The supplied evidence does not support every required answer slot under the stated conditions."
        )
    return answer
