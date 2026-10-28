import json
from collections import defaultdict
from pathlib import Path

from .domain import Domain
from .utils import load, save


def reference_answer(query, facts, cfg):
    domain = Domain(query["domain"], cfg["domain_limit"])
    query_mask = domain.mask(query["scope"])
    masks = {fact["id"]: domain.mask(fact["scope"]) for fact in facts}
    known = set(query.get("known_prerequisites", []))
    applicable = {}
    for fact in facts:
        residual = "entailed" if set(fact.get("residual", [])) <= known else "unknown"
        applicable[fact["id"]] = domain.applicability(
            query_mask, masks[fact["id"]], residual, fact.get("parsed", False)
        )
    groups, unanswered = {}, []
    for i, cell in enumerate(domain.configurations):
        if not query_mask[i]:
            continue
        support = [
            f
            for f in facts
            if masks[f["id"]][i]
            and f.get("parsed", False)
            and set(f.get("residual", [])) <= known
        ]
        values = {}
        for slot in query["slots"]:
            options = {
                json.dumps(f["values"][slot], sort_keys=True): f["values"][slot]
                for f in support
                if slot in f.get("values", {})
            }
            if len(options) != 1:
                break
            values[slot] = next(iter(options.values()))
        if len(values) != len(query["slots"]):
            unanswered.append(cell)
            continue
        key = json.dumps(values, sort_keys=True)
        branch = groups.setdefault(key, {"conditions": [], "values": values})
        branch["conditions"].append(cell)
    branches = list(groups.values())
    if not branches or len(branches) > cfg["max_bundles"]:
        return {
            "status": "clarification",
            "branches": [],
            "unanswered": domain.explicit(query_mask),
            "applicability": applicable,
        }
    status = "direct" if len(branches) == 1 and not unanswered else "conditional"
    return {
        "status": status,
        "branches": branches,
        "unanswered": unanswered,
        "applicability": applicable,
    }


def build(args):
    from .llm import LocalModel

    index = load(Path(args.index) / "index.json")
    cfg = load(args.config)
    llm = LocalModel(args.model, args.revision, max_generated=4096, max_total=30000)
    families = defaultdict(list)
    for fact in index["facts"]:
        if "product_family" in fact:
            families[fact["product_family"]].append(fact)
    splits = {
        "PyTorch": "train",
        "PostgreSQL": "train",
        "NumPy": "development",
        "Node.js": "test",
        "Apache Spark": "test",
    }
    tasks, labels, annotations, oracle = [], [], [], []
    for family, split in splits.items():
        facts = families[family]
        if not facts:
            raise ValueError("No indexed documentation for " + family)
        pair_questions = (
            24 if split == "train" else 32 if split == "development" else 20
        )
        ordinary = (40 - pair_questions) // 2
        requests = (
            ["pair"] * (pair_questions // 2)
            + ["missing"] * ordinary
            + ["ordinary"] * ordinary
        )
        for number, kind in enumerate(requests):
            start = (number * 4) % len(facts)
            pool = (facts + facts)[start : start + min(20, len(facts))]
            llm.reset()
            result = llm.json(
                "Author source-grounded conditional technical questions of category "
                + kind
                + ". Return {questions:[{question,scope,domain,slots,known_prerequisites}]}. For pair return exactly two questions differing in one explicitly stated prerequisite; otherwise one. Missing questions omit a necessary condition. Ordinary questions specify all needed conditions. Use only these source facts, never invent a restriction. Domain must list finite values for version,platform,hardware,backend,settings. Scope uses allowed sets, numeric version_range, or relation rows. Slots match source fact values. Use 'unspecified' for absent categorical evidence. Keep at most eight domain cells.\nFACTS="
                + json.dumps(pool),
                "diagnostic_authoring",
                3000,
            )
            questions = result["questions"]
            if len(questions) != (2 if kind == "pair" else 1):
                raise ValueError(
                    "Diagnostic authoring returned an incomplete question unit"
                )
            for offset, query in enumerate(questions):
                identifier = f"{family}_{number}_{offset}"
                query.update(
                    id=identifier,
                    dataset="scopediag",
                    split=split,
                    product_family=family,
                    document_ids=sorted(
                        {f["chunk_id"].rsplit(":", 1)[0] for f in facts}
                    ),
                )
                domain = Domain(query["domain"], 8)
                if not query["slots"] or not domain.mask(query["scope"]).any():
                    raise ValueError(
                        "Authored query needs nonempty slots and a consistent source domain"
                    )
                answer = reference_answer(query, pool, cfg)
                label = {
                    "id": identifier,
                    "review_status": "unreviewed",
                    "category": kind,
                    "structured_answer": answer,
                    "applicability": answer["applicability"],
                    "eligible_branches": [b["conditions"] for b in answer["branches"]],
                    "source_fact_ids": [f["id"] for f in pool],
                }
                if kind == "pair":
                    label["pair_id"] = f"{family}_{number}"
                tasks.append(query)
                labels.append(label)
                oracle.append(
                    {
                        "id": identifier,
                        "facts": {f["id"]: {"scope": f["scope"]} for f in facts},
                    }
                )
                if split != "test":
                    for fact in pool:
                        annotations.append(
                            {
                                "query": query["question"],
                                "fact": fact["assertion"],
                                "qualifier": fact.get("qualifier", ""),
                                "product_family": family,
                                "split": split,
                                "applicability": answer["applicability"][fact["id"]],
                                "review_status": "unreviewed",
                                "source_span": fact["source_span"],
                                "url": fact["url"],
                            }
                        )
                        for slot in query["slots"]:
                            annotations.append(
                                {
                                    "query": query["question"],
                                    "fact": fact["assertion"],
                                    "qualifier": fact.get("qualifier", ""),
                                    "slot": slot,
                                    "product_family": family,
                                    "split": split,
                                    "slot_support": int(slot in fact.get("values", {})),
                                    "review_status": "unreviewed",
                                    "source_span": fact["source_span"],
                                    "url": fact["url"],
                                }
                            )
            output = Path(args.output)
            save(output / "tasks.json", tasks)
            save(output / "labels.json", labels)
            save(output / "annotations.json", annotations)
            save(output / "scopes.json", oracle)
