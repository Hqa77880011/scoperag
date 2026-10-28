import argparse
import csv
import json
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from transformers import AutoTokenizer

from .method import METHODS, Verifier, generate, infer, verify
from .model import ApplicabilityModel, Heads
from .utils import load, plot_results, save, seed_all


def prepare(args):
    if args.source in ("confact", "conflictbank"):
        from .sources import public_pack

        public_pack(args.source, args.output, args.input, args.count)
        return
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.source == "versionqa":
        with open(
            source / "data/test/evaluation_set.csv", encoding="utf-8-sig", newline=""
        ) as handle:
            raw = list(csv.DictReader(handle))
        tasks = [
            {
                "id": "versionqa_" + str(i),
                "question": r["Question"],
                "split": "test",
                "dataset": "versionqa",
                "type": r["Type"],
            }
            for i, r in enumerate(raw)
        ]
        labels = [{"id": t["id"], "answer": r["Answer"]} for t, r in zip(tasks, raw)]
        documents = []
        for p in sorted((source / "data/raw").iterdir()):
            if p.suffix.lower() not in (".md", ".txt", ".pdf"):
                continue
            if p.suffix.lower() == ".pdf":
                import fitz

                with fitz.open(p) as pdf:
                    text = "\n".join(page.get_text() for page in pdf)
            else:
                text = p.read_text(encoding="utf-8-sig")
            documents.append(
                {
                    "id": p.stem,
                    "text": text,
                    "url": "https://github.com/danielhuwiler/versionrag/blob/main/data/raw/"
                    + p.name,
                    "capture_date": date.today().isoformat(),
                    "version": next(
                        iter(re.findall(r"\d+\.\d+(?:\.\d+)?", p.stem)), "0"
                    ),
                }
            )
        save(output / "documents.json", documents)
    else:
        payload = load(source)
        tasks, labels = payload["tasks"], payload.get("labels", [])
        if payload.get("documents"):
            save(output / "documents.json", payload["documents"])
        for task in tasks:
            if any(key in task for key in ("gold", "answer", "reference")):
                raise ValueError("Keep gold answers outside runtime tasks")
    save(output / "tasks.json", tasks)
    if labels:
        save(output / "labels.json", labels)


def chunks(text, size=1400, overlap=200):
    if size <= overlap:
        raise ValueError("Chunk size must exceed overlap")
    return [text[i : i + size] for i in range(0, len(text), size - overlap)]


def index(args):
    from .llm import LocalModel
    from sentence_transformers import SentenceTransformer

    llm = LocalModel(
        args.model,
        args.revision,
        max_generated=1024,
        max_total=16384,
        device=args.device,
    )
    indexed, facts = [], []
    for document in load(args.documents):
        for number, text in enumerate(chunks(document["text"])):
            identifier = document["id"] + ":" + str(number)
            llm.reset()
            result = llm.json(
                "Extract at most two atomic source facts. Return {facts:[{assertion,qualifier,source_span,scope:{allowed:{version:[...],platform:[...],hardware:[...],backend:[...],settings:[...]},version_range:{min,max,max_inclusive},relation:[{field:value}],implications:[{if:{field:[values]},then:{field:[values]}}]},residual:[unparsed prerequisite strings],parsed:boolean,credibility:source-reliability score from 0 to 1,values:{answer_slot:asserted_value}}]}. Omit unused scope keys. The source_span and qualifier must be exact source substrings. Preserve coupled conditions as full finite relations. parsed=true only when all restrictions are represented; omission is not universal unless the source states universal applicability. Do not invent values or source restrictions.\nDOCUMENT_VERSION="
                + document.get("version", "unknown")
                + "\nSOURCE="
                + text,
                "extract",
                1000,
            )
            chunk_facts = []
            for offset, fact in enumerate(result.get("facts", [])[:2]):
                if (
                    not fact.get("source_span")
                    or fact["source_span"] not in text
                    or (fact.get("qualifier") and fact["qualifier"] not in text)
                ):
                    continue
                fact.update(
                    id=identifier + ":" + str(offset),
                    chunk_id=identifier,
                    url=document["url"],
                    capture_date=document["capture_date"],
                    version=document.get("version", "0"),
                    product_family=document.get("product_family"),
                    credibility=float(
                        document.get("credibility", fact.get("credibility", 0.5))
                    ),
                )
                chunk_facts.append(fact)
            indexed.append(
                {
                    "id": identifier,
                    "document_id": document["id"],
                    "text": text,
                    "fact_ids": [f["id"] for f in chunk_facts],
                }
            )
            facts.extend(chunk_facts)
    if not indexed or not facts:
        raise ValueError("No source-grounded facts extracted")
    encoder = SentenceTransformer(args.encoder, device=args.device)
    embeddings = encoder.encode(
        ["passage: " + c["text"] for c in indexed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embeddings)
    save(
        output / "index.json",
        {
            "chunks": indexed,
            "facts": facts,
            "encoder": args.encoder,
            "generator": args.model,
            "revision": args.revision,
        },
    )


def train(args):
    cfg, rows = load(args.config), load(args.annotations)
    seed_all(args.seed)
    if any(r.get("review_status") == "unreviewed" for r in rows):
        raise ValueError(
            "Review generated annotations against their source spans before marking review_status=verified"
        )
    family_splits = defaultdict(set)
    for row in rows:
        family_splits[row["product_family"]].add(row["split"])
    if any(len(values) > 1 for values in family_splits.values()):
        raise ValueError(
            "Product families must be isolated across training, development and test"
        )
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] in ("development", "validation")]
    if not training or not validation:
        raise ValueError("Need source-verified training and development annotations")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, revision=args.revision)
    model = ApplicabilityModel(args.encoder, args.revision).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = -float("inf"), []
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        order = np.random.permutation(len(training))
        for start in range(0, len(order), cfg["batch_size"]):
            batch = [training[i] for i in order[start : start + cfg["batch_size"]]]
            tokens = tokenizer(
                [r["query"] + "\nSLOT: " + r.get("slot", "") for r in batch],
                [r["fact"] + "\nQUALIFIERS: " + r.get("qualifier", "") for r in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(args.device)
            applicability, support = model(tokens)
            app_indices = [i for i, r in enumerate(batch) if "applicability" in r]
            slot_indices = [i for i, r in enumerate(batch) if "slot_support" in r]
            terms = []
            if app_indices:
                labels = torch.tensor(
                    [
                        ("entailed", "inconsistent", "unknown").index(
                            batch[i]["applicability"]
                        )
                        for i in app_indices
                    ],
                    device=args.device,
                )
                terms.append(F.cross_entropy(applicability[app_indices], labels))
            if slot_indices:
                labels = torch.tensor(
                    [float(batch[i]["slot_support"]) for i in slot_indices],
                    device=args.device,
                )
                terms.append(
                    F.binary_cross_entropy_with_logits(support[slot_indices], labels)
                )
            if not terms:
                raise ValueError(
                    "Annotation needs applicability or slot_support labels"
                )
            optimizer.zero_grad()
            loss = sum(terms)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        predicted, gold = [], []
        with torch.inference_mode():
            for row in validation:
                if "applicability" not in row:
                    continue
                tokens = tokenizer(
                    row["query"] + "\nSLOT: " + row.get("slot", ""),
                    row["fact"] + "\nQUALIFIERS: " + row.get("qualifier", ""),
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(args.device)
                predicted.append(int(model(tokens)[0].argmax()))
                gold.append(
                    ("entailed", "inconsistent", "unknown").index(row["applicability"])
                )
        if not gold:
            raise ValueError(
                "Development applicability labels are required for checkpoint selection"
            )
        score = float(f1_score(gold, predicted, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "development_macro_f1": score,
            }
        )
        if score > best:
            best = score
            torch.save(model.state_dict(), output / "heads.pt")
            tokenizer.save_pretrained(output / "tokenizer")
            model.encoder.config.save_pretrained(output / "encoder_config")
        save(
            output / "model.json",
            {
                "encoder": args.encoder,
                "revision": args.revision,
                "config": cfg,
                "seed": args.seed,
            },
        )
        save(output / "history.json", history)


def bm25(question, texts):
    tokens = [re.findall(r"\w+", t.lower()) for t in texts]
    query = re.findall(r"\w+", question.lower())
    average = np.mean([len(t) for t in tokens]) if tokens else 1
    scores = []
    for words in tokens:
        score = 0.0
        for term in query:
            df = sum(term in candidate for candidate in tokens)
            frequency = words.count(term)
            idf = np.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
            score += (
                idf
                * frequency
                * 2.2
                / (frequency + 1.2 * (0.25 + 0.75 * len(words) / max(average, 1)))
            )
        scores.append(score)
    return np.asarray(scores)


def finite_values(facts):
    values = {
        key: set() for key in ("version", "platform", "hardware", "backend", "settings")
    }
    for fact in facts:
        scope = fact["scope"]
        values["version"].add(fact.get("version", "0"))
        for key, items in scope.get("allowed", {}).items():
            if key in values:
                values[key].update(items)
        for row in scope.get("relation", []):
            for key, value in row.items():
                if key in values:
                    values[key].add(value)
        for bound in ("min", "max"):
            if bound in scope.get("version_range", {}):
                values["version"].add(scope["version_range"][bound])
    return {
        key: sorted(items, key=str) if items else ["unspecified"]
        for key, items in values.items()
    }


def run(args):
    from .llm import LocalModel
    from sentence_transformers import SentenceTransformer

    cfg = load(args.config)
    index_dir = Path(args.index)
    indexed = load(index_dir / "index.json")
    embedding = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    encoder = SentenceTransformer(indexed["encoder"], device=args.device)
    heads = Heads(args.heads, args.device)
    llm = LocalModel(
        args.model,
        args.revision,
        seed=args.seed,
        max_generated=2000,
        max_total=30000,
        device=args.device,
    )
    verifier = (
        None if args.method == "no_verifier" else Verifier(args.verifier, args.device)
    )
    oracle = (
        {r["id"]: r for r in load(args.oracle_scopes)} if args.oracle_scopes else {}
    )
    traces = []
    for task in load(args.tasks):
        if task["split"] != args.split:
            continue
        llm.reset()
        started = time.perf_counter()
        stage_started = started
        stages = {}
        question = task["question"]
        similarity = (
            bm25(question, [c["text"] for c in indexed["chunks"]])
            if args.method == "bm25"
            else embedding
            @ encoder.encode("query: " + question, normalize_embeddings=True)
        )
        permitted = set(task.get("document_ids", []))
        if permitted:
            similarity = np.asarray(similarity).copy()
            for i, chunk in enumerate(indexed["chunks"]):
                if chunk["document_id"] not in permitted:
                    similarity[i] = -np.inf
        selected = [
            int(i) for i in np.argsort(-similarity) if np.isfinite(similarity[i])
        ][: args.top_k]
        ids = {indexed["chunks"][i]["id"] for i in selected}
        facts = [f for f in indexed["facts"] if f["chunk_id"] in ids]
        rank = {indexed["chunks"][i]["id"]: j for j, i in enumerate(selected)}
        facts.sort(key=lambda f: rank[f["chunk_id"]])
        if args.method == "human_scope":
            if task["id"] not in oracle:
                raise ValueError(
                    "human_scope requires --oracle-scopes with evaluator-provided conditions"
                )
            facts = [
                dict(
                    f,
                    scope=oracle[task["id"]]["facts"][f["id"]]["scope"],
                    parsed=True,
                    residual=[],
                )
                for f in facts
            ]
        stages["retrieval"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        query = dict(task)
        if "scope" not in query or "domain" not in query or "slots" not in query:
            values = finite_values(facts)
            declaration = llm.json(
                "Parse only explicitly supplied deployment restrictions and requested answer slots. Return {scope:{allowed:{field:[values]}},slots:[slot identifiers],known_prerequisites:[explicit textual conditions]}. Omit missing query fields; do not fill a default deployment. Available finite field values: "
                + json.dumps(values)
                + "\nQUESTION="
                + question,
                "query_parse",
                512,
            )
            query.update(declaration, domain=values)
        stages["query_parsing"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        scores = heads.score(question, facts, query["slots"])
        stages["scoring"] = time.perf_counter() - stage_started
        if args.method == "faithful":
            prior = llm.chat(
                "Give your current technical answer and the conditions you believe it requires.\nQUESTION="
                + question,
                "parametric_facts",
                256,
            )
            alignment = llm.json(
                "Align the proposed answer with these source facts. Prefer source-grounded corrections when parametric knowledge conflicts. Return {priority_fact_ids:[...]} in useful evidence order.\nQUESTION="
                + question
                + "\nPROPOSED="
                + prior
                + "\nFACTS="
                + json.dumps(
                    [{k: f[k] for k in ("id", "assertion", "qualifier")} for f in facts]
                ),
                "context_alignment",
                256,
            )
            order = {
                identifier: i
                for i, identifier in enumerate(alignment.get("priority_fact_ids", []))
            }
            for fact in facts:
                fact["alignment_priority"] = len(order) - order.get(
                    fact["id"], len(order)
                )
        elif args.method == "probe":
            probe = llm.json(
                "Identify misleading or conflicting evidence under the question conditions. Preserve valid evidence for distinct conditions. Return {keep_fact_ids:[...]} for a conflict-pruned candidate pool; do not add new facts.\nQUESTION="
                + question
                + "\nFACTS="
                + json.dumps(
                    [{k: f[k] for k in ("id", "assertion", "qualifier")} for f in facts]
                ),
                "conflict_probe",
                256,
            )
            keep = set(probe.get("keep_fact_ids", []))
            selected_pairs = [(f, s) for f, s in zip(facts, scores) if f["id"] in keep]
            facts, scores = (
                [p[0] for p in selected_pairs],
                [p[1] for p in selected_pairs],
            )
        try:
            answer, prepared, domain = infer(
                query, facts, scores, cfg, args.method, llm.tokens
            )
            stages["selection"] = answer.get("selection_seconds", 0.0)
            stage_started = time.perf_counter()
            answer = generate(answer, prepared, llm, query, cfg)
            stages["generation"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if verifier:
                answer = verify(
                    answer, prepared, domain, verifier, cfg["verifier_threshold"]
                )
            stages["verification"] = time.perf_counter() - stage_started
            traces.append(
                {
                    "stages_seconds": stages,
                    "total_seconds": time.perf_counter() - started,
                    "top_k": args.top_k,
                    "id": task["id"],
                    "dataset": task["dataset"],
                    "method": args.method,
                    "seed": args.seed,
                    "answer": answer,
                    "query": query,
                    "ledger": llm.ledger.copy(),
                }
            )
        except (ValueError, KeyError, RuntimeError) as error:
            stages["verification"] = time.perf_counter() - stage_started
            traces.append(
                {
                    "stages_seconds": stages,
                    "total_seconds": time.perf_counter() - started,
                    "top_k": args.top_k,
                    "id": task["id"],
                    "dataset": task["dataset"],
                    "method": args.method,
                    "seed": args.seed,
                    "answer": {
                        "status": "clarification",
                        "branches": [],
                        "claims": [],
                        "text": str(error),
                    },
                    "error": str(error),
                    "query": query,
                    "ledger": llm.ledger.copy(),
                }
            )
        save(Path(args.output) / "traces.json", traces)


def canonical(answer):
    return {
        "status": answer["status"],
        "branches": sorted(
            [
                {
                    "conditions": sorted(
                        [json.dumps(c, sort_keys=True) for c in b["conditions"]]
                    ),
                    "values": b.get("values", {}),
                }
                for b in answer.get("branches", [])
            ],
            key=lambda b: json.dumps(b, sort_keys=True),
        ),
    }


def evaluate(args):
    labels = {r["id"]: r for r in load(args.labels)}
    if any(r.get("review_status") == "unreviewed" for r in labels.values()):
        raise ValueError(
            "Source-generated labels require source-binding review before evaluation"
        )
    judgments = (
        {
            (
                "|".join(str(r[k]) for k in ("id", "method", "seed"))
                if "method" in r and "seed" in r
                else r["id"]
            ): r
            for r in load(args.judgments)
        }
        if args.judgments
        else {}
    )
    judge = None
    if args.judge_model:
        from .llm import LocalModel

        judge = LocalModel(args.judge_model, max_generated=32, max_total=16000)
    rows = [r for path in args.inputs for r in load(path)]
    identities = [
        (r["method"], r.get("top_k", 20), r["id"], r.get("seed", 17)) for r in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate method/top-k/question/seed observations")
    summary = []
    for method, top_k in sorted({(r["method"], r.get("top_k", 20)) for r in rows}):
        traces = [
            r for r in rows if r["method"] == method and r.get("top_k", 20) == top_k
        ]
        correctness, pairs, datasets = {}, defaultdict(list), defaultdict(list)
        unsupported, claims, valid_links, links = 0, 0, 0, 0
        application, conflicts, suppressed = [], [], []
        token_cost, latency = [], []
        modes = defaultdict(list)
        categories = defaultdict(list)
        completeness = []
        stage_values = defaultdict(list)
        pair_strata = {}
        for trace in traces:
            label, answer = labels[trace["id"]], trace["answer"]
            if trace.get("error"):
                correct = False
            elif "structured_answer" in label:
                correct = canonical(answer) == canonical(label["structured_answer"])
            elif judge:
                judge.reset()
                response = judge.chat(
                    "Judge correctness against the reference answer. Correct necessary clarification is allowed only if the reference requires it. Return True or False.\nQUESTION="
                    + trace["query"]["question"]
                    + "\nREFERENCE="
                    + label["answer"]
                    + "\nANSWER="
                    + answer["text"],
                    "sealed_evaluation",
                    8,
                    temperature=0,
                )
                correct = response.strip().lower() == "true"
            else:
                raise ValueError(
                    "Free-text benchmark answers require --judge-model or structured labels"
                )
            correctness[(trace["id"], trace.get("seed", 17))] = bool(correct)
            modes[answer["status"]].append(bool(correct))
            for category in label.get(
                "categories", [label.get("category", "ordinary")]
            ):
                categories[category].append(bool(correct))
            for stage, seconds in trace.get("stages_seconds", {}).items():
                stage_values[stage].append(seconds)
            datasets[trace["dataset"]].append(correct)
            if label.get("pair_id"):
                pairs[(label["pair_id"], trace.get("seed", 17))].append(correct)
                pair_strata[label["pair_id"]] = label.get(
                    "pair_stratum", label.get("stratum", "unspecified")
                )
            for fact_id, gold in label.get("applicability", {}).items():
                application.append(answer.get("applicability", {}).get(fact_id) == gold)
            if label.get("same_scope_conflict"):
                conflicts.append(correct)
            eligible = label.get("eligible_branches", [])
            observed = {
                json.dumps(b.get("conditions"), sort_keys=True)
                for b in answer.get("branches", [])
            }
            for condition in eligible:
                suppressed.append(json.dumps(condition, sort_keys=True) not in observed)
            judgment = judgments.get(
                "|".join(str(trace[k]) for k in ("id", "method", "seed"))
            )
            if judgment is None:
                judgment = judgments.get(trace["id"])
            if judgment and "claim_has_valid_citation" in judgment:
                completeness.extend(judgment["claim_has_valid_citation"])
            if judgment:
                for field in ["claim_supported", "citation_valid"]:
                    if any(type(value) is not bool for value in judgment[field]):
                        raise ValueError("Independent audit flags must be booleans")
                for flag in judgment["claim_supported"]:
                    claims += 1
                    unsupported += not flag
                for flag in judgment["citation_valid"]:
                    links += 1
                    valid_links += bool(flag)
            token_cost.append(
                sum(r["prompt_tokens"] + r["generated_tokens"] for r in trace["ledger"])
            )
            latency.append(
                trace.get("total_seconds", sum(r["seconds"] for r in trace["ledger"]))
            )
        if any(len(values) != 2 for values in pairs.values()):
            raise ValueError(
                "Paired accuracy requires both questions exactly once for every pair/seed"
            )
        strata = {}
        for stratum in set(pair_strata.values()):
            selected = [
                (key, values)
                for key, values in pairs.items()
                if pair_strata[key[0]] == stratum
            ]
            strata[stratum] = {
                "both": sum(all(v) for _, v in selected),
                "seed_pairs": len(selected),
                "source_pairs": len({k[0] for k, _ in selected}),
                "accuracy_percent": 100 * np.mean([all(v) for _, v in selected]),
            }
        result = {
            "method": method,
            "top_k": top_k,
            "pair_strata": strata,
            "runtime_error_count": sum(bool(r.get("error")) for r in traces),
            "pair_both": sum(all(v) for v in pairs.values() if len(v) == 2),
            "pair_single": sum(sum(v) == 1 for v in pairs.values() if len(v) == 2),
            "pair_neither": sum(not any(v) for v in pairs.values() if len(v) == 2),
            "citation_completeness": float(np.mean(completeness))
            if completeness
            else None,
            "answer_coverage": sum(
                len(v) for k, v in modes.items() if k != "clarification"
            )
            / len(traces),
            "selective_risk": (
                1
                - float(
                    np.mean(
                        [x for k, v in modes.items() if k != "clarification" for x in v]
                    )
                )
            )
            if any(k != "clarification" for k in modes)
            else None,
            "answer_modes": {k: len(v) for k, v in modes.items()},
            **{
                k + "_accuracy_percent": 100 * float(np.mean(v))
                for k, v in categories.items()
            },
            **{k + "_seconds": float(np.mean(v)) for k, v in stage_values.items()},
            "prompt_tokens": float(
                np.mean([sum(c["prompt_tokens"] for c in t["ledger"]) for t in traces])
            ),
            "generated_tokens": float(
                np.mean(
                    [sum(c["generated_tokens"] for c in t["ledger"]) for t in traces]
                )
            ),
            "answer_accuracy_percent": 100 * float(np.mean(list(correctness.values()))),
            "paired_accuracy_percent": 100
            * float(np.mean([all(v) for v in pairs.values() if len(v) == 2]))
            if any(len(v) == 2 for v in pairs.values())
            else None,
            "unsupported_claim_rate": unsupported / claims if claims else None,
            "citation_correctness": valid_links / links if links else None,
            "applicability_accuracy": float(np.mean(application))
            if application
            else None,
            "conflict_resolution_accuracy": float(np.mean(conflicts))
            if conflicts
            else None,
            "valid_branch_suppression": float(np.mean(suppressed))
            if suppressed
            else None,
            "tokens": float(np.mean(token_cost)),
            "total_seconds": float(np.mean(latency)),
            "model_seconds": float(
                np.mean([sum(c["seconds"] for c in t["ledger"]) for t in traces])
            ),
        }
        for dataset, values in datasets.items():
            result[dataset + "_accuracy_percent"] = 100 * float(np.mean(values))
        summary.append(result)
    save(args.output, {"summary": summary})


def main():
    parser = argparse.ArgumentParser(prog="scoperag")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--source",
        choices=["versionqa", "confact", "conflictbank", "manifest"],
        required=True,
    )
    prep.add_argument("--input")
    prep.add_argument("--count", type=int, default=1000)
    prep.add_argument("--output", required=True)
    fetch = sub.add_parser("fetch-documents")
    fetch.add_argument("--sources", default="configs/sources.json")
    fetch.add_argument("--output", required=True)
    diagnostic = sub.add_parser("build-diagnostics")
    diagnostic.add_argument("--index", required=True)
    diagnostic.add_argument("--config", default="configs/default.json")
    diagnostic.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    diagnostic.add_argument("--revision", default="main")
    diagnostic.add_argument("--output", required=True)
    indexing = sub.add_parser("index")
    indexing.add_argument("--documents", required=True)
    indexing.add_argument("--encoder", default="intfloat/e5-base-v2")
    indexing.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--annotations", required=True)
    training.add_argument("--encoder", default="microsoft/deberta-v3-base")
    training.add_argument("--revision", default="main")
    training.add_argument("--config", default="configs/default.json")
    training.add_argument("--output", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--tasks", required=True)
    execution.add_argument("--index", required=True)
    execution.add_argument("--heads", required=True)
    execution.add_argument("--method", choices=METHODS, default="scoperag")
    execution.add_argument(
        "--split", choices=["validation", "development", "test"], default="test"
    )
    execution.add_argument("--top-k", type=int, default=20)
    execution.add_argument("--config", default="configs/default.json")
    execution.add_argument("--verifier", default="cross-encoder/nli-deberta-v3-base")
    execution.add_argument("--oracle-scopes")
    execution.add_argument("--output", required=True)
    for item in (indexing, execution):
        item.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
        item.add_argument("--revision", default="main")
    for item in (indexing, training, execution):
        item.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        item.add_argument("--seed", type=int, default=17)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--inputs", nargs="+", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--judgments")
    evaluation.add_argument("--judge-model")
    evaluation.add_argument("--output", required=True)
    audit = sub.add_parser("judge-citations")
    audit.add_argument("--traces", required=True)
    audit.add_argument("--index", required=True)
    audit.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    audit.add_argument("--revision", default="main")
    audit.add_argument("--output", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "fetch-documents":
        from .sources import download_documents

        download_documents(args.sources, args.output)
    elif args.command == "build-diagnostics":
        from .diagnostics import build

        build(args)
    elif args.command == "index":
        index(args)
    elif args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)
    elif args.command == "judge-citations":
        from .audit import judge_citations

        judge_citations(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
