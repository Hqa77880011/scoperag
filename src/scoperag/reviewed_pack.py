"""Materialize/check a small explicitly source-reviewed pack without model downloads."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from .domain import Domain, parsed_scope
from .method import infer
from .sources import TextParser
from .utils import load, save

SPLITS = {
    "PyTorch": "train",
    "PostgreSQL": "train",
    "NumPy": "development",
    "Node.js": "test",
    "Apache Spark": "test",
}


def validate(pack, registry):
    """Reject unreviewed inputs; this command never upgrades review status."""
    if pack["historical_benchmark"] or "Codex" not in pack["reviewer"]:
        raise ValueError("This pack must identify the new Codex source review")
    sources = {row["id"]: row for row in registry}
    if {f["source_id"] for f in pack["families"]} != set(sources):
        raise ValueError("Pack must bind all five registered official sources")
    for family in pack["families"]:
        if family.get("review_status") != "reviewed" or any(
            c.get("review_status") != "reviewed" for c in family["cases"]
        ):
            raise ValueError(
                "Source facts and case labels require explicit source review"
            )
        if (
            family["split"] != SPLITS[family["product_family"]]
            or sources[family["source_id"]]["product_family"]
            != family["product_family"]
        ):
            raise ValueError("Keep the declared product-family split")
        if (
            family["source_span"] not in family["excerpt"]
            or not family["review_rationale"]
        ):
            raise ValueError("Missing source-span binding or reviewer rationale")
        domain = Domain(family["domain"], 8)
        if len(domain.configurations) != 2:
            raise ValueError(
                "This reviewed micro-pack has exactly two declared cells per family"
            )
        axis = family["axis"]
        if family["domain"][axis] != [family["supported"], family["outside"]]:
            raise ValueError(
                "Manual cell indices must preserve the reviewed domain order"
            )
        if {case["kind"] for case in family["cases"]} != {
            "specified",
            "missing",
            "outside",
        }:
            raise ValueError("Reviewed specified/missing/outside cases are required")
    return sources


def capture(pack, registry, output):
    """Fetch actual official pages and locate reviewed excerpts in normalized text."""
    sources = validate(pack, registry)
    captures = {}
    for family in pack["families"]:
        source = sources[family["source_id"]]
        url = family.get("capture_url", source["url"])
        request = urllib.request.Request(
            url, headers={"User-Agent": "ScopeRAG-source-review/0.1"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            parser = TextParser()
            parser.feed(
                response.read().decode(
                    response.headers.get_content_charset() or "utf-8"
                )
            )
            text = " ".join(" ".join(parser.parts).split())
            start = text.find(family["excerpt"])
            if start < 0:
                raise ValueError(
                    "Reviewed excerpt no longer matches live source: " + url
                )
            if (
                family["product_family"] == "Node.js"
                and ("v" + family["domain"]["version"][0]) not in text
            ):
                raise ValueError(
                    "Node's latest-v22 page changed release; review its binding again"
                )
            fact_start = text.find(family["source_span"], start)
            metadata = dict(
                source_url=source["url"],
                capture_url=url,
                effective_url=response.geturl(),
                http_status=response.status,
                captured_at_utc=datetime.now(timezone.utc).isoformat(),
                normalization="HTML text with script/style removed; whitespace collapsed",
                normalized_text_length=len(text),
                excerpt_char_span=[start, start + len(family["excerpt"])],
                fact_char_span=[fact_start, fact_start + len(family["source_span"])],
            )
        captures[family["source_id"]] = dict(metadata=metadata, text=text)
        save(output, captures)
        print("Captured and source-bound: " + family["source_id"], flush=True)
    return captures


def materialize(pack, registry, output, captures=None):
    """Write standard files from literal reviewed facts/cases, never solver-derived gold."""
    sources = validate(pack, registry)
    output = Path(output)
    documents, facts, tasks, labels, annotations = [], [], [], [], []
    review = dict(
        reviewer=pack["reviewer"],
        review_date=pack["review_date"],
        data_kind="source_reviewed_micro",
        historical_benchmark=False,
    )
    for family in pack["families"]:
        identifier = family["source_id"]
        source = sources[identifier]
        metadata = (
            captures[identifier]["metadata"] if captures else family.get("capture")
        )
        if metadata is None:
            raise ValueError("Capture the sources before binding this new pack")
        if metadata["source_url"] != source["url"]:
            raise ValueError("Capture provenance must bind the registered source URL")
        if captures:
            text = captures[identifier]["text"]
            begin, end = metadata["excerpt_char_span"]
            if text[begin:end] != family["excerpt"]:
                raise ValueError("Captured source span disagrees with reviewed excerpt")
        documents.append(
            dict(
                id=identifier,
                text=family["excerpt"],
                url=source["url"],
                product_family=family["product_family"],
                section=family["section"],
                capture=metadata,
                review_status=family["review_status"],
                **review,
            )
        )
        allowed = {
            key: values
            for key, values in family["domain"].items()
            if key != family["axis"] and "unspecified" not in values
        }
        allowed[family["axis"]] = [family["supported"]]
        fact = dict(
            id=identifier + ":reviewed_fact",
            chunk_id=identifier + ":0",
            product_family=family["product_family"],
            split=family["split"],
            assertion=family["assertion"],
            qualifier=family["qualifier"],
            source_span=family["source_span"],
            url=source["url"],
            capture=metadata,
            scope={"allowed": allowed},
            parsed=True,
            residual=[],
            values=family["values"],
            review_rationale=family["review_rationale"],
            review_status=family["review_status"],
            **review,
        )
        facts.append(fact)
        domain = Domain(family["domain"], 8)
        for case in family["cases"]:
            scope = (
                {}
                if case["kind"] == "missing"
                else {
                    "allowed": {
                        family["axis"]: [
                            family["supported"]
                            if case["kind"] == "specified"
                            else family["outside"]
                        ]
                    }
                }
            )
            task = dict(
                id=identifier + ":" + case["kind"],
                question=case["question"],
                scope=scope,
                domain=family["domain"],
                slots=list(family["values"]),
                known_prerequisites=[],
                document_ids=[identifier],
                dataset="source_reviewed_micro",
                product_family=family["product_family"],
                split=family["split"],
                fact_ids=[fact["id"]],
                review_status=case["review_status"],
                **review,
            )
            # Status, cell IDs, values and applicability are manual reviewed declarations.
            branches = (
                [
                    dict(
                        conditions=[
                            domain.configurations[i] for i in case["gold_cells"]
                        ],
                        values=family["values"],
                    )
                ]
                if case["gold_cells"]
                else []
            )
            label = dict(
                id=task["id"],
                category=case["kind"],
                product_family=family["product_family"],
                split=family["split"],
                structured_answer=dict(
                    status=case["gold_status"],
                    branches=branches,
                    unanswered=[
                        domain.configurations[i] for i in case["gold_unanswered"]
                    ],
                ),
                applicability={fact["id"]: case["gold_applicability"]},
                source_fact_ids=[fact["id"]],
                review_rationale=family["review_rationale"],
                scope_note=pack["scope_note"],
                review_status=case["review_status"],
                **review,
            )
            tasks.append(task)
            labels.append(label)
            if family["split"] != "test":
                annotations.append(
                    dict(
                        query=task["question"],
                        fact=fact["assertion"],
                        qualifier=fact["qualifier"],
                        source_span=fact["source_span"],
                        url=fact["url"],
                        slot=task["slots"][0],
                        slot_support=1,
                        applicability=case["gold_applicability"],
                        product_family=family["product_family"],
                        split=family["split"],
                        source_fact_id=fact["id"],
                        review_status=case["review_status"],
                        **review,
                    )
                )
    for name, rows in (
        ("documents", documents),
        ("facts", facts),
        ("tasks", tasks),
        ("labels", labels),
        ("annotations", annotations),
    ):
        save(output / (name + ".json"), rows)
    return documents, facts, tasks, labels


def canonical(answer):
    return dict(
        status=answer["status"],
        branches=sorted(
            [
                json.dumps(
                    dict(
                        conditions=sorted(
                            b["conditions"], key=lambda x: json.dumps(x, sort_keys=True)
                        ),
                        values=b["values"],
                    ),
                    sort_keys=True,
                )
                for b in answer["branches"]
            ]
        ),
        unanswered=sorted(
            [json.dumps(x, sort_keys=True) for x in answer["unanswered"]]
        ),
    )


def check(pack, registry, output, captures=None, config="configs/default.json"):
    """Check source spans, family splits, manual Domain labels and exact search agreement."""
    documents, facts, tasks, labels = materialize(pack, registry, output, captures)
    indexed = {fact["id"]: fact for fact in facts}
    docs = {doc["id"]: doc for doc in documents}
    cfg = load(config)
    traces = []
    for task, label in zip(tasks, labels):
        candidates = [indexed[identifier] for identifier in task["fact_ids"]]
        fact = candidates[0]
        assert fact["source_span"] in docs[fact["chunk_id"].rsplit(":", 1)[0]]["text"]
        assert not parsed_scope(fact["scope"])[1]
        domain = Domain(task["domain"], 8)
        application = domain.applicability(
            domain.mask(task["scope"]), domain.mask(fact["scope"])
        )
        assert application == label["applicability"][fact["id"]], task["id"]
        # Reviewed slot support isolates the finite-domain/search check; it is not a learned prediction.
        scores = [
            dict(
                semantic_probabilities=[1.0, 0.0, 0.0],
                slot_scores=[1.0] * len(task["slots"]),
            )
        ]
        answers = []
        for method in ("scoperag", "exact_atomic"):
            answer, _, _ = infer(
                task, candidates, scores, cfg, method, lambda s: len(s.split())
            )
            assert canonical(answer) == canonical(label["structured_answer"]), (
                task["id"],
                method,
            )
            assert answer["applicability"] == label["applicability"]
            answers.append(answer)
            traces.append(
                dict(
                    id=task["id"],
                    method=method,
                    dataset=task["dataset"],
                    seed=17,
                    query=task,
                    answer=answer,
                    ledger=[],
                    support_source="reviewed_slot_support_for_search_check",
                    historical_benchmark=False,
                )
            )
        assert canonical(answers[0]) == canonical(answers[1])
    invalid = copy.deepcopy(pack)
    invalid["families"][0]["cases"][0]["review_status"] = "unreviewed"
    try:
        validate(invalid, registry)
    except ValueError as error:
        assert "review" in str(error)
    else:
        raise AssertionError("An unreviewed case was accepted")
    result = dict(
        purpose=pack["purpose"],
        reviewer=pack["reviewer"],
        historical_benchmark=False,
        official_sources=len(documents),
        source_bound_facts=len(facts),
        reviewed_cases=len(tasks),
        solver_checks=len(traces),
        domain_classes=["entailed", "unknown", "inconsistent"],
        family_split=SPLITS,
        live_capture_verified=captures is not None,
        unreviewed_gate_preserved=True,
        model_downloads=0,
        support_source="reviewed_slot_support_for_search_check",
    )
    save(Path(output) / "traces.json", traces)
    save(Path(output) / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="configs/source_reviewed_pack.json")
    parser.add_argument("--sources", default="configs/sources.json")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="outputs/source-reviewed")
    parser.add_argument("--capture-live", action="store_true")
    args = parser.parse_args()
    pack, registry = load(args.pack), load(args.sources)
    captures = (
        capture(pack, registry, Path(args.output) / "captures.json")
        if args.capture_live
        else None
    )
    check(pack, registry, args.output, captures, args.config)


if __name__ == "__main__":
    main()
