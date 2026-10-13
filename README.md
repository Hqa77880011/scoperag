# ScopeRAG

Implementation of **ScopeRAG: Joint Evidence Applicability for Conditional Technical Question Answering**.

ScopeRAG retains source qualifiers, distinguishes entailed/inconsistent/unknown applicability, and propagates a shared configuration domain while selecting evidence. A feasible bundle produces a direct answer only when the user's context entails every prerequisite; otherwise it produces explicit branches or a clarification.

## Installation and sources

Use Python 3.11 and PyTorch for your hardware. From an activated virtual environment:

```bash
python -m pip install -e .
git clone https://github.com/danielhuwiler/versionrag data/raw/versionrag
scoperag prepare --source versionqa --input data/raw/versionrag --output data/versionqa
```

The [VersionQA](https://github.com/danielhuwiler/versionrag) adapter imports original questions and answers separately and extracts released documentation text. PDF documentation requires `PyMuPDF` (`python -m pip install PyMuPDF`). Source artifacts retain their release paths, versions and capture dates.

Use the direct import commands below for ConflictBank and CONFACT, or `prepare --source manifest --input pack.json --output data/custom` for a curated source-grounded pack. `pack.json` contains separate `tasks`, `labels` and `documents` arrays. Documents have `id`, `text`, `url`, `capture_date`, and optional `version`. Tasks have `id`, `question`, `dataset`, `split`, and optionally `scope`, `domain`, `slots`, `known_prerequisites`. Keep native questions and labels; a source-constrained sample must preserve its original IDs.

For ScopeDiag, product families remain disjoint: training PyTorch/PostgreSQL, development NumPy, and test Node.js/Apache Spark. Store paired questions with an evaluator-only `pair_id`; keep both members together. Source-verified applicability and slot labels are required for head training, rather than treating arbitrary version substitutions as verified negatives.

## Build the source diagnostic

```bash
scoperag fetch-documents --sources configs/sources.json --output data/technical_documents.json
scoperag index --documents data/technical_documents.json --output data/technical_index
scoperag build-diagnostics --index data/technical_index --output data/scopediag
```

The source registry covers PyTorch, PostgreSQL, NumPy, Node.js and Apache Spark. The authoring command creates 200 questions with product-isolated 80/40/80 splits, intact condition-changing pairs, missing-condition questions and ordinary controls. A separate exhaustive reference calculation derives labels from source scopes, independent of the learned selector. The exact quotations, scope labels and slot supervision remain inspectable in `labels.json`, `annotations.json` and the index; check their interpretation before fixing the cohort. Question wording is generated from the indexed source material. This is a new diagnostic cohort with the paper's experiment structure.

## Indexing and head training

```bash
scoperag index --documents data/versionqa/documents.json --output data/index
scoperag train --annotations data/scopediag/annotations.json --output checkpoints/heads
```

Indexing uses Qwen2.5-7B-Instruct to extract at most two facts per chunk and E5-base-v2 embeddings with `query:`/`passage:` prefixes. An exact source-span check rejects ungrounded quotations. Facts retain assertion, original qualifier, URL, capture date, parsed scope, unresolved textual prerequisites and asserted slot values. `index.json` stores chunks and facts; `embeddings.npy` stores normalized dense vectors. Pin the generator with `--revision`; `--encoder` chooses the retriever.

Training annotations contain `query`, `fact`, optional `qualifier`, optional `slot`, `product_family`, `split`, and `applicability` (`entailed`, `inconsistent`, `unknown`) and/or binary `slot_support`. Labels must come from source-verified conditions. Training isolates product families and fits a shared DeBERTa-v3-base encoder with three-way applicability and slot-support heads. The default five epochs use AdamW at 0.00002 and batch 16. Development applicability macro-F1 selects `heads.pt`; `model.json` records the encoder revision and configuration. Use seeds 17, 29 and 43 for independently fitted heads.

## Scope representation

Five finite fields are `version`, `platform`, `hardware`, `backend` and `settings`. A query `domain` lists the possible values of every field. A scope supports:

- `allowed`: field → permitted value set.
- `version_range`: numeric tuple bounds, with an inclusive minimum and an exclusive maximum unless `max_inclusive` is true.
- `relation`: allowed complete or partial configurations, preserving coupled conditions.
- `implications`: `if` and `then` field predicates.

The configuration domain is explicitly enumerated and capped at 100,000 cells. Omitted query fields stay unresolved. When task parsing is needed, finite values are derived from retrieved source scopes and document releases; `unspecified` represents an unobserved categorical domain. For precise diagnostics, provide the intended finite domain directly in the task pack.

## Run and evaluate

```bash
scoperag run --tasks data/versionqa/tasks.json --index data/index --heads checkpoints/heads --method scoperag --output runs/scoperag
scoperag evaluate --inputs runs/scoperag/traces.json --labels data/versionqa/labels.json --judge-model Qwen/Qwen2.5-7B-Instruct --output runs/metrics.json
scoperag plot --input runs/metrics.json --output runs/figures
```

Each query uses the same dense top-20 candidate pool unless `--top-k` changes it. `configs/default.json` sets the 3,000-token evidence cap, six-fact limit, beam width 16, three retained bundles, eight explicit output cells and 512-token answer cap. Head/verifier thresholds are 0.80/0.90. Objective penalties are 0.15 for unknown applicability, 0.30 for weak support and 0.05 for duplicate slot-value coverage.

`traces.json` stores parsed queries, candidate IDs, applicability decisions, branch configurations, atomic claims, citations, verification outcomes and model-token ledgers. Opposing values in one branch are rejected. Residual opaque prerequisites cause clarification. The NLI verifier checks original source support while deterministic inclusion checks the cited scope. Rejected statements are removed and the output is reduced to clarification when no supported claims remain.

Structured diagnostic labels use `structured_answer` with `status`, `branches`, explicit `conditions` and asserted `values`. `evaluate` compares these exact fields. Free-text public tasks use a separately loaded judge through `--judge-model`; its model should be named with the scores. Optional evaluator labels include fact-level `applicability`, `same_scope_conflict`, and `eligible_branches`.

For unsupported-claim and citation metrics, supply `--judgments judgments.json` with independently reviewed `id`, `claim_supported` and `citation_valid` boolean arrays. These labels are never runtime inputs. Output metrics cover answer accuracy per dataset, paired accuracy, unsupported rate, citation correctness, applicability accuracy, conflict resolution, valid-branch suppression, token cost and measured model time. Use the independent citation-audit command below to create these evaluation inputs, or supply reviewed labels in the same format.

## Comparisons

`run --method` supports dense or BM25 retrieval, latest-first, credibility-first, version/platform filtering, fact NLI, fixed-pool VersionRAG, FaithfulRAG/ProbeRAG selection adaptations, `individual`, `all_conditions` and `exact_atomic`. The structured individual comparator omits domain propagation during selection and cannot rerun selection after output rejection. Exact atomic filtering is an equivalence reference on exhaustive parsed cells.

The FaithfulRAG adaptation produces parametric facts and aligns them with the fixed contextual pool before selection. The ProbeRAG adaptation performs an explicit conflict probe and prunes that pool. The fixed-pool VersionRAG control prioritizes requested releases and slot support; it does not rebuild the original version graph. All-condition prompting retains every fact that fits the evidence-token cap. These mechanisms use the same source qualifiers and final commitment checks.

Component removals are `no_platform`, `no_uncertainty`, `no_joint` and `no_verifier`. `human_scope --oracle-scopes scopes.json` supplies evaluator-verified source scopes to diagnose extraction errors. Candidate sweeps use `--top-k` consistently across methods.

The implementation uses finite configuration domains and slot-wise support probabilities. Its source extraction, local public-task judge and bounded selection adaptations define the operational comparison. Diagnostic source annotations and citation judgments are stored separately from runtime inputs.

## Public sources and all metrics

```bash
scoperag prepare --source conflictbank --count 1000 --output data/conflictbank
scoperag prepare --source confact --output data/confact
scoperag index --documents data/conflictbank/documents.json --output data/conflictbank_index
scoperag index --documents data/confact/documents.json --output data/confact_index
scoperag run --tasks data/scopediag/tasks.json --index data/technical_index --heads checkpoints/heads --output runs/scopediag
scoperag judge-citations --traces runs/scopediag/traces.json --index data/technical_index --model Qwen/Qwen2.5-14B-Instruct --output runs/scopediag/judgments.json
scoperag evaluate --inputs runs/scopediag/traces.json --labels data/scopediag/labels.json --judgments runs/scopediag/judgments.json --output runs/scopediag/metrics.json
```

ConflictBank sampling uses a fixed seed and a 334/333/333 misinformation/temporal/semantic allocation. CONFACT retains the complete HumC source pack. Retrieval is limited to each public task's own evidence documents, preventing evidence from another question entering its pool. Replace the task/index/label paths in `run` and `evaluate` to evaluate those cohorts. Free-text public answers use `--judge-model`.

| Analysis | Output |
| --- | --- |
| Public and diagnostic answers | Dataset accuracy and category accuracy |
| Condition-changing pairs | Paired accuracy, both/single/neither counts, grouped separately by seed |
| Answer conditions | Applicability accuracy, same-scope conflict resolution, missing/ordinary accuracy, valid-branch suppression |
| Claims and citations | Unsupported rate, scope-valid citation correctness, citation completeness |
| Coverage | Direct/conditional/clarification counts, substantive answer coverage and selective risk |
| Efficiency | Input/output/total tokens; retrieval, parsing, scoring, selection, generation, verification and total elapsed time |

For same-scope conflict and other curated mechanism subsets, use evaluator-only `same_scope_conflict` and `categories` fields in `labels.json`. Citation auditing uses a separate model and saves its judgments and token ledger for review. To combine judgments from several methods/seeds, retain their `id`, `method` and `seed` keys. Candidate sweeps change `--top-k`; evaluation keeps each pool size separate. `plot` produces PNG and SVG outputs.
