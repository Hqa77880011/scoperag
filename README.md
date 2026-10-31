# ScopeRAG

Implementation of **ScopeRAG: Joint Evidence Applicability for Conditional Technical Question Answering**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Import [VersionQA](https://github.com/danielhuwiler/versionrag), then fetch the technical sources listed in `configs/sources.json`. The fetcher records captured document text and provenance. Diagnostic annotations are generated declarations and need source-binding review before fitting.

`build-diagnostics` marks generated annotations and gold labels as `review_status: unreviewed`. Bind each assertion, qualifier and target answer to its captured source, correct it, and mark reviewed records `review_status: reviewed` before training or evaluation. The commands reject unreviewed records. A generated declaration is not an independently verified label. Keep product families in the declared train/development/test splits; do not tune thresholds using the test labels.

```bash
git clone https://github.com/danielhuwiler/versionrag data/raw/versionrag
scoperag prepare --source versionqa --input data/raw/versionrag --output data/versionqa
scoperag fetch-documents --sources configs/sources.json --output data/technical_documents.json
scoperag index --documents data/technical_documents.json --output data/technical_index
scoperag build-diagnostics --index data/technical_index --output data/scopediag
scoperag index --documents data/versionqa/documents.json --output data/index
```

### Small source-reviewed pack

`configs/source_reviewed_pack.json` is a new, explicitly **Codex source-reviewed** micro-pack. It is not human annotation or the paper's historical ScopeDiag data. On 2026-11-01, all five registered official pages were fetched successfully and their selected excerpts were checked against the original text. Capture records retain the registry URL, effective URL, UTC time, normalized-page character offsets and the selected source span. PyTorch's moving stable link resolved to 2.15, so this pack pins that documentation; Node's v22 capture was checked as v22.23.4.

| Official source | Reviewed fact and retained restriction | Split |
| --- | --- | --- |
| [PyTorch CUDA](https://docs.pytorch.org/docs/2.15/notes/cuda.html#asynchronous-execution) | GPU work is asynchronous under the documented default execution settings. | train |
| [PostgreSQL 16](https://www.postgresql.org/docs/16/parallel-safety.html) | An unmarked user-defined function defaults to parallel UNSAFE; explicit marking is an exception. | train |
| [NumPy 2.0.0](https://numpy.org/doc/2.0/release/2.0.0-notes.html) | The Windows default integer dtype changed to int64. | development |
| [Node.js v22 workers](https://nodejs.org/docs/latest-v22.x/api/worker_threads.html) | Worker threads can help CPU-intensive JavaScript; this fact does not assert an I/O prohibition or measured speedup. | test |
| [Spark SQL 3.5.6](https://spark.apache.org/docs/3.5.6/sql-performance-tuning.html#join-strategy-hints-for-sql-queries) | A supplied join-strategy hint does not guarantee the chosen strategy. | test |

Each fact has three explicitly reviewed cases: the stated condition, a missing condition, and a configuration outside that fact's declared scope. Gold status, applicability and finite-domain cell indices are declarations in the checked-in pack, not outputs of `reference_answer`, a generator or the solver under test. Outside cases concern this selected reviewed fact pool: they do not assert the opposite technical answer, or that the complete official page lacks other relevant information. No generated/unreviewed annotation is automatically marked reviewed.

```bash
python -m scoperag.reviewed_pack --output outputs/source-reviewed
python -m scoperag.reviewed_pack --capture-live --output outputs/source-reviewed-live
```

The first command uses the checked-in reviewed excerpts and capture provenance; the second re-fetches all five official pages and checks the excerpt/character bindings against actual captured text. A changed Node release or missing excerpt fails for review. Both commands write standard `documents.json`, `facts.json`, `tasks.json`, `labels.json`, train/development-only `annotations.json`, search traces and `verification.json`. Live captures remain under ignored `outputs/`. The original generated-data review gate remains in `train` and `evaluate`; this module also rejects an unreviewed fact or case before writing outputs.

Actual CPU checks passed for five source spans, five facts, 15 manually declared cases and 30 ScopeRAG/exact_atomic comparisons, covering all three Domain applicability states, conditional branches and unanswered cells. The product-family split is unchanged. These checks supply reviewed slot support to isolate source binding and finite-domain selection; they do not measure learned-head accuracy, retrieval, generation or independent citation judging. There are no model downloads. CI repeats this small offline check with NumPy only.

`prepare --source conflictbank --count 1000 --output data/conflictbank` and `prepare --source confact --output data/confact` import the other supported sources. Index each prepared `documents.json` separately. Custom packs use `--source manifest --input pack.json` with separate tasks, documents and labels.

## Training and inference

Local encoders, verifier and language model download Hugging Face weights. `configs/default.json` sets training and generation budgets; use `--revision` where supported to pin model revisions. Keep the fitting and execution seeds explicit.

The applicability and slot-support heads share the encoder. Training sums cross entropy for entailed/inconsistent/unknown and binary cross entropy for slot support. Saved heads include the encoder configuration and tokenizer, so reloading does not require the original encoder download path. `--device cpu` is supported for small checks; the default models require substantially more memory.

```bash
scoperag train --annotations data/scopediag/annotations.json --config configs/default.json --seed 17 --output checkpoints/heads
scoperag run --tasks data/versionqa/tasks.json --index data/index --heads checkpoints/heads --method scoperag --seed 17 --output runs/scoperag
```

ScopeRAG represents version, platform, hardware, backend and settings as a finite coupled domain. Slot scores represent joint applicability/support; max-over-fact coverage is not an additive count. Selection uses lambda=0.15, mu=0.3, rho=0.05, six facts, beam width 16 and at most three bundles. Unknown fields or unparsed restrictions retain an unresolved status and can require clarification. Conditions are intersected jointly before generation. The evidence cap applies to the complete serialized evidence payload across bundles (3,000 tokens), with a 512-token answer cap; source spans are not silently truncated to fit.

## Evaluation

```bash
scoperag evaluate --inputs runs/scoperag/traces.json --labels data/versionqa/labels.json --judge-model Qwen/Qwen2.5-7B-Instruct --output runs/metrics.json
scoperag plot --input runs/metrics.json --output runs/figures
```

Structured diagnostic answers use exact canonical conditions/values. Free-text benchmarks need the independent answer judge shown above. Runtime errors count as incorrect, including errors whose fallback text requests clarification. Paired accuracy requires both questions correct for the same source pair and seed; missing or duplicate pair observations are rejected. The summary also reports answer accuracy by dataset/category/mode, applicability accuracy, same-scope conflict accuracy, branch suppression, answer coverage/selective risk, inference tokens and stage/total latency. Citation validity, completeness and unsupported-claim rates require independent citation judgments; absent judgments remain unavailable rather than being estimated from the inference verifier.

## Ablations

Use the same tasks/index/heads and seed with `run --method no_platform`, `no_uncertainty`, `no_joint` or `no_verifier`, saving distinct output directories. `--top-k` controls the candidate pool. Optional citation auditing uses a separately named judge.

Other entries are `bm25`, `dense`, `latest`, `credibility`, `version_platform`, `fact_nli`, `faithful`, `probe`, `versionrag`, `individual`, `all_conditions` and `exact_atomic`. They are this implementation's controls/adaptations, rather than bundled upstream reproductions. `individual`/`no_joint` precommit the ranked independent bundles before rejecting joint inconsistencies; they do not search lower-ranked replacements after the rejection. `exact_atomic` is the finite-domain reference and is expensive for large domains. `human_scope` is an evaluator-assisted diagnostic requiring `--oracle-scopes`.

```bash
scoperag judge-citations --traces runs/scoperag/traces.json --index data/index --model Qwen/Qwen2.5-14B-Instruct --output runs/judgments.json
scoperag evaluate --inputs runs/scoperag/traces.json --labels data/versionqa/labels.json --judgments runs/judgments.json --output runs/audited_metrics.json
```

## Outputs

Data preparation writes `tasks.json`, `documents.json` and separate `labels.json`. Diagnostic construction writes `annotations.json`; indexing writes its index files. Training saves `heads.pt`, tokenizer files, `model.json` with config/seed, and `history.json`. Runs save `traces.json`; evaluation and citation auditing write the specified JSONs, and plotting produces PNG/SVG.

| Paper module or metric | Code entry | Configuration/command |
| --- | --- | --- |
| Captured evidence and qualifier provenance | `sources.py`, `cli.index` | `fetch-documents`, `index` |
| Three-way applicability and slot support | `model.py`, `cli.train` | `train --annotations`, `configs/default.json` |
| Coupled scopes and joint bundle search | `domain.py`, `method.infer` | `run --method scoperag`, `exact_atomic` |
| Conditional answer, evidence budget and verification | `method.generate`, `method.verify`, `llm.py` | `evidence_tokens`, `answer_tokens`, `verifier_threshold` |
| Answer/paired/applicability/conflict/coverage/risk metrics | `cli.evaluate` | `evaluate --labels`, optional `--judge-model` |
| Unsupported claims and citation metrics | `audit.py`, `cli.evaluate` | `judge-citations`, `evaluate --judgments` |
| Baselines, removals and plots | `method.METHODS`, `utils.plot_results` | `run --method`, `--top-k`, `plot` |

## Verification

Rechecked on 2026-11-01 using the existing `python tests/smoke.py` with one CPU thread. Optional TensorFlow/Flax backends were disabled (`USE_TF=0`, `USE_FLAX=0`); local test-only dependencies were isolated from the shared Python environment.

The existing implementation was retained and repaired for scope parsing, joint controls, evidence accounting, checkpoint portability and evaluation. Neither its presence nor the new checks establishes provenance for the paper's recorded results.

Run the offline functional check from the repository root:

```bash
python tests/smoke.py
```

Actually checked on Windows, Python 3.13.5, torch 2.7.1+cu118 (CPU execution), NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, transformers 4.57.6 and tokenizers 0.22.1. It creates random tiny BERT/GPT-2 fixtures locally, fits the applicability/support heads for two optimizer steps, checks finite gradients and parameter changes, reloads without the original encoder path, and runs actual causal forward/generation and token limits. Hand-constructed coupled domains check a higher-order joint incompatibility, exact-reference agreement, the independent control, unknown restrictions and evidence limits. Hand-counted paired evaluation checks that an error cannot earn clarification credit. Fixtures and weights stay under ignored `outputs/smoke/`.

The neural smoke fixtures are synthetic functional checks. The separate reviewed micro-pack above adds real source-binding examples. The original full annotation/benchmark pack, full 7B retrieval/indexing/generation/verifier pipeline, benchmark scores, and 512/4,096/65,536-state scaling sweep have not been recovered or rerun. Source collection still depends on the upstream sites and access conditions. No paper performance table is populated from these checks.
