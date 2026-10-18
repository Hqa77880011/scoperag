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

```bash
git clone https://github.com/danielhuwiler/versionrag data/raw/versionrag
scoperag prepare --source versionqa --input data/raw/versionrag --output data/versionqa
scoperag fetch-documents --sources configs/sources.json --output data/technical_documents.json
scoperag index --documents data/technical_documents.json --output data/technical_index
scoperag build-diagnostics --index data/technical_index --output data/scopediag
scoperag index --documents data/versionqa/documents.json --output data/index
```

`prepare --source conflictbank --count 1000 --output data/conflictbank` and `prepare --source confact --output data/confact` import the other supported sources. Index each prepared `documents.json` separately. Custom packs use `--source manifest --input pack.json` with separate tasks, documents and labels.

## Training and inference

Local encoders, verifier and language model download Hugging Face weights. `configs/default.json` sets training and generation budgets; use `--revision` where supported to pin model revisions. Keep the fitting and execution seeds explicit.

```bash
scoperag train --annotations data/scopediag/annotations.json --config configs/default.json --seed 17 --output checkpoints/heads
scoperag run --tasks data/versionqa/tasks.json --index data/index --heads checkpoints/heads --method scoperag --seed 17 --output runs/scoperag
```

## Evaluation

```bash
scoperag evaluate --inputs runs/scoperag/traces.json --labels data/versionqa/labels.json --judge-model Qwen/Qwen2.5-7B-Instruct --output runs/metrics.json
scoperag plot --input runs/metrics.json --output runs/figures
```

## Ablations

Use the same tasks/index/heads and seed with `run --method no_platform`, `no_uncertainty`, `no_joint` or `no_verifier`, saving distinct output directories. `--top-k` controls the candidate pool. Optional citation auditing uses a separately named judge.

```bash
scoperag judge-citations --traces runs/scoperag/traces.json --index data/index --model Qwen/Qwen2.5-14B-Instruct --output runs/judgments.json
scoperag evaluate --inputs runs/scoperag/traces.json --labels data/versionqa/labels.json --judgments runs/judgments.json --output runs/audited_metrics.json
```

## Outputs

Data preparation writes `tasks.json`, `documents.json` and separate `labels.json`. Diagnostic construction writes `annotations.json`; indexing writes its index files. Training saves `heads.pt`, tokenizer files, `model.json` with config/seed, and `history.json`. Runs save `traces.json`; evaluation and citation auditing write the specified JSONs, and plotting produces PNG/SVG.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
