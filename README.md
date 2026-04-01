# ACE-Atlas

ACE-Atlas is a research codebase for a `hybrid recurrent-memory-MoE` language model with verifier-driven reasoning.

The immediate project goal is practical, not speculative:

- run a stable dense baseline,
- run a stable hybrid baseline,
- train both on a real small corpus,
- and compare quality, speed, and training behavior before scaling.

## Current State

The repository now supports the first real-data training milestone:

- dense and hybrid training entrypoints,
- synthetic smoke training for fast bring-up,
- tokenized JSONL training for real corpora,
- config-driven train/validation split paths,
- validation loss logging,
- checkpoint save/resume,
- tokenizer and manifest utilities,
- benchmark smoke adapters for recall and verifier tasks.

What it does not support yet:

- distributed training,
- large-scale data pipelines,
- production inference,
- or large-model orchestration.

## Start Here

If you are trying to get the first non-toy run working, use these in order:

1. [First Real TinyStories Run](docs/FIRST_REAL_TINYSTORIES_RUN.md)
2. [Cloud Training Prep](docs/CLOUD_PREP.md)
3. [Execution Roadmap](docs/ROADMAP.md)

Reference docs:

- [Project Dossier](docs/PROJECT_DOSSIER.md)
- [Architecture Spec](docs/ARCHITECTURE_SPEC.md)

## Repository Layout

```text
ace-atlas/
  configs/
  docs/
  scripts/
  src/ace_atlas/
  tests/
```

## First Real-Data Commands

Prepare TinyStories:

```bash
python -m pip install -e '.[dev,data]'
python scripts/prepare_tinystories.py --output-dir data/tinystories
```

Run the dense baseline:

```bash
python scripts/train_dense_baseline.py \
  --config configs/dense_small.json \
  --train-config configs/train_tinystories_smoke.json \
  --run-name tinystories_dense_smoke
```

Run the hybrid baseline:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_small.json \
  --train-config configs/train_tinystories_smoke.json \
  --run-name tinystories_hybrid_smoke
```

Artifacts are written under `artifacts/<run_name>/` with:

- `run.json`
- `metrics.json`
- `checkpoints/latest.pt`

## Verification

Fast checks that should work before a longer run:

```bash
python scripts/preflight_check.py
python scripts/run_benchmark_recall.py
python scripts/run_benchmark_verifier.py
pytest -q
```
