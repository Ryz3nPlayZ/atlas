# ACE-Atlas

ACE-Atlas is a research codebase for a `hybrid recurrent-memory-MoE` language model aimed at beating strong dense baselines in the same parameter class on a low-budget training path.

## Current Status

As of `April 1, 2026`, the accepted baseline is:

- `~300M` dense baseline
- `~300M` hybrid baseline with a `GRU-fused` recurrent core

The hybrid has beaten the same-size dense baseline on:

- TinyStories training/validation loss
- WikiText-2 held-out evaluation
- CodeSearchNet Python held-out evaluation

What has **not** happened yet:

- task-style code benchmark wins
- larger-than-300M accepted hybrid runs
- production inference or deployment work

## Current Claim

The strongest honest claim supported by the repo today is:

`ACE-Atlas appears to beat same-size dense baselines on cross-domain modeling quality at 50M, 100M, and 300M.`

The strongest honest limitation is:

`That cross-domain loss advantage has not yet turned into executable-code wins on HumanEval or MBPP.`

## Accepted Results

### TinyStories

At roughly matched size:

- `50M`: hybrid beat dense
- `100M`: hybrid beat dense
- `300M`: hybrid beat dense

Accepted `300M` TinyStories result:

- dense final val loss: `2.0015`
- hybrid final val loss: `1.3264`

### WikiText-2 Checkpoint Eval

Fixed `300M` checkpoints evaluated out of domain:

- dense validation loss: `3.7954`
- hybrid validation loss: `3.4819`
- dense test loss: `3.8185`
- hybrid test loss: `3.5030`

### CodeSearchNet Python Checkpoint Eval

Fixed `300M` checkpoints evaluated on held-out Python:

- dense validation loss: `6.3760`
- hybrid validation loss: `5.8332`
- dense test loss: `6.6820`
- hybrid test loss: `6.0925`

### Task-Style Code Benchmarks

Current accepted `300M` checkpoints do **not** yet show executable-code wins:

- HumanEval subset: dense `0/10`, hybrid `0/10`
- MBPP sanitized subset: dense `0/20`, hybrid `0/20`

Code-focused continuation improved output alignment, but not pass rate.

## Why This Project Exists

This repo is not trying to beat frontier models overall on a shoestring budget. The target is narrower and more realistic:

- beat strong models in the same weight class
- improve quality per parameter
- improve quality per dollar
- test whether a hybrid recurrent-memory design can outperform a plain dense baseline without requiring frontier-scale compute

## Start Here

If you want the current project state instead of the historical bring-up path, use these docs in order:

1. [Project Dossier](docs/PROJECT_DOSSIER.md)
2. [Execution Roadmap](docs/ROADMAP.md)
3. [Code Continuation Plan](docs/CODE_CONTINUATION_PLAN.md)
4. [Cloud Training Prep](docs/CLOUD_PREP.md)

Operational docs:

- [First Real TinyStories Run](docs/FIRST_REAL_TINYSTORIES_RUN.md)
- [How To Interpret First Results](docs/INTERPRETING_FIRST_RESULTS.md)
- [Architecture Spec](docs/ARCHITECTURE_SPEC.md)

## Practical Commands

Report parameter counts:

```bash
python scripts/report_model_stats.py --model-name dense_baseline --config configs/dense_300m.json
python scripts/report_model_stats.py --model-name ace_atlas --config configs/hybrid_300m_gru.json
```

Prepare TinyStories:

```bash
python -m pip install -e '.[dev,data]'
python scripts/prepare_tinystories.py --output-dir data/tinystories
```

Run the accepted `300M` hybrid recipe on T4:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_300m_gru.json \
  --train-config configs/train_tinystories_300m_hybrid_t4_fp32_no_ckpt.json \
  --run-name tinystories_hybrid_300m_gru_t4_fp32_no_ckpt
```

Compare two completed runs:

```bash
python scripts/compare_runs.py \
  artifacts/tinystories_dense_300m \
  artifacts/tinystories_hybrid_300m_gru_t4_fp32_no_ckpt
```

Evaluate a checkpoint on WikiText-2 or tokenized JSONL:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint artifacts/tinystories_hybrid_300m_gru_t4_fp32_no_ckpt/checkpoints/latest.pt \
  --config configs/hybrid_300m_gru.json \
  --data artifacts/data/wikitext2_validation_tokens.jsonl
```

## Accepted Presets

Roughly budget-matched config tiers:

- `configs/dense_50m.json`: `48.66M`
- `configs/hybrid_50m_gru.json`: `51.08M`
- `configs/dense_100m.json`: `91.38M`
- `configs/hybrid_100m_gru.json`: `93.19M`
- `configs/dense_300m.json`: `295.83M`
- `configs/hybrid_300m_gru.json`: `295.68M`

## What Needs To Happen Next

Do **not** scale beyond `300M` yet.

The current bottleneck is:

- not basic training stability
- not same-size dense comparisons
- not held-out loss quality

It is:

`turning the hybrid's modeling advantage into executable-code task wins`

The next recommended move is stronger task-aligned code continuation on the accepted `300M` hybrid checkpoint, not more scale.

## Verification

Fast repo checks:

```bash
python scripts/preflight_check.py
python scripts/run_benchmark_recall.py
python scripts/run_benchmark_verifier.py
pytest -q
```
