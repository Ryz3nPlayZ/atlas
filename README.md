# ACE-Atlas

ACE-Atlas is a research implementation of a `hybrid recurrent-memory-MoE` language model with verifier-driven reasoning.

The v1 thesis is narrow:

- keep exact local attention where recall matters most,
- move long-range context handling into a cheaper recurrent path,
- use sparse experts for active capacity,
- add bounded learned memory instead of brute-force context stuffing,
- and escalate to executable reasoning only when the model is uncertain.

This repository is being built in phases. The current state is:

- detailed architecture and roadmap docs,
- a first-pass Python package scaffold,
- bootstrap model modules for the backbone, memory, arbiter, and tools,
- bootstrap training entrypoints and benchmark adapters.

## Repository Layout

```text
ace-atlas/
  docs/
  scripts/
  src/ace_atlas/
  tests/
```

## Documents

- [Architecture Spec](docs/ARCHITECTURE_SPEC.md)
- [Execution Roadmap](docs/ROADMAP.md)
- [Cloud Training Prep](docs/CLOUD_PREP.md)

## Current Implementation Focus

The first implementation target is not the full research agenda. It is:

- hybrid backbone,
- bounded memory,
- memory arbiter,
- verifier path,
- benchmark harness.

Deferred research bets include:

- full online plasticity,
- NCA as the main backbone,
- HDC as the main representation family,
- full active inference control.

## Status

The environment used to create this scaffold does not have `torch` installed, so the code has only been syntax-checked, not executed end-to-end.

The following scripts run without `torch`:

- `scripts/run_benchmark_recall.py`
- `scripts/run_benchmark_verifier.py`
- `scripts/build_manifest.py`
- `scripts/tokenize_corpus.py`
- `scripts/preflight_check.py`

The following training scripts require `torch` and print a clear message if it is missing:

- `scripts/train_dense_baseline.py`
- `scripts/train_hybrid.py`
