# Cloud Training Prep

Date: March 31, 2026

For the operational TinyStories workflow, start with [FIRST_REAL_TINYSTORIES_RUN.md](./FIRST_REAL_TINYSTORIES_RUN.md) and [INTERPRETING_FIRST_RESULTS.md](./INTERPRETING_FIRST_RESULTS.md).

## Local Machine Reality

An i5-8210Y with integrated graphics is not a serious training environment for this project.

It is still useful for:

- repository scaffolding,
- config and artifact design,
- dataset manifesting,
- tokenization pipeline bring-up,
- benchmark adapter development,
- pure Python smoke tests,
- and preflight validation.

It is not useful for:

- meaningful PyTorch model training,
- long-context throughput testing,
- MoE performance work,
- or large benchmark sweeps.

## What Should Be Finished Locally Before Cloud

- architecture spec and roadmap
- package layout
- config presets
- dataset manifest schema
- tokenizer interface
- corpus preparation scripts
- benchmark adapters
- preflight environment check
- training entrypoints and artifact layout

## What Still Requires Cloud

- installing full ML dependencies
- training dense and hybrid baselines
- profiling recurrent and MoE kernels
- long-context evaluation at realistic batch sizes
- ablation sweeps
- distributed orchestration

## Recommended Cloud Sequence

1. Provision a small GPU instance first, not a large cluster.
2. Install the package and dependencies.
3. Run the dense baseline for a tiny synthetic smoke test.
4. Run the hybrid model for the same smoke test.
5. Move to a tiny real-data TinyStories run.
6. Confirm train and validation metrics plus checkpoints are written correctly.
7. Only then increase run length or model size.

## Cloud Readiness Checklist

- `python scripts/preflight_check.py`
- `python scripts/run_benchmark_recall.py`
- `python scripts/run_benchmark_verifier.py`
- install `torch`
- run `python scripts/train_dense_baseline.py --steps 2`
- run `python scripts/train_hybrid.py --steps 2`
- run the TinyStories smoke flow from `docs/FIRST_REAL_TINYSTORIES_RUN.md`

## Practical Rule

Do not spend cloud money discovering local packaging bugs.
