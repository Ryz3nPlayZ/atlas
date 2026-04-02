# Cloud Training Prep

Date: April 1, 2026

This doc is now historical-plus-operational.

The repo is past the original "can it run at all?" stage. The accepted low-budget cloud path is:

- a single T4-class GPU
- real tokenized JSONL training
- same-size dense vs hybrid comparisons
- accepted `300M` hybrid runs via microbatching and accumulation

For the current project state, pair this doc with:

- [Project Dossier](./PROJECT_DOSSIER.md)
- [Roadmap](./ROADMAP.md)
- [Code Continuation Plan](./CODE_CONTINUATION_PLAN.md)

## What A Local Machine Is Still Good For

A weak local machine is still fine for:

- repo edits
- config changes
- dataset prep scripts
- tokenization
- pure Python verification
- benchmark harness development

It is still not the place for:

- meaningful model training
- throughput profiling
- `300M` hybrid runs

## Accepted Low-Budget Cloud Setup

The repo has already been validated on a T4-class path.

Current accepted practical target:

- single T4 GPU
- PyTorch with CUDA
- fp32 training for the accepted `300M` hybrid recipe

## Accepted 300M Hybrid T4 Recipe

The working recipe that fit and completed is:

- batch size: `12`
- microbatch size: `4`
- grad accumulation: `3`
- mixed precision: `none`
- activation checkpointing: `false`

This matters because the naive `300M` hybrid recipe did **not** fit on the T4 until the microbatch/accumulation split was used.

## What The Cloud Path Has Already Proved

Completed on the low-budget path:

- dense and hybrid smoke runs
- real TinyStories training
- same-size comparisons at `50M`, `100M`, and `300M`
- accepted `300M` hybrid checkpoint training
- checkpoint evaluation on WikiText-2 and CodeSearchNet Python
- small HumanEval and MBPP task-style evaluation
- code-focused continuation runs

## What The Cloud Path Has Not Proved

Not yet proven:

- code-task benchmark wins
- justification for scaling past `300M`
- production inference readiness

## Recommended Current Cloud Sequence

If you are reproducing the project from scratch:

1. run the TinyStories prep flow
2. run a small dense vs hybrid comparison
3. confirm checkpoints and evaluation scripts work
4. reproduce the accepted `300M` hybrid recipe
5. evaluate the saved checkpoints
6. only then attempt code-focused continuation

## Practical Rule

Do not spend cloud money on bigger scale until:

- the accepted `300M` result is reproduced cleanly
- and the code-task alignment bottleneck is better understood
