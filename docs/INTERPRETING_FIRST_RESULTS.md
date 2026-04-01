# How To Interpret First Results

Date: April 1, 2026

This guide is for the first low-budget TinyStories dense vs hybrid comparison.

The goal is not to declare a final architecture winner. The goal is to decide whether the hybrid is earning the next run.

## What Counts As Success

- the hybrid trains stably without exploding loss or repeated resume failures
- the hybrid reaches a validation loss that is competitive with the dense baseline at the same budget tier
- the hybrid shows either:
  - better validation loss, or
  - similar validation loss with better step time or throughput, or
  - similar validation loss with a clearly stronger scaling path worth testing next

## What Counts As Failure

- validation loss is materially worse than dense and stays worse
- the hybrid is unstable across checkpoints or resumes
- the hybrid is much slower without a quality benefit
- the hybrid needs repeated babysitting just to finish a small run

## When To Scale

Scale from `~50M` to `~100M` only if:

- both dense and hybrid complete cleanly
- validation curves are believable
- checkpoints and resume work
- the hybrid is at least competitive enough to justify more compute

Scale from `~100M` to `~300M` only if:

- the first comparison still looks promising
- runtime on the T4 is manageable
- the hybrid is not obviously losing on both quality and speed

## When To Stop And Fix Training

Stop and fix the setup before spending more money if:

- train loss does not move at all
- validation loss is flat or worsening immediately
- throughput is far below expectation for both models
- checkpoint files are missing or resume is broken
- one model has a clearly mismatched parameter budget

## The First Practical Decision Rule

For the first serious low-budget run:

- use the `~50M` pair first
- compare with `scripts/compare_runs.py`
- if the hybrid is clearly worse, do not scale yet
- if the hybrid is competitive, move to the `~100M` pair

## What To Look At In The Artifacts

- `run.json`: parameter count, device metadata, training config
- `metrics.json`: train and validation loss trend, step time, tokens per second
- `checkpoints/latest.pt`: confirms the run can resume

## Recommended Mindset

This stage is about decision quality, not about headline metrics.

The right outcome is:

- a clean comparison,
- a justified next run,
- or a clear reason to pause and fix training before scaling.
