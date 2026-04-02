# How To Interpret First Results

Date: April 1, 2026

This document is now historical guidance for the first same-size comparisons.

The first-result stage has already been passed. The repo has now completed:

- `~50M` dense vs hybrid
- `~100M` dense vs hybrid
- `~300M` dense vs hybrid

But the original decision rules are still useful when reproducing or extending the work.

## What Counted As Success

The hybrid had to show:

- stable training
- clean checkpoints
- believable validation curves
- enough quality gain to justify another run

That happened.

## What The Early Results Actually Meant

The early TinyStories comparisons did **not** prove a breakthrough.

They proved something narrower:

- the hybrid architecture was worth taking seriously
- the original recurrent implementation was too slow
- the project needed a systems pivot before more scale

## Why The GRU Pivot Was The Key Decision

The accepted recurrent decision was not the first one.

The original recurrent path had promising quality but poor hardware behavior.
The `GRU-fused` replacement made the low-budget path workable enough to reach `100M` and `300M`.

So if you are repeating the historical progression, the lesson is:

`quality alone is not enough; the architecture has to earn the next dollar`

## What Counts As Success Now

At the current project stage, success is no longer:

- "the run completed"
- or "validation loss got better on TinyStories"

Success now means:

- preserving the cross-domain advantage
- and improving actual task behavior, especially on code benchmarks

## What Counts As Failure Now

Failure now is:

- more scale without stronger evidence
- more loss wins without task movement
- more code continuation that improves style but not correctness

## Current Practical Rule

Do not scale past `300M` until at least one cheap executable-code benchmark moves in the right direction.
