# ACE-Atlas Project Dossier

Date: April 1, 2026

## Executive Summary

ACE-Atlas is a custom language-model research project built around a hybrid decoder that combines:

- local attention
- a recurrent sequence path
- sparse MoE blocks
- bounded memory-oriented structure

The practical goal is not "beat frontier labs overall." It is:

`beat strong dense baselines in the same weight class on a low-budget path`

As of this update, the project has reached a real milestone:

- the hybrid beat the same-size dense baseline at `50M`, `100M`, and `300M`
- the advantage holds on TinyStories, WikiText-2, and held-out Python code likelihood
- the accepted low-budget hybrid baseline is now the `~300M GRU-fused` variant

What is still missing:

- executable-code benchmark wins
- proof that the architecture beats strong public models in the same weight class
- evidence beyond the current low-budget benchmark set

## Project Purpose

The project exists to test whether a small-team, low-budget architecture can outperform plain dense baselines without relying on frontier-scale compute.

The bet is:

- architectural efficiency can matter at fixed parameter count
- a hybrid recurrent-memory model can win on quality per parameter
- low-budget experimentation can still produce meaningful same-size results if the comparison is disciplined

## Architecture Direction

The accepted backbone direction is:

`decoder-only hybrid recurrent-memory-MoE language model`

Current practical form:

- local attention for exact short-range recall
- GRU-fused recurrent core for sequence mixing
- sparse MoE blocks for active-capacity efficiency
- bounded memory path
- next-token and multi-token heads

Important non-goals for the accepted baseline:

- not a pure Transformer
- not a pure SSM/Mamba stack
- not a JEPA text backbone
- not a kitchen-sink architecture with every speculative idea in the critical path

## What Was Built

The repository now includes:

- dense and hybrid model paths
- parameter-matched config tiers at `~50M`, `~100M`, and `~300M`
- real tokenized JSONL training
- train/validation split support
- validation logging
- checkpoint save/resume
- TinyStories prep and comparison workflow
- checkpoint evaluation scripts
- small HumanEval and MBPP subset harnesses
- CodeSearchNet Python and WikiText-2 prep/eval helpers

Repo:

- `https://github.com/Ryz3nPlayZ/atlas`

Accepted source of truth at the time of this doc update:

- `origin/main`

## What Was Proven So Far

### Same-Size TinyStories Results

The hybrid beat the dense baseline at each accepted size tier:

- `~50M`
- `~100M`
- `~300M`

Accepted `300M` TinyStories comparison:

- dense final val loss: `2.0015`
- hybrid final val loss: `1.3264`

### Cross-Domain Held-Out Evaluation

Fixed `300M` checkpoints were evaluated out of training domain.

#### WikiText-2

- dense validation loss: `3.7954`
- hybrid validation loss: `3.4819`
- dense test loss: `3.8185`
- hybrid test loss: `3.5030`

#### CodeSearchNet Python

- dense validation loss: `6.3760`
- hybrid validation loss: `5.8332`
- dense test loss: `6.6820`
- hybrid test loss: `6.0925`

Interpretation:

- the hybrid's advantage is not confined to TinyStories
- the current evidence supports a real cross-domain likelihood advantage

## What Was Not Proven

Two cheap task-style code evaluations were run on the accepted `300M` checkpoints:

### HumanEval subset

- dense: `0/10`
- hybrid: `0/10`

### MBPP sanitized subset

- dense: `0/20`
- hybrid: `0/20`

Interpretation:

- the architecture advantage has **not** yet turned into executable-code wins
- the current checkpoint family is still weak at code-task behavior

## Code Continuation Findings

The project tested code-focused continuation instead of scaling further.

### Small continuation

- CodeSearchNet Python continuation improved output behavior
- pass rate did not improve

### Stronger continuation

Accepted stronger run:

- `code_mix_300m_hybrid_continue_v2`
- `600` steps
- `50,000` CodeSearchNet Python functions
- `6,800` MBPP-style supervised examples

Result:

- outputs became more code-like
- prose leakage decreased
- pass rate still stayed at `0/20` on MBPP and `0/10` on HumanEval

Current conclusion:

- architecture is not the blocker
- scale is not the blocker
- code-task alignment is the blocker

## Why The GRU Pivot Mattered

The original recurrent path showed strong quality but was too slow on the T4 budget path.

The accepted pivot replaced the original recurrent implementation with a `GRU-fused` core.

That change:

- materially improved throughput
- preserved or slightly improved quality
- unlocked the `100M` and `300M` hybrid path as a practical experiment

This is the most important systems decision in the project so far.

## Current Strongest Honest Claim

The strongest honest claim supported by the current evidence is:

`ACE-Atlas beats same-size dense baselines on cross-domain modeling quality at 50M, 100M, and 300M.`

What cannot be claimed yet:

- that ACE-Atlas beats top public models in the same weight class
- that it has a task-benchmark code advantage
- that it is a frontier-level breakthrough

## Current Limitation

The main limitation is no longer basic training or memory fit.

The main limitation is:

`the 300M hybrid still does not reliably generate executable benchmark-grade Python`

That is why more scale is not the next move.

## Accepted 300M T4 Recipe

The working low-budget `300M` hybrid recipe is:

- batch size: `12`
- microbatch size: `4`
- grad accumulation: `3`
- mixed precision: `none`
- activation checkpointing: `false`

This fit the T4 cleanly and preserved stable training behavior.

## What Needs To Happen Next

The project should **not** scale past `300M` yet.

The next milestone is:

`task-aligned code finetuning that tries to convert the hybrid's modeling advantage into benchmark pass-rate wins`

That means:

- keep the accepted `300M` GRU-hybrid fixed
- keep the current same-size result intact
- shift effort from bigger scale to better code-task alignment

## Why This Matters

Right now the project is no longer a toy scaffold, but it is not yet a breakthrough.

The honest placement is:

- real architecture signal: yes
- same-size dense wins: yes
- cross-domain held-out wins: yes
- benchmark-grade code generation: no
- breakthrough-level result: not yet

That is still a meaningful place to be. It means the project has earned another phase of focused work rather than more speculative architecture assembly.
