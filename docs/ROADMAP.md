# ACE-Atlas Roadmap

Date: April 1, 2026

## Roadmap Principle

The project does **not** earn scale just by training a bigger model.

It earns scale only if each stage proves something useful at fixed budget:

1. stable same-size baseline comparison
2. real quality advantage
3. cross-domain validation
4. task-level evidence
5. then, and only then, bigger scale

## Completed Phases

### Phase 0: Repo and Bring-Up

Completed:

- research repo scaffold
- docs and config surfaces
- dense and hybrid entrypoints
- manifest and tokenization utilities
- smoke training path
- benchmark smoke checks

### Phase 1: Real-Data TinyStories Pipeline

Completed:

- tokenized JSONL training
- train/validation split support
- checkpoint save/resume
- TinyStories preparation flow
- first real dense vs hybrid comparisons

### Phase 2: Systems Triage

Completed:

- profiling of the original recurrent bottleneck
- recurrent optimization passes
- replacement of the original recurrent core with the accepted `GRU-fused` variant

Outcome:

- hybrid quality signal preserved
- throughput improved enough to justify further same-size scaling

### Phase 3: Same-Size Scaling Validation

Completed:

- `~50M` dense vs hybrid
- `~100M` dense vs hybrid
- `~300M` dense vs hybrid

Outcome:

- hybrid beat dense at all three accepted size tiers

### Phase 4: Cross-Domain Held-Out Validation

Completed:

- WikiText-2 checkpoint evaluation
- CodeSearchNet Python checkpoint evaluation

Outcome:

- hybrid still beat the same-size dense baseline outside TinyStories

## Current Phase

### Phase 5: Task-Aligned Code Capability

Status: `in progress`

What happened already:

- HumanEval subset evaluation: both dense and hybrid failed
- MBPP subset evaluation: both dense and hybrid failed
- code-focused continuation improved code-like behavior but not pass rate

Current diagnosis:

- architecture is not the immediate blocker
- scale is not the immediate blocker
- task alignment is the blocker

## Current Accepted Baseline

The accepted baseline to work from is:

- `configs/hybrid_300m_gru.json`
- `configs/train_tinystories_300m_hybrid_t4_fp32_no_ckpt.json`

Accepted model class:

- `~300M` GRU-hybrid

Accepted practical claim:

- beats same-size dense baselines on cross-domain modeling quality

Unaccepted claim:

- beats same-size models on executable-code task benchmarks

## Next Phase

### Phase 6: Task-Style Code Finetuning

Goal:

- turn the current modeling advantage into measurable code-task wins

Immediate tasks:

- keep architecture fixed at `300M`
- continue from the accepted hybrid checkpoint lineage
- shift training mix toward task-style code supervision
- improve prompt/completion formatting for code tasks
- re-evaluate on MBPP and HumanEval subsets

Acceptance gate:

- any real pass-rate improvement over the current `0/20` MBPP and `0/10` HumanEval baseline

Kill criteria:

- if repeated task-aligned continuation still fails to produce any benchmark movement, pause and revisit whether the architecture advantage is only a likelihood-level advantage

## Deferred Phases

These phases are now explicitly deferred until task-aligned code capability improves.

### Phase 7: Scale Beyond 300M

Deferred because:

- the project already has enough same-size evidence for now
- current value comes from tightening the claim, not more size

### Phase 8: New Architecture Mechanism Work

Deferred because:

- the current architecture still has room to prove itself
- changing architecture before resolving the current task-alignment bottleneck would muddy attribution

## Practical Decision Rules

### Scale again only if:

- task-style code results improve meaningfully
- cross-domain held-out advantage stays intact
- the T4-aware training recipe remains operational

### Do not scale if:

- pass rates stay flat at zero after another serious task-aligned continuation
- new continuation runs only improve style, not correctness
- code-task wins still do not materialize

## End State Worth Reaching

The next meaningful project state is not:

- a larger model

It is:

`a fixed 300M hybrid that beats a same-size dense baseline on both cross-domain loss and at least one cheap executable-code benchmark`

If the project reaches that state, then scaling past `300M` becomes much easier to justify.
