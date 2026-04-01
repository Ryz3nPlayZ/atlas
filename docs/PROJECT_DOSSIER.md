# ACE-Atlas Project Dossier

Date: March 31, 2026

## 1. Executive Summary

ACE-Atlas is a research project to build a more efficient and more capable long-context language model than a dense Transformer baseline at comparable active compute.

The guiding idea is not to replace Transformers with a single new primitive. It is to build a hybrid system that uses:

- exact local attention where recall and token precision matter,
- recurrent or linear state for cheaper long-range sequence handling,
- sparse MoE blocks for active-capacity efficiency,
- bounded learned memory instead of brute-force context stuffing,
- and verifier-driven escalation for code, math, and structured reasoning.

The project is deliberately staged. The current repository is not a finished model training system. It is a research scaffold with the architecture, roadmap, code skeleton, benchmark adapters, data-manifest utilities, and cloud/kaggle setup guidance needed to move into the next implementation phase.

## 2. Project Goal and Purpose

The project exists to answer one practical question:

`Can we build an LLM that preserves useful capability while reducing long-context cost and enabling better reasoning workflows than a dense Transformer baseline?`

This is narrower than "build AGI" and more ambitious than "make a cheaper Transformer."

The concrete goals are:

- beat or match dense local-attention baselines on useful quality at similar active compute,
- handle longer contexts more efficiently,
- support verifier-driven reasoning where it materially improves correctness,
- establish a research platform for memory arbitration and efficient hybrid sequence modeling.

## 3. Final Architecture Direction Chosen In This Project

The project explored several architecture families conceptually:

- GPT / dense Transformers
- Mamba / SSM / recurrent families
- JEPA-style latent predictive models
- external memory systems
- program-aided / verifier-driven reasoning
- more speculative ideas such as active inference, NCA, HDC, and online weight plasticity

The final near-term decision is:

`Do not build a kitchen-sink ACE stack. Build ACE-Atlas as a hybrid recurrent-memory-MoE decoder with verifier-driven escalation.`

That means:

- keep exact attention in the system,
- do not bet the entire model on a pure SSM or pure linear family,
- do not use JEPA as the text backbone in v1,
- do not put NCA, HDC, or full active inference into the critical path,
- and treat verifier tools as a specialist reasoning path, not the core model.

## 4. Current Thesis

The current working thesis is:

- dense Transformers still have the strongest local exact-recall behavior,
- pure efficient models still risk losing some of that recall,
- MoE helps capacity but does not solve long-context scaling alone,
- memory has to be explicit rather than hidden inside an ever-growing attention context,
- and hard reasoning should not force every token through maximum compute.

The consequence is a staged hybrid system.

## 5. The Core Model Idea

ACE-Atlas is a `decoder-only hybrid recurrent-memory-MoE model`.

High-level flow:

```text
tokens
  -> embeddings
  -> hybrid blocks:
       recurrent / linear path
       optional local exact attention
       sparse MoE feedforward
  -> periodic memory bus
  -> next-token and multi-token heads
  -> uncertainty / escalation score

if hard problem:
  -> more memory
  -> optional deliberation
  -> optional tool/program execution
  -> revised answer
```

This is a system architecture, not just a block choice.

## 6. Key Design Decisions From The Conversation

### 6.1 What We Explicitly Decided To Keep

- local exact attention
- recurrent or linear sequence mixer
- sparse MoE in the middle and upper model
- bounded memory
- verifier-driven path
- a simple research-first codebase rather than a production training stack

### 6.2 What We Explicitly Decided To Defer

- full online weight plasticity
- NCA as the main compute substrate
- HDC as the main representation family
- full active inference as the control theory
- JEPA as the text-generation backbone
- distributed training infrastructure before single-machine smoke validation

### 6.3 Why Those Choices Matter

They reduce research risk and implementation risk at the same time:

- fewer moving parts,
- easier ablations,
- clearer attribution when something works or fails,
- and a realistic path from local development to cloud smoke tests.

## 7. Repository Status

The repository is already live on GitHub and contains the first serious scaffold.

Current repo: `https://github.com/Ryz3nPlayZ/atlas`

The repository currently includes:

- top-level docs
- architecture spec
- roadmap
- cloud prep notes
- model config system
- hybrid model scaffold
- dense baseline model scaffold
- memory and arbiter modules
- benchmark adapters
- tokenizer and corpus-prep utilities
- training harness bootstrap
- smoke scripts

## 8. What Has Been Done

### 8.1 Documentation Already Written

- [ARCHITECTURE_SPEC.md](./ARCHITECTURE_SPEC.md)
- [ROADMAP.md](./ROADMAP.md)
- [CLOUD_PREP.md](./CLOUD_PREP.md)

### 8.2 Core Code Already Implemented

Repository entrypoints:

- `/README.md`
- `/pyproject.toml`

Config surface:

- `src/ace_atlas/config.py`
- `src/ace_atlas/train/config.py`

Model code:

- `src/ace_atlas/model/attention.py`
- `src/ace_atlas/model/recurrent.py`
- `src/ace_atlas/model/moe.py`
- `src/ace_atlas/model/memory.py`
- `src/ace_atlas/model/arbiter.py`
- `src/ace_atlas/model/backbone.py`
- `src/ace_atlas/model/dense_baseline.py`
- `src/ace_atlas/model/types.py`

Training code:

- `src/ace_atlas/train/data.py`
- `src/ace_atlas/train/objectives.py`
- `src/ace_atlas/train/harness.py`

Tools:

- `src/ace_atlas/tools/python_executor.py`

Benchmarks:

- `src/ace_atlas/benchmarks/recall.py`
- `src/ace_atlas/benchmarks/verifier.py`

Data and tokenizer utilities:

- `src/ace_atlas/data/manifest.py`
- `src/ace_atlas/data/corpus.py`
- `src/ace_atlas/tokenizer/byte_level.py`

Scripts:

- `scripts/train_dense_baseline.py`
- `scripts/train_hybrid.py`
- `scripts/train_baseline.py`
- `scripts/run_benchmark_recall.py`
- `scripts/run_benchmark_verifier.py`
- `scripts/build_manifest.py`
- `scripts/tokenize_corpus.py`
- `scripts/preflight_check.py`

Tests:

- `tests/test_config.py`
- `tests/test_benchmarks.py`
- `tests/test_manifest_and_tokenizer.py`

### 8.3 What Was Verified Locally

The following local checks were performed:

- repo-wide `python3 -m compileall` succeeded
- recall benchmark smoke script ran successfully
- verifier benchmark smoke script ran successfully
- manifest generation and tokenization worked on a sample local corpus
- preflight script correctly reported environment state

### 8.4 What Could Not Be Verified Locally

The environment used during initial development did not have:

- `torch`
- `pytest`
- a CUDA-capable GPU

As a result:

- full runtime model execution was not verified locally,
- training entrypoints were only validated to the point of clean failure when `torch` is absent,
- and the original scaffold could not complete real GPU-backed runs until later Lightning validation work.

## 9. Current Known Limitations

This is the most important status section in the entire project.

The training harness now supports both:

- `synthetic` random-token debug training
- `tokenized` JSONL training on real corpora

That means:

- smoke training remains available for bring-up,
- first real-data TinyStories runs are now possible,
- and the repository is ready for small dense vs hybrid comparisons on Lightning.

Other limitations:

- no distributed training,
- no mixed-precision or throughput tuning pass yet,
- no real tokenizer beyond the minimal byte-level pipeline,
- no long-context production kernels,
- no serious benchmark loaders for standard public suites yet,
- no artifact dashboarding beyond JSON outputs,
- no production inference path.

## 10. Ideal Specification

The ideal near-term ACE-Atlas system should eventually have:

### 10.1 Model

- hybrid recurrent-memory-MoE backbone
- local exact attention with efficient kernels
- bounded multi-tier memory
- learned memory arbiter
- multi-token prediction
- verifier-aware escalation

### 10.2 Data

- real tokenized shard pipeline
- manifest-backed train/validation splits
- code, math, general text, and long-structure data
- synthetic recall curricula

### 10.3 Training

- single-GPU smoke training
- checkpointing
- validation loop
- reproducible config artifacts
- later: multi-GPU and long-context scaling

### 10.4 Evaluation

- perplexity
- recall tasks
- long-context tasks
- code tasks
- math tasks
- verifier utility metrics
- systems metrics

### 10.5 Infrastructure

- clean cloud smoke path
- later: scalable GPU training path
- cost-aware experimentation

## 11. What Needs To Be Done Next

These are the highest-priority remaining tasks, in order.

### Phase 1: First Real Corpus Comparison

The single most important next task is:

`run the first real TinyStories dense vs hybrid comparison and review the results`

That includes:

- preparing TinyStories into tokenized JSONL,
- running the dense baseline on Lightning,
- running the hybrid baseline on Lightning,
- inspecting validation curves and checkpoints,
- and deciding whether the current small-model recipe earns a longer low-budget run.

### Phase 2: Add Minimal Real Training Requirements

After real data loading:

- add checkpoint save/resume
- add held-out validation loss
- add artifact and metrics writing that is stable across restarts

### Phase 3: Cloud Smoke Validation

Once real data loading exists:

- provision one GPU machine
- install dependencies
- run dense smoke
- run hybrid smoke
- compare artifacts and losses

### Phase 4: Hybrid Quality/Speed Validation

After smoke training succeeds:

- compare dense vs hybrid on small real runs
- verify that the hybrid is not obviously worse at similar active compute
- profile memory and throughput

### Phase 5: Memory Value Validation

Only after the backbone is stable:

- ablate memory on/off
- ablate heuristics vs arbiter
- verify whether bounded memory actually helps

## 12. Detailed Next-Step Backlog

### 12.1 Must-Do Engineering

- real token dataset loader
- tokenized corpus reader
- checkpoint save
- checkpoint load
- validation dataloader
- validation loop
- training/validation metric separation
- config-driven data paths

### 12.2 Must-Do Research Infrastructure

- baseline experiment naming
- experiment registry or manifest
- reproducible artifact layout
- benchmark result aggregation

### 12.3 Nice-To-Have After That

- richer tokenizer integration
- benchmark dataset download helpers
- better uncertainty calibration
- richer verifier task cases
- GPU profiling scripts

## 13. Recommended Operational Guidance

### 13.1 Local Machine

The local laptop or CPU-only environment should be used for:

- repo development
- docs
- tests that do not need torch
- manifesting and tokenization
- benchmark adapter work
- preflight verification

It should not be used for:

- meaningful model training
- throughput evaluation
- real GPU experiments

### 13.2 Kaggle

Kaggle is acceptable only for:

- smoke tests
- install verification
- very tiny experiments

It is not a good place for:

- sustained custom training
- checkpoint-heavy workflows
- real research iteration on long jobs

### 13.3 Google Cloud

Google Cloud is a reasonable next step if using:

- one small GPU VM first
- smoke validation before real training
- and ideally startup credits if available

But cloud spend should wait until the random-data training path is replaced.

## 14. Exact Current Blocking Issue

If someone asks "what is the next step?" the honest answer is:

`wire real tokenized data into training`

That is the blocker between "good scaffold" and "actual training system."

Everything else before that is secondary.

## 15. Success Criteria For The Next Milestone

The next milestone is complete only when:

- a real tokenized manifest can be loaded into training,
- a dense baseline can train on real data for a short run,
- a hybrid model can train on real data for a short run,
- artifacts are written,
- and the run can be resumed or at least re-run reproducibly.

## 16. Source Of Truth Files

If someone needs to understand the project quickly, start here:

1. `/README.md`
2. `/docs/PROJECT_DOSSIER.md`
3. `/docs/ARCHITECTURE_SPEC.md`
4. `/docs/ROADMAP.md`
5. `/docs/CLOUD_PREP.md`

If someone needs to understand the current implementation:

1. `src/ace_atlas/config.py`
2. `src/ace_atlas/model/backbone.py`
3. `src/ace_atlas/model/dense_baseline.py`
4. `src/ace_atlas/train/harness.py`
5. `src/ace_atlas/data/manifest.py`
6. `src/ace_atlas/data/corpus.py`

## 17. Bottom Line

ACE-Atlas currently has:

- a clear architecture direction,
- a GitHub repository,
- a detailed spec,
- a roadmap,
- a functioning scaffold,
- verified pure-Python utilities,
- and a practical cloud-prep path.

ACE-Atlas does not yet have:

- real-data training,
- meaningful GPU training results,
- or evidence that the hybrid model beats a dense baseline.

So the project is in a good state for a research scaffold, and not yet in a good state for a real training campaign.

The next serious move is not more architecture writing.

The next serious move is:

`implement manifest-backed real-data training`
