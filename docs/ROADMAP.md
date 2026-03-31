# ACE-Atlas Roadmap

Date: March 31, 2026

## 1. Roadmap Principle

This roadmap is designed to reduce research risk before scale risk.

The sequence is:

1. establish reproducible baselines,
2. validate the hybrid backbone,
3. validate bounded memory,
4. validate arbitration,
5. validate verifier-driven escalation,
6. then scale.

At every stage, the project must earn the right to continue.

## 2. Phase Breakdown

## Phase 0: Repository Foundation

Goal:

- create a usable research repo with docs, config surfaces, and module boundaries.

Deliverables:

- architecture spec,
- roadmap,
- package layout,
- core configs,
- bootstrap modules for attention, recurrent mixer, MoE, memory, arbiter, and backbone,
- skeleton training and benchmark entrypoints.

Exit criteria:

- repository is coherent,
- module interfaces are stable enough to build against,
- basic syntax checks pass.

## Phase 1: Baseline Reproduction

Goal:

- reproduce a dense sliding-window baseline and at least one efficient baseline.

Targets:

- dense local-attention LM,
- xLSTM-style baseline,
- optional Mamba baseline.

Deliverables:

- training harness,
- evaluation harness,
- reproducible config files,
- first throughput vs quality chart.

Acceptance gates:

- dense baseline is stable,
- efficient baseline runs at the expected speedup range,
- evaluation scripts are trusted.

Kill criteria:

- if the team cannot produce stable baselines, do not proceed to architecture invention.

## Phase 2: Hybrid Backbone

Goal:

- build the core recurrent + local attention + MoE hybrid.

Tasks:

- implement hybrid block stack,
- support configurable attention cadence,
- compare xLSTM-first vs KDA-first recurrent slots,
- instrument routing metrics.

Deliverables:

- 150M and 400M hybrid runs,
- block ablation table,
- first hybrid Pareto comparison against dense and efficient baselines.

Acceptance gates:

- quality loss versus dense baseline is acceptable,
- speedup versus dense baseline is real,
- training is stable over full runs.

## Phase 3: Memory Bus

Goal:

- add bounded memory in a way that is measurable and ablatable.

Tasks:

- implement episodic memory state,
- implement read path,
- implement write path,
- add memory interval scheduling,
- benchmark memory-on vs memory-off.

Deliverables:

- memory stress tests,
- bounded memory behavior plots,
- long-context recall results.

Acceptance gates:

- memory helps or at least stays neutral on long-context tasks,
- memory does not destabilize training,
- memory overhead is bounded and measurable.

Kill criteria:

- if bounded memory behaves like a noisy cache and loses to simple retrieval heuristics, pause before building arbiter logic.

## Phase 4: Memory Arbiter

Goal:

- turn memory from a static mechanism into a cost-aware controller.

Tasks:

- implement action logits and telemetry,
- start with heuristic supervision,
- progress toward learned memory utility targets,
- compare against recency, attention-mass, and similarity heuristics.

Deliverables:

- arbitration policy reports,
- memory action histograms,
- ablations by task family.

Acceptance gates:

- arbiter beats simple heuristics on at least one important benchmark family,
- compute overhead is justified.

Kill criteria:

- if the arbiter cannot outperform trivial rules, freeze memory policy and scale the hybrid without it.

## Phase 5: Multi-Token and Uncertainty Heads

Goal:

- improve training and enable adaptive inference.

Tasks:

- add multi-token prediction heads,
- add uncertainty or escalation head,
- calibrate thresholds on held-out reasoning tasks.

Deliverables:

- MTP ablation,
- escalation calibration curves,
- compute-versus-accuracy study.

Acceptance gates:

- MTP improves throughput-quality tradeoff or downstream performance,
- escalation avoids unnecessary tool use on easy tasks.

## Phase 6: Verifier Path

Goal:

- integrate program- and tool-based reasoning.

Tasks:

- Python executor,
- deterministic calculator,
- basic retrieval interface,
- structured result injection,
- verifier-aware trace formatting.

Deliverables:

- code and math benchmark deltas,
- execution latency report,
- failure taxonomy for verifier use.

Acceptance gates:

- verifier path materially improves correctness on code and math,
- latency is acceptable when gated by uncertainty.

Kill criteria:

- if tool use only helps under unrealistic prompting, keep the executor for research but do not treat it as a core advantage.

## Phase 7: Mid-Scale Validation

Goal:

- validate the winning configuration at 1.3B to 7B active scale.

Tasks:

- choose the best recurrent family,
- choose the best memory policy,
- choose the best MoE depth and width,
- run full context curriculum.

Deliverables:

- mid-scale model card,
- comprehensive benchmark sheet,
- scale recommendation.

Acceptance gates:

- clear win on at least one meaningful Pareto frontier,
- no hidden systems bottleneck that would block frontier scale.

## Phase 8: Frontier Candidate

Goal:

- train the first model that could plausibly be competitive beyond research prototypes.

Prerequisites:

- stable backbone,
- justified memory,
- useful verifier path,
- reproducible data pipeline,
- acceptable hardware budget.

Deliverables:

- full training run,
- model card,
- deployment notes,
- comparison against best available baselines.

## 3. Detailed Workstreams

### 3.1 Model Architecture

Immediate:

- lock config schema,
- implement bootstrap modules,
- benchmark naive versions.

Next:

- optimize recurrent path,
- refine attention cadence,
- improve memory fusion.

Later:

- custom kernels,
- fused routing,
- better state caching for inference.

### 3.2 Data

Immediate:

- define tokenizer assumptions,
- prepare data manifest format,
- identify corpora for code, math, long documents.

Next:

- build shard and resume logic,
- add synthetic recall tasks,
- add tool-trace formatting.

Later:

- curriculum-aware mixing,
- deduplication and contamination audits,
- memory-target annotation.

### 3.3 Evaluation

Immediate:

- implement benchmark adapters,
- define system metrics collection.

Next:

- add long-context and code repository tests,
- add escalation telemetry.

Later:

- add agentic evaluations,
- add failure clustering and trace inspection.

### 3.4 Systems

Immediate:

- choose training launcher,
- define config and artifact layout.

Next:

- add distributed checkpointing,
- add profiling hooks,
- optimize memory state movement.

Later:

- Triton kernels,
- inference engine integration,
- large-scale serving support.

## 4. Detailed Milestone Checklist

### Milestone A: Repo Is Real

- docs are written
- package is importable
- model config exists
- model modules exist
- scripts exist
- syntax checks pass

### Milestone B: Baselines Are Trusted

- dense baseline runs
- recurrent baseline runs
- metrics scripts work
- results are reproducible

### Milestone C: Hybrid Wins Somewhere

- hybrid beats dense on speed at useful quality
- hybrid beats pure recurrent on recall

### Milestone D: Memory Is Worth Keeping

- memory helps long-context retrieval
- memory cost is bounded

### Milestone E: Arbiter Is Real

- learned arbitration beats heuristics

### Milestone F: Verifier Adds Value

- code/math correctness materially increases

### Milestone G: Scale Is Justified

- 1.3B to 7B active model still shows the same advantages

## 5. Roles Needed

Minimum:

- research lead,
- model engineer,
- systems engineer,
- eval and data engineer.

Preferred:

- kernel engineer,
- RL specialist,
- safety engineer.

## 6. Decision Log Requirements

Every major experiment should record:

- exact config,
- training tokens,
- context curriculum,
- systems metrics,
- benchmark summary,
- whether the result changes roadmap direction.

The repo should accumulate decisions, not just checkpoints.

## 7. Immediate Next Actions

The next concrete engineering actions are:

1. finish the repository scaffold,
2. implement the first-pass hybrid backbone,
3. add benchmark-ready interfaces,
4. wire in a baseline training config,
5. and add syntax or smoke validation in CI.

## 8. Time Horizon

If executed seriously:

- repository scaffold: days,
- baseline reproduction: 2 to 4 weeks,
- hybrid backbone and memory prototype: 4 to 8 weeks,
- verifier path and mid-scale validation: 6 to 12 weeks after that,
- frontier candidate: only after the earlier gates clear.

The fastest way to fail is to skip baseline reproduction and jump directly to large-scale training.

