# ACE-Atlas Architecture Spec

Date: April 1, 2026

Current implementation note:

- the accepted low-budget recurrent core in the repo is now the `GRU-fused` variant
- this document still describes the broader architectural thesis, not only the currently accepted T4-oriented implementation choice

## 1. Objective

ACE-Atlas is intended to be a more efficient and more capable long-context language model than dense Transformer baselines at comparable active compute.

The design target is not merely lower perplexity. It is a better system-level Pareto point across:

- short-context quality,
- long-context recall,
- decode speed,
- prefill cost,
- tool-verified reasoning,
- and memory efficiency.

## 2. Design Thesis

No single existing family is sufficient:

- dense Transformers preserve recall but scale poorly,
- pure recurrent or linear models scale well but still risk recall loss,
- MoE improves capacity but not long-context memory by itself,
- retrieval alone does not solve online continuity,
- tool use helps reasoning but is not a backbone.

Therefore ACE-Atlas is a `memory-native hybrid causal LM`.

The central architectural principle is:

`use the cheapest computation and memory substrate that preserves future performance`

That means the model should be able to choose among:

- exact local attention,
- recurrent state,
- bounded learned memory,
- or symbolic / executable escalation.

## 3. Non-Goals for v1

The first implementation does not attempt to solve:

- general multimodal world modeling,
- true online modification of slow weights,
- embodied control,
- or full active-inference planning.

Those are future research branches. The v1 goal is a strong text-first model family.

## 4. Model Family

ACE-Atlas is a `decoder-only hybrid recurrent-memory-MoE architecture`.

Each block includes:

- a recurrent or linear sequence mixer,
- optional local exact attention,
- sparse MoE feedforward,
- residual and normalization structure.

Every few layers, the model runs a memory bus:

- summarize current chunk state,
- retrieve candidate memory,
- fuse retrieved memory,
- decide whether to write or update memory.

The output stack provides:

- next-token prediction,
- multi-token prediction,
- uncertainty / escalation scoring.

## 5. High-Level Data Flow

```text
input ids
  -> token embedding
  -> stack of hybrid blocks
  -> periodic memory bus
  -> lm head and mtp head
  -> uncertainty score

if uncertainty > threshold:
  -> retrieve more memory and/or tools
  -> optionally generate code or tool calls
  -> execute
  -> condition on result
  -> revise answer
```

## 6. Core Components

### 6.1 Token Embedding and Positional Treatment

Inputs:

- `input_ids: [batch, seq]`

Outputs:

- `hidden: [batch, seq, d_model]`

Requirements:

- support large vocabularies,
- preserve causal decoding behavior,
- remain compatible with local attention windows and recurrent scanning.

The initial implementation can use a standard learned embedding table and leave rotary or more advanced position encodings as a follow-up improvement.

### 6.2 Local Exact Attention

Purpose:

- preserve short-range exact recall,
- handle token-level disambiguation,
- maintain strong local language modeling quality.

Design:

- grouped-query or multi-query attention,
- sliding-window causal mask,
- no full-prefix dense attention in v1 except for baseline comparisons.

Contract:

- input: `[B, T, D]`
- output: `[B, T, D]`

Why local-only:

- long-range dependencies should not force quadratic cost at every layer,
- but exact attention remains essential for associative recall.

### 6.3 Recurrent / Linear Sequence Mixer

Purpose:

- carry long-range context cheaply,
- provide a long-horizon state path whose cost scales more gently than dense attention.

Candidate families:

- xLSTM Large style recurrent block,
- KDA / linear-attention style block,
- Mamba-like state-space mixer as a baseline.

Current implementation decision:

- keep the recurrent mixer interface modular
- use the accepted `GRU-fused` recurrent core for the current practical baseline
- preserve the ability to swap in other recurrent or linear families later if they earn their way in experimentally

Contract:

- input: `[B, T, D]`
- optional state: implementation-defined
- output: `[B, T, D]`, updated recurrent state

### 6.4 Sparse MoE Feedforward

Purpose:

- increase effective model capacity without paying full dense FFN cost on every token.

Placement:

- middle and upper third of the network only.

Design:

- shared experts for universal processing,
- routed experts for specialization,
- top-2 routing,
- router telemetry for debugging and load balancing.

Contract:

- input: `[B, T, D]`
- output: `[B, T, D]`
- aux: routing logits, selected experts, route probabilities

### 6.5 Bounded Memory

Purpose:

- replace unbounded KV growth with an explicit bounded memory substrate,
- preserve important context beyond the local attention window,
- support test-time continuity without modifying slow weights.

Memory tiers:

- `episodic`: recent chunk summaries and high-value latent states
- `semantic`: compact longer-lived entries that can be updated

Each memory slot stores:

- key vector,
- value vector,
- confidence,
- age,
- source metadata,
- write reason.

Operations:

- `read(query)` returns retrieved memory context and scores,
- `write(key, value, score)` inserts under budget,
- `update(slot, key, value)` revises an existing memory,
- `decay()` reduces stale low-utility entries.

The v1 code should implement the interface and a basic bounded tensor store even if the sophisticated policies come later.

### 6.6 Memory Arbiter

Purpose:

- decide whether the current information should remain transient, be stored, or trigger retrieval.

Action space:

- `keep_state`
- `write_ep`
- `update_sem`
- `retrieve`
- `ignore`

Training target:

The long-term target is:

```text
argmin_a E[future loss | action a] + lambda * compute_cost(a)
```

The initial implementation does not need full counterfactual estimation on day one. It needs:

- a clean arbiter module,
- action logits,
- probability outputs,
- and a path to plug in richer supervision later.

### 6.7 Multi-Token Prediction

Purpose:

- improve optimization efficiency,
- encourage the model to represent slightly longer future structure,
- align with evidence from current frontier training recipes.

Contract:

- primary logits: `[B, T, vocab]`
- optional mtp logits: `[B, T, horizon, vocab]`

The initial implementation can produce `mtp_horizon` separate prediction heads.

### 6.8 Uncertainty and Escalation

Purpose:

- avoid paying maximal compute or tool cost on every token or task.

The model should estimate whether the current pass is likely sufficient.

Escalation targets:

- more memory retrieval,
- more recurrent deliberation steps,
- external verifier or tool path.

The initial implementation only needs:

- an uncertainty head,
- an escalation threshold config,
- an executor interface.

### 6.9 Verifier / Tool Path

Purpose:

- convert faux reasoning into executable, checkable reasoning on the subset of tasks where that is possible.

Initial tool set:

- Python execution,
- deterministic calculator,
- retrieval over local indexed corpora,
- test runner for code tasks.

Contract:

- model emits tool request,
- executor runs it in a bounded sandbox,
- structured result returns to the model.

This is a `specialist path`, not the primary forward path of the model.

## 7. Block Specification

### 7.1 Hybrid Block

Each hybrid block contains:

- `pre_norm_recurrent`
- recurrent mixer
- residual add
- optional `pre_norm_attention`
- local attention
- residual add
- `pre_norm_moe`
- sparse MoE
- residual add

Interface:

```python
hidden, state, aux = block(hidden, state=state, attention_mask=mask)
```

Where `aux` can include:

- routing information,
- attention statistics,
- recurrent state summaries.

### 7.2 Memory Bus Layer

Every `memory_every_n_layers`, run:

1. summarize current chunk,
2. obtain arbiter action logits,
3. optionally read memory,
4. fuse retrieved context,
5. optionally write or update memory.

The memory bus should be its own module, not hand-written inside the forward loop, so it can be benchmarked and ablated.

## 8. Configuration Surface

The public config object should include:

- vocabulary and embedding size,
- layer count,
- attention cadence,
- recurrent kind,
- MoE sizes,
- memory sizes,
- arbiter cost weight,
- escalation threshold,
- multi-token prediction horizon.

Every config value must be serializable so experiment artifacts can be reproduced exactly.

## 9. Training Objectives

### 9.1 Base Losses

- next-token cross entropy,
- multi-token prediction loss,
- optional router regularization,
- optional memory write supervision loss,
- optional uncertainty calibration loss.

### 9.2 Curriculum

Train context in stages:

- 8k,
- 32k,
- 128k,
- 512k,
- 1M only after smaller contexts are stable.

### 9.3 Data Mixture

Mix:

- general high-quality text,
- code,
- mathematics and formal reasoning,
- long-structure documents,
- synthetic long-context recall tasks.

### 9.4 Post-Training

- SFT on instruction, code, math, and tool traces,
- verifier-aware RL,
- refusal and safety tuning.

## 10. Evaluation Requirements

The model is only worth scaling if it improves the Pareto frontier.

Required metrics:

- perplexity,
- long-context recall,
- code correctness,
- math accuracy,
- verifier success rate,
- prefill throughput,
- decode throughput,
- peak memory,
- KV footprint.

## 11. Implementation Constraints

### 11.1 Initial Framework

- Python
- PyTorch
- Triton-ready architecture boundaries

### 11.2 Code Quality

Requirements:

- explicit dataclass configs,
- modular blocks,
- no hidden global state,
- clear separation of model code and tool code,
- easy ablations.

### 11.3 Verification Reality

The initial codebase will be a scaffold and bootstrap model, not a production-quality high-performance implementation.

That means:

- interfaces should be stable,
- internals can begin naive,
- performance kernels come later.

## 12. What Must Be True For This To Be A Breakthrough

ACE-Atlas is interesting only if the following become true experimentally:

- hybrid backbone matches or exceeds dense baselines at similar active compute,
- bounded memory beats simple retrieval or recency heuristics,
- the arbiter learns useful compute-memory tradeoffs,
- verifier use materially increases correctness,
- and the combined system wins on cost per solved problem.

If those conditions fail, the repo should still produce a strong hybrid research baseline, but not a new model family.
