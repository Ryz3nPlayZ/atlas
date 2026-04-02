# Code Continuation Plan

Date: April 1, 2026

## Current Conclusion

The accepted `300M` GRU-hybrid checkpoint has a real cross-domain modeling advantage, but that has not yet turned into executable-code wins.

Task-style benchmark status:

- HumanEval subset: `0/10`
- MBPP subset: `0/20`

So the current bottleneck is:

`task-aligned code capability`

## Accepted Checkpoint Lineage

Baseline checkpoint:

- `artifacts/tinystories_hybrid_300m_gru_t4_fp32_no_ckpt/checkpoints/latest.pt`

Code continuation lineage:

- small pilot: `artifacts/codesearchnet_300m_hybrid_continue_v1/checkpoints/latest.pt`
- stronger continuation: `artifacts/code_mix_300m_hybrid_continue_v2/checkpoints/latest.pt`

## What Was Tried

### v1: Raw CodeSearchNet Python continuation

Recipe:

- dataset: `code_search_net/python`
- field: `func_code_string`
- steps: `120`
- effective batch: `12`
- microbatch: `4`
- grad accumulation: `3`
- sequence length: `256`
- precision: `fp32`

Outcome:

- outputs became more code-like
- pass rate did not improve

### v2: CodeSearchNet + MBPP-style supervised mix

Recipe:

- steps: `600`
- effective batch: `12`
- microbatch: `4`
- grad accumulation: `3`
- sequence length: `256`
- learning rate: `3e-5`
- weight decay: `0.01`
- precision: `fp32`
- activation checkpointing: `off`

Dataset mix:

- `50,000` CodeSearchNet Python functions
- `6,800` MBPP-style supervised examples

Outcome:

- outputs shifted further away from prose-like failures
- pass rate still stayed at `0/20` on MBPP and `0/10` on HumanEval

## Current Read

What improved:

- less prose leakage
- better code-like formatting
- more benchmark-shaped outputs

What did not improve enough:

- executable correctness
- pass rate

So more raw code tokens alone are probably not enough.

## Next Recommended Direction

Do **not** scale.
Do **not** change architecture.

Do:

- keep the accepted `300M` GRU-hybrid fixed
- continue from the current checkpoint lineage
- bias training more strongly toward task-style code supervision
- optimize for benchmark-compatible completions, not just code-like text

## What The Next Continuation Should Prioritize

Highest-value changes:

- more task-format examples
- stronger prompt -> function-body supervision
- better completion formatting for benchmark-style evaluation
- less reliance on raw code as the majority signal if it dilutes task alignment

## Success Criteria

The next continuation phase succeeds only if it produces:

- any non-zero MBPP or HumanEval movement
- or a very clear improvement on a cheap executable-code benchmark

Lower held-out loss by itself is no longer enough.
