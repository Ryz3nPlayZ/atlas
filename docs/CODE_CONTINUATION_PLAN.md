# Code Continuation Plan

Goal: improve code-task capability of the accepted 300M GRU-hybrid checkpoint without changing model family or scale.

Baseline checkpoint:
- `artifacts/tinystories_hybrid_300m_gru_t4_fp32_no_ckpt/checkpoints/latest.pt`

Low-budget continued-training recipe:
- Dataset: `code_search_net/python`
- Field: `func_code_string`
- Intervention: continue training from checkpoint weights, do not resume the old optimizer state
- Effective batch: 12
- Microbatch: 4
- Gradient accumulation: 3
- Sequence length: 256
- Precision: fp32
- Activation checkpointing: off
- Learning rate: `5e-5`
- Weight decay: `0.01`
- Steps: `120`

Why this recipe:
- Reuses the accepted T4-safe 300M hybrid training setup
- Keeps the intervention small and reversible
- Uses a cheap Python-heavy corpus already compatible with the repo tokenizer and dataloader flow
- Tests whether code-focused continuation can convert the hybrid's loss advantage into executable-code task gains

Evaluation after continuation:
- MBPP sanitized subset via `scripts/evaluate_mbpp_subset.py`
- HumanEval subset via `scripts/evaluate_humaneval_subset.py`

Success criteria for this stage:
- Any improvement in task-style code pass rate over the pre-continuation baseline
- Or a clear shift from prose-like generations toward syntactically valid Python
