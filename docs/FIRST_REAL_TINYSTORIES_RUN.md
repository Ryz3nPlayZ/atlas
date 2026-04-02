# First Real TinyStories Run

Date: April 1, 2026

This is the operational guide for the first real ACE-Atlas training milestone.

Historical note:

- this doc is for reproducing the first clean real-data workflow
- the accepted current hybrid baseline later moved to the `GRU-fused` configs and the accepted `300M` T4 recipe

The target is simple:

- prepare TinyStories,
- move the prepared tokenized files onto Lightning,
- run a dense baseline,
- run a hybrid baseline,
- and compare metrics plus checkpoints.

This is not the final model recipe. It is the first clean low-budget real-data workflow.

For result interpretation after the runs finish, see [INTERPRETING_FIRST_RESULTS.md](./INTERPRETING_FIRST_RESULTS.md).

## 1. Kaggle Prep

In a Kaggle notebook:

```bash
!git clone https://github.com/Ryz3nPlayZ/atlas.git
%cd atlas
!python -m pip install -e '.[data]'
!python scripts/prepare_tinystories.py \
  --output-dir /kaggle/working/tinystories \
  --train-limit 200000 \
  --val-limit 5000 \
  --sequence-length 256
!tar -czf /kaggle/working/tinystories_prepped.tgz -C /kaggle/working tinystories
```

What this produces:

- `/kaggle/working/tinystories/train.jsonl`
- `/kaggle/working/tinystories/val.jsonl`
- `/kaggle/working/tinystories/manifest.json`
- `/kaggle/working/tinystories/train_tokens.jsonl`
- `/kaggle/working/tinystories/val_tokens.jsonl`
- `/kaggle/working/tinystories_prepped.tgz`

Notes:

- `--train-limit` and `--val-limit` are optional. Use `0` for the full split.
- The tokenized JSONL files are the actual training inputs for the current trainer.
- The manifest is kept alongside them so the split source of truth is explicit.

## 2. Move Prepared Data To Lightning

Download `tinystories_prepped.tgz` from Kaggle notebook output, upload it into the repo root on Lightning, then run:

```bash
cd /teamspace/studios/this_studio/atlas
mkdir -p data
tar -xzf /teamspace/studios/this_studio/atlas/tinystories_prepped.tgz -C data
```

After extraction, these paths should exist:

- `data/tinystories/train_tokens.jsonl`
- `data/tinystories/val_tokens.jsonl`
- `data/tinystories/manifest.json`

## 3. Verify The Lightning Environment

```bash
cd /teamspace/studios/this_studio/atlas
python -m pip install -e '.[dev,data]'
python scripts/preflight_check.py
python scripts/run_benchmark_recall.py
python scripts/run_benchmark_verifier.py
```

## 4. Run The Dense Baseline

Check the parameter count first:

```bash
python scripts/report_model_stats.py --model-name dense_baseline --config configs/dense_50m.json
```

Small real-data smoke:

```bash
python scripts/train_dense_baseline.py \
  --config configs/dense_tiny_debug.json \
  --train-config configs/train_tinystories_smoke.json \
  --run-name tinystories_dense_smoke
```

First serious low-budget comparison run:

```bash
python scripts/train_dense_baseline.py \
  --config configs/dense_50m.json \
  --train-config configs/train_tinystories_first_compare.json \
  --run-name tinystories_dense_50m
```

Longer low-budget T4 comparison run:

```bash
python scripts/train_dense_baseline.py \
  --config configs/dense_50m.json \
  --train-config configs/train_tinystories_t4.json \
  --run-name tinystories_dense_t4
```

## 5. Run The Hybrid Baseline

Check the parameter count first:

```bash
python scripts/report_model_stats.py --model-name ace_atlas --config configs/hybrid_50m_gru.json
```

Small real-data smoke:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_tiny_debug.json \
  --train-config configs/train_tinystories_smoke.json \
  --run-name tinystories_hybrid_smoke
```

First serious low-budget comparison run:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_50m_gru.json \
  --train-config configs/train_tinystories_first_compare.json \
  --run-name tinystories_hybrid_50m
```

Longer low-budget T4 comparison run:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_50m_gru.json \
  --train-config configs/train_tinystories_t4.json \
  --run-name tinystories_hybrid_t4
```

## 6. Inspect Metrics And Checkpoints

For any run name, inspect:

```bash
find artifacts/tinystories_dense_smoke -maxdepth 3 -type f | sort
cat artifacts/tinystories_dense_smoke/run.json
cat artifacts/tinystories_dense_smoke/metrics.json
```

Expected outputs:

- `run.json`: model config plus training config used for the run
- `metrics.json`: interleaved `train` and `val` loss records
- `checkpoints/latest.pt`: resumable checkpoint
- `checkpoints/step_*.pt`: periodic checkpoints

Compare two runs with one command:

```bash
python scripts/compare_runs.py \
  artifacts/tinystories_dense_50m \
  artifacts/tinystories_hybrid_50m
```

## 7. Resume A Run

Dense example:

```bash
python scripts/train_dense_baseline.py \
  --config configs/dense_50m.json \
  --train-config configs/train_tinystories_first_compare.json \
  --run-name tinystories_dense_50m_resume \
  --resume-from artifacts/tinystories_dense_50m/checkpoints/latest.pt
```

Hybrid example:

```bash
python scripts/train_hybrid.py \
  --config configs/hybrid_50m_gru.json \
  --train-config configs/train_tinystories_first_compare.json \
  --run-name tinystories_hybrid_50m_resume \
  --resume-from artifacts/tinystories_hybrid_50m/checkpoints/latest.pt
```

## 8. What To Compare First

For the first dense vs hybrid comparison, keep the recipe fixed except for the model config:

- same `train_config`
- same tokenized TinyStories files
- same step count
- same batch size
- same sequence length
- same validation cadence

Look at:

- parameter count
- train loss trend
- validation loss trend
- step time
- tokens per second
- checkpoint stability
- GPU memory use

Recommended first serious comparison:

- dense: `configs/dense_50m.json`
- hybrid: `configs/hybrid_50m_gru.json`
- train config: `configs/train_tinystories_first_compare.json`

## 9. Practical Scope

This milestone is done when:

- both dense and hybrid can train on tokenized TinyStories,
- validation is logged during the run,
- checkpoints are written and resume works,
- and the repo can support a first low-budget comparison without extra infrastructure.
