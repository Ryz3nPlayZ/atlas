from __future__ import annotations

import json
from pathlib import Path

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.experiment import build_model, count_parameters, format_parameter_count, summarize_run_dir


def test_count_parameters_returns_positive_counts() -> None:
    model = build_model("dense_baseline", ACEAtlasConfig.small())
    stats = count_parameters(model)
    assert stats["total"] > 0
    assert stats["trainable"] == stats["total"]
    assert stats["non_embedding"] < stats["total"]


def test_format_parameter_count_uses_compact_units() -> None:
    assert format_parameter_count(1_234) == "1.23K"
    assert format_parameter_count(12_345_678) == "12.35M"


def test_summarize_run_dir_collects_comparison_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "model_name": "dense_baseline",
                "model_stats": {"parameter_count": 12345678, "parameter_count_human": "12.35M"},
                "system": {"device_name": "cpu"},
                "training_config": {"resume_from": "artifacts/other/latest.pt"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            [
                {"phase": "train", "step": 1.0, "loss": 3.0, "step_time_sec": 0.5, "tokens_per_sec": 200.0},
                {"phase": "train", "step": 1.5, "loss": 2.8, "step_time_sec": 0.4, "tokens_per_sec": 220.0, "peak_memory_mb": 512.0},
                {"phase": "val", "step": 1.0, "loss": 2.5},
                {"phase": "train", "step": 2.0, "loss": 2.0, "step_time_sec": 0.25, "tokens_per_sec": 400.0, "peak_memory_mb": 768.0},
                {"phase": "val", "step": 2.0, "loss": 1.5},
            ]
        ),
        encoding="utf-8",
    )
    (checkpoints / "latest.pt").write_bytes(b"checkpoint")
    (checkpoints / "step_000002.pt").write_bytes(b"checkpoint")

    summary = summarize_run_dir(run_dir)

    assert summary["final_train_loss"] == 2.0
    assert summary["final_validation_loss"] == 1.5
    assert summary["best_validation_loss"] == 1.5
    assert summary["checkpoint_count"] == 2
    assert summary["resume_metadata_exists"] is True
    assert summary["avg_step_time_sec"] == 0.3833333333333333
    assert summary["avg_tokens_per_sec"] == 273.3333333333333
    assert summary["max_peak_memory_mb"] == 768.0
