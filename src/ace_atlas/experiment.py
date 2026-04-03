from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.atlas_transformer import ACEAtlasTransformerModel
from ace_atlas.model.backbone import ACEAtlasModel
from ace_atlas.model.dense_baseline import DenseCausalTransformer


MODEL_REGISTRY = {
    "dense_baseline": DenseCausalTransformer,
    "ace_atlas": ACEAtlasModel,
    "ace_atlas_transformer": ACEAtlasTransformerModel,
}


def build_model(model_name: str, config: ACEAtlasConfig) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model_name: {model_name}") from exc
    return model_cls(config)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_embedding = total - sum(parameter.numel() for name, parameter in model.named_parameters() if "embed_tokens" in name)
    return {
        "total": total,
        "trainable": trainable,
        "non_embedding": non_embedding,
    }


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def collect_system_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_device_count"] = torch.cuda.device_count()
    else:
        metadata["device_name"] = str(device)
        metadata["cuda_device_count"] = 0
    return metadata


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_run_dir(path: str | Path) -> dict[str, object]:
    run_dir = Path(path)
    run_payload = load_json(run_dir / "run.json")
    metrics = load_json(run_dir / "metrics.json")
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").exists() else []

    train_records = [record for record in metrics if record.get("phase") == "train"]
    val_records = [record for record in metrics if record.get("phase") == "val"]

    final_train = train_records[-1] if train_records else None
    final_val = val_records[-1] if val_records else None
    best_val = min((record["loss"] for record in val_records), default=None)

    train_step_times = [record["step_time_sec"] for record in train_records if "step_time_sec" in record]
    train_tokens_per_sec = [
        record["tokens_per_sec"] for record in train_records if "tokens_per_sec" in record
    ]
    peak_memory_values = [record["peak_memory_mb"] for record in train_records if "peak_memory_mb" in record]

    return {
        "run_dir": str(run_dir),
        "model_name": run_payload["model_name"],
        "parameter_count": run_payload.get("model_stats", {}).get("parameter_count"),
        "parameter_count_human": run_payload.get("model_stats", {}).get("parameter_count_human"),
        "device_name": run_payload.get("system", {}).get("device_name"),
        "final_train_loss": final_train["loss"] if final_train else None,
        "final_validation_loss": final_val["loss"] if final_val else None,
        "best_validation_loss": best_val,
        "checkpoint_count": len(checkpoints),
        "resume_metadata_exists": bool(run_payload.get("training_config", {}).get("resume_from")),
        "avg_step_time_sec": sum(train_step_times) / len(train_step_times) if train_step_times else None,
        "avg_tokens_per_sec": (
            sum(train_tokens_per_sec) / len(train_tokens_per_sec) if train_tokens_per_sec else None
        ),
        "max_peak_memory_mb": max(peak_memory_values) if peak_memory_values else None,
    }
