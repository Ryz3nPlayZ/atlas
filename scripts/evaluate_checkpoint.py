from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.experiment import build_model, count_parameters, format_parameter_count
from ace_atlas.train.data import build_tokenized_lm_dataloader
from ace_atlas.train.objectives import total_training_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on tokenized JSONL.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--runtime-profile-every", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = ACEAtlasConfig.from_dict(payload["model_config"])
    model_name = payload["model_name"]
    model = build_model(model_name, config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dataloader = build_tokenized_lm_dataloader(
        path=args.data,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        shuffle=False,
    )

    totals = {"loss": 0.0, "lm_loss": 0.0, "mtp_loss": 0.0}
    batch_times: list[float] = []
    tokens_per_sec: list[float] = []
    peak_memory_values: list[float] = []
    runtime_samples: list[dict[str, float]] = []
    batches_seen = 0

    with torch.no_grad():
        for batches_seen, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            collect_runtime = args.runtime_profile_every > 0 and batches_seen % args.runtime_profile_every == 0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = model(input_ids, collect_runtime_stats=collect_runtime)
            losses = total_training_loss(output, labels)
            step_time = time.perf_counter() - started

            for name, value in losses.items():
                totals[name] += float(value.detach().cpu().item())
            batch_times.append(step_time)
            tokens = input_ids.numel()
            tokens_per_sec.append(tokens / step_time if step_time > 0 else 0.0)
            if device.type == "cuda":
                peak_memory_values.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if output.runtime_stats is not None:
                runtime_samples.append({f"runtime_{k}_sec": v for k, v in output.runtime_stats.items()})

            if args.max_batches and batches_seen >= args.max_batches:
                break

    if batches_seen == 0:
        raise ValueError("Evaluation dataloader yielded no batches")

    metrics = {
        "model_name": model_name,
        "parameter_count": count_parameters(model)["total"],
        "parameter_count_human": format_parameter_count(count_parameters(model)["total"]),
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "batches": batches_seen,
        "loss": totals["loss"] / batches_seen,
        "lm_loss": totals["lm_loss"] / batches_seen,
        "mtp_loss": totals["mtp_loss"] / batches_seen,
        "perplexity": math.exp(totals["loss"] / batches_seen),
        "avg_batch_time_sec": sum(batch_times) / len(batch_times),
        "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
        "max_peak_memory_mb": max(peak_memory_values) if peak_memory_values else None,
        "runtime_breakdown_samples": runtime_samples,
    }

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
