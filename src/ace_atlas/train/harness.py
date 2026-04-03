from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.experiment import build_model, collect_system_metadata, count_parameters, format_parameter_count, load_json
from ace_atlas.model.atlas_transformer import ACEAtlasTransformerModel
from ace_atlas.model.backbone import ACEAtlasModel
from ace_atlas.model.dense_baseline import DenseCausalTransformer
from ace_atlas.train.config import TrainingConfig
from ace_atlas.train.data import build_random_lm_dataloader, build_tokenized_lm_dataloader
from ace_atlas.train.objectives import total_training_loss


MODEL_REGISTRY = {
    "dense_baseline": DenseCausalTransformer,
    "ace_atlas": ACEAtlasModel,
    "ace_atlas_transformer": ACEAtlasTransformerModel,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class Trainer:
    def __init__(self, model_name: str, model_config: ACEAtlasConfig, training_config: TrainingConfig) -> None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.model_name = model_name
        self.model_config = model_config
        self.training_config = training_config
        set_seed(training_config.seed)
        self.device = resolve_device(training_config.device)
        self.model: nn.Module = build_model(model_name, model_config).to(self.device)
        if isinstance(self.model, (ACEAtlasModel, ACEAtlasTransformerModel)):
            self.model.enable_activation_checkpointing(training_config.activation_checkpointing)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.micro_batch_size = training_config.micro_batch_size or training_config.batch_size
        if training_config.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        expected_accum = training_config.batch_size // self.micro_batch_size
        if training_config.grad_accum_steps != expected_accum:
            raise ValueError(
                "grad_accum_steps must equal batch_size // micro_batch_size for a stable effective batch"
            )
        self.autocast_enabled = self.device.type == "cuda" and training_config.mixed_precision in {"fp16", "bf16"}
        self.autocast_dtype = (
            torch.float16 if training_config.mixed_precision == "fp16" else torch.bfloat16
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and training_config.mixed_precision == "fp16",
        )
        self.model_stats = count_parameters(self.model)
        self.system_metadata = collect_system_metadata(self.device)
        self.teacher_model: nn.Module | None = None
        if (
            training_config.distill_weight > 0.0
            and training_config.teacher_model_name
            and training_config.teacher_config_path
            and training_config.teacher_checkpoint_path
        ):
            teacher_config = ACEAtlasConfig.from_dict(load_json(training_config.teacher_config_path))
            self.teacher_model = build_model(training_config.teacher_model_name, teacher_config).to(self.device)
            teacher_payload = torch.load(training_config.teacher_checkpoint_path, map_location=self.device)
            self.teacher_model.load_state_dict(teacher_payload["model_state_dict"])
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)

    def write_run_metadata(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "model_stats": {
                "parameter_count": self.model_stats["total"],
                "parameter_count_human": format_parameter_count(self.model_stats["total"]),
                "trainable_parameter_count": self.model_stats["trainable"],
                "trainable_parameter_count_human": format_parameter_count(self.model_stats["trainable"]),
                "non_embedding_parameter_count": self.model_stats["non_embedding"],
                "non_embedding_parameter_count_human": format_parameter_count(
                    self.model_stats["non_embedding"]
                ),
            },
            "system": self.system_metadata,
        }
        (output_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        config = self.training_config
        if config.data_mode == "synthetic":
            train_loader = build_random_lm_dataloader(
                vocab_size=self.model_config.vocab_size,
                sequence_length=config.sequence_length,
                batch_size=self.micro_batch_size,
                total_examples=config.steps * config.batch_size,
            )
            val_loader = None
            return train_loader, val_loader

        if config.data_mode != "tokenized":
            raise ValueError(f"Unsupported data_mode: {config.data_mode}")
        if not config.train_data_path:
            raise ValueError("train_data_path is required when data_mode='tokenized'")

        train_loader = build_tokenized_lm_dataloader(
            path=config.train_data_path,
            sequence_length=config.sequence_length,
            batch_size=self.micro_batch_size,
            shuffle=True,
        )
        val_loader = None
        if config.val_data_path:
            val_loader = build_tokenized_lm_dataloader(
                path=config.val_data_path,
                sequence_length=config.sequence_length,
                batch_size=self.micro_batch_size,
                shuffle=False,
            )
        return train_loader, val_loader

    def cycle_batches(self, dataloader: DataLoader) -> Iterator[dict[str, Tensor]]:
        while True:
            for batch in dataloader:
                yield batch

    def run_step(
        self,
        batch: dict[str, Tensor],
        training: bool,
        collect_runtime_profile: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, float] | None]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.device)
        mode_ids = batch.get("mode_ids")
        if mode_ids is not None:
            mode_ids = mode_ids.to(self.device)
        teacher_logits = None
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            output = self.model(
                input_ids,
                segment_ids=segment_ids,
                mode_ids=mode_ids,
                collect_runtime_stats=collect_runtime_profile,
            )
            if training and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_output = self.teacher_model(
                        input_ids,
                        segment_ids=segment_ids,
                        mode_ids=mode_ids,
                        collect_runtime_stats=False,
                    )
                teacher_logits = teacher_output.logits
            losses = total_training_loss(
                output,
                labels,
                teacher_logits=teacher_logits,
                distill_weight=self.training_config.distill_weight,
                distill_temperature=self.training_config.distill_temperature,
            )

        return losses, output.runtime_stats

    def evaluate(self, dataloader: DataLoader, step: int) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {"loss": 0.0, "lm_loss": 0.0, "mtp_loss": 0.0, "distill_loss": 0.0}
        batches_seen = 0
        max_batches = self.training_config.validation_batches
        started = time.perf_counter()

        with torch.no_grad():
            for batches_seen, batch in enumerate(dataloader, start=1):
                losses, _ = self.run_step(batch, training=False)
                for name, value in losses.items():
                    totals[name] += float(value.detach().cpu().item())
                if max_batches and batches_seen >= max_batches:
                    break

        if batches_seen == 0:
            raise ValueError("Validation dataloader yielded no batches")

        metrics = {name: total / batches_seen for name, total in totals.items()}
        metrics["step"] = float(step)
        metrics["phase"] = "val"
        metrics["elapsed_time_sec"] = time.perf_counter() - started
        self.model.train()
        return metrics

    def checkpoint_payload(self, step: int, metrics: list[dict[str, float]]) -> dict:
        return {
            "model_name": self.model_name,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

    def save_checkpoint(self, output_dir: Path, step: int, metrics: list[dict[str, float]]) -> None:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload(step, metrics)
        latest_path = checkpoint_dir / "latest.pt"
        step_path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(payload, latest_path)
        try:
            torch.save(payload, step_path)
        except RuntimeError:
            step_path.unlink(missing_ok=True)

    def load_checkpoint(self, checkpoint_path: str | Path) -> tuple[int, list[dict[str, float]]]:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return int(payload.get("step", 0)), list(payload.get("metrics", []))

    def initialize_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        load_result = self.model.load_state_dict(
            payload["model_state_dict"],
            strict=self.training_config.init_strict,
        )
        if not self.training_config.init_strict:
            if load_result.missing_keys:
                print(
                    f"[{self.model_name}] init_from missing_keys={len(load_result.missing_keys)}"
                )
            if load_result.unexpected_keys:
                print(
                    f"[{self.model_name}] init_from unexpected_keys={len(load_result.unexpected_keys)}"
                )

    def maybe_run_validation(
        self,
        val_loader: DataLoader | None,
        step: int,
        metrics: list[dict[str, float]],
    ) -> None:
        if val_loader is None or self.training_config.validation_every <= 0:
            return
        if step % self.training_config.validation_every != 0:
            return

        val_record = self.evaluate(val_loader, step=step)
        metrics.append(val_record)
        print(
            f"[{self.model_name}] step={step} "
            f"val_loss={val_record['loss']:.4f} "
            f"val_lm={val_record['lm_loss']:.4f} "
            f"val_mtp={val_record['mtp_loss']:.4f}"
        )

    def write_metrics(self, output_dir: Path, metrics: list[dict[str, float]]) -> None:
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def train(self) -> list[dict[str, float]]:
        output_dir = Path(self.training_config.output_dir) / self.training_config.run_name
        self.write_run_metadata(output_dir)
        print(
            f"[{self.model_name}] device={self.system_metadata['device_name']} "
            f"params={format_parameter_count(self.model_stats['total'])} "
            f"trainable={format_parameter_count(self.model_stats['trainable'])}"
        )
        if self.teacher_model is not None:
            print(
                f"[{self.model_name}] teacher={self.training_config.teacher_model_name} "
                f"distill_weight={self.training_config.distill_weight}"
            )

        train_loader, val_loader = self.build_dataloaders()
        train_batches = self.cycle_batches(train_loader)
        metrics: list[dict[str, float]] = []
        start_step = 0

        if self.training_config.init_from:
            self.initialize_from_checkpoint(self.training_config.init_from)

        if self.training_config.resume_from:
            start_step, metrics = self.load_checkpoint(self.training_config.resume_from)

        self.model.train()
        for step in range(start_step + 1, self.training_config.steps + 1):
            collect_runtime_profile = (
                self.training_config.runtime_profile_every > 0
                and step % self.training_config.runtime_profile_every == 0
            )
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            runtime_stats: dict[str, float] | None = None
            aggregated_losses = {"loss": 0.0, "lm_loss": 0.0, "mtp_loss": 0.0, "distill_loss": 0.0}
            for micro_step in range(self.training_config.grad_accum_steps):
                batch = next(train_batches)
                micro_losses, micro_runtime_stats = self.run_step(
                    batch,
                    training=True,
                    collect_runtime_profile=collect_runtime_profile and micro_step == self.training_config.grad_accum_steps - 1,
                )
                loss = micro_losses["loss"] / self.training_config.grad_accum_steps
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                for name, value in micro_losses.items():
                    aggregated_losses[name] += float(value.detach().cpu().item())
                if micro_runtime_stats is not None:
                    runtime_stats = micro_runtime_stats

            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            step_time = time.perf_counter() - started

            record = {
                name: total / self.training_config.grad_accum_steps for name, total in aggregated_losses.items()
            }
            record["step"] = float(step)
            record["phase"] = "train"
            record["step_time_sec"] = step_time
            tokens_per_step = self.training_config.batch_size * self.training_config.sequence_length
            record["tokens_per_sec"] = tokens_per_step / step_time if step_time > 0 else 0.0
            if self.device.type == "cuda":
                record["peak_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if runtime_stats is not None:
                for key, value in runtime_stats.items():
                    record[f"runtime_{key}_sec"] = value
            metrics.append(record)

            if step % self.training_config.log_every == 0:
                print(
                    f"[{self.model_name}] step={step} "
                    f"loss={record['loss']:.4f} "
                    f"lm={record['lm_loss']:.4f} "
                    f"mtp={record['mtp_loss']:.4f} "
                    f"distill={record['distill_loss']:.4f} "
                    f"step_time={record['step_time_sec']:.3f}s "
                    f"tok/s={record['tokens_per_sec']:.1f}"
                )

            self.maybe_run_validation(val_loader, step=step, metrics=metrics)

            if self.training_config.checkpoint_every > 0 and step % self.training_config.checkpoint_every == 0:
                self.save_checkpoint(output_dir, step=step, metrics=metrics)
                self.write_metrics(output_dir, metrics)

        self.save_checkpoint(output_dir, step=self.training_config.steps, metrics=metrics)
        self.write_metrics(output_dir, metrics)
        return metrics
