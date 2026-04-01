from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.backbone import ACEAtlasModel
from ace_atlas.model.dense_baseline import DenseCausalTransformer
from ace_atlas.train.config import TrainingConfig
from ace_atlas.train.data import build_random_lm_dataloader, build_tokenized_lm_dataloader
from ace_atlas.train.objectives import total_training_loss


MODEL_REGISTRY = {
    "dense_baseline": DenseCausalTransformer,
    "ace_atlas": ACEAtlasModel,
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
        self.model: nn.Module = MODEL_REGISTRY[model_name](model_config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

    def write_run_metadata(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
        }
        (output_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        config = self.training_config
        if config.data_mode == "synthetic":
            train_loader = build_random_lm_dataloader(
                vocab_size=self.model_config.vocab_size,
                sequence_length=config.sequence_length,
                batch_size=config.batch_size,
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
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = None
        if config.val_data_path:
            val_loader = build_tokenized_lm_dataloader(
                path=config.val_data_path,
                sequence_length=config.sequence_length,
                batch_size=config.batch_size,
                shuffle=False,
            )
        return train_loader, val_loader

    def cycle_batches(self, dataloader: DataLoader) -> Iterator[dict[str, Tensor]]:
        while True:
            for batch in dataloader:
                yield batch

    def run_step(self, batch: dict[str, Tensor], training: bool) -> dict[str, Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        output = self.model(input_ids)
        losses = total_training_loss(output, labels)

        if training:
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
            self.optimizer.step()

        return losses

    def evaluate(self, dataloader: DataLoader, step: int) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {"loss": 0.0, "lm_loss": 0.0, "mtp_loss": 0.0}
        batches_seen = 0
        max_batches = self.training_config.validation_batches

        with torch.no_grad():
            for batches_seen, batch in enumerate(dataloader, start=1):
                losses = self.run_step(batch, training=False)
                for name, value in losses.items():
                    totals[name] += float(value.detach().cpu().item())
                if max_batches and batches_seen >= max_batches:
                    break

        if batches_seen == 0:
            raise ValueError("Validation dataloader yielded no batches")

        metrics = {name: total / batches_seen for name, total in totals.items()}
        metrics["step"] = float(step)
        metrics["phase"] = "val"
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
        torch.save(payload, step_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> tuple[int, list[dict[str, float]]]:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return int(payload.get("step", 0)), list(payload.get("metrics", []))

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

        train_loader, val_loader = self.build_dataloaders()
        train_batches = self.cycle_batches(train_loader)
        metrics: list[dict[str, float]] = []
        start_step = 0

        if self.training_config.resume_from:
            start_step, metrics = self.load_checkpoint(self.training_config.resume_from)

        self.model.train()
        for step in range(start_step + 1, self.training_config.steps + 1):
            batch = next(train_batches)
            losses = self.run_step(batch, training=True)

            record = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
            record["step"] = float(step)
            record["phase"] = "train"
            metrics.append(record)

            if step % self.training_config.log_every == 0:
                print(
                    f"[{self.model_name}] step={step} "
                    f"loss={record['loss']:.4f} "
                    f"lm={record['lm_loss']:.4f} "
                    f"mtp={record['mtp_loss']:.4f}"
                )

            self.maybe_run_validation(val_loader, step=step, metrics=metrics)

            if self.training_config.checkpoint_every > 0 and step % self.training_config.checkpoint_every == 0:
                self.save_checkpoint(output_dir, step=step, metrics=metrics)
                self.write_metrics(output_dir, metrics)

        self.save_checkpoint(output_dir, step=self.training_config.steps, metrics=metrics)
        self.write_metrics(output_dir, metrics)
        return metrics
