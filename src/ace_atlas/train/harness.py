from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import AdamW

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.backbone import ACEAtlasModel
from ace_atlas.model.dense_baseline import DenseCausalTransformer
from ace_atlas.train.config import TrainingConfig
from ace_atlas.train.data import build_random_lm_dataloader
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

    def train(self) -> list[dict[str, float]]:
        self.model.train()
        dataloader = build_random_lm_dataloader(
            vocab_size=self.model_config.vocab_size,
            sequence_length=self.training_config.sequence_length,
            batch_size=self.training_config.batch_size,
            total_examples=self.training_config.steps * self.training_config.batch_size,
        )
        metrics: list[dict[str, float]] = []
        output_dir = Path(self.training_config.output_dir) / self.training_config.run_name
        self.write_run_metadata(output_dir)

        for step, batch in enumerate(dataloader, start=1):
            if step > self.training_config.steps:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(input_ids)
            losses = total_training_loss(output, labels)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
            self.optimizer.step()

            record = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
            record["step"] = float(step)
            metrics.append(record)
            if step % self.training_config.log_every == 0:
                print(
                    f"[{self.model_name}] step={step} "
                    f"loss={record['loss']:.4f} "
                    f"lm={record['lm_loss']:.4f} "
                    f"mtp={record['mtp_loss']:.4f}"
                )

        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

