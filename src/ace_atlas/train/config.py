from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainingConfig:
    run_name: str = "dev"
    steps: int = 10
    batch_size: int = 4
    sequence_length: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 1
    output_dir: str = "artifacts"
    seed: int = 7
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        return cls(**data)

