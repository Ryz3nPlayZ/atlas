from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainingConfig:
    run_name: str = "dev"
    steps: int = 10
    batch_size: int = 4
    micro_batch_size: int | None = None
    grad_accum_steps: int = 1
    sequence_length: int = 128
    data_mode: str = "synthetic"
    train_data_path: str | None = None
    val_data_path: str | None = None
    tokenizer_name: str = "byte"
    tokenizer_path: str | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 1
    validation_every: int = 0
    validation_batches: int = 0
    checkpoint_every: int = 0
    runtime_profile_every: int = 0
    resume_from: str | None = None
    init_from: str | None = None
    init_strict: bool = True
    teacher_model_name: str | None = None
    teacher_config_path: str | None = None
    teacher_checkpoint_path: str | None = None
    distill_weight: float = 0.0
    distill_temperature: float = 1.0
    output_dir: str = "artifacts"
    seed: int = 7
    device: str = "cuda"
    mixed_precision: str = "none"
    activation_checkpointing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        return cls(**data)
