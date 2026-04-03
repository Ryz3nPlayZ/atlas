from __future__ import annotations

import json
from pathlib import Path

import torch

from ace_atlas.config import ACEAtlasConfig, AttentionConfig, ArbiterConfig, MemoryConfig, MoEConfig, RecurrentConfig
from ace_atlas.train.config import TrainingConfig
from ace_atlas.train.harness import Trainer


def tiny_model_config() -> ACEAtlasConfig:
    return ACEAtlasConfig(
        vocab_size=256,
        model_dim=32,
        num_layers=2,
        attention_every_n=1,
        max_position_embeddings=256,
        mtp_horizon=2,
        attention=AttentionConfig(window_size=32, num_heads=4, num_kv_heads=2),
        recurrent=RecurrentConfig(state_dim=32),
        moe=MoEConfig(enabled=False, num_shared_experts=1, num_routed_experts=2, hidden_dim=64),
        memory=MemoryConfig(enabled=False, key_dim=32, value_dim=32),
        arbiter=ArbiterConfig(enabled=False, hidden_dim=32),
    )


def tiny_gru_model_config() -> ACEAtlasConfig:
    config = tiny_model_config()
    config.recurrent = RecurrentConfig(kind="gru_fused", state_dim=24, expansion_factor=2)
    return config


def tiny_gru_segment_adapter_config() -> ACEAtlasConfig:
    config = tiny_gru_model_config()
    config.answer_span_embeddings = True
    config.completion_adapter_dim = 8
    return config


def write_tokenized_file(path: Path, records: list[list[int]]) -> None:
    payload = "\n".join(json.dumps({"tokens": record}) for record in records) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_trainer_builds_distinct_train_and_val_loaders(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_tokenized_file(train_path, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    write_tokenized_file(val_path, [[21, 22, 23, 24, 25]])

    config = TrainingConfig(
        run_name="split_check",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        output_dir=str(tmp_path / "artifacts"),
        device="cpu",
    )

    trainer = Trainer("dense_baseline", tiny_model_config(), config)
    train_loader, val_loader = trainer.build_dataloaders()

    assert len(train_loader.dataset) == 2
    assert val_loader is not None
    assert len(val_loader.dataset) == 1


def test_trainer_supports_tokenized_validation_and_resume(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "artifacts"

    write_tokenized_file(
        train_path,
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
    )
    write_tokenized_file(
        val_path,
        [
            [21, 22, 23, 24, 25, 26, 27, 28, 29],
        ],
    )

    first_config = TrainingConfig(
        run_name="tokenized_dev",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        validation_every=1,
        checkpoint_every=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    trainer = Trainer("dense_baseline", tiny_model_config(), first_config)
    first_metrics = trainer.train()

    checkpoint_path = output_dir / "tokenized_dev" / "checkpoints" / "latest.pt"
    run_path = output_dir / "tokenized_dev" / "run.json"
    assert checkpoint_path.exists()
    assert any(record["phase"] == "val" for record in first_metrics)
    assert [record["phase"] for record in first_metrics] == ["train", "val"]
    assert "step_time_sec" in first_metrics[0]
    assert "tokens_per_sec" in first_metrics[0]

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["model_stats"]["parameter_count"] > 0
    assert "torch_version" in run_payload["system"]
    assert run_payload["training_config"]["tokenizer_name"] == "byte"

    resumed_config = TrainingConfig(
        run_name="tokenized_dev",
        steps=2,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        validation_every=1,
        checkpoint_every=1,
        resume_from=str(checkpoint_path),
        output_dir=str(output_dir),
        device="cpu",
    )

    resumed_trainer = Trainer("dense_baseline", tiny_model_config(), resumed_config)
    resumed_metrics = resumed_trainer.train()

    assert resumed_metrics[-1]["step"] == 2.0
    saved_metrics = json.loads((output_dir / "tokenized_dev" / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics[-1]["step"] == 2.0

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["step"] == 2


def test_trainer_supports_gru_recurrent_core(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "artifacts"

    write_tokenized_file(
        train_path,
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
    )
    write_tokenized_file(val_path, [[21, 22, 23, 24, 25, 26, 27, 28, 29]])

    config = TrainingConfig(
        run_name="gru_dev",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        validation_every=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    trainer = Trainer("ace_atlas", tiny_gru_model_config(), config)
    metrics = trainer.train()

    assert metrics[0]["phase"] == "train"
    assert metrics[-1]["phase"] == "val"


def test_trainer_supports_init_from_checkpoint(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "artifacts"

    write_tokenized_file(
        train_path,
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
    )
    write_tokenized_file(val_path, [[21, 22, 23, 24, 25, 26, 27, 28, 29]])

    base_config = TrainingConfig(
        run_name="base_ckpt",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        checkpoint_every=1,
        output_dir=str(output_dir),
        device="cpu",
    )
    base_trainer = Trainer("ace_atlas", tiny_gru_model_config(), base_config)
    base_trainer.train()

    checkpoint_path = output_dir / "base_ckpt" / "checkpoints" / "latest.pt"

    init_config = TrainingConfig(
        run_name="init_ckpt",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        init_from=str(checkpoint_path),
        output_dir=str(output_dir),
        device="cpu",
    )
    init_trainer = Trainer("ace_atlas", tiny_gru_model_config(), init_config)
    metrics = init_trainer.train()

    assert metrics[0]["phase"] == "train"
    assert metrics[0]["step"] == 1.0


def test_trainer_supports_nonstrict_init_for_new_code_params(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "artifacts"

    write_tokenized_file(
        train_path,
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
    )
    write_tokenized_file(val_path, [[21, 22, 23, 24, 25, 26, 27, 28, 29]])

    base_config = TrainingConfig(
        run_name="base_ckpt_partial",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        checkpoint_every=1,
        output_dir=str(output_dir),
        device="cpu",
    )
    base_trainer = Trainer("ace_atlas", tiny_gru_model_config(), base_config)
    base_trainer.train()

    checkpoint_path = output_dir / "base_ckpt_partial" / "checkpoints" / "latest.pt"

    init_config = TrainingConfig(
        run_name="init_ckpt_partial",
        steps=1,
        batch_size=1,
        sequence_length=4,
        data_mode="tokenized",
        train_data_path=str(train_path),
        val_data_path=str(val_path),
        init_from=str(checkpoint_path),
        init_strict=False,
        output_dir=str(output_dir),
        device="cpu",
    )
    init_trainer = Trainer("ace_atlas", tiny_gru_segment_adapter_config(), init_config)
    metrics = init_trainer.train()

    assert metrics[0]["phase"] == "train"
    assert metrics[0]["step"] == 1.0
