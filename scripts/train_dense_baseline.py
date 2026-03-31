from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ACE-Atlas dense baseline.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dense_small.json")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train_dev.json")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--run-name", type=str, default="dense_baseline_dev")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    try:
        from ace_atlas.config import ACEAtlasConfig
        from ace_atlas.train.config import TrainingConfig
        from ace_atlas.train.harness import Trainer
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("Training requires PyTorch. Install project dependencies first, then rerun.")
            return
        raise

    args = parse_args()
    model_config = ACEAtlasConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    train_config = TrainingConfig.from_dict(json.loads(args.train_config.read_text(encoding="utf-8")))
    train_config.run_name = args.run_name
    train_config.steps = args.steps
    train_config.batch_size = args.batch_size
    train_config.sequence_length = args.sequence_length
    train_config.output_dir = str(ROOT / "artifacts")
    train_config.device = args.device
    trainer = Trainer("dense_baseline", model_config, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
