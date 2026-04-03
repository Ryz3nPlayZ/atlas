from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.experiment import build_model, count_parameters, format_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report parameter counts for an ACE-Atlas model config.")
    parser.add_argument("--model-name", required=True, choices=["dense_baseline", "ace_atlas", "ace_atlas_transformer"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ACEAtlasConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    model = build_model(args.model_name, config)
    stats = count_parameters(model)
    payload = {
        "model_name": args.model_name,
        "config_path": str(args.config),
        "parameter_count": stats["total"],
        "parameter_count_human": format_parameter_count(stats["total"]),
        "trainable_parameter_count": stats["trainable"],
        "trainable_parameter_count_human": format_parameter_count(stats["trainable"]),
        "non_embedding_parameter_count": stats["non_embedding"],
        "non_embedding_parameter_count_human": format_parameter_count(stats["non_embedding"]),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Model: {payload['model_name']}")
    print(f"Config: {payload['config_path']}")
    print(
        f"Parameters: {payload['parameter_count_human']} "
        f"({payload['parameter_count']:,})"
    )
    print(
        f"Trainable: {payload['trainable_parameter_count_human']} "
        f"({payload['trainable_parameter_count']:,})"
    )
    print(
        f"Non-embedding: {payload['non_embedding_parameter_count_human']} "
        f"({payload['non_embedding_parameter_count']:,})"
    )


if __name__ == "__main__":
    main()
