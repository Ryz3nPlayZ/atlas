from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.experiment import summarize_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ACE-Atlas run directories.")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def print_summary(label: str, summary: dict[str, object]) -> None:
    parameter_count = summary["parameter_count"]
    if parameter_count is None:
        parameter_text = "unknown"
    else:
        parameter_text = f"{summary['parameter_count_human']} ({parameter_count:,})"
    print(f"{label}: {summary['run_dir']}")
    print(f"  model: {summary['model_name']}")
    print(f"  params: {parameter_text}")
    print(f"  device: {summary['device_name']}")
    print(f"  final train loss: {summary['final_train_loss']}")
    print(f"  final val loss: {summary['final_validation_loss']}")
    print(f"  best val loss: {summary['best_validation_loss']}")
    print(f"  checkpoints: {summary['checkpoint_count']}")
    print(f"  resume metadata: {summary['resume_metadata_exists']}")
    print(f"  avg step time (s): {summary['avg_step_time_sec']}")
    print(f"  avg tokens/sec: {summary['avg_tokens_per_sec']}")


def main() -> None:
    args = parse_args()
    summary_a = summarize_run_dir(args.run_a)
    summary_b = summarize_run_dir(args.run_b)
    payload = {"run_a": summary_a, "run_b": summary_b}

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print_summary("Run A", summary_a)
    print()
    print_summary("Run B", summary_b)


if __name__ == "__main__":
    main()
