from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.data.manifest import DatasetManifest, infer_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dataset manifest for ACE-Atlas.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--name", type=str, default="local-corpus")
    parser.add_argument("--tokenizer", type=str, default="byte")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "dataset_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted([p for p in args.input_dir.rglob("*") if p.suffix in {".txt", ".jsonl"}])
    entries = [
        infer_entry(path, split="train", text_key=args.text_key if path.suffix == ".jsonl" else None)
        for path in paths
    ]
    manifest = DatasetManifest(
        name=args.name,
        tokenizer=args.tokenizer,
        sequence_length=args.sequence_length,
        entries=entries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(args.output)
    print(f"Wrote dataset manifest with {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()

