from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.data.manifest import DatasetEntry, DatasetManifest
from ace_atlas.tokenizer.byte_level import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyStories text, manifest, and tokenized JSONL.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "tinystories")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means use the full train split.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means use the full validation split.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream TinyStories from Hugging Face instead of materializing the full split locally.",
    )
    return parser.parse_args()


def load_split(split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TinyStories prep requires the optional 'data' dependencies. "
            "Install with: python -m pip install -e '.[data]'"
        ) from exc
    return load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)


def write_jsonl_split(dataset, output_path: Path, limit: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get("text")
            if not text:
                continue
            normalized = str(text).strip()
            if not normalized:
                continue
            handle.write(json.dumps({"text": normalized}) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def write_tokenized_split(jsonl_path: Path, output_path: Path, tokenizer: ByteTokenizer) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            sink.write(json.dumps({"tokens": tokenizer.encode(text)}) + "\n")
            count += 1
    return count


def write_manifest(output_dir: Path, sequence_length: int) -> Path:
    manifest = DatasetManifest(
        name="tinystories",
        tokenizer="byte",
        sequence_length=sequence_length,
        entries=[
            DatasetEntry(path=str(output_dir / "train.jsonl"), format="jsonl", split="train", text_key="text"),
            DatasetEntry(path=str(output_dir / "val.jsonl"), format="jsonl", split="val", text_key="text"),
        ],
    )
    manifest_path = output_dir / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_split("train", streaming=args.streaming)
    val_dataset = load_split("validation", streaming=args.streaming)

    train_jsonl_path = args.output_dir / "train.jsonl"
    val_jsonl_path = args.output_dir / "val.jsonl"
    train_count = write_jsonl_split(train_dataset, train_jsonl_path, limit=args.train_limit)
    val_count = write_jsonl_split(val_dataset, val_jsonl_path, limit=args.val_limit)

    manifest_path = write_manifest(args.output_dir, sequence_length=args.sequence_length)
    tokenizer = ByteTokenizer()
    train_token_path = args.output_dir / "train_tokens.jsonl"
    val_token_path = args.output_dir / "val_tokens.jsonl"
    tokenized_train = write_tokenized_split(train_jsonl_path, train_token_path, tokenizer)
    tokenized_val = write_tokenized_split(val_jsonl_path, val_token_path, tokenizer)

    print(f"Wrote train JSONL: {train_count} stories -> {train_jsonl_path}")
    print(f"Wrote val JSONL: {val_count} stories -> {val_jsonl_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote tokenized train: {tokenized_train} examples -> {train_token_path}")
    print(f"Wrote tokenized val: {tokenized_val} examples -> {val_token_path}")


if __name__ == "__main__":
    main()
