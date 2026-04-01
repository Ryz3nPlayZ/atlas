from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.tokenizer.byte_level import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 raw text and tokenized JSONL.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "wikitext2")
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means full validation split.")
    parser.add_argument("--test-limit", type=int, default=0, help="0 means full test split.")
    return parser.parse_args()


def load_split(split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WikiText-2 prep requires the optional 'data' dependencies. "
            "Install with: python -m pip install -e '.[data]'"
        ) from exc
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


def write_jsonl_split(dataset, output_path: Path, limit: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            handle.write(json.dumps({"text": text}) + "\n")
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


def prepare_split(split: str, limit: int, output_dir: Path, tokenizer: ByteTokenizer) -> tuple[int, int]:
    dataset = load_split(split)
    jsonl_path = output_dir / f"{split}.jsonl"
    token_path = output_dir / f"{split}_tokens.jsonl"
    count = write_jsonl_split(dataset, jsonl_path, limit=limit)
    token_count = write_tokenized_split(jsonl_path, token_path, tokenizer)
    return count, token_count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    for split, limit in (
        ("train", args.train_limit),
        ("validation", args.val_limit),
        ("test", args.test_limit),
    ):
        count, token_count = prepare_split(split, limit, args.output_dir, tokenizer)
        print(f"Wrote {split} JSONL: {count} rows -> {args.output_dir / f'{split}.jsonl'}")
        print(f"Wrote {split} tokenized: {token_count} rows -> {args.output_dir / f'{split}_tokens.jsonl'}")


if __name__ == "__main__":
    main()
