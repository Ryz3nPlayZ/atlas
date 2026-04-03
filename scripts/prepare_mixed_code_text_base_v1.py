from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local code-heavy mixed-base tokenized dataset from existing tokenized corpora."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "mixed_code_text_base_v1")
    parser.add_argument(
        "--code-train-path",
        type=Path,
        default=ROOT / "data" / "code_search_net_python_ft" / "train_tokens.jsonl",
    )
    parser.add_argument(
        "--code-val-path",
        type=Path,
        default=ROOT / "data" / "code_search_net_python_ft" / "validation_tokens.jsonl",
    )
    parser.add_argument(
        "--text-train-path",
        type=Path,
        default=ROOT / "data" / "wikitext2" / "train_tokens.jsonl",
    )
    parser.add_argument(
        "--text-val-path",
        type=Path,
        default=ROOT / "data" / "wikitext2" / "validation_tokens.jsonl",
    )
    parser.add_argument("--code-repeat", type=int, default=2)
    parser.add_argument("--text-repeat", type=int, default=1)
    parser.add_argument("--text-val-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    code_train = read_jsonl(args.code_train_path)
    code_val = read_jsonl(args.code_val_path)
    text_train = read_jsonl(args.text_train_path)
    text_val = read_jsonl(args.text_val_path)[: args.text_val_limit]

    train_records = code_train * args.code_repeat + text_train * args.text_repeat
    rng.shuffle(train_records)

    val_records = code_val + text_val
    rng.shuffle(val_records)

    train_path = args.output_dir / "train_tokens.jsonl"
    val_path = args.output_dir / "validation_tokens.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"Wrote train tokenized: {len(train_records)} rows -> {train_path}")
    print(f"Wrote validation tokenized: {len(val_records)} rows -> {val_path}")
    print(
        "Train composition: "
        f"code={len(code_train) * args.code_repeat} text={len(text_train) * args.text_repeat}"
    )
    print(f"Validation composition: code={len(code_val)} text={len(text_val)}")


if __name__ == "__main__":
    main()
