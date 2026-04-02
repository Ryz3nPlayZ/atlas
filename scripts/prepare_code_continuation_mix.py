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
    parser = argparse.ArgumentParser(
        description="Prepare a low-budget code continuation mix from CodeSearchNet Python plus MBPP."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "code_mix_ft")
    parser.add_argument("--codesearchnet-train-limit", type=int, default=50000)
    parser.add_argument("--codesearchnet-validation-limit", type=int, default=1000)
    parser.add_argument("--mbpp-repeat", type=int, default=40)
    return parser.parse_args()


def load_codesearchnet(split: str, limit: int):
    from datasets import load_dataset

    split_spec = f"{split}[:{limit}]" if limit else split
    return load_dataset("code_search_net", "python", split=split_spec)


def load_mbpp(split: str):
    from datasets import load_dataset

    return load_dataset("mbpp", "sanitized", split=split)


def serialize_codesearchnet_train(rows) -> list[str]:
    serialized: list[str] = []
    for row in rows:
        code = str(row.get("func_code_string", "")).strip()
        if code:
            serialized.append(code)
    return serialized


def serialize_mbpp_examples(rows, repeat: int) -> list[str]:
    serialized: list[str] = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        code = str(row.get("code", "")).strip()
        if not prompt or not code:
            continue
        text = f"# Task: {prompt}\n{code}\n"
        for _ in range(repeat):
            serialized.append(text)
    return serialized


def write_jsonl(records: list[str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in records:
            handle.write(json.dumps({"text": text}) + "\n")
    return len(records)


def write_tokenized(jsonl_path: Path, token_path: Path, tokenizer: ByteTokenizer) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as source, token_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            sink.write(json.dumps({"tokens": tokenizer.encode(text)}) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    cs_train = serialize_codesearchnet_train(load_codesearchnet("train", args.codesearchnet_train_limit))
    mbpp_train = serialize_mbpp_examples(load_mbpp("train"), repeat=args.mbpp_repeat)
    mbpp_val = serialize_mbpp_examples(load_mbpp("validation"), repeat=args.mbpp_repeat)
    mbpp_prompt = serialize_mbpp_examples(load_mbpp("prompt"), repeat=args.mbpp_repeat)
    train_records = cs_train + mbpp_train + mbpp_val + mbpp_prompt

    val_records = serialize_codesearchnet_train(
        load_codesearchnet("validation", args.codesearchnet_validation_limit)
    )

    train_jsonl = args.output_dir / "train.jsonl"
    val_jsonl = args.output_dir / "validation.jsonl"
    train_tokens = args.output_dir / "train_tokens.jsonl"
    val_tokens = args.output_dir / "validation_tokens.jsonl"

    train_count = write_jsonl(train_records, train_jsonl)
    val_count = write_jsonl(val_records, val_jsonl)
    train_token_count = write_tokenized(train_jsonl, train_tokens, tokenizer)
    val_token_count = write_tokenized(val_jsonl, val_tokens, tokenizer)

    print(f"Wrote train JSONL: {train_count} rows -> {train_jsonl}")
    print(f"Wrote train tokenized: {train_token_count} rows -> {train_tokens}")
    print(f"Wrote validation JSONL: {val_count} rows -> {val_jsonl}")
    print(f"Wrote validation tokenized: {val_token_count} rows -> {val_tokens}")


if __name__ == "__main__":
    main()
