from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.tokenizer.factory import build_tokenizer
from ace_atlas.modes import resolve_mode_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a code-heavy mixed-base tokenized dataset with a non-byte tokenizer."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "mixed_code_text_base_sp_v1")
    parser.add_argument("--tokenizer-name", type=str, default="sentencepiece_bpe")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT / "data" / "tokenizers" / "code_text_sp32k_v1" / "code_text_sp32k_v1.model",
    )
    parser.add_argument(
        "--code-train-path",
        type=Path,
        default=ROOT / "data" / "code_search_net_python_ft" / "train.jsonl",
    )
    parser.add_argument(
        "--code-val-path",
        type=Path,
        default=ROOT / "data" / "code_search_net_python_ft" / "validation.jsonl",
    )
    parser.add_argument(
        "--text-train-path",
        type=Path,
        default=ROOT / "data" / "wikitext2" / "train.jsonl",
    )
    parser.add_argument(
        "--text-val-path",
        type=Path,
        default=ROOT / "data" / "wikitext2" / "validation.jsonl",
    )
    parser.add_argument("--code-repeat", type=int, default=2)
    parser.add_argument("--text-repeat", type=int, default=1)
    parser.add_argument("--text-val-limit", type=int, default=1000)
    parser.add_argument("--emit-mode-ids", action="store_true")
    parser.add_argument("--code-mode", type=str, default="code")
    parser.add_argument("--text-mode", type=str, default="general")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_text_jsonl(path: Path, limit: int | None = None) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Expected 'text' field in {path}")
            records.append(text)
            if limit is not None and len(records) >= limit:
                break
    return records


def write_tokenized(records: list[tuple[str, int]], path: Path, tokenizer, emit_mode_ids: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text, mode_id in records:
            token_ids = tokenizer.encode(text)
            payload = {"tokens": token_ids}
            if emit_mode_ids:
                payload["mode_ids"] = [mode_id] * len(token_ids)
            handle.write(json.dumps(payload) + "\n")
    return len(records)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = build_tokenizer(args.tokenizer_name, args.tokenizer_path)
    code_mode_id = resolve_mode_id(args.code_mode)
    text_mode_id = resolve_mode_id(args.text_mode)

    code_train = read_text_jsonl(args.code_train_path)
    code_val = read_text_jsonl(args.code_val_path)
    text_train = read_text_jsonl(args.text_train_path)
    text_val = read_text_jsonl(args.text_val_path, limit=args.text_val_limit)

    train_records = (
        [(text, code_mode_id) for text in code_train] * args.code_repeat
        + [(text, text_mode_id) for text in text_train] * args.text_repeat
    )
    rng.shuffle(train_records)
    val_records = [(text, code_mode_id) for text in code_val] + [(text, text_mode_id) for text in text_val]
    rng.shuffle(val_records)

    train_path = args.output_dir / "train_tokens.jsonl"
    val_path = args.output_dir / "validation_tokens.jsonl"
    write_tokenized(train_records, train_path, tokenizer, args.emit_mode_ids)
    write_tokenized(val_records, val_path, tokenizer, args.emit_mode_ids)

    metadata = {
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_path": str(args.tokenizer_path),
        "emit_mode_ids": args.emit_mode_ids,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_rows": len(train_records),
        "validation_rows": len(val_records),
        "train_composition": {
            "code": len(code_train) * args.code_repeat,
            "text": len(text_train) * args.text_repeat,
        },
        "validation_composition": {
            "code": len(code_val),
            "text": len(text_val),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
