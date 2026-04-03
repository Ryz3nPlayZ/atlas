from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a code-aware SentencePiece BPE tokenizer on mixed text+code corpora."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "tokenizers" / "code_text_sp32k_v1",
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--code-jsonl",
        type=Path,
        default=ROOT / "data" / "code_search_net_python_ft" / "train.jsonl",
    )
    parser.add_argument(
        "--text-jsonl",
        type=Path,
        default=ROOT / "data" / "wikitext2" / "train.jsonl",
    )
    parser.add_argument(
        "--task-jsonl",
        type=Path,
        default=ROOT / "data" / "executable_solution_sft_v7" / "train.jsonl",
    )
    parser.add_argument("--task-limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_text_records(path: Path, limit: int | None = None) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" in record:
                records.append(record["text"])
            elif "prompt" in record and "completion" in record:
                records.append(record["prompt"] + record["completion"])
            else:
                raise ValueError(f"Unsupported record schema in {path}")
            if limit is not None and len(records) >= limit:
                break
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    code_records = read_text_records(args.code_jsonl)
    text_records = read_text_records(args.text_jsonl)
    task_records = read_text_records(args.task_jsonl)
    rng.shuffle(task_records)
    if args.task_limit > 0:
        task_records = task_records[: args.task_limit]

    corpus_records = code_records + text_records + task_records
    rng.shuffle(corpus_records)

    corpus_path = args.output_dir / "training_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for text in corpus_records:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.write("\n")

    model_prefix = args.output_dir / "code_text_sp32k_v1"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )

    metadata = {
        "tokenizer_name": "sentencepiece_bpe",
        "model_path": str(model_prefix.with_suffix(".model")),
        "vocab_path": str(model_prefix.with_suffix(".vocab")),
        "vocab_size": args.vocab_size,
        "sources": {
            "code_jsonl": str(args.code_jsonl),
            "text_jsonl": str(args.text_jsonl),
            "task_jsonl": str(args.task_jsonl),
            "task_limit": args.task_limit,
        },
        "corpus_counts": {
            "code_records": len(code_records),
            "text_records": len(text_records),
            "task_records": len(task_records),
            "total_records": len(corpus_records),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
