from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.tokenizer.factory import build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize prompt/completion JSONL into answer-only loss-mask JSONL."
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--tokenizer-name", type=str, default="sentencepiece_bpe")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT / "data" / "tokenizers" / "code_text_sp32k_v1" / "code_text_sp32k_v1.model",
    )
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--completion-key", type=str, default="completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args.tokenizer_name, args.tokenizer_path)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with args.input_jsonl.open("r", encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record[args.prompt_key]
            completion = record[args.completion_key]
            prompt_tokens = tokenizer.encode(prompt, add_eos=False)
            completion_tokens = tokenizer.encode(completion, add_eos=True)
            tokens = prompt_tokens + completion_tokens
            loss_mask = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
            sink.write(json.dumps({"tokens": tokens, "loss_mask": loss_mask}) + "\n")
            rows += 1

    print(
        json.dumps(
            {
                "tokenizer_name": args.tokenizer_name,
                "tokenizer_path": str(args.tokenizer_path),
                "rows": rows,
                "output_jsonl": str(args.output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
