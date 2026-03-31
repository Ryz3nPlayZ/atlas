from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ace_atlas.data.corpus import iter_manifest_texts
from ace_atlas.data.manifest import DatasetManifest
from ace_atlas.tokenizer.byte_level import ByteTokenizer


TOKENIZERS = {
    "byte": ByteTokenizer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a manifest-backed corpus.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "tokenized.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples, 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = DatasetManifest.load(args.manifest)
    tokenizer_cls = TOKENIZERS.get(manifest.tokenizer)
    if tokenizer_cls is None:
        raise ValueError(f"Unsupported tokenizer: {manifest.tokenizer}")

    tokenizer = tokenizer_cls()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for text in iter_manifest_texts(manifest, split=args.split):
            tokens = tokenizer.encode(text)
            handle.write(json.dumps({"tokens": tokens}) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} tokenized examples to {args.output}")


if __name__ == "__main__":
    main()

