from __future__ import annotations

from pathlib import Path
from typing import Any

from ace_atlas.tokenizer.base import Tokenizer
from ace_atlas.tokenizer.byte_level import ByteTokenizer
from ace_atlas.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def build_tokenizer(name: str, model_path: str | Path | None = None) -> Tokenizer:
    if name == "byte":
        return ByteTokenizer()
    if name in {"sentencepiece", "sentencepiece_bpe"}:
        if model_path is None:
            raise ValueError("sentencepiece tokenizer requires a model_path")
        return SentencePieceTokenizer(model_path)
    raise ValueError(f"Unsupported tokenizer: {name}")


def build_tokenizer_from_training_config(training_config: dict[str, Any] | None) -> Tokenizer:
    if not training_config:
        return ByteTokenizer()
    tokenizer_name = training_config.get("tokenizer_name", "byte")
    tokenizer_path = training_config.get("tokenizer_path")
    return build_tokenizer(tokenizer_name, tokenizer_path)
