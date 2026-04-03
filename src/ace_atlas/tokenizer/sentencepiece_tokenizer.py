from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    """SentencePiece-backed tokenizer for code-aware tokenizer experiments."""

    name = "sentencepiece_bpe"

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)
        self.pad_token_id = int(self.processor.pad_id())
        self.eos_token_id = int(self.processor.eos_id())
        self.bos_token_id = int(self.processor.bos_id())
        self.vocab_size = int(self.processor.vocab_size())

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = list(self.processor.encode(text, out_type=int))
        if add_eos and self.eos_token_id >= 0:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, tokens: list[int]) -> str:
        filtered = [
            token
            for token in tokens
            if token not in {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        ]
        return self.processor.decode(filtered)
