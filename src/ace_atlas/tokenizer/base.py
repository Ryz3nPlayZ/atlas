from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    name: str
    pad_token_id: int
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...
