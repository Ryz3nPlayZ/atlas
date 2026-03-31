from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

