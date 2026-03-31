from __future__ import annotations


class ByteTokenizer:
    """Minimal byte-level tokenizer for local pipeline validation."""

    name = "byte"
    pad_token_id = 256
    eos_token_id = 257
    vocab_size = 258

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) + [self.eos_token_id]

    def decode(self, tokens: list[int]) -> str:
        byte_values = bytes(token for token in tokens if 0 <= token < 256)
        return byte_values.decode("utf-8", errors="replace")

