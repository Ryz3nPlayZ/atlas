from __future__ import annotations


class ByteTokenizer:
    """Minimal byte-level tokenizer for local pipeline validation."""

    name = "byte"
    pad_token_id = 256
    eos_token_id = 257
    vocab_size = 258

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, tokens: list[int]) -> str:
        byte_values = bytes(token for token in tokens if 0 <= token < 256)
        return byte_values.decode("utf-8", errors="replace")
