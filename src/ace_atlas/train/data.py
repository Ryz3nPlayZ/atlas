from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class RandomTokenDataset(Dataset):
    """Bootstrap dataset for training loop validation.

    This is only for harness bring-up. Real corpora and tokenized shards come later.
    """

    def __init__(self, vocab_size: int, sequence_length: int, total_examples: int) -> None:
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.total_examples = total_examples

    def __len__(self) -> int:
        return self.total_examples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        tokens = torch.randint(0, self.vocab_size, (self.sequence_length + 1,), dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def build_random_lm_dataloader(
    vocab_size: int,
    sequence_length: int,
    batch_size: int,
    total_examples: int,
) -> DataLoader:
    dataset = RandomTokenDataset(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        total_examples=total_examples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TokenizedJsonlDataset(Dataset):
    """Language-modeling dataset backed by tokenized JSONL records.

    Each line is expected to contain a ``tokens`` array produced by
    ``scripts/tokenize_corpus.py``.
    """

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.examples: list[tuple[Tensor, Tensor | None]] = []
        self._load_examples()
        if not self.examples:
            raise ValueError(
                f"No token sequences of length >= {sequence_length + 1} were found in {self.path}"
            )

    def _load_examples(self) -> None:
        window = self.sequence_length + 1
        for line_number, raw_line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            tokens = record.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError(f"Missing tokens list in {self.path}:{line_number}")
            loss_mask = record.get("loss_mask")
            if loss_mask is not None:
                if not isinstance(loss_mask, list):
                    raise ValueError(f"Missing loss_mask list in {self.path}:{line_number}")
                if len(loss_mask) != len(tokens):
                    raise ValueError(f"loss_mask length mismatch in {self.path}:{line_number}")
            if len(tokens) < window:
                continue
            for start in range(0, len(tokens) - window + 1, self.sequence_length):
                chunk = tokens[start : start + window]
                if len(chunk) == window:
                    chunk_mask = None
                    if loss_mask is not None:
                        chunk_mask = torch.tensor(loss_mask[start : start + window], dtype=torch.bool)
                        if not torch.any(chunk_mask[1:]):
                            continue
                    self.examples.append((torch.tensor(chunk, dtype=torch.long), chunk_mask))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        tokens, loss_mask = self.examples[index]
        labels = tokens[1:].clone()
        segment_ids = None
        if loss_mask is not None:
            labels[~loss_mask[1:]] = -100
            segment_ids = loss_mask[:-1].to(torch.long)
        batch = {
            "input_ids": tokens[:-1],
            "labels": labels,
        }
        if segment_ids is not None:
            batch["segment_ids"] = segment_ids
        return batch


def build_tokenized_lm_dataloader(
    path: str | Path,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TokenizedJsonlDataset(path=path, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
