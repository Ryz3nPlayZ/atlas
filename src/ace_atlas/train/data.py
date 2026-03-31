from __future__ import annotations

from dataclasses import dataclass

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

