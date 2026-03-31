from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ace_atlas.model.types import ModelOutput


def language_model_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def multi_token_prediction_loss(mtp_logits: Tensor | None, labels: Tensor) -> Tensor:
    if mtp_logits is None:
        return labels.new_zeros((), dtype=torch.float32)

    batch, seq_len, horizon, vocab = mtp_logits.shape
    total = mtp_logits.new_zeros(())
    count = 0
    for offset in range(horizon):
        if offset + 1 >= seq_len:
            continue
        pred = mtp_logits[:, : seq_len - offset - 1, offset]
        target = labels[:, offset + 1 :]
        total = total + F.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
        count += 1
    if count == 0:
        return total
    return total / count


def total_training_loss(output: ModelOutput, labels: Tensor, mtp_weight: float = 0.2) -> dict[str, Tensor]:
    lm = language_model_loss(output.logits, labels)
    mtp = multi_token_prediction_loss(output.mtp_logits, labels)
    total = lm + mtp_weight * mtp
    return {"loss": total, "lm_loss": lm, "mtp_loss": mtp}

