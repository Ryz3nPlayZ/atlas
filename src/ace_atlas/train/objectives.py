from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ace_atlas.model.types import ModelOutput


def language_model_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


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
        flat_target = target.reshape(-1)
        valid = flat_target.ne(-100)
        if not torch.any(valid):
            continue
        total = total + F.cross_entropy(pred.reshape(-1, vocab), flat_target, ignore_index=-100)
        count += 1
    if count == 0:
        return total
    return total / count


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor | None, labels: Tensor, temperature: float) -> Tensor:
    if teacher_logits is None:
        return labels.new_zeros((), dtype=torch.float32)
    student = student_logits.reshape(-1, student_logits.size(-1))
    teacher = teacher_logits.reshape(-1, teacher_logits.size(-1))
    valid = labels.reshape(-1).ne(-100)
    if not torch.any(valid):
        return student.new_zeros(())
    student = student[valid] / temperature
    teacher = teacher[valid] / temperature
    return (
        F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
        * (temperature ** 2)
    )


def total_training_loss(
    output: ModelOutput,
    labels: Tensor,
    mtp_weight: float = 0.2,
    teacher_logits: Tensor | None = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
) -> dict[str, Tensor]:
    lm = language_model_loss(output.logits, labels)
    mtp = multi_token_prediction_loss(output.mtp_logits, labels)
    distill = distillation_loss(output.logits, teacher_logits, labels, distill_temperature)
    total = lm + mtp_weight * mtp + distill_weight * distill
    return {"loss": total, "lm_loss": lm, "mtp_loss": mtp, "distill_loss": distill}
