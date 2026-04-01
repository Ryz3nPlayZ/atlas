from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(slots=True)
class ModelOutput:
    logits: Tensor
    mtp_logits: Tensor | None
    uncertainty: Tensor | None
    memory_state: Any | None
    arbiter_outputs: list[Any]
    memory_reads: list[Any]
    block_aux: list[Any]
    runtime_stats: dict[str, float] | None = None
