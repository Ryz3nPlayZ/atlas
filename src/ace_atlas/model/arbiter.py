from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn

from ace_atlas.config import ArbiterConfig


class MemoryAction(IntEnum):
    KEEP_STATE = 0
    WRITE_EPISODIC = 1
    UPDATE_SEMANTIC = 2
    RETRIEVE = 3
    IGNORE = 4


@dataclass(slots=True)
class ArbiterOutput:
    logits: Tensor
    probs: Tensor
    actions: Tensor
    expected_cost: Tensor


class MemoryArbiter(nn.Module):
    def __init__(self, model_dim: int, config: ArbiterConfig) -> None:
        super().__init__()
        self.enabled = config.enabled
        self.cost_weight = config.cost_weight
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(MemoryAction)),
        )
        self.register_buffer("action_costs", torch.tensor([0.10, 0.35, 0.45, 0.25, 0.05]))

    def forward(self, chunk_repr: Tensor) -> ArbiterOutput:
        logits = self.mlp(chunk_repr)
        probs = torch.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        expected_cost = (probs * self.action_costs.to(probs)).sum(dim=-1) * self.cost_weight
        return ArbiterOutput(logits=logits, probs=probs, actions=actions, expected_cost=expected_cost)

