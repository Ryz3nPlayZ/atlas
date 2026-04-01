from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ace_atlas.config import RecurrentConfig


@dataclass(slots=True)
class RecurrentState:
    hidden: Tensor


@torch.jit.script
def _recurrent_scan(
    projected: Tensor,
    recurrent_state: Tensor,
    state_proj_weight: Tensor,
    state_proj_bias: Tensor,
    state_update_weight: Tensor,
    state_update_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, projected_dim = projected.shape
    inner_dim = projected_dim // 3
    outputs = projected.new_empty((batch, seq_len, inner_dim))

    for t in range(seq_len):
        token_u, token_g, token_v = projected[:, t].chunk(3, dim=-1)
        state_u, state_g = F.linear(recurrent_state, state_proj_weight, state_proj_bias).chunk(2, dim=-1)
        candidate = torch.tanh(token_u + state_u)
        gate = torch.sigmoid(token_g + state_g)
        mixed = gate * candidate + (1.0 - gate) * torch.tanh(token_v)
        recurrent_state = F.linear(mixed, state_update_weight, state_update_bias)
        outputs[:, t] = mixed

    return outputs, recurrent_state


class BootstrapRecurrentMixer(nn.Module):
    """A simple gated recurrent mixer used as a bootstrap implementation.

    This is not an optimized xLSTM or KDA implementation. It provides the correct
    interface and a replaceable module boundary for later kernel-backed versions.
    """

    def __init__(self, model_dim: int, config: RecurrentConfig) -> None:
        super().__init__()
        inner_dim = model_dim * config.expansion_factor
        self.in_proj = nn.Linear(model_dim, inner_dim * 3)
        self.state_proj = nn.Linear(config.state_dim, inner_dim * 2)
        self.out_proj = nn.Linear(inner_dim, model_dim)
        self.state_update = nn.Linear(inner_dim, config.state_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.state_dim = config.state_dim

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        return RecurrentState(hidden=torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype))

    def forward(self, hidden: Tensor, state: RecurrentState | None = None) -> tuple[Tensor, RecurrentState]:
        batch, seq_len, _ = hidden.shape
        if state is None:
            state = self.initial_state(batch, hidden.device, hidden.dtype)

        projected = self.in_proj(hidden)
        outputs, recurrent_state = _recurrent_scan(
            projected,
            state.hidden,
            self.state_proj.weight,
            self.state_proj.bias,
            self.state_update.weight,
            self.state_update.bias,
        )

        return self.dropout(self.out_proj(outputs)), RecurrentState(hidden=recurrent_state)
