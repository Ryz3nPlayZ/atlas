from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ace_atlas.config import RecurrentConfig


@dataclass(slots=True)
class RecurrentState:
    hidden: Tensor


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

        outputs = []
        recurrent_state = state.hidden
        for t in range(seq_len):
            token = hidden[:, t]
            token_u, token_g, token_v = self.in_proj(token).chunk(3, dim=-1)
            state_u, state_g = self.state_proj(recurrent_state).chunk(2, dim=-1)
            candidate = torch.tanh(token_u + state_u)
            gate = torch.sigmoid(token_g + state_g)
            mixed = gate * candidate + (1.0 - gate) * torch.tanh(token_v)
            recurrent_state = self.state_update(mixed)
            outputs.append(mixed)

        stacked = torch.stack(outputs, dim=1)
        return self.dropout(self.out_proj(stacked)), RecurrentState(hidden=recurrent_state)

