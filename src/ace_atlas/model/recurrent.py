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


class FusedGRUMixer(nn.Module):
    """A CUDA-friendly recurrent mixer that delegates the scan to PyTorch GRU kernels."""

    def __init__(self, model_dim: int, config: RecurrentConfig) -> None:
        super().__init__()
        self.hidden_size = config.state_dim * config.expansion_factor
        self.in_proj = nn.Linear(model_dim, self.hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.hidden_size, model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        return RecurrentState(hidden=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype))

    def forward(self, hidden: Tensor, state: RecurrentState | None = None) -> tuple[Tensor, RecurrentState]:
        batch, _, _ = hidden.shape
        if state is None:
            state = self.initial_state(batch, hidden.device, hidden.dtype)
        if hidden.is_cuda:
            self.gru.flatten_parameters()

        projected = self.in_proj(hidden)
        outputs, recurrent_state = self.gru(projected, state.hidden.unsqueeze(0))
        outputs = self.out_proj(outputs)
        return self.dropout(outputs), RecurrentState(hidden=recurrent_state.squeeze(0))


def build_recurrent_mixer(model_dim: int, config: RecurrentConfig) -> nn.Module:
    if config.kind == "xlstm_bootstrap":
        return BootstrapRecurrentMixer(model_dim, config)
    if config.kind == "gru_fused":
        return FusedGRUMixer(model_dim, config)
    raise ValueError(f"Unsupported recurrent.kind: {config.kind}")
