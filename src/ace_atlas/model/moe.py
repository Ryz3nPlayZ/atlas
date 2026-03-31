from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ace_atlas.config import MoEConfig


@dataclass(slots=True)
class MoEAux:
    router_logits: Tensor
    topk_indices: Tensor
    topk_probs: Tensor


class ExpertFFN(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden)


class SparseMoE(nn.Module):
    """Naive all-experts evaluation MoE.

    This is intentionally simple and suitable only for bootstrap development.
    """

    def __init__(self, model_dim: int, config: MoEConfig) -> None:
        super().__init__()
        self.enabled = config.enabled
        self.top_k = config.top_k
        self.num_routed = config.num_routed_experts
        self.shared_experts = nn.ModuleList(
            [ExpertFFN(model_dim, config.hidden_dim, config.dropout) for _ in range(config.num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(model_dim, config.hidden_dim, config.dropout) for _ in range(config.num_routed_experts)]
        )
        self.router = nn.Linear(model_dim, config.num_routed_experts)

    def forward(self, hidden: Tensor) -> tuple[Tensor, MoEAux | None]:
        if not self.enabled:
            return hidden, None

        batch, seq_len, dim = hidden.shape
        flat = hidden.reshape(batch * seq_len, dim)
        router_logits = self.router(flat)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        shared_out = 0.0
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)
        if self.shared_experts:
            shared_out = shared_out / len(self.shared_experts)

        routed_outputs = torch.stack([expert(flat) for expert in self.routed_experts], dim=1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, dim)
        selected = torch.gather(routed_outputs, dim=1, index=gather_index)
        mixed = (selected * topk_probs.unsqueeze(-1)).sum(dim=1)

        output = (shared_out + mixed).reshape(batch, seq_len, dim)
        aux = MoEAux(
            router_logits=router_logits.reshape(batch, seq_len, self.num_routed),
            topk_indices=topk_indices.reshape(batch, seq_len, self.top_k),
            topk_probs=topk_probs.reshape(batch, seq_len, self.top_k),
        )
        return output, aux

