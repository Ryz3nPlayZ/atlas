from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.attention import LocalCausalSelfAttention
from ace_atlas.model.types import ModelOutput


@dataclass(slots=True)
class DenseBlockAux:
    attention_only: bool = True


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: ACEAtlasConfig) -> None:
        super().__init__()
        hidden_dim = config.model_dim * 4
        self.attn_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.ffn_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.attn = LocalCausalSelfAttention(config.model_dim, config.attention)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.model_dim),
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, DenseBlockAux]:
        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, DenseBlockAux()


class DenseCausalTransformer(nn.Module):
    """Dense local-attention baseline sharing the ACE-Atlas config surface."""

    def __init__(self, config: ACEAtlasConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DenseTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.model_dim, config.vocab_size, bias=False) for _ in range(config.mtp_horizon)]
        )
        self.uncertainty_head = nn.Linear(config.model_dim, 1)

    def forward(self, input_ids: Tensor, collect_runtime_stats: bool = False) -> ModelOutput:
        hidden = self.dropout(self.embed_tokens(input_ids))
        block_aux = []
        for layer in self.layers:
            hidden, aux = layer(hidden)
            block_aux.append(aux)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = None
        if self.mtp_heads:
            mtp_logits = torch.stack([head(hidden) for head in self.mtp_heads], dim=2)
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden.mean(dim=1))).squeeze(-1)
        return ModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            uncertainty=uncertainty,
            memory_state=None,
            arbiter_outputs=[],
            memory_reads=[],
            block_aux=block_aux,
            runtime_stats=None,
        )
