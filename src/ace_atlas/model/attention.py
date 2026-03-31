from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ace_atlas.config import AttentionConfig


def build_local_causal_mask(seq_len: int, window_size: int, device: torch.device) -> Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0
    return mask


class LocalCausalSelfAttention(nn.Module):
    """Naive bootstrap implementation of local causal attention.

    This module is intentionally simple and correctness-oriented. It is the interface
    target for later FlashAttention or custom kernel replacements.
    """

    def __init__(self, model_dim: int, config: AttentionConfig) -> None:
        super().__init__()
        if model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = config.num_heads
        self.head_dim = model_dim // config.num_heads
        self.window_size = config.window_size
        self.dropout = config.dropout

        self.q_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = build_local_causal_mask(seq_len, self.window_size, hidden.device)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.model_dim)
        return self.out_proj(attn)

