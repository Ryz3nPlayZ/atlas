from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ace_atlas.config import ACEAtlasConfig, AttentionConfig, ModeConditioningConfig, MoEConfig
from ace_atlas.model.attention import build_local_causal_mask
from ace_atlas.model.moe import ExpertFFN, MoEAux
from ace_atlas.model.types import ModelOutput


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: Tensor) -> Tensor:
        scale = torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return hidden * scale * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}

    def get_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        key = (seq_len, str(device))
        cached = self._cache.get(key)
        if cached is None:
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            cached = (cos, sin)
            self._cache[key] = cached
        return cached

    def apply(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self.get_cos_sin(query.size(-2), query.device)

        def rotate(x: Tensor) -> Tensor:
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            x_rotated = torch.stack(
                [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
                dim=-1,
            )
            return x_rotated.flatten(-2)

        return rotate(query), rotate(key)


def maybe_repeat_kv(hidden: Tensor, num_heads: int, num_kv_heads: int) -> Tensor:
    if num_heads == num_kv_heads:
        return hidden
    repeat_factor = num_heads // num_kv_heads
    return hidden.repeat_interleave(repeat_factor, dim=1)


def apply_qk_norm(hidden: Tensor, eps: float = 1e-6) -> Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)


class LocalGQAAttention(nn.Module):
    def __init__(self, model_dim: int, config: AttentionConfig) -> None:
        super().__init__()
        if model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.model_dim = model_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = model_dim // config.num_heads
        self.window_size = config.window_size
        self.dropout = config.dropout
        self.use_qk_norm = config.qk_norm

        self.q_proj = nn.Linear(model_dim, config.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(model_dim, config.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(model_dim, config.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_base_local)
        self._mask_cache: dict[tuple[int, str], Tensor] = {}

    def get_local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        key = (seq_len, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = build_local_causal_mask(seq_len, self.window_size, device)
            self._mask_cache[key] = mask
        return mask

    def forward(self, hidden: Tensor) -> Tensor:
        batch, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = self.rotary.apply(query, key)
        if self.use_qk_norm:
            query = apply_qk_norm(query)
            key = apply_qk_norm(key)

        key = maybe_repeat_kv(key, self.num_heads, self.num_kv_heads)
        value = maybe_repeat_kv(value, self.num_heads, self.num_kv_heads)
        mask = self.get_local_mask(seq_len, hidden.device)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.model_dim)
        return self.out_proj(attn)


class LatentGlobalAttention(nn.Module):
    def __init__(self, model_dim: int, config: AttentionConfig) -> None:
        super().__init__()
        if model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if config.global_latent_dim <= 0:
            raise ValueError("global_latent_dim must be positive for global latent attention")
        self.model_dim = model_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = model_dim // config.num_heads
        self.dropout = config.dropout
        self.use_qk_norm = config.qk_norm
        latent_dim = config.global_latent_dim

        self.q_proj = nn.Linear(model_dim, config.num_heads * self.head_dim, bias=config.use_bias)
        self.kv_down = nn.Linear(model_dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=config.use_bias)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_base_global)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_down(hidden)
        key = self.k_up(latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_up(latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = self.rotary.apply(query, key)
        if self.use_qk_norm:
            query = apply_qk_norm(query)
            key = apply_qk_norm(key)

        key = maybe_repeat_kv(key, self.num_heads, self.num_kv_heads)
        value = maybe_repeat_kv(value, self.num_heads, self.num_kv_heads)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.model_dim)
        return self.out_proj(attn)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(model_dim, hidden_dim)
        self.value_proj = nn.Linear(model_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden)) * self.value_proj(hidden)
        return self.out_proj(self.dropout(gated))


class ModeConditionedSparseMoE(nn.Module):
    def __init__(
        self,
        model_dim: int,
        moe_config: MoEConfig,
        mode_config: ModeConditioningConfig,
    ) -> None:
        super().__init__()
        self.top_k = moe_config.top_k
        self.num_routed = moe_config.num_routed_experts
        self.shared_experts = nn.ModuleList(
            [ExpertFFN(model_dim, moe_config.hidden_dim, moe_config.dropout) for _ in range(moe_config.num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(model_dim, moe_config.hidden_dim, moe_config.dropout) for _ in range(moe_config.num_routed_experts)]
        )
        self.router = nn.Linear(model_dim, moe_config.num_routed_experts)
        self.mode_conditioned = mode_config.enabled
        self.mode_shared_experts: nn.ModuleList | None = None
        if self.mode_conditioned:
            self.mode_shared_experts = nn.ModuleList(
                [
                    nn.ModuleList(
                        [ExpertFFN(model_dim, moe_config.hidden_dim, moe_config.dropout) for _ in range(mode_config.shared_experts_per_mode)]
                    )
                    for _ in range(len(mode_config.modes))
                ]
            )

    def forward(self, hidden: Tensor, mode_ids: Tensor | None = None) -> tuple[Tensor, MoEAux]:
        batch, seq_len, dim = hidden.shape
        flat = hidden.reshape(batch * seq_len, dim)
        router_logits = self.router(flat)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        shared_out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(flat)
        if self.shared_experts:
            shared_out = shared_out / len(self.shared_experts)

        if self.mode_shared_experts is not None and mode_ids is not None:
            mode_flat = mode_ids.reshape(-1)
            mode_out = torch.zeros_like(flat)
            for mode_index, expert_group in enumerate(self.mode_shared_experts):
                token_mask = mode_flat == mode_index
                if not torch.any(token_mask):
                    continue
                token_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)
                mode_input = flat.index_select(0, token_indices)
                mode_sum = torch.zeros_like(mode_input)
                for expert in expert_group:
                    mode_sum = mode_sum + expert(mode_input)
                mode_sum = mode_sum / len(expert_group)
                mode_out.index_add_(0, token_indices, mode_sum)
            shared_out = shared_out + mode_out

        mixed = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.routed_experts):
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if not torch.any(expert_mask):
                continue
            token_indices = expert_mask.nonzero(as_tuple=False).squeeze(-1)
            expert_input = flat.index_select(0, token_indices)
            expert_output = expert(expert_input)
            expert_weights = (
                topk_probs[token_indices] * (topk_indices[token_indices] == expert_idx).to(topk_probs.dtype)
            ).sum(dim=-1, keepdim=True)
            mixed.index_add_(0, token_indices, expert_output * expert_weights)

        output = (shared_out + mixed).reshape(batch, seq_len, dim)
        aux = MoEAux(
            router_logits=router_logits.reshape(batch, seq_len, self.num_routed),
            topk_indices=topk_indices.reshape(batch, seq_len, self.top_k),
            topk_probs=topk_probs.reshape(batch, seq_len, self.top_k),
        )
        return output, aux


@dataclass(slots=True)
class TransformerBlockAux:
    attention_kind: str
    moe: MoEAux | None


class TransformerHybridBlock(nn.Module):
    def __init__(self, config: ACEAtlasConfig, attention_kind: str, use_moe: bool) -> None:
        super().__init__()
        norm_cls = RMSNorm if config.transformer.use_rms_norm else nn.LayerNorm
        norm_kwargs = {} if config.transformer.use_rms_norm else {"eps": config.norm_epsilon}
        self.attention_kind = attention_kind
        self.use_moe = use_moe
        self.attn_norm = norm_cls(config.model_dim, **norm_kwargs)
        self.ffn_norm = norm_cls(config.model_dim, **norm_kwargs)

        if attention_kind == "global":
            self.attention = LatentGlobalAttention(config.model_dim, config.attention)
        else:
            self.attention = LocalGQAAttention(config.model_dim, config.attention)

        if use_moe:
            self.feedforward = ModeConditionedSparseMoE(
                config.model_dim,
                config.moe,
                config.mode_conditioning,
            )
        else:
            self.feedforward = SwiGLUFeedForward(
                config.model_dim,
                config.transformer.dense_hidden_dim,
                config.dropout,
            )

    def forward(
        self,
        hidden: Tensor,
        mode_ids: Tensor | None = None,
        collect_runtime_stats: bool = False,
    ) -> tuple[Tensor, TransformerBlockAux, dict[str, float] | None]:
        timings = {"attention_local": 0.0, "attention_global": 0.0, "moe": 0.0} if collect_runtime_stats else None

        start = time.perf_counter() if collect_runtime_stats else 0.0
        hidden = hidden + self.attention(self.attn_norm(hidden))
        if timings is not None:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            key = "attention_global" if self.attention_kind == "global" else "attention_local"
            timings[key] += time.perf_counter() - start

        start = time.perf_counter() if collect_runtime_stats else 0.0
        ffn_input = self.ffn_norm(hidden)
        if self.use_moe:
            ffn_out, moe_aux = self.feedforward(ffn_input, mode_ids=mode_ids)
        else:
            ffn_out = self.feedforward(ffn_input)
            moe_aux = None
        hidden = hidden + ffn_out
        if timings is not None and self.use_moe:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            timings["moe"] += time.perf_counter() - start
        return hidden, TransformerBlockAux(self.attention_kind, moe_aux), timings


class ACEAtlasTransformerModel(nn.Module):
    def __init__(self, config: ACEAtlasConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_checkpointing = False

        if config.transformer.local_layers_per_global <= 0:
            raise ValueError("local_layers_per_global must be positive")
        moe_start_index = int(math.floor(config.num_layers * config.transformer.moe_start_fraction))

        layers: list[TransformerHybridBlock] = []
        pattern_period = config.transformer.local_layers_per_global + 1
        for layer_idx in range(config.num_layers):
            attention_kind = "global" if (layer_idx + 1) % pattern_period == 0 else "local"
            use_moe = layer_idx >= moe_start_index
            layers.append(
                TransformerHybridBlock(
                    config=config,
                    attention_kind=attention_kind,
                    use_moe=use_moe,
                )
            )
        self.layers = nn.ModuleList(layers)

        norm_cls = RMSNorm if config.transformer.use_rms_norm else nn.LayerNorm
        norm_kwargs = {} if config.transformer.use_rms_norm else {"eps": config.norm_epsilon}
        self.final_norm = norm_cls(config.model_dim, **norm_kwargs)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.model_dim, config.vocab_size, bias=False) for _ in range(config.mtp_horizon)]
        )
        self.uncertainty_head = nn.Linear(config.model_dim, 1)

    def enable_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = enabled

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        mode_ids: Tensor | None = None,
        collect_runtime_stats: bool = False,
    ) -> ModelOutput:
        del segment_ids
        runtime_stats = {"attention_local": 0.0, "attention_global": 0.0, "moe": 0.0, "heads": 0.0} if collect_runtime_stats else None
        hidden = self.dropout(self.embed_tokens(input_ids))
        if mode_ids is None:
            mode_ids = torch.zeros_like(input_ids)
        block_aux: list[TransformerBlockAux] = []

        for layer in self.layers:
            use_checkpoint = (
                self.activation_checkpointing
                and self.training
                and not collect_runtime_stats
                and not layer.use_moe
            )
            if use_checkpoint:
                hidden = checkpoint(
                    lambda h, m: layer(h, mode_ids=m, collect_runtime_stats=False)[0],
                    hidden,
                    mode_ids,
                    use_reentrant=False,
                )
                aux = TransformerBlockAux(layer.attention_kind, moe=None)
                layer_timings = None
            else:
                hidden, aux, layer_timings = layer(
                    hidden,
                    mode_ids=mode_ids,
                    collect_runtime_stats=collect_runtime_stats,
                )
            block_aux.append(aux)
            if runtime_stats is not None and layer_timings is not None:
                for key, value in layer_timings.items():
                    runtime_stats[key] += value

        start = time.perf_counter() if collect_runtime_stats else 0.0
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = None
        if self.mtp_heads:
            mtp_logits = torch.stack([head(hidden) for head in self.mtp_heads], dim=2)
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden.mean(dim=1))).squeeze(-1)
        if runtime_stats is not None:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            runtime_stats["heads"] += time.perf_counter() - start
        return ModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            uncertainty=uncertainty,
            memory_state=None,
            arbiter_outputs=[],
            memory_reads=[],
            block_aux=block_aux,
            runtime_stats=runtime_stats,
        )
