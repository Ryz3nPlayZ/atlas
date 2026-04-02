from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.arbiter import ArbiterOutput, MemoryAction, MemoryArbiter
from ace_atlas.model.attention import LocalCausalSelfAttention
from ace_atlas.model.memory import BoundedMemory, MemoryReadResult, MemoryState
from ace_atlas.model.moe import MoEAux, SparseMoE
from ace_atlas.model.recurrent import RecurrentState, build_recurrent_mixer
from ace_atlas.model.types import ModelOutput


@dataclass(slots=True)
class BlockAux:
    moe: MoEAux | None


class HybridBlock(nn.Module):
    def __init__(self, config: ACEAtlasConfig, use_attention: bool) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.has_completion_adapter = config.completion_adapter_dim > 0
        self.recurrent_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.attention_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.moe_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.recurrent = build_recurrent_mixer(config.model_dim, config.recurrent)
        self.attention = (
            LocalCausalSelfAttention(config.model_dim, config.attention) if use_attention else None
        )
        self.moe = SparseMoE(config.model_dim, config.moe)
        if self.has_completion_adapter:
            self.completion_adapter_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
            self.completion_adapter_down = nn.Linear(config.model_dim, config.completion_adapter_dim)
            self.completion_adapter_up = nn.Linear(config.completion_adapter_dim, config.model_dim)
            nn.init.zeros_(self.completion_adapter_up.weight)
            nn.init.zeros_(self.completion_adapter_up.bias)

    def forward(
        self,
        hidden: Tensor,
        recurrent_state: RecurrentState | None = None,
        segment_ids: Tensor | None = None,
        collect_runtime_stats: bool = False,
    ) -> tuple[Tensor, RecurrentState, BlockAux, dict[str, float] | None]:
        timings = {"recurrent": 0.0, "attention": 0.0, "moe": 0.0} if collect_runtime_stats else None

        start = time.perf_counter() if collect_runtime_stats else 0.0
        recurrent_out, recurrent_state = self.recurrent(self.recurrent_norm(hidden), recurrent_state)
        if collect_runtime_stats:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            timings["recurrent"] += time.perf_counter() - start
        hidden = hidden + recurrent_out

        if self.attention is not None:
            start = time.perf_counter() if collect_runtime_stats else 0.0
            hidden = hidden + self.attention(self.attention_norm(hidden))
            if collect_runtime_stats:
                if hidden.is_cuda:
                    torch.cuda.synchronize(hidden.device)
                timings["attention"] += time.perf_counter() - start

        start = time.perf_counter() if collect_runtime_stats else 0.0
        moe_out, moe_aux = self.moe(self.moe_norm(hidden))
        if collect_runtime_stats:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            timings["moe"] += time.perf_counter() - start
        hidden = hidden + moe_out

        if self.has_completion_adapter and segment_ids is not None:
            completion_mask = segment_ids.unsqueeze(-1).to(hidden.dtype)
            adapter_hidden = self.completion_adapter_norm(hidden)
            adapter_hidden = torch.nn.functional.silu(self.completion_adapter_down(adapter_hidden))
            hidden = hidden + completion_mask * self.completion_adapter_up(adapter_hidden)
        return hidden, recurrent_state, BlockAux(moe=moe_aux), timings


class ACEAtlasModel(nn.Module):
    def __init__(self, config: ACEAtlasConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.segment_embeddings = (
            nn.Embedding(2, config.model_dim) if config.answer_span_embeddings else None
        )
        if self.segment_embeddings is not None:
            nn.init.zeros_(self.segment_embeddings.weight)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                HybridBlock(
                    config=config,
                    use_attention=((layer_idx + 1) % config.attention_every_n == 0),
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.model_dim, config.vocab_size, bias=False) for _ in range(config.mtp_horizon)]
        )
        self.uncertainty_head = nn.Linear(config.model_dim, 1)

        self.memory = BoundedMemory(config.model_dim, config.memory) if config.memory.enabled else None
        self.memory_fuse = nn.Linear(config.model_dim, config.model_dim)
        self.arbiter = MemoryArbiter(config.model_dim, config.arbiter) if config.arbiter.enabled else None
        self.memory_every_n_layers = max(1, config.attention_every_n)
        self.activation_checkpointing = False

    def enable_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = enabled

    def _run_layer_checkpointed(
        self,
        layer: HybridBlock,
        hidden: Tensor,
        recurrent_hidden: Tensor,
        segment_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        recurrent_state = RecurrentState(hidden=recurrent_hidden)
        next_hidden, next_state, _, _ = layer(
            hidden,
            recurrent_state,
            segment_ids=segment_ids,
            collect_runtime_stats=False,
        )
        return next_hidden, next_state.hidden

    def _apply_memory_bus(
        self,
        hidden: Tensor,
        memory_state: MemoryState | None,
        arbiter_outputs: list[ArbiterOutput],
        memory_reads: list[MemoryReadResult],
    ) -> tuple[Tensor, MemoryState | None]:
        if self.memory is None:
            return hidden, memory_state

        if memory_state is None:
            memory_state = self.memory.initial_state(hidden.size(0), hidden.device, hidden.dtype)

        chunk_repr = hidden.mean(dim=1)
        arbiter_output = self.arbiter(chunk_repr) if self.arbiter is not None else None
        if arbiter_output is not None:
            arbiter_outputs.append(arbiter_output)

        read_result = self.memory.read(chunk_repr, memory_state)
        memory_reads.append(read_result)
        hidden = hidden + self.memory_fuse(read_result.context).unsqueeze(1)

        keys, values = self.memory.summarize(hidden)
        write_scores = read_result.scores.abs().mean(dim=-1)

        if arbiter_output is None:
            memory_state = self.memory.write_episodic(keys, values, write_scores, memory_state)
            return hidden, memory_state

        actions = arbiter_output.actions
        if torch.any(actions == int(MemoryAction.WRITE_EPISODIC)):
            episodic_mask = actions == int(MemoryAction.WRITE_EPISODIC)
            episodic_scores = write_scores * episodic_mask.to(write_scores.dtype)
            memory_state = self.memory.write_episodic(keys, values, episodic_scores, memory_state)
        if torch.any(actions == int(MemoryAction.UPDATE_SEMANTIC)):
            semantic_mask = actions == int(MemoryAction.UPDATE_SEMANTIC)
            semantic_scores = write_scores * semantic_mask.to(write_scores.dtype)
            memory_state = self.memory.update_semantic(keys, values, semantic_scores, memory_state)

        return hidden, memory_state

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor | None = None,
        memory_state: MemoryState | None = None,
        collect_runtime_stats: bool = False,
    ) -> ModelOutput:
        runtime_stats = {"recurrent": 0.0, "attention": 0.0, "moe": 0.0, "memory": 0.0, "heads": 0.0} if collect_runtime_stats else None
        hidden = self.dropout(self.embed_tokens(input_ids))
        if self.segment_embeddings is not None and segment_ids is not None:
            hidden = hidden + self.segment_embeddings(segment_ids)
        recurrent_state: RecurrentState | None = None
        arbiter_outputs: list[ArbiterOutput] = []
        memory_reads: list[MemoryReadResult] = []
        block_aux: list[BlockAux] = []

        for layer_idx, layer in enumerate(self.layers):
            use_checkpoint = (
                self.activation_checkpointing
                and self.training
                and not collect_runtime_stats
            )
            if use_checkpoint:
                if recurrent_state is None:
                    recurrent_state = layer.recurrent.initial_state(hidden.size(0), hidden.device, hidden.dtype)
                layer_segment_ids = segment_ids
                if layer_segment_ids is None:
                    layer_segment_ids = torch.zeros_like(input_ids)
                hidden, recurrent_hidden = checkpoint(
                    lambda h, r, s: self._run_layer_checkpointed(layer, h, r, s),
                    hidden,
                    recurrent_state.hidden,
                    layer_segment_ids,
                    use_reentrant=True,
                )
                recurrent_state = RecurrentState(hidden=recurrent_hidden)
                aux = BlockAux(moe=None)
                block_timings = None
            else:
                hidden, recurrent_state, aux, block_timings = layer(
                    hidden,
                    recurrent_state,
                    segment_ids=segment_ids,
                    collect_runtime_stats=collect_runtime_stats,
                )
            block_aux.append(aux)
            if runtime_stats is not None and block_timings is not None:
                for key, value in block_timings.items():
                    runtime_stats[key] += value
            if (layer_idx + 1) % self.memory_every_n_layers == 0:
                start = time.perf_counter() if collect_runtime_stats else 0.0
                hidden, memory_state = self._apply_memory_bus(
                    hidden,
                    memory_state,
                    arbiter_outputs=arbiter_outputs,
                    memory_reads=memory_reads,
                )
                if runtime_stats is not None:
                    if hidden.is_cuda:
                        torch.cuda.synchronize(hidden.device)
                    runtime_stats["memory"] += time.perf_counter() - start

        start = time.perf_counter() if collect_runtime_stats else 0.0
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = None
        if self.mtp_heads:
            mtp_logits = torch.stack([head(hidden) for head in self.mtp_heads], dim=2)

        pooled = hidden.mean(dim=1)
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled)).squeeze(-1)
        if runtime_stats is not None:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            runtime_stats["heads"] += time.perf_counter() - start
        return ModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            uncertainty=uncertainty,
            memory_state=memory_state,
            arbiter_outputs=arbiter_outputs,
            memory_reads=memory_reads,
            block_aux=block_aux,
            runtime_stats=runtime_stats,
        )
