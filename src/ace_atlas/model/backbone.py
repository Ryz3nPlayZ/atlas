from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ace_atlas.config import ACEAtlasConfig
from ace_atlas.model.arbiter import ArbiterOutput, MemoryAction, MemoryArbiter
from ace_atlas.model.attention import LocalCausalSelfAttention
from ace_atlas.model.memory import BoundedMemory, MemoryReadResult, MemoryState
from ace_atlas.model.moe import MoEAux, SparseMoE
from ace_atlas.model.recurrent import BootstrapRecurrentMixer, RecurrentState
from ace_atlas.model.types import ModelOutput


@dataclass(slots=True)
class BlockAux:
    moe: MoEAux | None


class HybridBlock(nn.Module):
    def __init__(self, config: ACEAtlasConfig, use_attention: bool) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.recurrent_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.attention_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.moe_norm = nn.LayerNorm(config.model_dim, eps=config.norm_epsilon)
        self.recurrent = BootstrapRecurrentMixer(config.model_dim, config.recurrent)
        self.attention = (
            LocalCausalSelfAttention(config.model_dim, config.attention) if use_attention else None
        )
        self.moe = SparseMoE(config.model_dim, config.moe)

    def forward(
        self,
        hidden: Tensor,
        recurrent_state: RecurrentState | None = None,
    ) -> tuple[Tensor, RecurrentState, BlockAux]:
        recurrent_out, recurrent_state = self.recurrent(self.recurrent_norm(hidden), recurrent_state)
        hidden = hidden + recurrent_out

        if self.attention is not None:
            hidden = hidden + self.attention(self.attention_norm(hidden))

        moe_out, moe_aux = self.moe(self.moe_norm(hidden))
        hidden = hidden + moe_out
        return hidden, recurrent_state, BlockAux(moe=moe_aux)


class ACEAtlasModel(nn.Module):
    def __init__(self, config: ACEAtlasConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
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
        memory_state: MemoryState | None = None,
    ) -> ModelOutput:
        hidden = self.dropout(self.embed_tokens(input_ids))
        recurrent_state: RecurrentState | None = None
        arbiter_outputs: list[ArbiterOutput] = []
        memory_reads: list[MemoryReadResult] = []
        block_aux: list[BlockAux] = []

        for layer_idx, layer in enumerate(self.layers):
            hidden, recurrent_state, aux = layer(hidden, recurrent_state)
            block_aux.append(aux)
            if (layer_idx + 1) % self.memory_every_n_layers == 0:
                hidden, memory_state = self._apply_memory_bus(
                    hidden,
                    memory_state,
                    arbiter_outputs=arbiter_outputs,
                    memory_reads=memory_reads,
                )

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = None
        if self.mtp_heads:
            mtp_logits = torch.stack([head(hidden) for head in self.mtp_heads], dim=2)

        pooled = hidden.mean(dim=1)
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled)).squeeze(-1)
        return ModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            uncertainty=uncertainty,
            memory_state=memory_state,
            arbiter_outputs=arbiter_outputs,
            memory_reads=memory_reads,
            block_aux=block_aux,
        )
