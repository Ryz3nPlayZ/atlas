from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ace_atlas.config import MemoryConfig


@dataclass(slots=True)
class MemoryState:
    episodic_keys: Tensor
    episodic_values: Tensor
    episodic_scores: Tensor
    semantic_keys: Tensor
    semantic_values: Tensor
    semantic_scores: Tensor
    episodic_ptr: Tensor
    semantic_ptr: Tensor


@dataclass(slots=True)
class MemoryReadResult:
    context: Tensor
    scores: Tensor


class BoundedMemory(nn.Module):
    def __init__(self, model_dim: int, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.key_proj = nn.Linear(model_dim, config.key_dim)
        self.value_proj = nn.Linear(model_dim, config.value_dim)
        self.read_proj = nn.Linear(config.value_dim, model_dim)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        c = self.config
        return MemoryState(
            episodic_keys=torch.zeros(batch_size, c.episodic_slots, c.key_dim, device=device, dtype=dtype),
            episodic_values=torch.zeros(batch_size, c.episodic_slots, c.value_dim, device=device, dtype=dtype),
            episodic_scores=torch.zeros(batch_size, c.episodic_slots, device=device, dtype=dtype),
            semantic_keys=torch.zeros(batch_size, c.semantic_slots, c.key_dim, device=device, dtype=dtype),
            semantic_values=torch.zeros(batch_size, c.semantic_slots, c.value_dim, device=device, dtype=dtype),
            semantic_scores=torch.zeros(batch_size, c.semantic_slots, device=device, dtype=dtype),
            episodic_ptr=torch.zeros(batch_size, device=device, dtype=torch.long),
            semantic_ptr=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def summarize(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        chunk_repr = hidden.mean(dim=1)
        return self.key_proj(chunk_repr), self.value_proj(chunk_repr)

    def read(self, query: Tensor, state: MemoryState) -> MemoryReadResult:
        query = self.key_proj(query)
        epi_scores = torch.einsum("bd,bsd->bs", query, state.episodic_keys)
        sem_scores = torch.einsum("bd,bsd->bs", query, state.semantic_keys)

        epi_k = min(self.config.read_top_k, state.episodic_keys.shape[1])
        sem_k = min(self.config.read_top_k, state.semantic_keys.shape[1])

        epi_vals = torch.zeros_like(state.episodic_values[:, :1].mean(dim=1))
        sem_vals = torch.zeros_like(state.semantic_values[:, :1].mean(dim=1))

        if epi_k > 0:
            epi_top_scores, epi_top_idx = torch.topk(epi_scores, k=epi_k, dim=-1)
            epi_selected = torch.gather(
                state.episodic_values,
                1,
                epi_top_idx.unsqueeze(-1).expand(-1, -1, state.episodic_values.size(-1)),
            )
            epi_vals = (epi_selected * torch.softmax(epi_top_scores, dim=-1).unsqueeze(-1)).sum(dim=1)

        if sem_k > 0:
            sem_top_scores, sem_top_idx = torch.topk(sem_scores, k=sem_k, dim=-1)
            sem_selected = torch.gather(
                state.semantic_values,
                1,
                sem_top_idx.unsqueeze(-1).expand(-1, -1, state.semantic_values.size(-1)),
            )
            sem_vals = (sem_selected * torch.softmax(sem_top_scores, dim=-1).unsqueeze(-1)).sum(dim=1)

        context = self.read_proj(epi_vals + sem_vals)
        all_scores = torch.cat([epi_scores, sem_scores], dim=-1)
        return MemoryReadResult(context=context, scores=all_scores)

    def write_episodic(self, keys: Tensor, values: Tensor, scores: Tensor, state: MemoryState) -> MemoryState:
        batch = keys.size(0)
        for b in range(batch):
            if float(scores[b].item()) <= 0.0:
                continue
            slot = int(state.episodic_ptr[b].item() % self.config.episodic_slots)
            state.episodic_keys[b, slot] = keys[b]
            state.episodic_values[b, slot] = values[b]
            state.episodic_scores[b, slot] = scores[b]
            state.episodic_ptr[b] = state.episodic_ptr[b] + 1
        return state

    def update_semantic(self, keys: Tensor, values: Tensor, scores: Tensor, state: MemoryState) -> MemoryState:
        batch = keys.size(0)
        for b in range(batch):
            if float(scores[b].item()) <= 0.0:
                continue
            slot = int(state.semantic_ptr[b].item() % self.config.semantic_slots)
            state.semantic_keys[b, slot] = keys[b]
            state.semantic_values[b, slot] = values[b]
            state.semantic_scores[b, slot] = scores[b]
            state.semantic_ptr[b] = state.semantic_ptr[b] + 1
        return state
