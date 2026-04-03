from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class AttentionConfig:
    window_size: int = 4096
    num_heads: int = 16
    num_kv_heads: int = 8
    dropout: float = 0.0
    use_bias: bool = False
    qk_norm: bool = False
    rope_base_local: float = 10_000.0
    rope_base_global: float = 1_000_000.0
    global_latent_dim: int = 0


@dataclass(slots=True)
class RecurrentConfig:
    kind: str = "xlstm_bootstrap"
    state_dim: int = 1024
    expansion_factor: int = 2
    dropout: float = 0.0


@dataclass(slots=True)
class MoEConfig:
    enabled: bool = True
    num_shared_experts: int = 2
    num_routed_experts: int = 16
    top_k: int = 2
    hidden_dim: int = 4096
    dropout: float = 0.0


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    episodic_slots: int = 64
    semantic_slots: int = 32
    key_dim: int = 1024
    value_dim: int = 1024
    read_top_k: int = 4
    max_writes_per_chunk: int = 2


@dataclass(slots=True)
class ArbiterConfig:
    enabled: bool = True
    hidden_dim: int = 1024
    cost_weight: float = 0.05


@dataclass(slots=True)
class EscalationConfig:
    enabled: bool = True
    uncertainty_threshold: float = 0.65
    max_deliberation_steps: int = 2
    max_tool_calls: int = 2


@dataclass(slots=True)
class TransformerConfig:
    local_layers_per_global: int = 5
    dense_hidden_dim: int = 4096
    moe_start_fraction: float = 0.5
    use_rms_norm: bool = True
    use_rotary_embeddings: bool = True


@dataclass(slots=True)
class ModeConditioningConfig:
    enabled: bool = False
    modes: list[str] = field(default_factory=lambda: ["general", "code", "answer"])
    shared_experts_per_mode: int = 1


@dataclass(slots=True)
class ACEAtlasConfig:
    architecture: str = "gru_hybrid"
    vocab_size: int = 50257
    model_dim: int = 1024
    num_layers: int = 16
    attention_every_n: int = 4
    max_position_embeddings: int = 32768
    dropout: float = 0.0
    norm_epsilon: float = 1e-5
    mtp_horizon: int = 2
    answer_span_embeddings: bool = False
    completion_adapter_dim: int = 0
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mode_conditioning: ModeConditioningConfig = field(default_factory=ModeConditioningConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ACEAtlasConfig":
        return cls(
            architecture=data.get("architecture", "gru_hybrid"),
            vocab_size=data.get("vocab_size", 50257),
            model_dim=data.get("model_dim", 1024),
            num_layers=data.get("num_layers", 16),
            attention_every_n=data.get("attention_every_n", 4),
            max_position_embeddings=data.get("max_position_embeddings", 32768),
            dropout=data.get("dropout", 0.0),
            norm_epsilon=data.get("norm_epsilon", 1e-5),
            mtp_horizon=data.get("mtp_horizon", 2),
            answer_span_embeddings=data.get("answer_span_embeddings", False),
            completion_adapter_dim=data.get("completion_adapter_dim", 0),
            attention=AttentionConfig(**data.get("attention", {})),
            recurrent=RecurrentConfig(**data.get("recurrent", {})),
            moe=MoEConfig(**data.get("moe", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            arbiter=ArbiterConfig(**data.get("arbiter", {})),
            escalation=EscalationConfig(**data.get("escalation", {})),
            transformer=TransformerConfig(**data.get("transformer", {})),
            mode_conditioning=ModeConditioningConfig(**data.get("mode_conditioning", {})),
        )

    @classmethod
    def small(cls) -> "ACEAtlasConfig":
        return cls(
            model_dim=512,
            num_layers=8,
            attention=AttentionConfig(window_size=2048, num_heads=8, num_kv_heads=4),
            recurrent=RecurrentConfig(state_dim=512),
            moe=MoEConfig(num_routed_experts=8, hidden_dim=2048),
            memory=MemoryConfig(key_dim=512, value_dim=512),
            arbiter=ArbiterConfig(hidden_dim=512),
        )
