"""고급편 설정 데이터클래스."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class AdvancedModelConfig:
    """고급 아키텍처를 포함한 모델 설정."""

    vocab_size: int = 32000
    context_length: int = 2048
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0
    # 최신 아키텍처 옵션
    use_rope: bool = True              # Rotary Position Embedding
    use_gqa: bool = False              # Grouped Query Attention
    n_kv_heads: int | None = None      # GQA 사용 시 KV 헤드 수
    use_swiglu: bool = True            # SwiGLU 활성화
    use_rmsnorm: bool = True           # RMSNorm
    # MoE
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    # 기본
    pad_token_id: int = 0
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdvancedModelConfig":
        return cls(**d)


@dataclass
class DistributedConfig:
    """분산 학습 설정."""

    backend: str = "gloo"             # "nccl" (GPU) 또는 "gloo" (CPU)
    world_size: int = 1               # 전체 프로세스 수
    rank: int = 0                     # 현재 프로세스 순위
    local_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    # FSDP
    fsdp_enabled: bool = False
    fsdp_shard_grad_op: bool = False  # ZeRO-2와 유사
    fsdp_full_shard: bool = True      # ZeRO-3와 유사
    # ZeRO
    zero_stage: int = 0               # 0=비활성, 1=옵티마이저, 2=그래디언트, 3=파라미터
    # 체크포인트
    activation_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistributedConfig":
        return cls(**d)


@dataclass
class QuantConfig:
    """양자화 설정."""

    method: str = "int8"              # "int8", "int4", "awq", "gptq"
    weight_bits: int = 8
    group_size: int = 128             # 그룹 양자화 크기
    compute_dtype: str = "fp16"       # "fp16", "bf16", "fp32"
    # AWQ
    awq_alpha: float = 1.0
    # GPTQ
    gptq_iterations: int = 20


@dataclass
class AlignmentConfig:
    """정렬(SFT/RLHF/DPO) 설정."""

    method: str = "sft"               # "sft", "rlhf", "dpo", "lora"
    # 공통
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_epochs: int = 3
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # DPO
    dpo_beta: float = 0.1
    # RLHF
    ppo_clip: float = 0.2
    ppo_kl_coef: float = 0.1
    reward_model_dim: int = 768


def save_config(cfg: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
