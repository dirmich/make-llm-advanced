"""LoRA (Low-Rank Adaptation).

Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"

핵심:
  - 사전학습된 가중치 W를 고정하고, 저랭크 행렬 ΔW = BA를 추가
  - A: [r, in], B: [out, r], r << min(in, out)
  - 학습 가능한 파라미터는 r * (in + out)로 크게 감소
  - 추론 시 W + BA로 합쳐서 오버헤드 0
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Iterable


class LoRAConfig:
    """LoRA 설정."""

    def __init__(self, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        # 스케일링 팩터: alpha / rank
        self.scaling = alpha / rank


class LoRALinear(nn.Module):
    """LoRA가 적용된 Linear 레이어.

    구조:
      y = W·x + (α/r) · B·A·x
      W: 고정된 사전학습 가중치
      A, B: 학습 가능한 저랭크 행렬
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원본 Linear (고정)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA 어댑터
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # A는 정규분포 초기화, B는 0 초기화 (초기 ΔW = 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_features] → [..., out_features]"""
        # 원본 출력 (고정 가중치)
        base_out = self.base(x)
        # 병합된 경우 어댑터가 base에 합쳐졌으므로 LoRA 연산 생략
        if self.lora_A is None or self.lora_B is None:
            return base_out
        # LoRA 출력
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A.T  # [..., rank]
        lora_out = lora_out @ self.lora_B.T  # [..., out_features]
        return base_out + self.scaling * lora_out

    def merge_weights(self) -> None:
        """LoRA 가중치를 base에 합성 (추론 최적화)."""
        with torch.no_grad():
            # W += scaling * B @ A
            self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            # 어댑터 제거 (선택적)
            self.lora_A = None
            self.lora_B = None

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """모델의 모든 Linear를 LoRALinear로 교체 (in-place)."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = LoRALinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha, dropout=dropout,
                bias=module.bias is not None,
            )
            # 원본 가중치 복사
            new_layer.base.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.base.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            apply_lora(module, rank, alpha, dropout)
    return model


def count_lora_parameters(model: nn.Module) -> dict:
    """LoRA 파라미터 통계."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
