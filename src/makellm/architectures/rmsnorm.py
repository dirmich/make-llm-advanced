"""RMSNorm (Root Mean Square Layer Normalization).

Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"

LayerNorm과 달리 평균을 빼지 않고 RMS(제곱평균제곱근)로만 정규화.
  - 계산량 감소 (mean 계산 생략)
  - 학습 안정성 향상 (LLaMA, Mistral 등에서 사용)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm.

    수식: y = x / sqrt(mean(x^2) + eps) * gamma
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] → [..., d_model]"""
        # float32로 계산 (수치 안정성)
        input_dtype = x.dtype
        x = x.float()
        # RMS 계산
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # 정규화
        x = x * torch.rsqrt(ms + self.eps)
        # 원래 dtype으로 복원 + 가중치 곱
        return (x.to(input_dtype) * self.weight)
