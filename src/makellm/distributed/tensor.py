"""Tensor Parallel — Megatron-LM 스타일.

Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter Language Models
Using Model Parallelism"

두 가지 핵심 연산:
  1. ColumnParallelLinear: 가중치를 열(column) 기준으로 분할
  2. RowParallelLinear: 가중치를 행(row) 기준으로 분할

MLP: ColumnParallel → activation → RowParallel (입출력이 d_model로 유지)
Attention: Q/K/V를 ColumnParallel로, output을 RowParallel로 분할
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddp import is_distributed, get_world_size, get_rank


def _get_parallel_info():
    """현재 rank와 world_size 반환. 분산이 아니면 0, 1."""
    if is_distributed():
        return get_rank(), get_world_size()
    return 0, 1


class ColumnParallelLinear(nn.Module):
    """열 병렬 선형 레이어.

    weight W: [out_features, in_features]를 world_size개로 분할:
      각 rank는 W[i*chunk:(i+1)*chunk, :]를 보관
      출력 차원이 줄어들지만, all-gather로 합쳐서 원래 차원 복원 가능.

    단순화: 로컬에서 전체 가중치를 보관하되 슬라이스만 사용 (시뮬레이션).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank, self.world_size = _get_parallel_info()
        # 각 rank가 담당할 출력 차원
        self.out_per_rank = (out_features + self.world_size - 1) // self.world_size
        # 로컬 가중치 (시뮬레이션: 전체를 보관하되 슬라이스 사용)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_features] → [..., out_per_rank]"""
        # 자기 rank가 담당하는 열 슬라이스
        start = self.rank * self.out_per_rank
        end = min(start + self.out_per_rank, self.out_features)
        w_local = self.weight[start:end]
        b_local = self.bias[start:end] if self.bias is not None else None
        return F.linear(x, w_local, b_local)


class RowParallelLinear(nn.Module):
    """행 병렬 선형 레이어.

    weight W: [out_features, in_features]를 행 기준으로 분할:
      각 rank는 W[:, i*chunk:(i+1)*chunk]를 보관
      입력 차원이 줄어들고, all-reduce로 출력 합산.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank, self.world_size = _get_parallel_info()
        self.in_per_rank = (in_features + self.world_size - 1) // self.world_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_per_rank] → [..., out_features]
        (모든 rank의 결과를 합산해야 정확하지만, 단일 rank 시뮬레이션에서는 그대로)
        """
        # 자기 rank가 담당하는 행 슬라이스
        start = self.rank * self.in_per_rank
        end = min(start + self.in_per_rank, self.in_features)
        w_local = self.weight[:, start:end]
        # 입력도 같은 슬라이스
        x_local = x[..., start:end] if x.shape[-1] == self.in_features else x
        out = F.linear(x_local, w_local)
        if self.bias is not None:
            out = out + self.bias
        return out
