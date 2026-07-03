"""Pipeline 병렬 — 단순화 구현.

Huang et al. (2019) "GPipe: Efficient Training of Giant Neural Networks
using Pipeline Parallelism"

핵심:
  - 모델을 여러 stage(레이어 그룹)로 분할
  - 각 stage를 다른 GPU에 배치
  - micro-batch로 나누어 파이프라인 처리
  - 버블(bubble)이 발생하지만 전체 처리량은 향상

여기서는 단일 프로세스에서 동작하는 시뮬레이션 구현.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence


class PipelineParallel(nn.Module):
    """파이프라인 병렬 래퍼.

    모델을 순차적 stage로 분할. micro-batch 단위로 forward 수행.
    """

    def __init__(self, stages: Sequence[nn.Module], n_micro_batches: int = 4):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.n_micro_batches = n_micro_batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순방향 파이프라인.

        micro-batch로 분할하여 각 stage를 순차 처리.
        단일 프로세스이므로 실제 병렬은 아니지만, 개념을 보여줌.
        """
        batch_size = x.shape[0]
        # 마이크로배치 분할
        micro_size = max(1, batch_size // self.n_micro_batches)
        micro_batches = [
            x[i : i + micro_size] for i in range(0, batch_size, micro_size)
        ]

        # 각 마이크로배치를 모든 stage에 통과
        outputs = []
        for mb in micro_batches:
            for stage in self.stages:
                mb = stage(mb)
            outputs.append(mb)

        # 결합
        return torch.cat(outputs, dim=0)

    def num_stages(self) -> int:
        return len(self.stages)
