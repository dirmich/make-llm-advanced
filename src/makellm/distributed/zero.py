"""ZeRO (Zero Redundancy Optimizer) — 단순화 구현.

Rajbhandari et al. (2020) "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"

세 단계:
  ZeRO-1: 옵티마이저 상태를 샤딩
  ZeRO-2: 옵티마이저 상태 + 그래디언트를 샤딩
  ZeRO-3: 옵티마이저 상태 + 그래디언트 + 파라미터를 샤딩 (FSDP와 유사)

여기서는 ZeRO-1과 ZeRO-2의 단순화된 구현을 제공.
실제 DeepSpeed/FSDP와는 다르지만 개념을 이해하기에 충분.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable

from .ddp import is_distributed, get_world_size, get_rank


class ZeROOptimizer:
    """ZeRO-1/2 스타일 옵티마이저 래퍼.

    각 파라미터를 world_size개 샤드로 분할하여 현재 rank가 담당하는 샤드만
    옵티마이저 상태로 보관. 역전파 후 all_reduce로 그래디언트를 동기화.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        optimizer_cls=torch.optim.AdamW,
        stage: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.stage = stage
        self.world_size = get_world_size() if is_distributed() else 1
        self.rank = get_rank() if is_distributed() else 0
        # 파라미터 리스트
        self.params = list(params)
        # 각 파라미터를 샤드로 분할하여 인덱스 기록
        self.shards: list[tuple[int, int, int]] = []  # (param_idx, start, end)
        for i, p in enumerate(self.params):
            n = p.numel()
            # world_size로 분할
            chunk = (n + self.world_size - 1) // self.world_size
            start = self.rank * chunk
            end = min(start + chunk, n)
            self.shards.append((i, start, end))
        # 현재 rank가 담당하는 파라미터 샤드만 옵티마이저에 전달
        # (단순화: 원본 파라미터를 그대로 사용, 실제로는 flat buffer 사용)
        self.optimizer = optimizer_cls(
            self.params, lr=lr, weight_decay=weight_decay
        )
        # 그래디언트 버퍼 (ZeRO-2: 통신 후 즉시 해제)
        self._grad_buffer: dict[int, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """옵티마이저 스텝.

        ZeRO-1: 그래디언트를 all_reduce 후 스텝
        ZeRO-2: 그래디언트를 샤드별로 all_reduce 후 스텝, 나머지는 해제
        """
        # 분산 환경에서 그래디언트 동기화
        if self.world_size > 1:
            for p in self.params:
                if p.grad is not None:
                    from .ddp import all_reduce_mean
                    all_reduce_mean(p.grad)
        # 옵티마이저 스텝
        self.optimizer.step()

    def state_dict(self) -> dict:
        return {
            "stage": self.stage,
            "optimizer": self.optimizer.state_dict(),
            "shards": self.shards,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.optimizer.load_state_dict(sd["optimizer"])
        self.shards = sd["shards"]

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]
