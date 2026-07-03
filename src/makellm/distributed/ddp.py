"""Distributed Data Parallel (DDP) 설정 유틸.

PyTorch의 torch.distributed를 래핑하여 단일 프로세스 환경에서도
동작하도록 폴백을 제공. 실제 다중 프로세스는 spawn 또는 torchrun으로 실행.
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """분산 환경이 초기화되었는지 확인."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """전체 프로세스 수. 분산이 아니면 1."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """현재 프로세스 순위. 분산이 아니면 0."""
    if is_distributed():
        return dist.get_rank()
    return 0


def setup_distributed(backend: str = "gloo") -> None:
    """분산 환경 초기화.

    환경 변수에서 world_size, rank, master_addr, master_port를 읽어 초기화.
    단일 프로세스 환경에서는 아무 작업도 수행하지 않음.
    """
    if dist.is_available() and not dist.is_initialized():
        # 환경 변수가 설정된 경우만 초기화
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
    # 단일 프로세스면 그냥 통과


def cleanup_distributed() -> None:
    """분산 환경 정리."""
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """모든 프로세스의 텐서 평균. 분산이 아니면 입력 그대로."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """모든 프로세스의 텐서를 모음. 분산이 아니면 입력 그대로."""
    if not is_distributed():
        return tensor
    output = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(output, tensor)
    return torch.stack(output)
