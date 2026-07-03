"""INT4 양자화 — GPTQ 스타일 단순화 구현.

4-bit 양자화는 INT8 대비 추가 2배 메모리 절약.
LLaMA, Mistral 등 최신 모델이 추론에 사용.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_int4(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """INT4 양자화 (대칭, 그룹별).

    INT4 범위: -8 ~ 7 (부호 있음)
    스케일 = absmax / 7
    """
    orig_shape = w.shape
    w_flat = w.reshape(-1)
    n_groups = (w_flat.numel() + group_size - 1) // group_size
    pad = n_groups * group_size - w_flat.numel()
    if pad > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad, device=w.device, dtype=w.dtype)])
    w_grouped = w_flat.reshape(n_groups, group_size)
    absmax = w_grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.round(w_grouped / scale).clamp(-8, 7).to(torch.int8)
    return q.reshape(-1)[:orig_shape.numel()].reshape(orig_shape), scale.squeeze(-1)


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """INT4 가중치를 float로 복원."""
    orig_shape = qweight.shape
    q_flat = qweight.reshape(-1).float()
    if scales.dim() == 0:
        # 단일 스케일
        return (q_flat * scales).reshape(orig_shape)
    # 각 스케일이 group_size만큼의 토큰을 담당
    group_size = (q_flat.numel() + scales.numel() - 1) // scales.numel()
    scales_expanded = scales.repeat_interleave(group_size)[: q_flat.numel()]
    return (q_flat * scales_expanded).reshape(orig_shape)
