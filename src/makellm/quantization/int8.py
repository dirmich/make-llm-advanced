"""INT8 양자화 — 대칭(absmax) 방식.

Frantar et al. (2022) "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
Dettmers et al. (2022) "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"

핵심:
  - 가중치를 INT8 (-128~127)로 양자화하여 메모리 4배 절약
  - absmax: 가중치의 최대 절댓값을 스케일로 사용
  - 그룹 단위 양자화로 정확도 손실 최소화
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_int8(w: torch.Tensor, group_size: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 양자화 (absmax 대칭 방식).

    Args:
        w: 양자화할 가중치 텐서
        group_size: 그룹 크기 (-1이면 채널별, 양수면 그룹별 양자화)
    Returns:
        (qweight, scales) — qweight는 int8, scales는 float
    """
    orig_shape = w.shape
    if group_size <= 0:
        # 채널별 양자화 (마지막 차원 기준)
        # w를 [out, in]으로 평탄화
        w_flat = w.reshape(-1, orig_shape[-1]) if w.dim() > 1 else w.unsqueeze(0)
        # 각 행의 최대 절댓값
        absmax = w_flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = absmax / 127.0
        q = torch.round(w_flat / scale).clamp(-128, 127).to(torch.int8)
        return q.reshape(orig_shape), scale.squeeze(-1)
    else:
        # 그룹별 양자화
        w_flat = w.reshape(-1)
        n_groups = (w_flat.numel() + group_size - 1) // group_size
        # 마지막 그룹 패딩
        pad = n_groups * group_size - w_flat.numel()
        if pad > 0:
            w_flat = torch.cat([w_flat, torch.zeros(pad, device=w.device, dtype=w.dtype)])
        w_grouped = w_flat.reshape(n_groups, group_size)
        absmax = w_grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = absmax / 127.0
        q = torch.round(w_grouped / scale).clamp(-128, 127).to(torch.int8)
        return q.reshape(-1)[:orig_shape.numel()].reshape(orig_shape), scale.squeeze(-1)


def dequantize_int8(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """INT8 양자화된 가중치를 float로 복원.

    Args:
        qweight: int8 텐서
        scales: 스케일 텐서
            - 스칼라: 단일 스케일
            - 1D [out_features]: 채널별 (qweight.shape[0]와 일치)
            - 1D [n_groups]: 그룹별 (qweight.numel() / group_size와 일치)
    """
    orig_shape = qweight.shape
    if scales.dim() == 0 or scales.numel() == 1:
        # 단일 스케일
        return qweight.float() * scales
    if qweight.dim() > 1 and scales.numel() == qweight.shape[0]:
        # 행별 스케일 (채널별)
        return qweight.float() * scales.unsqueeze(-1)
    if qweight.dim() > 1 and scales.numel() == qweight.shape[1]:
        # 열별 스케일
        return qweight.float() * scales.unsqueeze(0)
    # 그룹별 스케일: 각 스케일이 group_size만큼의 원소를 담당
    q_flat = qweight.reshape(-1).float()
    group_size = (q_flat.numel() + scales.numel() - 1) // scales.numel()
    scales_expanded = scales.repeat_interleave(group_size)[: q_flat.numel()]
    return (q_flat * scales_expanded).reshape(orig_shape)


class INT8Quantizer:
    """모델 전체를 INT8로 양자화하는 클래스."""

    def __init__(self, group_size: int = -1):
        self.group_size = group_size
        self.quantized_layers: dict[str, dict] = {}

    def quantize_model(self, model: nn.Module) -> None:
        """모델의 모든 Linear 레이어를 INT8로 양자화 (in-place)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear(name, module)

    def _quantize_linear(self, name: str, linear: nn.Linear) -> None:
        """단일 Linear 레이어 양자화."""
        w = linear.weight.data
        qw, scales = quantize_int8(w, self.group_size)
        # 원본 가중치를 양자화된 버전으로 교체
        # (실제로는 별도 저장하지만, 여기서는 dequantize한 값을 다시 넣어 시뮬레이션)
        linear.weight.data = dequantize_int8(qw, scales)
        self.quantized_layers[name] = {
            "scales": scales,
            "group_size": self.group_size,
            "orig_shape": w.shape,
        }

    def memory_savings(self, model: nn.Module) -> dict:
        """메모리 절약 효과 추정."""
        total_params = sum(p.numel() for p in model.parameters())
        # FP32: 4 bytes/param, INT8: 1 byte/param
        return {
            "fp32_mb": total_params * 4 / 1e6,
            "int8_mb": total_params * 1 / 1e6,
            "savings_ratio": 0.75,  # 75% 절약
        }
