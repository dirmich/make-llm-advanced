"""AWQ (Activation-aware Weight Quantization) — 단순화 구현.

Lin et al. (2023) "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

핵심:
  - 가중치의 중요도를 평가할 때 activation 통계를 활용
  - 큰 activation을 가지는 채널의 가중치는 더 높은 정밀도로 유지 (salient)
  - 전체는 INT4/INT8로 양자화하되, salient 채널은 스케일링으로 보정
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AWQQuantizer:
    """AWQ 양자화 — 단순화 구현.

    실제 AWQ는 calibration 데이터로 activation 통계를 수집해야 하지만,
    여기서는 가중치 통계로 대체하여 개념을 보여줌.
    """

    def __init__(self, n_bits: int = 4, group_size: int = 128, alpha: float = 1.0):
        self.n_bits = n_bits
        self.group_size = group_size
        self.alpha = alpha
        self.qmax = 2 ** (n_bits - 1) - 1  # 7 for INT4, 127 for INT8

    def quantize_linear(self, linear: nn.Linear) -> dict:
        """Linear 레이어 AWQ 양자화.

        Returns:
            메타데이터 (scales, zero_points 등)
        """
        w = linear.weight.data
        # 채널별 중요도 (가중치의 분산으로 대체 — 실제로는 activation 통계 사용)
        importance = w.abs().mean(dim=-1)  # [out_features]
        # 상위 1%를 salient로 간주
        threshold = torch.quantile(importance, 0.99)
        salient_mask = importance > threshold

        # Salient 채널은 스케일 팩터를 곱하여 정밀도 보정
        scale_factor = torch.where(
            salient_mask,
            torch.tensor(self.alpha, device=w.device),
            torch.tensor(1.0, device=w.device),
        )
        w_scaled = w * scale_factor.unsqueeze(-1)

        # 그룹별 양자화
        orig_shape = w_scaled.shape
        w_flat = w_scaled.reshape(-1)
        n_groups = (w_flat.numel() + self.group_size - 1) // self.group_size
        pad = n_groups * self.group_size - w_flat.numel()
        if pad > 0:
            w_flat = torch.cat([w_flat, torch.zeros(pad, device=w.device, dtype=w.dtype)])
        w_grouped = w_flat.reshape(n_groups, self.group_size)
        absmax = w_grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = absmax / self.qmax
        q = torch.round(w_grouped / scale).clamp(-self.qmax - 1, self.qmax).to(torch.int8)

        # 양자화된 가중치를 다시 float로 복원하여 저장 (시뮬레이션)
        dequant = q.float() * scale.repeat(1, self.group_size)
        dequant = dequant.reshape(-1)[:orig_shape.numel()].reshape(orig_shape)
        # 스케일 팩터로 나누어 원래 스케일로 복원
        linear.weight.data = dequant / scale_factor.unsqueeze(-1)

        return {
            "n_bits": self.n_bits,
            "group_size": self.group_size,
            "n_salient": int(salient_mask.sum().item()),
            "alpha": self.alpha,
        }

    def quantize_model(self, model: nn.Module) -> dict:
        """모델 전체 양자화."""
        stats = {"layers": {}}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                stats["layers"][name] = self.quantize_linear(module)
        return stats
