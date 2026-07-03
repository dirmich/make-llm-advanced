"""SwiGLU 활성화 함수.

Shazeer (2020) "GLU Variants Improve Transformer"

수식: SwiGLU(x) = Swish(x * W1) * (x * W3)
  - FFN의 단순 GELU/ReLU를 게이팅 메커니즘으로 대체
  - 3개의 선형 레이어 사용 (W1, W2, W3)
  - 일반적으로 d_ff를 2/3로 줄여 파라미터 수를 맞춤 (LLaMA 방식)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FFN.

    구조:
      x → Linear(d, d_ff) → Swish
                              ↘ multiply → Linear(d_ff, d)
      x → Linear(d, d_ff) → ⤴
    """

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = False):
        super().__init__()
        # d_ff가 None이면 2/3 * 4 * d_model = 8/3 * d_model (LLaMA 스타일)
        if d_ff is None:
            d_ff = int(2 * 4 * d_model / 3)
            # 8의 배수로 반올림 (하드웨어 효율)
            d_ff = ((d_ff + 7) // 8) * 8
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # up
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] → [..., d_model]"""
        gate = F.silu(self.w1(x))   # Swish(gate)
        up = self.w2(x)
        return self.w3(gate * up)
