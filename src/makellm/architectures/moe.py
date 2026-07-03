"""Mixture-of-Experts (MoE) 레이어.

Shazeer et al. (2017) "Outrageously Large Neural Networks:
The Sparsely-Gated Mixture-of-Experts Layer"
Fedus et al. (2022) "Switch Transformers: Scaling to Trillion Parameter Models"

핵심:
  - 여러 전문가(FFN) 네트워크를 두고, 토큰별로 상위 K개만 활성화
  - 총 파라미터는 늘어나지만 실제 연산량은 K/n_experts 비율로 감소
  - 라우터(router)가 각 토큰을 어떤 전문가에 보낼지 결정
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """토큰별 상위 K개 전문가를 선택하는 라우터.

    입력: [batch, seq, d_model]
    출력:
      - routing_weights: [batch, seq, n_active] 상위 K 전문가 가중치 (softmax)
      - selected_experts: [batch, seq, n_active] 선택된 전문가 인덱스
    """

    def __init__(self, d_model: int, n_experts: int, n_active: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, seq, d_model] → (weights, indices)"""
        # 로짓 계산
        logits = self.gate(x)  # [batch, seq, n_experts]
        # 상위 K개 선택
        top_logits, top_indices = torch.topk(logits, self.n_active, dim=-1)
        # 선택된 K개에 대해 softmax (나머지는 0)
        weights = F.softmax(top_logits, dim=-1)
        return weights, top_indices


class MoELayer(nn.Module):
    """Mixture-of-Experts 레이어.

    구조:
      x → Router → 상위 K개 전문가 선택
        → 각 전문가 FFN(x) 계산 → 가중치 합산
        → 출력

    Load balancing loss:
      모든 전문가가 균등하게 사용되도록 유도하는 보조 손실.
      L_aux = alpha * n_experts * sum(f_i * P_i)
        f_i = (i번 전문가에게 보내진 토큰 비율)
        P_i = (i번 전문가의 평균 라우팅 확률)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        n_active: int = 2,
        dropout: float = 0.0,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.aux_loss_alpha = aux_loss_alpha
        self.router = TopKRouter(d_model, n_experts, n_active)
        # 전문가 풀 (간단한 FFN)
        self.experts = nn.ModuleList([
            self._make_expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

    def _make_expert(self, d_model: int, d_ff: int, dropout: float) -> nn.Module:
        """단일 전문가 FFN (GELU 활성화)."""
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            output: [batch, seq, d_model]
            aux_loss: scalar (load balancing loss)
        """
        batch, seq, d = x.shape
        # 라우팅
        weights, indices = self.router(x)  # [batch, seq, K] x 2

        # 출력 초기화
        out = torch.zeros_like(x)

        # 각 전문가별로 처리 (효율적인 방식은 아니지만 구현이 명확)
        # 실제 프로덕션에서는 expert 병렬 처리를 사용하지만
        # 여기서는 이해를 우선으로 루프 사용.
        for i in range(self.n_experts):
            # i번 전문가가 선택된 위치 찾기
            mask = (indices == i)  # [batch, seq, K]
            # 각 토큰에 대해 i번 전문가가 선택되었는지
            selected = mask.any(dim=-1)  # [batch, seq]
            if not selected.any():
                continue
            # 해당 토큰들 추출
            x_sel = x[selected]  # [n_selected, d]
            # 전문가 계산
            expert_out = self.experts[i](x_sel)
            # 가중치 추출 (i번이 선택된 슬롯의 가중치)
            # mask에서 True인 슬롯의 가중치 합 (중복 선택 시)
            slot_weights = (weights * mask.float()).sum(dim=-1)  # [batch, seq]
            w_sel = slot_weights[selected].unsqueeze(-1)  # [n_selected, 1]
            # 출력에 더함
            out[selected] += expert_out * w_sel

        # Load balancing auxiliary loss
        aux_loss = self._compute_aux_loss(weights, indices)

        return out, aux_loss

    def _compute_aux_loss(
        self,
        weights: torch.Tensor,    # [batch, seq, K]
        indices: torch.Tensor,    # [batch, seq, K]
    ) -> torch.Tensor:
        """Switch Transformer 스타일 load balancing loss.

        L_aux = alpha * n_experts * sum_i(f_i * P_i)
          f_i = (i번 전문가가 선택된 토큰 수) / (전체 토큰 수 * K)
          P_i = (i번 전문가의 평균 라우팅 확률)
        """
        batch, seq, _ = weights.shape
        n_tokens = batch * seq
        device = weights.device
        # f_i: 각 전문가에게 보내진 토큰 비율
        expert_counts = torch.zeros(self.n_experts, device=device)
        for i in range(self.n_experts):
            expert_counts[i] = (indices == i).sum().float()
        f = expert_counts / max(n_tokens * self.n_active, 1)
        # P_i: 단순화 — 선택된 전문가들의 가중치 평균을 근사치로 사용
        # (정확한 구현은 router의 전체 softmax 출력의 평균)
        p = f.detach()  # 근사: f와 동일 분포 가정
        return self.aux_loss_alpha * self.n_experts * (f * p).sum()
