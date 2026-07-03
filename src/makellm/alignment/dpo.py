"""DPO (Direct Preference Optimization).

Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model"

RLHF의 복잡한 PPO 파이프라인 없이, 선호도 데이터만으로 직접 정렬.
보상 모델을 따로 학습할 필요 없이 정책 모델 자체가 보상을 근사.

핵심 손실:
  L_DPO = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    y_w: 선택된(winning) 응답
    y_l: 거절된(losing) 응답
    π: 현재 정책, π_ref: 참조 정책 (고정)
    β: 온도 파라미터
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def get_sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """시퀀스의 로그 확률 합 계산.

    Args:
        model: 언어 모델 (input_ids → logits 반환)
        input_ids: [batch, seq]
        target_ids: [batch, seq]
    Returns:
        logprobs: [batch] — 각 시퀀스의 총 로그 확률
    """
    logits = model(input_ids)  # [batch, seq, vocab]
    # 마지막 위치는 타겟이 없으므로 제외
    logits = logits[:, :-1, :]
    targets = target_ids[:, 1:]
    # 로그 확률
    log_probs = F.log_softmax(logits, dim=-1)
    # 타겟 토큰의 로그 확률 추출
    mask = (targets != ignore_index).float()
    targets_clamped = targets.clamp(min=0)
    token_logprobs = log_probs.gather(-1, targets_clamped.unsqueeze(-1)).squeeze(-1)
    # 마스크 적용하여 합산
    seq_logprobs = (token_logprobs * mask).sum(dim=-1)
    return seq_logprobs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO 손실 계산.

    L = -log σ(β · ((π(y_w)/π_ref(y_w)) - (π(y_l)/π_ref(y_l))))
      = -log σ(β · ((log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))))
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    return -F.logsigmoid(logits).mean()


class DPOTrainer:
    """DPO 학습 루프."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        beta: float = 0.1,
        lr: float = 5e-7,
        device: str = "cpu",
    ):
        self.policy = policy_model
        # 참조 모델은 고정
        self.reference = reference_model
        for p in self.reference.parameters():
            p.requires_grad = False
        self.beta = beta
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.reference.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=lr,
        )

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> float:
        """DPO 1스텝 학습."""
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)

        # 정책 모델 로그 확률
        policy_chosen_logps = get_sequence_logprobs(
            self.policy, chosen_ids, chosen_ids
        )
        policy_rejected_logps = get_sequence_logprobs(
            self.policy, rejected_ids, rejected_ids
        )

        # 참조 모델 로그 확률 (no_grad)
        with torch.no_grad():
            ref_chosen_logps = get_sequence_logprobs(
                self.reference, chosen_ids, chosen_ids
            )
            ref_rejected_logps = get_sequence_logprobs(
                self.reference, rejected_ids, rejected_ids
            )

        # DPO 손실
        loss = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
