"""Reward Model (RLHF용).

Stiennon et al. (2020) "Learning to summarize from human feedback"
Ouyang et al. (2022) "Training language models to follow instructions with human feedback"

언어 모델의 마지막에 헤드를 추가하여 시퀀스에 스칼라 보상을 부여.
인간 피드백 데이터(prompt, chosen, rejected)로 학습.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """보상 모델 — LM 백본 위에 스칼라 헤드 추가.

    구조:
      token_ids → [LM 백본] → hidden_states → [pooling] → [linear] → reward

    pooling: 보통 마지막 토큰의 hidden state 사용.
    """

    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.backbone = backbone
        # 스칼라 보상 출력 헤드
        self.reward_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [batch, seq] → rewards: [batch]"""
        # 백본 통과
        if hasattr(self.backbone, "forward"):
            out = self.backbone(input_ids)
            # GPT 클래스의 경우 (logits, hidden) 또는 logits 반환
            if isinstance(out, tuple):
                hidden = out[1]
            elif hasattr(out, "shape") and out.dim() == 3:
                # logits만 반환하는 경우, 마지막 차원이 vocab_size이면 hidden이 아님
                # 이 경우 backbone이 hidden을 반환한다고 가정
                hidden = out
            else:
                hidden = out
        else:
            hidden = self.backbone(input_ids)

        # 마지막 토큰의 hidden state 사용
        last_hidden = hidden[:, -1, :]  # [batch, d_model]
        # 보상 점수
        reward = self.reward_head(last_hidden)  # [batch, 1]
        return reward.squeeze(-1)  # [batch]

    def compute_pairwise_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> torch.Tensor:
        """순위 손실: chosen의 보상이 rejected보다 높아야 함.

        L = -log(sigmoid(r_chosen - r_rejected))
        """
        r_chosen = self.forward(chosen_ids)
        r_rejected = self.forward(rejected_ids)
        # Bradley-Terry 모델
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        return loss
