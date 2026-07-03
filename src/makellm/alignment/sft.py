"""SFT (Supervised Fine-Tuning) 트레이너.

기초편 Trainer를 확장하여 지도학습 파인튜닝을 지원.
명령-응답 쌍을 학습하여 모델이 프롬프트에 적절히 반응하도록 함.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable


class SFTDataset(Dataset):
    """SFT용 데이터셋.

    각 샘플은 (prompt_ids, response_ids) 형태.
    학습 시 prompt 부분은 loss에서 무시하고 response 부분만 계산.
    """

    def __init__(self, samples: list[tuple[list[int], list[int]]], context_length: int = 512):
        """samples: [(prompt_ids, response_ids), ...]"""
        self.samples = samples
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids, response_ids = self.samples[idx]
        # prompt + response 결합
        full_ids = prompt_ids + response_ids
        # context_length로 자르기
        full_ids = full_ids[: self.context_length]
        # 패딩 (간단화)
        pad_id = 0
        while len(full_ids) < self.context_length:
            full_ids.append(pad_id)
        # input: 전체, target: shift 1
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_ids[1:], dtype=torch.long)
        # loss 마스크: prompt 부분은 -100 (ignore_index)
        prompt_len = min(len(prompt_ids) - 1, self.context_length - 1)
        loss_mask = torch.full_like(target_ids, -100)
        loss_mask[prompt_len:] = target_ids[prompt_len:]
        return input_ids, loss_mask


class SFTTrainer:
    """SFT 학습 루프."""

    def __init__(
        self,
        model: nn.Module,
        dataset: SFTDataset,
        lr: float = 1e-5,
        batch_size: int = 4,
        device: str = "cpu",
    ):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs: int = 1, log_fn: Callable[[int, float], None] | None = None) -> list[float]:
        """학습 루프. 에포크별 손실 반환."""
        losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n = 0
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs)
                # cross-entropy (ignore_index=-100)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / max(n, 1)
            losses.append(avg)
            if log_fn:
                log_fn(epoch, avg)
        return losses
