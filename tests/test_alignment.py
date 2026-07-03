"""정렬(Alignment) 테스트: LoRA, SFT, RewardModel, DPO."""

import pytest
import torch
import torch.nn as nn

from makellm.alignment import (
    LoRALinear, LoRAConfig, apply_lora,
    SFTTrainer,
    RewardModel,
    DPOTrainer, dpo_loss,
)
from makellm.alignment.sft import SFTDataset
from makellm.alignment.lora import count_lora_parameters


class TestLoRA:
    def test_lora_linear_shape(self):
        layer = LoRALinear(32, 16, rank=4, alpha=8)
        x = torch.randn(2, 8, 32)
        out = layer(x)
        assert out.shape == (2, 8, 16)

    def test_lora_initial_zero_delta(self):
        """초기 상태에서는 ΔW = 0이어야 함 (B가 0으로 초기화)."""
        layer = LoRALinear(8, 8, rank=2, alpha=4)
        layer.eval()
        x = torch.randn(1, 4, 8)
        out_lora = layer(x)
        out_base = layer.base(x)
        assert torch.allclose(out_lora, out_base, atol=1e-6)

    def test_lora_trainable_params(self):
        layer = LoRALinear(64, 64, rank=8, alpha=16)
        stats = count_lora_parameters(layer)
        # trainable은 A와 B뿐
        # A: 8 * 64 = 512, B: 64 * 8 = 512 → 총 1024
        assert stats["trainable_params"] == 1024
        assert stats["trainable_pct"] < 50  # 50% 미만

    def test_lora_merge_weights(self):
        """가중치 합성 후 어댑터 제거."""
        layer = LoRALinear(8, 8, rank=2, alpha=4)
        # A, B 학습 (랜덤)
        layer.lora_A.data.normal_()
        layer.lora_B.data.normal_()
        layer.merge_weights()
        # 합성 후 어댑터 제거되어야 함
        assert layer.lora_A is None
        assert layer.lora_B is None

    def test_apply_lora_to_model(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
        apply_lora(model, rank=4, alpha=8)
        # LoRALinear로 교체되어야 함
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)


class TestSFT:
    def test_dataset(self):
        samples = [([1, 2, 3], [4, 5, 6])] * 10
        ds = SFTDataset(samples, context_length=8)
        inp, tgt = ds[0]
        # context_length=8이므로 input/target은 7 (shift 1)
        assert inp.shape == (7,)
        assert tgt.shape == (7,)
        # prompt 부분은 -100 (ignore)
        assert (tgt[:2] == -100).all()

    def test_sft_trainer_one_epoch(self):
        """1에포크 학습이 정상 동작해야 함."""
        torch.manual_seed(0)
        # 간단한 모델
        model = nn.Sequential(
            nn.Embedding(50, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 50),
        )
        samples = [([1, 2, 3], [4, 5])] * 8
        ds = SFTDataset(samples, context_length=8)
        trainer = SFTTrainer(model, ds, lr=1e-3, batch_size=4, device="cpu")
        losses = trainer.train(epochs=1)
        assert len(losses) == 1
        assert all(l > 0 for l in losses)


class TestRewardModel:
    def test_reward_shape(self):
        """보상 모델이 스칼라를 반환해야 함."""
        backbone = nn.Sequential(nn.Embedding(50, 16), nn.Linear(16, 16))
        # 마지막 hidden state를 반환하는 래퍼
        class HiddenBackbone(nn.Module):
            def __init__(self, emb, lin):
                super().__init__()
                self.emb = emb
                self.lin = lin
            def forward(self, ids):
                return self.lin(self.emb(ids))

        hb = HiddenBackbone(nn.Embedding(50, 16), nn.Linear(16, 16))
        rm = RewardModel(hb, d_model=16)
        ids = torch.randint(0, 50, (2, 8))
        rewards = rm(ids)
        assert rewards.shape == (2,)
        # 스칼라 보상
        assert rewards.dim() == 1

    def test_pairwise_loss(self):
        """순위 손실이 양수여야 함."""
        torch.manual_seed(0)
        class HiddenBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(50, 16)
                self.lin = nn.Linear(16, 16)
            def forward(self, ids):
                return self.lin(self.emb(ids))

        rm = RewardModel(HiddenBackbone(), d_model=16)
        chosen = torch.randint(0, 50, (2, 8))
        rejected = torch.randint(0, 50, (2, 8))
        loss = rm.compute_pairwise_loss(chosen, rejected)
        assert loss.item() > 0


class TestDPO:
    def test_dpo_loss_zero_when_equal(self):
        """정책과 참조가 동일하면 손실은 -log(0.5) ≈ 0.693."""
        logp = torch.zeros(4)
        loss = dpo_loss(logp, logp, logp, logp, beta=0.1)
        # chosen - rejected = 0 → sigmoid(0) = 0.5 → -log(0.5) = 0.693
        assert abs(loss.item() - 0.6931) < 0.01

    def test_dpo_loss_decreases_with_better_chosen(self):
        """chosen의 logprob가 높을수록 손실이 낮아야 함."""
        ref = torch.zeros(4)
        chosen = torch.tensor([1.0, 1.0, 1.0, 1.0])
        rejected = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        loss = dpo_loss(chosen, rejected, ref, ref, beta=0.1)
        assert loss.item() < 0.6931  # -log(0.5)보다 작아야 함

    def test_dpo_trainer_step(self):
        """DPO 1스텝 학습."""
        torch.manual_seed(0)
        # 간단한 정책 모델
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(50, 16)
                self.lin = nn.Linear(16, 50)
            def forward(self, ids):
                return self.lin(self.emb(ids))

        policy = TinyLM()
        reference = TinyLM()
        # 가중치 동기화
        reference.load_state_dict(policy.state_dict())

        trainer = DPOTrainer(policy, reference, beta=0.1, lr=1e-4)
        chosen = torch.randint(0, 50, (2, 8))
        rejected = torch.randint(0, 50, (2, 8))
        loss = trainer.train_step(chosen, rejected)
        assert loss > 0
