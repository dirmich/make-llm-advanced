"""LoRA 파인튜닝 데모: 작은 모델에 LoRA 어댑터 적용 후 학습.

이 스크립트는 2권의 LoRA 모듈을 활용하여:
  1. 작은 GPT 모델 생성 (1권 코드 사용)
  2. LoRA 어댑터 적용
  3. 어댑터만 학습 (원본 가중치는 고정)
  4. 파라미터 절약 효과 측정
  5. 어댑터 병합 (merge) 후 추론

실행:
    python scripts/lora_finetune.py
    python scripts/lora_finetune.py --rank 16 --alpha 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 패키지 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

# 1권 모델 (설치되어 있다고 가정; 없으면 직접 import)
try:
    from makellm_basic.makellm.model import GPT
    from makellm_basic.makellm.utils import ModelConfig, set_seed
    from makellm_basic.makellm.tokenizer import BPETokenizer
    HAS_BASIC = True
except ImportError:
    HAS_BASIC = False
    # 1권 패키지가 없으면 간단한 더미 모델 사용
    class ModelConfig:
        def __init__(self, vocab_size=100, d_model=64, n_heads=4, n_layers=2,
                     d_ff=256, context_length=32, dropout=0.0, pad_token_id=0):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.d_ff = d_ff
            self.context_length = context_length
            self.dropout = dropout
            self.pad_token_id = pad_token_id

    def set_seed(seed):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    class GPT(nn.Module):
        """간단한 더미 GPT (LoRA 데모용)."""
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_ff),
                    nn.GELU(),
                    nn.Linear(config.d_ff, config.d_model),
                ) for _ in range(config.n_layers)
            ])
            self.ln = nn.LayerNorm(config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size)

        def forward(self, x):
            h = self.embedding(x)
            for layer in self.layers:
                h = h + layer(h)
            h = self.ln(h)
            return self.head(h)

        def num_parameters(self):
            return sum(p.numel() for p in self.parameters())


# 2권 LoRA 모듈
from makellm.alignment import apply_lora
from makellm.alignment.lora import count_lora_parameters, LoRALinear


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a slow brown dog sleeps under the warm sun",
    "quick foxes are happy foxes",
    "lazy dogs sleep in the warm sun",
    "the brown fox and the lazy dog",
] * 10


def main():
    parser = argparse.ArgumentParser(description="Make LLM-advanced: LoRA finetuning demo")
    parser.add_argument("--rank", type=int, default=8, help="LoRA 랭크")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA 알파")
    parser.add_argument("--epochs", type=int, default=2, help="학습 에포크")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    args = parser.parse_args()

    print("=" * 60)
    print("  Make LLM-advanced: LoRA Fine-tuning Demo")
    print("=" * 60)

    set_seed(42)

    # 1. 기본 모델 생성
    print("\n[1/5] 기본 모델 구성...")
    config = ModelConfig(
        vocab_size=200, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, context_length=32, dropout=0.0,
    )
    model = GPT(config)
    print(f"    모델: {model.num_parameters():,} 파라미터")

    # 2. LoRA 적용 전 통계
    print("\n[2/5] LoRA 적용 전...")
    stats_before = count_lora_parameters(model)
    print(f"    총 파라미터: {stats_before['total_params']:,}")
    print(f"    학습 가능: {stats_before['trainable_params']:,} (100%)")

    # 3. LoRA 적용
    print(f"\n[3/5] LoRA 적용 (rank={args.rank}, alpha={args.alpha})...")
    # 모든 Linear를 LoRALinear로 교체
    apply_lora(model, rank=args.rank, alpha=args.alpha, dropout=0.0)
    stats_after = count_lora_parameters(model)
    print(f"    총 파라미터: {stats_after['total_params']:,}")
    print(f"    학습 가능: {stats_after['trainable_params']:,} ({stats_after['trainable_pct']:.2f}%)")
    reduction = 100.0 * (stats_before['trainable_params'] - stats_after['trainable_params']) / stats_before['trainable_params']
    print(f"    학습 파라미터 절감: {reduction:.2f}%")

    # 4. LoRA 학습 (간단한 루프)
    print(f"\n[4/5] LoRA 어댑터 학습 ({args.epochs}에포크)...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    # 더미 데이터로 학습
    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for _ in range(20):
            inputs = torch.randint(0, config.vocab_size, (4, config.context_length))
            targets = torch.randint(0, config.vocab_size, (4, config.context_length))
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n
        losses.append(avg)
        print(f"    에포크 {epoch+1}: loss={avg:.4f}")

    # 5. 어댑터 병합 (선택적)
    print("\n[5/5] 어댑터 병합 (추론 최적화)...")
    # 병합 전 추론 테스트
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, config.vocab_size, (1, 8))
        out_before = model(test_input)
        print(f"    병합 전 추론: 입력 {test_input.shape} → 출력 {out_before.shape}")

    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                module.merge_weights()
                merged += 1
    print(f"    {merged}개 레이어 병합 완료")
    print(f"    병합 후 추론 시 LoRA 오버헤드 0")

    # 병합 후 추론 테스트 (결과는 동일해야 함)
    with torch.no_grad():
        out_after = model(test_input)
        diff = (out_before - out_after).abs().max().item()
        print(f"    병합 후 추론: 출력 shape {out_after.shape}")
        print(f"    병합 전후 최대 차이: {diff:.8f} (0에 가까워야 함)")

    print("\n" + "=" * 60)
    print("  LoRA 데모 완료!")
    print("=" * 60)
    print(f"\n요약:")
    print(f"  - 원본 학습 파라미터: {stats_before['trainable_params']:,}")
    print(f"  - LoRA 학습 파라미터: {stats_after['trainable_params']:,} ({stats_after['trainable_pct']:.2f}%)")
    print(f"  - 절감률: {reduction:.2f}%")
    print(f"  - 최종 손실: {losses[-1]:.4f}")
    print(f"\n다음 단계:")
    print(f"  - rank를 4, 16, 32로 바꿔가며 품질/효율 트레이드오프 실험")
    print(f"  - 실제 SFT 데이터셋으로 파인튜닝")
    print(f"  - 어댑터를 저장/로드하여 다른 태스크에 재사용")


if __name__ == "__main__":
    main()
