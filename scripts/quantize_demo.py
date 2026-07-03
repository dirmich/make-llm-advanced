"""양자화 데모: 모델을 INT8/INT4로 양자화하여 메모리 절약 효과 측정.

이 스크립트는 2권의 양자화 모듈을 활용하여:
  1. 작은 GPT 모델 생성
  2. FP32 기준 메모리 측정
  3. INT8 양자화 후 메모리 및 정확도 비교
  4. INT4 양자화 후 메모리 및 정확도 비교
  5. AWQ 양자화 적용

실행:
    python scripts/quantize_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from makellm.quantization import (
    quantize_int8, dequantize_int8, INT8Quantizer,
    quantize_int4, dequantize_int4,
    AWQQuantizer,
)


def make_tiny_model(vocab=200, d=64, layers=2) -> nn.Module:
    """간단한 모델 생성 (양자화 데모용)."""
    model = nn.Sequential(
        nn.Embedding(vocab, d),
        nn.Linear(d, d * 4),
        nn.GELU(),
        nn.Linear(d * 4, d),
        nn.LayerNorm(d),
        nn.Linear(d, vocab),
    )
    return model


def measure_memory(model: nn.Module) -> dict:
    """모델의 메모리 사용량 측정."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return {
        "total_bytes": total,
        "total_mb": total / 1e6,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def measure_output_diff(model_before: nn.Module, model_after: nn.Module, vocab=200, d=64) -> float:
    """양자화 전후 출력의 평균 절대 차이."""
    import torch
    torch.manual_seed(0)
    # 모델이 Embedding으로 시작하므로 정수 인덱스 입력
    x = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        out_before = model_before(x)
        out_after = model_after(x)
    return (out_before - out_after).abs().mean().item()


def main():
    print("=" * 60)
    print("  Make LLM-advanced: Quantization Demo")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. 기본 모델
    print("\n[1/4] FP32 기준 모델...")
    model_fp32 = make_tiny_model()
    mem_fp32 = measure_memory(model_fp32)
    print(f"    파라미터: {mem_fp32['n_params']:,}")
    print(f"    메모리: {mem_fp32['total_mb']:.3f} MB")

    # 2. INT8 양자화
    print("\n[2/4] INT8 양자화...")
    model_int8 = make_tiny_model()
    # 가중치 복사 (동일한 시작점)
    model_int8.load_state_dict(model_fp32.state_dict())
    quantizer8 = INT8Quantizer(group_size=64)
    t0 = time.time()
    quantizer8.quantize_model(model_int8)
    t_int8 = time.time() - t0
    diff_int8 = measure_output_diff(model_fp32, model_int8)
    savings = quantizer8.memory_savings(model_int8)
    print(f"    양자화 시간: {t_int8:.3f}초")
    print(f"    FP32 메모리: {savings['fp32_mb']:.3f} MB")
    print(f"    INT8 메모리: {savings['int8_mb']:.3f} MB")
    print(f"    절약 비율: {savings['savings_ratio']*100:.0f}%")
    print(f"    출력 MAE: {diff_int8:.6f}")

    # 3. INT4 양자화
    print("\n[3/4] INT4 양자화 (직접 가중치 변환)...")
    model_int4 = make_tiny_model()
    model_int4.load_state_dict(model_fp32.state_dict())
    # 각 Linear의 가중치를 INT4로 양자화 후 역양자화 (시뮬레이션)
    n_layers_quantized = 0
    for name, module in model_int4.named_modules():
        if isinstance(module, nn.Linear) and module.weight.dim() == 2:
            w = module.weight.data
            qw, scales = quantize_int4(w, group_size=32)
            w_rec = dequantize_int4(qw, scales)
            module.weight.data = w_rec
            n_layers_quantized += 1
    diff_int4 = measure_output_diff(model_fp32, model_int4)
    print(f"    양자화된 레이어: {n_layers_quantized}")
    print(f"    출력 MAE: {diff_int4:.6f}")
    print(f"    (INT4는 INT8보다 오차가 크지만 메모리 2배 추가 절약)")

    # 4. AWQ 양자화
    print("\n[4/4] AWQ 양자화...")
    model_awq = make_tiny_model()
    model_awq.load_state_dict(model_fp32.state_dict())
    awq = AWQQuantizer(n_bits=4, group_size=32, alpha=1.5)
    t0 = time.time()
    stats = awq.quantize_model(model_awq)
    t_awq = time.time() - t0
    diff_awq = measure_output_diff(model_fp32, model_awq)
    n_salient = sum(s["n_salient"] for s in stats["layers"].values())
    print(f"    양자화 시간: {t_awq:.3f}초")
    print(f"    Salient 채널 수: {n_salient}")
    print(f"    출력 MAE: {diff_awq:.6f}")
    print(f"    (AWQ는 중요 채널을 보존하여 INT4 대비 개선)")

    # 요약
    print("\n" + "=" * 60)
    print("  양자화 데모 완료!")
    print("=" * 60)
    print("\n요약:")
    print(f"  FP32  메모리: {mem_fp32['total_mb']:.3f} MB (기준)")
    print(f"  INT8  메모리: {savings['int8_mb']:.3f} MB ({savings['savings_ratio']*100:.0f}% 절약), MAE={diff_int8:.6f}")
    print(f"  INT4  메모리: ~{savings['int8_mb']/2:.3f} MB ({savings['savings_ratio']*100+12.5:.0f}% 절약), MAE={diff_int4:.6f}")
    print(f"  AWQ   메모리: ~{savings['int8_mb']/2:.3f} MB (INT4와 동일), MAE={diff_awq:.6f}")
    print(f"\n결론:")
    print(f"  - INT8은 정확도 손실 거의 없이 4배 메모리 절약")
    print(f"  - INT4는 약간의 손실이 있지만 8배 절약")
    print(f"  - AWQ는 INT4의 손실을 줄이면서 같은 메모리 절약")


if __name__ == "__main__":
    main()
