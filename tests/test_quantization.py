"""양자화 테스트: INT8, INT4, AWQ."""

import pytest
import torch
import torch.nn as nn

from makellm.quantization import (
    quantize_int8, dequantize_int8, INT8Quantizer,
    quantize_int4, dequantize_int4,
    AWQQuantizer,
)


class TestINT8:
    def test_quantize_shape(self):
        w = torch.randn(16, 32)
        qw, scales = quantize_int8(w)
        assert qw.shape == w.shape
        assert scales.shape == (16,)

    def test_quantize_range(self):
        """INT8 범위 -128~127 내에 있어야 함."""
        w = torch.randn(8, 16) * 5
        qw, _ = quantize_int8(w)
        assert qw.min() >= -128
        assert qw.max() <= 127

    def test_dequantize_close_to_original(self):
        """역양자화 결과가 원본과 가까워야 함 (오차 허용)."""
        torch.manual_seed(0)
        w = torch.randn(8, 32)
        qw, scales = quantize_int8(w)
        w_rec = dequantize_int8(qw, scales)
        # 평균 절대 오차가 0.1 이내
        mae = (w - w_rec).abs().mean()
        assert mae < 0.1

    def test_group_quantization(self):
        """그룹별 양자화."""
        w = torch.randn(16, 64)
        qw, scales = quantize_int8(w, group_size=16)
        assert qw.shape == w.shape
        # 그룹 수 = 16 * 64 / 16 = 64
        assert scales.shape == (64,)

    def test_quantize_model(self):
        """모델의 Linear 레이어를 양자화할 수 있어야 함."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        quantizer = INT8Quantizer()
        quantizer.quantize_model(model)
        # 양자화된 레이어가 기록되어야 함
        assert len(quantizer.quantized_layers) >= 2

    def test_memory_savings(self):
        model = nn.Linear(100, 100)
        quantizer = INT8Quantizer()
        savings = quantizer.memory_savings(model)
        assert savings["savings_ratio"] > 0.5


class TestINT4:
    def test_quantize_shape(self):
        w = torch.randn(8, 32)
        qw, scales = quantize_int4(w, group_size=16)
        assert qw.shape == w.shape

    def test_quantize_range(self):
        """INT4 범위 -8~7 내에 있어야 함."""
        w = torch.randn(8, 16) * 3
        qw, _ = quantize_int4(w, group_size=8)
        assert qw.min() >= -8
        assert qw.max() <= 7

    def test_dequantize(self):
        torch.manual_seed(0)
        w = torch.randn(4, 32)
        qw, scales = quantize_int4(w, group_size=16)
        w_rec = dequantize_int4(qw, scales)
        # INT4는 INT8보다 오차가 큼
        mae = (w - w_rec).abs().mean()
        assert mae < 0.5


class TestAWQ:
    def test_awq_quantize_linear(self):
        linear = nn.Linear(32, 32)
        quantizer = AWQQuantizer(n_bits=4, group_size=16)
        info = quantizer.quantize_linear(linear)
        assert info["n_bits"] == 4
        assert info["n_salient"] >= 0

    def test_awq_quantize_model(self):
        model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16))
        quantizer = AWQQuantizer(n_bits=4, group_size=8)
        stats = quantizer.quantize_model(model)
        assert len(stats["layers"]) >= 2

    def test_awq_preserves_output_shape(self):
        linear = nn.Linear(16, 32)
        x = torch.randn(2, 16)
        out_before = linear(x)
        quantizer = AWQQuantizer(n_bits=4, group_size=8)
        quantizer.quantize_linear(linear)
        out_after = linear(x)
        # 출력 shape은 동일
        assert out_before.shape == out_after.shape
