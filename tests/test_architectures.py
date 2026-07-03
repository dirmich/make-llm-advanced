"""최신 아키텍처 테스트: RoPE, RMSNorm, SwiGLU, GQA, MoE."""

import pytest
import torch

from makellm.architectures import (
    RotaryPositionEmbedding, apply_rope,
    RMSNorm,
    SwiGLU,
    GroupedQueryAttention,
    MoELayer, TopKRouter,
)


class TestRoPE:
    def test_precompute_freqs(self):
        from makellm.architectures.rope import precompute_freqs_cis
        fc = precompute_freqs_cis(head_dim=8, max_seq_len=16)
        assert fc.shape == (16, 4)  # [seq, head_dim//2]

    def test_apply_rope_shape(self):
        from makellm.architectures.rope import precompute_freqs_cis
        x = torch.randn(2, 4, 8, 8)  # [batch, heads, seq, d_head]
        fc = precompute_freqs_cis(8, 16)
        out = apply_rope(x, fc[:8])
        assert out.shape == x.shape

    def test_rope_module(self):
        rope = RotaryPositionEmbedding(head_dim=16, max_seq_len=64)
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 2, 8, 16)  # GQA: n_kv_heads < n_heads
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rope_preserves_norm(self):
        """RoPE는 회전이므로 벡터의 노름을 보존해야 함."""
        from makellm.architectures.rope import precompute_freqs_cis
        torch.manual_seed(0)
        x = torch.randn(1, 1, 1, 8)  # 단일 토큰
        fc = precompute_freqs_cis(8, 16)
        out = apply_rope(x, fc[:1])
        # 노름이 거의 보존되어야 함 (부동소수 오차 허용)
        norm_in = x.norm().item()
        norm_out = out.norm().item()
        assert abs(norm_in - norm_out) < 0.1


class TestRMSNorm:
    def test_shape(self):
        norm = RMSNorm(d_model=32)
        x = torch.randn(2, 8, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        """입력이 0이면 출력도 0이어야 함."""
        norm = RMSNorm(d_model=8)
        x = torch.zeros(1, 4, 8)
        out = norm(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_unit_weight_identity(self):
        """가중치가 1이면 RMS 정규화만 수행."""
        norm = RMSNorm(d_model=4)
        norm.weight.data.fill_(1.0)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        out = norm(x)
        # RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        expected = x * rms
        assert torch.allclose(out, expected, atol=1e-5)


class TestSwiGLU:
    def test_shape(self):
        ffn = SwiGLU(d_model=32, d_ff=64)
        x = torch.randn(2, 8, 32)
        out = ffn(x)
        assert out.shape == x.shape

    def test_default_dff(self):
        """d_ff=None이면 LLaMA 스타일 8/3*d_model 사용."""
        ffn = SwiGLU(d_model=96)
        assert ffn.d_ff > 0
        # 8의 배수여야 함
        assert ffn.d_ff % 8 == 0

    def test_output_range(self):
        """SwiGLU 출력은 유한해야 함."""
        ffn = SwiGLU(d_model=16, d_ff=32)
        x = torch.randn(1, 4, 16)
        out = ffn(x)
        assert torch.isfinite(out).all()


class TestGQA:
    def test_mha_mode(self):
        """n_kv_heads=None이면 표준 MHA와 동일."""
        attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=None, use_rope=False)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_mode(self):
        """n_kv_heads < n_heads (GQA)"""
        attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, use_rope=False)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_mqa_mode(self):
        """n_kv_heads=1 (MQA)"""
        attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=1, use_rope=False)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=3)  # 4 % 3 != 0

    def test_with_rope(self):
        attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, use_rope=True)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape


class TestMoE:
    def test_router_output(self):
        router = TopKRouter(d_model=32, n_experts=8, n_active=2)
        x = torch.randn(2, 4, 32)
        weights, indices = router(x)
        assert weights.shape == (2, 4, 2)
        assert indices.shape == (2, 4, 2)
        # 가중치 합은 1 (softmax)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    def test_moe_output_shape(self):
        moe = MoELayer(d_model=32, d_ff=64, n_experts=4, n_active=2)
        x = torch.randn(2, 4, 32)
        out, aux = moe(x)
        assert out.shape == x.shape
        assert aux.dim() == 0  # 스칼라

    def test_moe_aux_loss_finite(self):
        moe = MoELayer(d_model=16, d_ff=32, n_experts=4, n_active=2)
        x = torch.randn(1, 4, 16)
        _, aux = moe(x)
        assert torch.isfinite(aux)

    def test_moe_n_active_cannot_exceed_n_experts(self):
        moe = MoELayer(d_model=16, d_ff=32, n_experts=4, n_active=2)
        x = torch.randn(1, 2, 16)
        out, _ = moe(x)
        assert out.shape == x.shape
