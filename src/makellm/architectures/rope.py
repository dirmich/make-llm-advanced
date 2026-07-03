"""Rotary Position Embedding (RoPE).

Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"

핵심 아이디어:
  위치 정보를 쿼리/키에 복소수 회전으로 주입.
  - d_head 차원을 2개씩 짝지어 회전 적용
  - 상대적 위치만큼 회전 각도가 비례
  - 외삽(extrapolation)이 자연스럽고 긴 시퀀스에서 우수
"""

from __future__ import annotations

import torch
import torch.nn as nn


def precompute_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """회전 주파수 미리 계산.

    Returns:
        freqs_cis: [max_seq_len, head_dim//2] 복소수 (cos + i*sin)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # 각 차원 쌍에 대한 주파수: theta_i = base^(-2i/d) for i in [0, d/2)
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    # [max_seq_len, half]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    # 복소수 표현: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """RoPE 적용.

    Args:
        x: [batch, n_heads, seq, head_dim]
        freqs_cis: [seq, head_dim//2] 복소수
    Returns:
        [batch, n_heads, seq, head_dim]
    """
    batch, n_heads, seq, head_dim = x.shape
    half = head_dim // 2
    # x를 복소수로 재해석: 마지막 차원을 (half, 2) → 복소수 half개로
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], half, 2))
    # freqs_cis를 브로드캐스트: [seq, half] → [1, 1, seq, half]
    freqs_cis = freqs_cis[:seq].view(1, 1, seq, half)
    # 회전: 복소수 곱
    x_rotated = torch.view_as_real(x_complex * freqs_cis)
    # [batch, n_heads, seq, half, 2] → [batch, n_heads, seq, head_dim]
    x_out = x_rotated.flatten(-2)
    return x_out.type_as(x)


class RotaryPositionEmbedding(nn.Module):
    """RoPE 모듈 — precompute된 freqs_cis를 버퍼로 보관."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, base)
        # 복소수 텐서는 buffer로 저장 (실수/허수 두 텐서로 분해하거나 그대로 보관)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k에 RoPE 적용.

        Args:
            q: [batch, n_heads, seq, head_dim]
            k: [batch, n_kv_heads, seq, head_dim]
        Returns:
            (q_rotated, k_rotated) — k의 헤드 수가 다를 수 있음 (GQA)
        """
        seq = q.shape[2]
        freqs = self.freqs_cis[:seq]
        q_out = apply_rope(q, freqs)
        k_out = apply_rope(k, freqs)
        return q_out, k_out
