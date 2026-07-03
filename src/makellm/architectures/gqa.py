"""Grouped Query Attention (GQA) & Multi-Query Attention (MQA).

Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints"

핵심:
  - Q 헤드 수는 그대로, K/V 헤드 수를 줄여 메모리 절약
  - n_kv_heads = n_heads: 표준 MHA
  - n_kv_heads = 1: MQA (가장 극단적)
  - 1 < n_kv_heads < n_heads: GQA
  - KV cache 메모리가 선형적으로 감소하여 추론 속도 향상
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionEmbedding


class GroupedQueryAttention(nn.Module):
    """GQA — KV 헤드 수를 줄인 멀티헤드 어텐션.

    n_kv_heads개의 KV 헤드가 있고, 각 KV 헤드는 n_heads // n_kv_heads개의
    Q 헤드를 담당. 예: n_heads=8, n_kv_heads=2 → 각 KV 헤드가 4개 Q 헤드 담당.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads({n_heads}) must be divisible by n_kv_heads({self.n_kv_heads})"
        )
        self.n_rep = self.n_heads // self.n_kv_heads  # KV 반복 횟수

        # Q, K, V 프로젝션 (K/V는 더 작음)
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=bias)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

        self.dropout_p = dropout
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.d_head, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [batch, seq, d_model] → [batch, seq, d_model]"""
        batch, seq, _ = x.shape

        # 프로젝션
        q = self.wq(x).view(batch, seq, self.n_heads, self.d_head)
        k = self.wk(x).view(batch, seq, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(batch, seq, self.n_kv_heads, self.d_head)

        # [batch, n_heads, seq, d_head]로 전치
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE 적용
        if self.use_rope:
            q, k = self.rope(q, k)

        # KV 반복 (GQA): [batch, n_kv_heads, seq, d_head] → [batch, n_heads, seq, d_head]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 어텐션
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=mask is None,
        )

        # [batch, seq, n_heads * d_head]로 합치기
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out)
