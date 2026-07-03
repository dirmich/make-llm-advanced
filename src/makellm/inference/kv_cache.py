"""KV Cache — 어텐션 추론 가속.

Kwon et al. (2023) "Efficient Memory Management for Large Language Model Serving
with PagedAttention" (vLLM)

핵심:
  - 자기회귀 생성에서 이전 토큰의 K, V를 재사용
  - 매 스텝 새 토큰의 K, V만 계산하여 캐시에 추가
  - 복잡도 O(seq^2) → O(seq) (스텝당)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """단순 KV 캐시.

    각 어텐션 헤드에 대해 K, V 텐서를 보관.
    새 토큰이 오면 K, V를 계산하여 append.
    """

    def __init__(self, n_heads: int, d_head: int, max_seq_len: int = 2048, device: str = "cpu"):
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.device = device
        # 캐시 텐서 (처음엔 빈 것)
        self.k_cache: torch.Tensor | None = None  # [n_heads, 0, d_head]
        self.v_cache: torch.Tensor | None = None
        self.seq_len = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """새 K, V를 캐시에 추가하고 전체 반환.

        Args:
            new_k: [n_heads, new_len, d_head]
            new_v: [n_heads, new_len, d_head]
        Returns:
            (full_k, full_v) — [n_heads, total_len, d_head]
        """
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=1)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=1)
        self.seq_len = self.k_cache.shape[1]
        return self.k_cache, self.v_cache

    def reset(self) -> None:
        """캐시 초기화."""
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    @property
    def total_tokens(self) -> int:
        return self.seq_len


class CachedAttention(nn.Module):
    """KV 캐시를 사용하는 어텐션 (추론용).

    학습 시에는 일반 멀티헤드 어텐션과 동일.
    추론 시에는 캐시를 사용하여 이전 토큰의 재계산을 피함.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.kv_cache: KVCache | None = None

    def init_cache(self, max_seq_len: int = 2048, device: str = "cpu") -> None:
        """추론용 KV 캐시 초기화."""
        self.kv_cache = KVCache(self.n_heads, self.d_head, max_seq_len, device)

    def reset_cache(self) -> None:
        if self.kv_cache is not None:
            self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """x: [batch, seq, d_model] → [batch, seq, d_model]

        use_cache=True이면 KV 캐시 사용 (추론 모드).
        """
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

        if use_cache and self.kv_cache is not None:
            # 캐시 업데이트 (batch 차원 무시, 단일 시퀀스 가정)
            k_full, v_full = self.kv_cache.update(k[0], v[0])  # [n_heads, total, d_head]
            k_full = k_full.unsqueeze(0)  # [1, n_heads, total, d_head]
            v_full = v_full.unsqueeze(0)
            # q는 현재 스텝의 것만 사용
            # 어텐션
            out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        else:
            # 일반 어텐션 (학습용)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out)
