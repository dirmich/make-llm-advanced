"""PagedAttention (단순화) — vLLM의 핵심 아이디어.

Kwon et al. (2023) "Efficient Memory Management for Large Language Model Serving
with PagedAttention"

핵심:
  - KV 캐시를 고정 크기 블록(페이지)으로 분할
  - OS의 가상 메모리처럼 논리적 연속성을 물리적 블록으로 매핑
  - 단편화(fragmentation) 감소, 메모리 활용도 향상
  - 여러 요청의 KV를 공통 풀에서 관리

여기서는 개념 이해를 위한 단순화된 구현.
"""

from __future__ import annotations

import torch
from typing import Dict, List


class PagedKVCache:
    """페이지 단위 KV 캐시.

    block_size 단위로 K, V를 저장. 각 시퀀스는 블록 테이블을 통해
    실제 블록 인덱스에 접근.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        block_size: int = 16,
        max_blocks: int = 256,
        device: str = "cpu",
    ):
        self.n_heads = n_heads
        self.d_head = d_head
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        # 블록 풀: [max_blocks, n_heads, block_size, d_head]
        self.k_blocks = torch.zeros(
            max_blocks, n_heads, block_size, d_head, device=device
        )
        self.v_blocks = torch.zeros_like(self.k_blocks)
        self.free_blocks: list[int] = list(range(max_blocks))
        # 시퀀스별 블록 테이블
        self.block_tables: Dict[int, List[int]] = {}
        # 시퀀스별 현재 길이
        self.seq_lens: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int) -> None:
        """새 시퀀스에 첫 블록 할당."""
        if seq_id in self.block_tables:
            return
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")
        block_id = self.free_blocks.pop(0)
        self.block_tables[seq_id] = [block_id]
        self.seq_lens[seq_id] = 0

    def append_kv(
        self,
        seq_id: int,
        new_k: torch.Tensor,  # [n_heads, new_len, d_head]
        new_v: torch.Tensor,
    ) -> None:
        """시퀀스에 새 K, V 추가."""
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)
        new_len = new_k.shape[1]
        cur_len = self.seq_lens[seq_id]
        # 현재 블록에 남은 공간
        cur_block_idx = cur_len // self.block_size
        cur_offset = cur_len % self.block_size
        # 필요 시 새 블록 할당
        while cur_block_idx >= len(self.block_tables[seq_id]):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available")
            self.block_tables[seq_id].append(self.free_blocks.pop(0))
        # 새 K, V를 블록에 복사
        for i in range(new_len):
            block_id = self.block_tables[seq_id][cur_block_idx]
            slot = cur_offset + i
            if slot >= self.block_size:
                cur_block_idx += 1
                while cur_block_idx >= len(self.block_tables[seq_id]):
                    if not self.free_blocks:
                        raise RuntimeError("No free blocks available")
                    self.block_tables[seq_id].append(self.free_blocks.pop(0))
                block_id = self.block_tables[seq_id][cur_block_idx]
                slot = slot % self.block_size
            self.k_blocks[block_id, :, slot] = new_k[:, i]
            self.v_blocks[block_id, :, slot] = new_v[:, i]
        self.seq_lens[seq_id] += new_len

    def get_kv(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """시퀀스의 전체 K, V를 가져옴 (concat)."""
        if seq_id not in self.block_tables:
            return torch.zeros(self.n_heads, 0, self.d_head, device=self.device), \
                   torch.zeros(self.n_heads, 0, self.d_head, device=self.device)
        seq_len = self.seq_lens[seq_id]
        # 모든 블록에서 K, V 수집
        ks, vs = [], []
        for block_id in self.block_tables[seq_id]:
            ks.append(self.k_blocks[block_id])
            vs.append(self.v_blocks[block_id])
        k = torch.cat(ks, dim=1)[:, :seq_len]
        v = torch.cat(vs, dim=1)[:, :seq_len]
        return k, v

    def free_sequence(self, seq_id: int) -> None:
        """시퀀스 해제 — 블록을 free pool로 반환."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            del self.seq_lens[seq_id]

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.max_blocks - len(self.free_blocks)
