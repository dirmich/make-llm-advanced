"""MinHash 기반 중복 제거.

Broder (1997) "On the Resemblance and Containment of Documents"

핵심:
  - 문서의 n-gram 집합의 Jaccard 유사도를 근사
  - 해시 함수 k개를 사용하여 각 문서의 signature 생성
  - signature 비교로 빠르게 중복 탐지
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def _hash(s: str) -> int:
    """SHA-1의 앞 8바이트를 정수로."""
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _shingles(text: str, k: int = 5) -> set[str]:
    """텍스트를 k-shingle 집합으로 변환."""
    # 단어 단위로 정규화
    text = re.sub(r"\s+", " ", text.lower().strip())
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class MinHashDedup:
    """MinHash 기반 문서 중복 제거."""

    def __init__(self, n_hashes: int = 16, k: int = 5, threshold: float = 0.8):
        self.n_hashes = n_hashes
        self.k = k
        self.threshold = threshold
        # 각 해시 함수의 seed
        self.seeds = list(range(1, n_hashes + 1))

    def compute_signature(self, text: str) -> tuple[int, ...]:
        """문서의 MinHash signature 계산."""
        shingles = _shingles(text, self.k)
        if not shingles:
            return tuple([0] * self.n_hashes)
        sig = []
        for seed in self.seeds:
            # 각 seed마다 shingle의 해시 최솟값
            min_h = min(
                _hash(f"{seed}:{s}") for s in shingles
            )
            sig.append(min_h)
        return tuple(sig)

    def jaccard_estimate(self, sig1: tuple, sig2: tuple) -> float:
        """두 signature의 Jaccard 유사도 추정."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def dedup(self, corpus: list[str]) -> list[str]:
        """코퍼스에서 중복 문서 제거.

        Returns:
            중복이 제거된 문서 리스트
        """
        unique: list[str] = []
        signatures: list[tuple] = []
        for doc in corpus:
            sig = self.compute_signature(doc)
            is_dup = False
            for existing_sig in signatures:
                if self.jaccard_estimate(sig, existing_sig) >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(doc)
                signatures.append(sig)
        return unique

    def find_duplicates(self, corpus: list[str]) -> list[tuple[int, int, float]]:
        """중복 문서 쌍과 유사도 반환."""
        sigs = [self.compute_signature(d) for d in corpus]
        dups = []
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                sim = self.jaccard_estimate(sigs[i], sigs[j])
                if sim >= self.threshold:
                    dups.append((i, j, sim))
        return dups
