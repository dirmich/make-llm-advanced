"""데이터 품질 필터."""
from __future__ import annotations
import re
from typing import Callable


class LengthFilter:
    """길이 기반 필터."""

    def __init__(self, min_len: int = 10, max_len: int = 100_000):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, text: str) -> bool:
        n = len(text)
        return self.min_len <= n <= self.max_len


class LanguageFilter:
    """언어 감지 기반 필터 (단순화)."""

    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang

    def __call__(self, text: str) -> bool:
        if self.target_lang == "en":
            # 영어: 알파벳 비율이 높은지 확인
            alpha = sum(1 for c in text if c.isascii() and c.isalpha())
            return alpha / max(len(text), 1) > 0.5
        elif self.target_lang == "ko":
            # 한국어: 한글 비율
            hangul = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
            return hangul / max(len(text), 1) > 0.3
        return True


class QualityFilter:
    """품질 기반 필터 — 반복 텍스트, 이상 문자 등 제거."""

    def __init__(self, max_repeat_ratio: float = 0.5, min_alpha_ratio: float = 0.3):
        self.max_repeat_ratio = max_repeat_ratio
        self.min_alpha_ratio = min_alpha_ratio

    def __call__(self, text: str) -> bool:
        # 반복 비율: 같은 단어가 반복되는지
        words = text.split()
        if not words:
            return False
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < (1 - self.max_repeat_ratio):
            return False
        # 알파벳/기호 비율
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / max(len(text), 1) < self.min_alpha_ratio:
            return False
        return True


class Pipeline:
    """필터 파이프라인."""

    def __init__(self, filters: list[Callable[[str], bool]]):
        self.filters = filters

    def __call__(self, text: str) -> bool:
        return all(f(text) for f in self.filters)

    def filter_corpus(self, corpus: list[str]) -> list[str]:
        return [t for t in corpus if self(t)]
