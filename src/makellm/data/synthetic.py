"""합성 데이터 생성 — 템플릿 기반.

간단한 템플릿을 사용하여 학습용 합성 데이터 생성.
실제 프로덕션에서는 강력한 모델을 사용해 생성하지만,
여기서는 규칙 기반으로 개념을 보여줌.
"""

from __future__ import annotations

import random
from typing import Iterable


class TemplateGenerator:
    """템플릿 기반 합성 데이터 생성기."""

    def __init__(self, templates: list[str], vocab: list[str] | None = None):
        """templates: "{subject} {verb} {object}" 형식의 문자열 리스트.
        vocab: 템플릿 변수에 채울 단어 사전.
        """
        self.templates = templates
        self.vocab = vocab or [
            "the cat", "a dog", "the bird", "a fish",
            "runs", "jumps", "sleeps", "eats",
            "the ball", "the food", "the tree", "the water",
        ]
        self.rng = random.Random(42)

    def _fill_template(self, template: str) -> str:
        """템플릿 변수를 어휘로 채움."""
        result = template
        # {0}, {1}, ... 변수를 무작위 어휘로 치환
        while "{" in result and "}" in result:
            start = result.index("{")
            end = result.index("}", start)
            word = self.rng.choice(self.vocab)
            result = result[:start] + word + result[end + 1:]
        return result

    def generate(self, n: int) -> list[str]:
        """n개의 합성 문장 생성."""
        results = []
        for _ in range(n):
            tpl = self.rng.choice(self.templates)
            results.append(self._fill_template(tpl))
        return results

    def generate_pairs(self, n: int) -> list[tuple[str, str]]:
        """(prompt, response) 쌍 생성 (SFT용)."""
        prompts = self.generate(n)
        # 간단한 규칙: prompt 다음에 올 문장을 response로
        responses = self.generate(n)
        return list(zip(prompts, responses))


# 기본 템플릿 모음
DEFAULT_TEMPLATES = [
    "{0} {1} {2}.",
    "the {0} is {1}ing.",
    "when the {0} {1}s, the {2} also {1}s.",
    "a story about {0} that {1}s {2}.",
]
