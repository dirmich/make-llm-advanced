"""추론 최적화 & 데이터 파이프라인 테스트."""

import pytest
import torch

from makellm.inference.kv_cache import KVCache, CachedAttention
from makellm.inference.paged import PagedKVCache
from makellm.data import (
    QualityFilter, LengthFilter, LanguageFilter,
    MinHashDedup,
    TemplateGenerator,
)


class TestKVCache:
    def test_cache_update(self):
        cache = KVCache(n_heads=4, d_head=8, max_seq_len=64)
        k = torch.randn(4, 2, 8)
        v = torch.randn(4, 2, 8)
        full_k, full_v = cache.update(k, v)
        assert full_k.shape == (4, 2, 8)
        assert cache.total_tokens == 2

        # 추가
        k2 = torch.randn(4, 1, 8)
        v2 = torch.randn(4, 1, 8)
        full_k2, _ = cache.update(k2, v2)
        assert full_k2.shape == (4, 3, 8)
        assert cache.total_tokens == 3

    def test_cache_reset(self):
        cache = KVCache(n_heads=2, d_head=4)
        k = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 4)
        cache.update(k, v)
        cache.reset()
        assert cache.total_tokens == 0
        assert cache.k_cache is None


class TestCachedAttention:
    def test_attention_shape(self):
        attn = CachedAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_cache_mode(self):
        attn = CachedAttention(d_model=16, n_heads=2)
        attn.init_cache(max_seq_len=64)
        x = torch.randn(1, 4, 16)
        out = attn(x, use_cache=True)
        assert out.shape == x.shape
        assert attn.kv_cache.total_tokens == 4


class TestPagedKVCache:
    def test_allocate(self):
        cache = PagedKVCache(n_heads=4, d_head=8, block_size=4, max_blocks=8)
        cache.allocate_sequence(seq_id=0)
        assert cache.num_used_blocks == 1
        assert cache.num_free_blocks == 7

    def test_append_and_get(self):
        cache = PagedKVCache(n_heads=4, d_head=8, block_size=4, max_blocks=8)
        cache.allocate_sequence(seq_id=0)
        k = torch.randn(4, 3, 8)
        v = torch.randn(4, 3, 8)
        cache.append_kv(0, k, v)
        k_out, v_out = cache.get_kv(0)
        assert k_out.shape == (4, 3, 8)

    def test_free_sequence(self):
        cache = PagedKVCache(n_heads=2, d_head=4, block_size=4, max_blocks=4)
        cache.allocate_sequence(seq_id=0)
        assert cache.num_used_blocks == 1
        cache.free_sequence(seq_id=0)
        assert cache.num_used_blocks == 0
        assert cache.num_free_blocks == 4

    def test_multiple_sequences(self):
        cache = PagedKVCache(n_heads=2, d_head=4, block_size=4, max_blocks=8)
        cache.allocate_sequence(seq_id=0)
        cache.allocate_sequence(seq_id=1)
        assert cache.num_used_blocks == 2
        k = torch.randn(2, 2, 4)
        v = torch.randn(2, 2, 4)
        cache.append_kv(0, k, v)
        cache.append_kv(1, k, v)
        assert cache.seq_lens[0] == 2
        assert cache.seq_lens[1] == 2


class TestFilters:
    def test_length_filter(self):
        f = LengthFilter(min_len=5, max_len=20)
        assert f("hello world")  # 11 chars
        assert not f("hi")  # too short
        assert not f("a" * 30)  # too long

    def test_language_filter_english(self):
        f = LanguageFilter(target_lang="en")
        assert f("the quick brown fox jumps over the lazy dog")
        assert not f("안녕하세요 만나서 반갑습니다")

    def test_language_filter_korean(self):
        f = LanguageFilter(target_lang="ko")
        assert f("안녕하세요 만나서 반갑습니다 한국어를 연습합니다")
        assert not f("the quick brown fox jumps")

    def test_quality_filter(self):
        f = QualityFilter()
        assert f("the cat sat on the mat and slept all day")
        # 반복 텍스트는 거르기
        assert not f("the the the the the the the the")

    def test_pipeline(self):
        from makellm.data.filter import Pipeline
        p = Pipeline([LengthFilter(5, 100), QualityFilter()])
        assert p("hello world this is a test")
        assert not p("aaa")


class TestMinHash:
    def test_signature_shape(self):
        m = MinHashDedup(n_hashes=16, k=3)
        sig = m.compute_signature("the quick brown fox")
        assert len(sig) == 16

    def test_identical_documents(self):
        m = MinHashDedup(n_hashes=32, k=3)
        text = "the quick brown fox jumps over the lazy dog"
        sig1 = m.compute_signature(text)
        sig2 = m.compute_signature(text)
        assert m.jaccard_estimate(sig1, sig2) == 1.0

    def test_similar_documents(self):
        m = MinHashDedup(n_hashes=32, k=3, threshold=0.5)
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy cat"
        sig1 = m.compute_signature(text1)
        sig2 = m.compute_signature(text2)
        sim = m.jaccard_estimate(sig1, sig2)
        # 비슷한 문서는 높은 유사도
        assert sim > 0.3

    def test_dedup_removes_duplicates(self):
        m = MinHashDedup(n_hashes=16, k=2, threshold=0.7)
        corpus = [
            "the cat sat on the mat",
            "the cat sat on the mat",  # 중복
            "dogs are great animals",
            "the cat sat on the mat",  # 중복
        ]
        unique = m.dedup(corpus)
        assert len(unique) < len(corpus)
        assert len(unique) >= 2


class TestSyntheticData:
    def test_template_generator(self):
        gen = TemplateGenerator(
            templates=["{0} {1} {2}.", "the {0} is {1}ing."],
            vocab=["cat", "dog", "runs", "jumps", "fast", "slow"],
        )
        data = gen.generate(10)
        assert len(data) == 10
        assert all(isinstance(s, str) for s in data)

    def test_generate_pairs(self):
        gen = TemplateGenerator(templates=["{0} {1}."])
        pairs = gen.generate_pairs(5)
        assert len(pairs) == 5
        assert all(len(p) == 2 for p in pairs)
