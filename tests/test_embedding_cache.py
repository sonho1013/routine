import pytest
from panoramix_core.embedding_cache import EmbeddingCache


def test_set_get_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    vec = [0.1, 0.2, 0.3]
    cache.set(model="ada-002", text="hello", vector=vec, provider="openai")
    assert cache.get(model="ada-002", text="hello") == vec


def test_get_miss_returns_none(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    assert cache.get(model="ada-002", text="hello") is None


def test_get_or_raise_miss(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    with pytest.raises(KeyError):
        cache.get_or_raise(model="ada-002", text="hello")


def test_read_only_blocks_writes(tmp_path):
    p = tmp_path / "e.json"
    EmbeddingCache(p).set(model="m", text="t", vector=[1.0], provider="openai")
    ro = EmbeddingCache(p, read_only=True)
    ro.set(model="m", text="u", vector=[2.0], provider="openai")
    assert EmbeddingCache(p).get(model="m", text="u") is None
