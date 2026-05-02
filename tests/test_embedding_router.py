import pytest
from unittest.mock import MagicMock
from panoramix_core.embedding_router import EmbeddingRouter
from panoramix_core.embedding_cache import EmbeddingCache
from panoramix_core.providers.base import ProviderError
from panoramix_core.demo_mode import set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(False)


def _ok(vec):
    p = MagicMock()
    p.embed.return_value = vec
    p.embed_batch.return_value = [vec]
    p.name = "mock"
    return p


def _fail():
    p = MagicMock()
    p.embed.side_effect = ProviderError("nope")
    p.embed_batch.side_effect = ProviderError("nope")
    p.name = "mock"
    return p


def test_first_wins(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_ok([1.0]), _ok([2.0])], cache=cache)
    assert r.embed("t", "ada") == [1.0]


def test_falls_through_on_error(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_fail(), _ok([2.0])], cache=cache)
    assert r.embed("t", "ada") == [2.0]


def test_all_fail_uses_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="t", vector=[9.0], provider="seed")
    r = EmbeddingRouter(providers=[_fail()], cache=cache)
    assert r.embed("t", "ada") == [9.0]


def test_success_writes_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_ok([1.0])], cache=cache)
    r.embed("t", "ada")
    assert cache.get(model="ada", text="t") == [1.0]


def test_force_cache_skips_providers(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="t", vector=[9.0], provider="seed")
    p = _ok([1.0])
    r = EmbeddingRouter(providers=[p], cache=cache)
    set_force_cache_for_test(True)
    try:
        assert r.embed("t", "ada") == [9.0]
        p.embed.assert_not_called()
    finally:
        set_force_cache_for_test(False)


def test_embed_batch_partial_cache_hit(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="cached", vector=[7.0], provider="seed")
    p = MagicMock()
    p.name = "mock"
    p.embed_batch.return_value = [[8.0]]  # only the missing one
    r = EmbeddingRouter(providers=[p], cache=cache)
    out = r.embed_batch(["cached", "missing"], "ada")
    assert out == [[7.0], [8.0]]
    # provider should be asked only for the missing one
    # (router calls embed_batch with kwargs)
    assert p.embed_batch.call_args.kwargs["texts"] == ["missing"]
    assert p.embed_batch.call_args.kwargs["model"] == "ada"
