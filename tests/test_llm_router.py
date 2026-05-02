import pytest
from unittest.mock import MagicMock
from panoramix_core.llm_router import LLMRouter
from panoramix_core.llm_cache import LLMCache
from panoramix_core.providers.base import (
    ProviderError, ProviderTimeout, ProviderAuthError,
)
from panoramix_core.demo_mode import set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(False)


def _ok(text="ok"):
    p = MagicMock()
    p.invoke.return_value = text
    p.name = "mock"
    return p


def _fail(exc=ProviderError("nope")):
    p = MagicMock()
    p.invoke.side_effect = exc
    p.name = "mock"
    return p


def test_first_provider_wins(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _ok("A"), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "A"
    b.invoke.assert_not_called()


def test_falls_through_to_second_on_error(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_falls_through_on_timeout(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(ProviderTimeout("slow")), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_falls_through_on_auth_error(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(ProviderAuthError("401")), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_all_providers_fail_uses_cache(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="hi", temperature=0.0,
              response="seeded", provider="seed")
    r = LLMRouter(providers=[_fail(), _fail()], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "seeded"


def test_all_fail_no_cache_raises(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    r = LLMRouter(providers=[_fail(), _fail()], cache=cache)
    with pytest.raises(KeyError):
        r.invoke("hi", "m", 0.0)


def test_success_writes_cache(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a = _ok("A"); a.name = "openai"
    r = LLMRouter(providers=[a], cache=cache)
    r.invoke("hi", "m", 0.0)
    assert cache.get(model="m", prompt="hi", temperature=0.0) == "A"


def test_force_cache_skips_providers(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="hi", temperature=0.0,
              response="cached", provider="seed")
    a = _ok("FRESH")
    r = LLMRouter(providers=[a], cache=cache)
    set_force_cache_for_test(True)
    try:
        assert r.invoke("hi", "m", 0.0) == "cached"
        a.invoke.assert_not_called()
    finally:
        set_force_cache_for_test(False)


def test_force_cache_miss_raises_without_calling_providers(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a = _ok("FRESH")
    r = LLMRouter(providers=[a], cache=cache)
    set_force_cache_for_test(True)
    try:
        with pytest.raises(KeyError):
            r.invoke("hi", "m", 0.0)
        a.invoke.assert_not_called()
    finally:
        set_force_cache_for_test(False)
