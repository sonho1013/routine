import json
import pytest
from pathlib import Path
from panoramix_core.llm_cache import LLMCache


def test_set_and_get_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="gpt-4.1", prompt="hi", temperature=0.2,
              response="hello", provider="openai")
    assert cache.get(model="gpt-4.1", prompt="hi", temperature=0.2) == "hello"


def test_get_miss_returns_none(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    assert cache.get(model="gpt-4.1", prompt="hi", temperature=0.2) is None


def test_persists_to_disk(tmp_path):
    p = tmp_path / "c.json"
    LLMCache(p).set(model="m", prompt="p", temperature=0.0,
                    response="r", provider="openai")
    raw = json.loads(p.read_text())
    assert len(raw) == 1
    entry = next(iter(raw.values()))
    assert entry["response"] == "r"
    assert entry["provider"] == "openai"
    assert "cached_at" in entry


def test_get_or_raise_hits(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="p", temperature=0.0,
              response="r", provider="openai")
    assert cache.get_or_raise(model="m", prompt="p", temperature=0.0) == "r"


def test_get_or_raise_miss_raises(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    with pytest.raises(KeyError):
        cache.get_or_raise(model="m", prompt="p", temperature=0.0)


def test_read_only_mode_blocks_writes(tmp_path):
    p = tmp_path / "c.json"
    LLMCache(p).set(model="m", prompt="p", temperature=0.0,
                    response="r", provider="openai")
    ro = LLMCache(p, read_only=True)
    ro.set(model="m", prompt="p2", temperature=0.0,
           response="r2", provider="openai")
    raw = json.loads(p.read_text())
    assert len(raw) == 1


def test_key_distinguishes_temperature(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="p", temperature=0.0,
              response="A", provider="openai")
    cache.set(model="m", prompt="p", temperature=0.7,
              response="B", provider="openai")
    assert cache.get(model="m", prompt="p", temperature=0.0) == "A"
    assert cache.get(model="m", prompt="p", temperature=0.7) == "B"
