import pytest
from unittest.mock import MagicMock
from panoramix_core.providers.openrouter_provider import (
    OpenRouterProvider, MODEL_NOT_SUPPORTED,
)
from panoramix_core.providers.base import ProviderError


def _client_returning(text):
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=text))
    ]
    return c


def test_translates_model_via_map(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text("gpt-4.1: openai/gpt-4-turbo\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    p.invoke("hi", "gpt-4.1", 0.2)
    args, kwargs = p.client.chat.completions.create.call_args
    assert kwargs["model"] == "openai/gpt-4-turbo"


def test_passthrough_when_no_mapping(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text("gpt-4.1: openai/gpt-4-turbo\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    p.invoke("hi", "claude-3-5-sonnet", 0.2)
    assert p.client.chat.completions.create.call_args.kwargs["model"] == "claude-3-5-sonnet"


def test_not_supported_raises(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text(f"text-embedding-ada-002: {MODEL_NOT_SUPPORTED}\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    with pytest.raises(ProviderError):
        p.invoke("hi", "text-embedding-ada-002", 0.0)
