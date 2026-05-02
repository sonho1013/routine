import httpx
import pytest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from panoramix_core.providers.openai_provider import OpenAIProvider
from panoramix_core.providers.base import (
    ProviderError, ProviderTimeout, ProviderAuthError,
)


def _mk_client_returning(text):
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=text))
    ]
    client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2])
    ]
    return client


def test_invoke_returns_text():
    p = OpenAIProvider(api_key="x")
    p.client = _mk_client_returning("hi there")
    assert p.invoke("hello", "gpt-4.1", 0.2) == "hi there"


def test_invoke_timeout_maps_to_provider_timeout():
    p = OpenAIProvider(api_key="x")
    p.client = MagicMock()
    p.client.chat.completions.create.side_effect = APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    )
    with pytest.raises(ProviderTimeout):
        p.invoke("hi", "m", 0.0)


def test_invoke_auth_maps_to_auth_error():
    p = OpenAIProvider(api_key="x")
    p.client = MagicMock()
    p.client.chat.completions.create.side_effect = AuthenticationError(
        message="bad key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
        ),
        body=None,
    )
    with pytest.raises(ProviderAuthError):
        p.invoke("hi", "m", 0.0)


def test_invoke_rate_limit_maps_to_provider_error():
    p = OpenAIProvider(api_key="x")
    p.client = MagicMock()
    p.client.chat.completions.create.side_effect = RateLimitError(
        message="rate",
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
        ),
        body=None,
    )
    with pytest.raises(ProviderError):
        p.invoke("hi", "m", 0.0)


def test_invoke_connection_error_maps_to_provider_error():
    p = OpenAIProvider(api_key="x")
    p.client = MagicMock()
    p.client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    )
    with pytest.raises(ProviderError):
        p.invoke("hi", "m", 0.0)


def test_embed_returns_vector():
    p = OpenAIProvider(api_key="x")
    p.client = _mk_client_returning("ignored")
    assert p.embed("text", "ada") == [0.1, 0.2]


def test_constructor_sets_explicit_timeout():
    with patch("panoramix_core.providers.openai_provider.OpenAI") as oai:
        OpenAIProvider(api_key="x", connect_timeout=5.0, read_timeout=15.0)
        kwargs = oai.call_args.kwargs
        timeout = kwargs["timeout"]
        assert timeout.connect == 5.0
        assert timeout.read == 15.0
