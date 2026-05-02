"""OpenAI provider wrapping the OpenAI SDK with explicit timeouts."""
from __future__ import annotations

import os
import httpx
from openai import (
    OpenAI,
    APIConnectionError, APITimeoutError, AuthenticationError,
    RateLimitError, BadRequestError, PermissionDeniedError,
    APIError,
)
from panoramix_core.providers.base import (
    LLMProvider, EmbeddingProvider,
    ProviderError, ProviderTimeout, ProviderAuthError,
)


def _fix_socks_proxy() -> None:
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
                "all_proxy", "https_proxy", "http_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks://"):
            os.environ[var] = val.replace("socks://", "socks5://", 1)


class OpenAIProvider(LLMProvider, EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
    ) -> None:
        _fix_socks_proxy()
        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout,
                                write=read_timeout, pool=read_timeout)
        kwargs = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def invoke(self, prompt: str, model: str, temperature: float) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except APITimeoutError as e:
            raise ProviderTimeout(f"{self.name}: {e}") from e
        except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
            raise ProviderAuthError(f"{self.name}: {e}") from e
        except (RateLimitError, APIConnectionError, APIError) as e:
            raise ProviderError(f"{self.name}: {e}") from e

    def embed(self, text: str, model: str) -> list[float]:
        return self.embed_batch([text], model)[0]

    def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            resp = self.client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in resp.data]
        except APITimeoutError as e:
            raise ProviderTimeout(f"{self.name}: {e}") from e
        except (AuthenticationError, PermissionDeniedError, BadRequestError) as e:
            raise ProviderAuthError(f"{self.name}: {e}") from e
        except (RateLimitError, APIConnectionError, APIError) as e:
            raise ProviderError(f"{self.name}: {e}") from e
