"""Provider interface used by LLMRouter and EmbeddingRouter."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Raised on any provider failure that should trigger fallback."""


class ProviderTimeout(ProviderError):
    """Raised when a provider exceeds its configured timeout."""


class ProviderAuthError(ProviderError):
    """Auth/quota/billing failure (4xx). Skip this provider, try next."""


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def invoke(self, prompt: str, model: str, temperature: float) -> str:
        """Return the LLM response text or raise ProviderError."""


class EmbeddingProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def embed(self, text: str, model: str) -> list[float]:
        """Return the embedding vector or raise ProviderError."""

    @abstractmethod
    def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        ...
