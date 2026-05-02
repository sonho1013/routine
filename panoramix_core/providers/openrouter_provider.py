"""OpenRouter LLM provider — drop-in OpenAI-compatible API with model map."""
from __future__ import annotations

import yaml
import httpx
from pathlib import Path
from openai import OpenAI

from panoramix_core.providers.openai_provider import OpenAIProvider
from panoramix_core.providers.base import ProviderError

MODEL_NOT_SUPPORTED = "<not_supported>"


class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
        model_map_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key, base_url=base_url,
            connect_timeout=connect_timeout, read_timeout=read_timeout,
        )
        self.model_map = {}
        if model_map_path and Path(model_map_path).exists():
            self.model_map = yaml.safe_load(Path(model_map_path).read_text()) or {}

    def _translate(self, model: str) -> str:
        target = self.model_map.get(model, model)
        if target == MODEL_NOT_SUPPORTED:
            raise ProviderError(
                f"openrouter: model {model!r} is not supported via OpenRouter"
            )
        return target

    def invoke(self, prompt: str, model: str, temperature: float) -> str:
        return super().invoke(prompt, self._translate(model), temperature)

    def embed(self, text: str, model: str) -> list[float]:
        # OpenRouter does not provide embeddings; force the router to skip us.
        raise ProviderError("openrouter: embeddings are not supported")

    def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        raise ProviderError("openrouter: embeddings are not supported")
