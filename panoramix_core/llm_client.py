"""LLM client — public façade preserving the original `.invoke(prompt) -> str` signature.

Internals delegate to :class:`panoramix_core.llm_router.LLMRouter`, which walks
OpenAI → OpenRouter → tunnel → cache and writes successes back to the cache.
"""
from __future__ import annotations

import logging
from panoramix_core.config import (
    HABIT_DETECTION_LLM_MODEL,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    TUNNEL_BASE_URL, TUNNEL_OPENAI_KEY,
    OPENAI_TIMEOUT_CONNECT, OPENAI_TIMEOUT_READ,
    OPENROUTER_TIMEOUT_CONNECT, OPENROUTER_TIMEOUT_READ,
    TUNNEL_TIMEOUT_CONNECT, TUNNEL_TIMEOUT_READ,
    LLM_CACHE_PATH, LLM_CACHE_READ_ONLY,
    OPENROUTER_MODEL_MAP_PATH,
)
from panoramix_core.llm_cache import LLMCache
from panoramix_core.llm_router import LLMRouter
from panoramix_core.providers.openai_provider import OpenAIProvider
from panoramix_core.providers.openrouter_provider import OpenRouterProvider
from panoramix_core.providers.tunnel_provider import TunnelProvider

log = logging.getLogger(__name__)


def _build_router() -> LLMRouter:
    providers = []
    providers.append(OpenAIProvider(
        base_url=OPENAI_BASE_URL,
        connect_timeout=OPENAI_TIMEOUT_CONNECT,
        read_timeout=OPENAI_TIMEOUT_READ,
    ))
    if OPENROUTER_API_KEY:
        providers.append(OpenRouterProvider(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            connect_timeout=OPENROUTER_TIMEOUT_CONNECT,
            read_timeout=OPENROUTER_TIMEOUT_READ,
            model_map_path=OPENROUTER_MODEL_MAP_PATH,
        ))
    if TUNNEL_BASE_URL and TUNNEL_OPENAI_KEY:
        providers.append(TunnelProvider(
            api_key=TUNNEL_OPENAI_KEY,
            base_url=TUNNEL_BASE_URL,
            connect_timeout=TUNNEL_TIMEOUT_CONNECT,
            read_timeout=TUNNEL_TIMEOUT_READ,
        ))
    cache = LLMCache(LLM_CACHE_PATH, read_only=LLM_CACHE_READ_ONLY)
    return LLMRouter(providers=providers, cache=cache)


class LLMClient:
    """Backwards-compatible façade for HabitsDetector and friends."""

    def __init__(self, model: str | None = None, temperature: float = 0.2) -> None:
        self.model = model or HABIT_DETECTION_LLM_MODEL
        self.temperature = temperature
        self._router = _build_router()
        log.info("LLMClient ready: model=%s providers=%s",
                 self.model, [p.name for p in self._router.providers])

    def invoke(self, prompt: str) -> str:
        return self._router.invoke(prompt=prompt, model=self.model,
                                   temperature=self.temperature)
