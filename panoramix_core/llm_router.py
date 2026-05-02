"""LLM router — try providers in order, fall back to cache, opportunistic write."""
from __future__ import annotations

import logging
from panoramix_core.llm_cache import LLMCache
from panoramix_core.providers.base import LLMProvider, ProviderError
from panoramix_core.demo_mode import is_force_cache

log = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, providers: list[LLMProvider], cache: LLMCache) -> None:
        self.providers = providers
        self.cache = cache

    def invoke(self, prompt: str, model: str, temperature: float) -> str:
        if is_force_cache():
            return self.cache.get_or_raise(model=model, prompt=prompt,
                                           temperature=temperature)
        for p in self.providers:
            try:
                resp = p.invoke(prompt=prompt, model=model, temperature=temperature)
                self.cache.set(model=model, prompt=prompt, temperature=temperature,
                               response=resp, provider=p.name)
                return resp
            except ProviderError as e:
                log.warning("provider %s failed: %s — trying next", p.name, e)
                continue
        log.warning("all providers failed; falling back to cache")
        return self.cache.get_or_raise(model=model, prompt=prompt,
                                       temperature=temperature)
