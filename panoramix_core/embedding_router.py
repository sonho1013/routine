"""Embedding router — chain over providers, cache fallback, batch-aware."""
from __future__ import annotations

import logging
from panoramix_core.embedding_cache import EmbeddingCache
from panoramix_core.providers.base import EmbeddingProvider, ProviderError
from panoramix_core.demo_mode import is_force_cache

log = logging.getLogger(__name__)


class EmbeddingRouter:
    def __init__(self, providers: list[EmbeddingProvider],
                 cache: EmbeddingCache) -> None:
        self.providers = providers
        self.cache = cache

    def embed(self, text: str, model: str) -> list[float]:
        if is_force_cache():
            return self.cache.get_or_raise(model=model, text=text)
        for p in self.providers:
            try:
                vec = p.embed(text=text, model=model)
                self.cache.set(model=model, text=text, vector=vec, provider=p.name)
                return vec
            except ProviderError as e:
                log.warning("embedding provider %s failed: %s", p.name, e)
                continue
        return self.cache.get_or_raise(model=model, text=text)

    def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []

        # Cache lookup first (always — saves cost even outside emergency mode)
        for i, t in enumerate(texts):
            v = self.cache.get(model=model, text=t)
            if v is not None:
                out[i] = v
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if not missing_texts:
            return [v for v in out]  # type: ignore[return-value]

        if is_force_cache():
            raise KeyError(
                f"embedding cache miss for {len(missing_texts)} texts in force-cache mode"
            )

        last_exc: Exception | None = None
        for p in self.providers:
            try:
                vecs = p.embed_batch(texts=missing_texts, model=model)
                for j, idx in enumerate(missing_idx):
                    out[idx] = vecs[j]
                    self.cache.set(model=model, text=missing_texts[j],
                                   vector=vecs[j], provider=p.name)
                return [v for v in out]  # type: ignore[return-value]
            except ProviderError as e:
                last_exc = e
                log.warning("embedding provider %s failed: %s", p.name, e)
                continue
        raise ProviderError(
            f"all embedding providers failed for batch; last_error={last_exc}"
        )
