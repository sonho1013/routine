"""Embedder — singleton façade preserving `.embed(text)` and `.embed_batch(texts)`.

Delegates to :class:`panoramix_core.embedding_router.EmbeddingRouter`, which
walks OpenAI → tunnel → cache. (OpenRouter does not provide embeddings.)
"""
from __future__ import annotations

import logging
from typing import List
import numpy as np

from panoramix_core.config import (
    EMBEDDING_MODEL,
    OPENAI_BASE_URL,
    TUNNEL_BASE_URL, TUNNEL_OPENAI_KEY,
    OPENAI_TIMEOUT_CONNECT, OPENAI_TIMEOUT_READ,
    TUNNEL_TIMEOUT_CONNECT, TUNNEL_TIMEOUT_READ,
    EMBEDDING_CACHE_PATH, LLM_CACHE_READ_ONLY,
)
from panoramix_core.embedding_cache import EmbeddingCache
from panoramix_core.embedding_router import EmbeddingRouter
from panoramix_core.providers.openai_provider import OpenAIProvider
from panoramix_core.providers.tunnel_provider import TunnelProvider


def _build_router() -> EmbeddingRouter:
    providers = [OpenAIProvider(
        base_url=OPENAI_BASE_URL,
        connect_timeout=OPENAI_TIMEOUT_CONNECT,
        read_timeout=OPENAI_TIMEOUT_READ,
    )]
    if TUNNEL_BASE_URL and TUNNEL_OPENAI_KEY:
        providers.append(TunnelProvider(
            api_key=TUNNEL_OPENAI_KEY,
            base_url=TUNNEL_BASE_URL,
            connect_timeout=TUNNEL_TIMEOUT_CONNECT,
            read_timeout=TUNNEL_TIMEOUT_READ,
        ))
    cache = EmbeddingCache(EMBEDDING_CACHE_PATH, read_only=LLM_CACHE_READ_ONLY)
    return EmbeddingRouter(providers=providers, cache=cache)


class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = EMBEDDING_MODEL
        self._router = _build_router()
        self._initialized = True
        logging.info("Embedder initialized: model=%s", self.model)

    def embed(self, text: str) -> List[float]:
        return self._router.embed(text=text, model=self.model)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array(self._router.embed_batch(texts=texts, model=self.model))
