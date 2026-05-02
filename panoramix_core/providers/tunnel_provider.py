"""Tunnel provider — calls the home FastAPI proxy via cloudflared."""
from __future__ import annotations

from panoramix_core.providers.openai_provider import OpenAIProvider


class TunnelProvider(OpenAIProvider):
    name = "tunnel"
    # Inherits invoke/embed/embed_batch unchanged — base_url is the only difference.
