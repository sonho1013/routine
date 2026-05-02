from unittest.mock import patch
from panoramix_core.providers.tunnel_provider import TunnelProvider


def test_tunnel_uses_configured_base_url():
    with patch("panoramix_core.providers.openai_provider.OpenAI") as oai:
        TunnelProvider(
            api_key="tk",
            base_url="https://abc.trycloudflare.com/v1",
            connect_timeout=5.0, read_timeout=10.0,
        )
        kwargs = oai.call_args.kwargs
        assert kwargs["base_url"] == "https://abc.trycloudflare.com/v1"
        assert kwargs["api_key"] == "tk"
        assert kwargs["timeout"].read == 10.0


def test_tunnel_name_is_tunnel():
    with patch("panoramix_core.providers.openai_provider.OpenAI"):
        p = TunnelProvider(api_key="tk",
                           base_url="https://x/v1",
                           connect_timeout=5, read_timeout=10)
    assert p.name == "tunnel"
