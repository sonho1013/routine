# Demo Backup & Portable Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add layered LLM/embedding fallback (OpenAI → OpenRouter → home tunnel → cache), build a USB-distributable migration package, and produce demo-day operational scripts and runbooks for a 3-week schedule ending with on-site Renault demos in France.

**Architecture:** Insert two router classes (`LLMRouter`, `EmbeddingRouter`) behind the existing `LLMClient.invoke()` and `Embedder.embed()/.embed_batch()` interfaces — call sites stay untouched. Routers walk an ordered provider list with explicit per-provider `httpx.Timeout`, opportunistically write a JSON cache on success, and fall back to cache when all providers fail. A URL query param `?cache_only=1` (read by Streamlit) and an env var `DEMO_FORCE_CACHE=1` switch the routers into a cache-only mode. The home tunnel is a tiny FastAPI proxy launched on the dev desktop and exposed via `cloudflared`. The migration unit is a self-contained folder copied via USB; offline `pip install` from a bundled `wheels/` directory removes pypi from the critical path.

**Tech Stack:** Python 3.12, OpenAI SDK ≥ 1.30, httpx, FastAPI + uvicorn (home proxy), cloudflared, Streamlit, ChromaDB, pytest, pip-tools.

**Spec:** `docs/superpowers/specs/2026-05-02-demo-backup-and-portable-package-design.md`

---

## File Structure (created or modified)

**New (`panoramix_core/`):**
- `panoramix_core/llm_cache.py` — `LLMCache` class (key = sha256 of model+prompt+temperature; JSON-backed)
- `panoramix_core/embedding_cache.py` — `EmbeddingCache` class (same shape, value is `list[float]`)
- `panoramix_core/providers/__init__.py`
- `panoramix_core/providers/base.py` — `Provider` abstract base + `ProviderError`, `ProviderTimeout`
- `panoramix_core/providers/openai_provider.py` — `OpenAIProvider`, explicit `httpx.Timeout(connect=5, read=15)`
- `panoramix_core/providers/openrouter_provider.py` — `OpenRouterProvider` + model-map loader
- `panoramix_core/providers/tunnel_provider.py` — `TunnelProvider` (OpenAI SDK with `base_url=$TUNNEL_BASE_URL`, key=`$TUNNEL_OPENAI_KEY`)
- `panoramix_core/llm_router.py` — `LLMRouter`, replaces internals of `LLMClient`
- `panoramix_core/embedding_router.py` — `EmbeddingRouter`, replaces internals of `Embedder`
- `panoramix_core/demo_mode.py` — `is_force_cache()` reader (URL param + env)

**Modified:**
- `panoramix_core/llm_client.py` — `invoke()` delegates to `LLMRouter`; constructor reads provider configs
- `panoramix_core/embedder.py` — `embed()/.embed_batch()` delegate to `EmbeddingRouter`
- `panoramix_core/config.py` — add `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `TUNNEL_BASE_URL`, `TUNNEL_OPENAI_KEY`, `LLM_CACHE_PATH`, `EMBEDDING_CACHE_PATH`, `LLM_CACHE_READ_ONLY`, `OPENAI_TIMEOUT_CONNECT`, `OPENAI_TIMEOUT_READ`, `OPENROUTER_TIMEOUT_CONNECT`, `OPENROUTER_TIMEOUT_READ`, `TUNNEL_TIMEOUT_CONNECT`, `TUNNEL_TIMEOUT_READ`, `OPENROUTER_MODEL_MAP_PATH`
- `simulator/app.py` — wire query-param reader for `?cache_only=1`; render footer indicator

**New (`simulator/`):**
- `simulator/components/cache_status.py` — `render_cache_status_footer()` returns `live ✓` or `CACHE ⚠`

**New (`config/`):**
- `config/openrouter_model_map.yaml` — `gpt-4.1: openai/gpt-4-turbo` etc

**New (`home_proxy/`):**
- `home_proxy/proxy.py` — FastAPI app forwarding `/v1/*` to OpenAI with bearer-token gate
- `home_proxy/start.sh` — start uvicorn + cloudflared, print public URL
- `home_proxy/README.md` — deployment instructions

**New (`scripts/`):**
- `scripts/setup.sh` — venv + offline pip install + unpack fixtures
- `scripts/verify.sh` — preflight checks
- `scripts/run_demo.sh` — start streamlit + sleep-disable + open browser
- `scripts/reset_demo.sh` — restore fixtures and restart streamlit
- `scripts/seed_demo_data.py` — copy `fixtures/` → `storage/`
- `scripts/warmup_llm_cache.py` — run scripted demo flow, populate cache
- `scripts/download_wheels.sh` — `pip download` for offline install

**New (root):**
- `requirements.txt` — pinned top-level deps
- `requirements-lock.txt` — `pip-compile` output
- `.env.example` — annotated template
- `Makefile` — convenience targets wrapping the scripts
- `README.md` — 5-step open-box guide
- `README_DEMO_RUNBOOK.md` — day-of checklist
- `docs/DEMO_BACKUP_PLAN.md` — stakeholder-readable backup strategy

**New tests:**
- `tests/test_llm_cache.py`
- `tests/test_embedding_cache.py`
- `tests/test_openai_provider.py`
- `tests/test_openrouter_provider.py`
- `tests/test_tunnel_provider.py`
- `tests/test_llm_router.py`
- `tests/test_embedding_router.py`
- `tests/test_demo_reset_idempotent.py`

---

## Phase 0 — Setup & guardrails

### Task 0: Confirm Python version and create skeleton

**Files:**
- Create: `panoramix_core/providers/__init__.py`

- [ ] **Step 1: Confirm Python 3.12 is the target**

Run:
```bash
python3 --version
```
Expected output contains `3.12.` — if it does not, stop and discuss with user (the `wheels/` and `setup.sh` plan all assume 3.12).

- [ ] **Step 2: Create empty providers package**

```bash
mkdir -p panoramix_core/providers
touch panoramix_core/providers/__init__.py
```

- [ ] **Step 3: Verify pytest discovers the project**

Run: `pytest --collect-only tests/ -q | tail -5`
Expected: existing tests collected without errors.

- [ ] **Step 4: Commit**

```bash
git add panoramix_core/providers/__init__.py
git commit -m "chore(providers): scaffold providers package"
```

---

## Phase 1 — Cache layer (foundation)

### Task 1: `LLMCache` — JSON-backed prompt→response cache

**Files:**
- Create: `panoramix_core/llm_cache.py`
- Create: `tests/test_llm_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_cache.py`:

```python
import json
import pytest
from pathlib import Path
from panoramix_core.llm_cache import LLMCache


def test_set_and_get_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="gpt-4.1", prompt="hi", temperature=0.2,
              response="hello", provider="openai")
    assert cache.get(model="gpt-4.1", prompt="hi", temperature=0.2) == "hello"


def test_get_miss_returns_none(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    assert cache.get(model="gpt-4.1", prompt="hi", temperature=0.2) is None


def test_persists_to_disk(tmp_path):
    p = tmp_path / "c.json"
    LLMCache(p).set(model="m", prompt="p", temperature=0.0,
                    response="r", provider="openai")
    raw = json.loads(p.read_text())
    assert len(raw) == 1
    entry = next(iter(raw.values()))
    assert entry["response"] == "r"
    assert entry["provider"] == "openai"
    assert "cached_at" in entry


def test_get_or_raise_hits(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="p", temperature=0.0,
              response="r", provider="openai")
    assert cache.get_or_raise(model="m", prompt="p", temperature=0.0) == "r"


def test_get_or_raise_miss_raises(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    with pytest.raises(KeyError):
        cache.get_or_raise(model="m", prompt="p", temperature=0.0)


def test_read_only_mode_blocks_writes(tmp_path):
    p = tmp_path / "c.json"
    LLMCache(p).set(model="m", prompt="p", temperature=0.0,
                    response="r", provider="openai")
    ro = LLMCache(p, read_only=True)
    ro.set(model="m", prompt="p2", temperature=0.0,
           response="r2", provider="openai")
    raw = json.loads(p.read_text())
    assert len(raw) == 1


def test_key_distinguishes_temperature(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="p", temperature=0.0,
              response="A", provider="openai")
    cache.set(model="m", prompt="p", temperature=0.7,
              response="B", provider="openai")
    assert cache.get(model="m", prompt="p", temperature=0.0) == "A"
    assert cache.get(model="m", prompt="p", temperature=0.7) == "B"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_llm_cache.py -v`
Expected: ImportError on `panoramix_core.llm_cache`.

- [ ] **Step 3: Implement `LLMCache`**

Create `panoramix_core/llm_cache.py`:

```python
"""JSON-backed prompt→response cache for the LLM fallback chain."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _key(model: str, prompt: str, temperature: float) -> str:
    raw = f"{model}\n{prompt}\n{temperature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self.path)

    def get(self, model: str, prompt: str, temperature: float) -> Optional[str]:
        with self._lock:
            data = self._load()
            entry = data.get(_key(model, prompt, temperature))
            return entry["response"] if entry else None

    def get_or_raise(self, model: str, prompt: str, temperature: float) -> str:
        v = self.get(model, prompt, temperature)
        if v is None:
            raise KeyError(
                f"LLM cache miss for model={model!r} prompt[:30]={prompt[:30]!r}"
            )
        return v

    def set(self, model: str, prompt: str, temperature: float,
            response: str, provider: str) -> None:
        if self.read_only:
            return
        with self._lock:
            data = self._load()
            data[_key(model, prompt, temperature)] = {
                "response": response,
                "provider": provider,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save(data)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_llm_cache.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/llm_cache.py tests/test_llm_cache.py
git commit -m "feat(cache): add JSON-backed LLMCache with read-only mode"
```

---

### Task 2: `EmbeddingCache` — same shape, vector value

**Files:**
- Create: `panoramix_core/embedding_cache.py`
- Create: `tests/test_embedding_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedding_cache.py`:

```python
import pytest
from panoramix_core.embedding_cache import EmbeddingCache


def test_set_get_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    vec = [0.1, 0.2, 0.3]
    cache.set(model="ada-002", text="hello", vector=vec, provider="openai")
    assert cache.get(model="ada-002", text="hello") == vec


def test_get_miss_returns_none(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    assert cache.get(model="ada-002", text="hello") is None


def test_get_or_raise_miss(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    with pytest.raises(KeyError):
        cache.get_or_raise(model="ada-002", text="hello")


def test_read_only_blocks_writes(tmp_path):
    p = tmp_path / "e.json"
    EmbeddingCache(p).set(model="m", text="t", vector=[1.0], provider="openai")
    ro = EmbeddingCache(p, read_only=True)
    ro.set(model="m", text="u", vector=[2.0], provider="openai")
    assert EmbeddingCache(p).get(model="m", text="u") is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_embedding_cache.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `EmbeddingCache`**

Create `panoramix_core/embedding_cache.py`:

```python
"""JSON-backed text→embedding cache for the embedding fallback chain."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _key(model: str, text: str) -> str:
    raw = f"{model}\n{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(self.path)

    def get(self, model: str, text: str) -> Optional[List[float]]:
        with self._lock:
            data = self._load()
            entry = data.get(_key(model, text))
            return entry["vector"] if entry else None

    def get_or_raise(self, model: str, text: str) -> List[float]:
        v = self.get(model, text)
        if v is None:
            raise KeyError(f"Embedding cache miss for text[:30]={text[:30]!r}")
        return v

    def set(self, model: str, text: str, vector: List[float], provider: str) -> None:
        if self.read_only:
            return
        with self._lock:
            data = self._load()
            data[_key(model, text)] = {
                "vector": list(vector),
                "provider": provider,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save(data)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_embedding_cache.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/embedding_cache.py tests/test_embedding_cache.py
git commit -m "feat(cache): add EmbeddingCache for vector fallback"
```

---

## Phase 2 — Provider classes

### Task 3: Provider base class + exceptions

**Files:**
- Create: `panoramix_core/providers/base.py`

- [ ] **Step 1: Create base module**

```python
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
```

- [ ] **Step 2: Sanity import**

Run:
```bash
python -c "from panoramix_core.providers.base import LLMProvider, EmbeddingProvider, ProviderError, ProviderTimeout, ProviderAuthError; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add panoramix_core/providers/base.py
git commit -m "feat(providers): add Provider base classes and exception hierarchy"
```

---

### Task 4: `OpenAIProvider` with explicit timeout

**Files:**
- Create: `panoramix_core/providers/openai_provider.py`
- Create: `tests/test_openai_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_openai_provider.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_openai_provider.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `OpenAIProvider`**

Create `panoramix_core/providers/openai_provider.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_openai_provider.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/providers/openai_provider.py tests/test_openai_provider.py
git commit -m "feat(providers): add OpenAIProvider with explicit timeout"
```

---

### Task 5: `OpenRouterProvider` with model map

**Files:**
- Create: `panoramix_core/providers/openrouter_provider.py`
- Create: `config/openrouter_model_map.yaml`
- Create: `tests/test_openrouter_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_openrouter_provider.py`:

```python
import pytest
from unittest.mock import MagicMock
from panoramix_core.providers.openrouter_provider import (
    OpenRouterProvider, MODEL_NOT_SUPPORTED,
)
from panoramix_core.providers.base import ProviderError


def _client_returning(text):
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=text))
    ]
    return c


def test_translates_model_via_map(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text("gpt-4.1: openai/gpt-4-turbo\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    p.invoke("hi", "gpt-4.1", 0.2)
    args, kwargs = p.client.chat.completions.create.call_args
    assert kwargs["model"] == "openai/gpt-4-turbo"


def test_passthrough_when_no_mapping(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text("gpt-4.1: openai/gpt-4-turbo\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    p.invoke("hi", "claude-3-5-sonnet", 0.2)
    assert p.client.chat.completions.create.call_args.kwargs["model"] == "claude-3-5-sonnet"


def test_not_supported_raises(tmp_path):
    map_path = tmp_path / "map.yaml"
    map_path.write_text(f"text-embedding-ada-002: {MODEL_NOT_SUPPORTED}\n")
    p = OpenRouterProvider(api_key="x", model_map_path=map_path)
    p.client = _client_returning("ok")
    with pytest.raises(ProviderError):
        p.invoke("hi", "text-embedding-ada-002", 0.0)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_openrouter_provider.py -v`
Expected: ImportError.

- [ ] **Step 3: Create model map config**

Create `config/openrouter_model_map.yaml`:

```yaml
# Renault demo OpenRouter model translation table.
# Left = the model name our code asks for; right = OpenRouter's model id.
gpt-4.1: openai/gpt-4-turbo
gpt-4: openai/gpt-4-turbo
text-embedding-ada-002: <not_supported>
```

- [ ] **Step 4: Implement `OpenRouterProvider`**

Create `panoramix_core/providers/openrouter_provider.py`:

```python
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
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `pytest tests/test_openrouter_provider.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add panoramix_core/providers/openrouter_provider.py config/openrouter_model_map.yaml tests/test_openrouter_provider.py
git commit -m "feat(providers): add OpenRouterProvider with model translation map"
```

---

### Task 6: `TunnelProvider` (cloudflared → home FastAPI proxy → OpenAI)

**Files:**
- Create: `panoramix_core/providers/tunnel_provider.py`
- Create: `tests/test_tunnel_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tunnel_provider.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_tunnel_provider.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `TunnelProvider`**

Create `panoramix_core/providers/tunnel_provider.py`:

```python
"""Tunnel provider — calls the home FastAPI proxy via cloudflared."""
from __future__ import annotations

from panoramix_core.providers.openai_provider import OpenAIProvider


class TunnelProvider(OpenAIProvider):
    name = "tunnel"
    # Inherits invoke/embed/embed_batch unchanged — base_url is the only difference.
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_tunnel_provider.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/providers/tunnel_provider.py tests/test_tunnel_provider.py
git commit -m "feat(providers): add TunnelProvider for home cloudflared proxy"
```

---

## Phase 3 — Demo mode reader

### Task 7: `demo_mode.is_force_cache()` reader

**Files:**
- Create: `panoramix_core/demo_mode.py`
- Create: `tests/test_demo_mode.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_demo_mode.py`:

```python
import os
import pytest
from panoramix_core.demo_mode import is_force_cache, set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(False)
    os.environ.pop("DEMO_FORCE_CACHE", None)


def test_default_off():
    assert is_force_cache() is False


def test_env_var_on():
    os.environ["DEMO_FORCE_CACHE"] = "1"
    assert is_force_cache() is True


def test_env_var_off_string():
    os.environ["DEMO_FORCE_CACHE"] = "0"
    assert is_force_cache() is False


def test_set_for_test_overrides_env():
    set_force_cache_for_test(True)
    assert is_force_cache() is True
    set_force_cache_for_test(False)
    assert is_force_cache() is False
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_demo_mode.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `demo_mode`**

Create `panoramix_core/demo_mode.py`:

```python
"""Reader for the emergency cache-only switch.

Three sources of truth, in priority order (highest first):

1. `set_force_cache_for_test(True)` — testing only
2. Streamlit URL query param `?cache_only=1` — set via `set_force_cache_runtime`
   from the Streamlit app at request time
3. Environment variable `DEMO_FORCE_CACHE=1` — set at process launch
"""
from __future__ import annotations

import os

_test_override: bool | None = None
_runtime_override: bool | None = None


def set_force_cache_for_test(value: bool | None) -> None:
    global _test_override
    _test_override = value


def set_force_cache_runtime(value: bool | None) -> None:
    global _runtime_override
    _runtime_override = value


def is_force_cache() -> bool:
    if _test_override is not None:
        return _test_override
    if _runtime_override is not None:
        return _runtime_override
    return os.environ.get("DEMO_FORCE_CACHE", "").strip() == "1"
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_demo_mode.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/demo_mode.py tests/test_demo_mode.py
git commit -m "feat(demo): add force-cache reader for emergency switch"
```

---

## Phase 4 — Routers

### Task 8: `LLMRouter` — chain runner with cache write-through

**Files:**
- Create: `panoramix_core/llm_router.py`
- Create: `tests/test_llm_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_router.py`:

```python
import pytest
from unittest.mock import MagicMock
from panoramix_core.llm_router import LLMRouter
from panoramix_core.llm_cache import LLMCache
from panoramix_core.providers.base import (
    ProviderError, ProviderTimeout, ProviderAuthError,
)
from panoramix_core.demo_mode import set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(False)


def _ok(text="ok"):
    p = MagicMock()
    p.invoke.return_value = text
    p.name = "mock"
    return p


def _fail(exc=ProviderError("nope")):
    p = MagicMock()
    p.invoke.side_effect = exc
    p.name = "mock"
    return p


def test_first_provider_wins(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _ok("A"), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "A"
    b.invoke.assert_not_called()


def test_falls_through_to_second_on_error(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_falls_through_on_timeout(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(ProviderTimeout("slow")), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_falls_through_on_auth_error(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a, b = _fail(ProviderAuthError("401")), _ok("B")
    r = LLMRouter(providers=[a, b], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "B"


def test_all_providers_fail_uses_cache(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="hi", temperature=0.0,
              response="seeded", provider="seed")
    r = LLMRouter(providers=[_fail(), _fail()], cache=cache)
    assert r.invoke("hi", "m", 0.0) == "seeded"


def test_all_fail_no_cache_raises(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    r = LLMRouter(providers=[_fail(), _fail()], cache=cache)
    with pytest.raises(KeyError):
        r.invoke("hi", "m", 0.0)


def test_success_writes_cache(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a = _ok("A"); a.name = "openai"
    r = LLMRouter(providers=[a], cache=cache)
    r.invoke("hi", "m", 0.0)
    assert cache.get(model="m", prompt="hi", temperature=0.0) == "A"


def test_force_cache_skips_providers(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    cache.set(model="m", prompt="hi", temperature=0.0,
              response="cached", provider="seed")
    a = _ok("FRESH")
    r = LLMRouter(providers=[a], cache=cache)
    set_force_cache_for_test(True)
    try:
        assert r.invoke("hi", "m", 0.0) == "cached"
        a.invoke.assert_not_called()
    finally:
        set_force_cache_for_test(False)


def test_force_cache_miss_raises_without_calling_providers(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    a = _ok("FRESH")
    r = LLMRouter(providers=[a], cache=cache)
    set_force_cache_for_test(True)
    try:
        with pytest.raises(KeyError):
            r.invoke("hi", "m", 0.0)
        a.invoke.assert_not_called()
    finally:
        set_force_cache_for_test(False)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_llm_router.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `LLMRouter`**

Create `panoramix_core/llm_router.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_llm_router.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/llm_router.py tests/test_llm_router.py
git commit -m "feat(router): add LLMRouter with chain + cache fallback + emergency mode"
```

---

### Task 9: `EmbeddingRouter` — same pattern, two methods

**Files:**
- Create: `panoramix_core/embedding_router.py`
- Create: `tests/test_embedding_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedding_router.py`:

```python
import pytest
from unittest.mock import MagicMock
from panoramix_core.embedding_router import EmbeddingRouter
from panoramix_core.embedding_cache import EmbeddingCache
from panoramix_core.providers.base import ProviderError
from panoramix_core.demo_mode import set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(False)


def _ok(vec):
    p = MagicMock()
    p.embed.return_value = vec
    p.embed_batch.return_value = [vec]
    p.name = "mock"
    return p


def _fail():
    p = MagicMock()
    p.embed.side_effect = ProviderError("nope")
    p.embed_batch.side_effect = ProviderError("nope")
    p.name = "mock"
    return p


def test_first_wins(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_ok([1.0]), _ok([2.0])], cache=cache)
    assert r.embed("t", "ada") == [1.0]


def test_falls_through_on_error(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_fail(), _ok([2.0])], cache=cache)
    assert r.embed("t", "ada") == [2.0]


def test_all_fail_uses_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="t", vector=[9.0], provider="seed")
    r = EmbeddingRouter(providers=[_fail()], cache=cache)
    assert r.embed("t", "ada") == [9.0]


def test_success_writes_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    r = EmbeddingRouter(providers=[_ok([1.0])], cache=cache)
    r.embed("t", "ada")
    assert cache.get(model="ada", text="t") == [1.0]


def test_force_cache_skips_providers(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="t", vector=[9.0], provider="seed")
    p = _ok([1.0])
    r = EmbeddingRouter(providers=[p], cache=cache)
    set_force_cache_for_test(True)
    try:
        assert r.embed("t", "ada") == [9.0]
        p.embed.assert_not_called()
    finally:
        set_force_cache_for_test(False)


def test_embed_batch_partial_cache_hit(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.set(model="ada", text="cached", vector=[7.0], provider="seed")
    p = MagicMock()
    p.name = "mock"
    p.embed_batch.return_value = [[8.0]]  # only the missing one
    r = EmbeddingRouter(providers=[p], cache=cache)
    out = r.embed_batch(["cached", "missing"], "ada")
    assert out == [[7.0], [8.0]]
    # provider should be asked only for the missing one
    # (router calls embed_batch with kwargs)
    assert p.embed_batch.call_args.kwargs["texts"] == ["missing"]
    assert p.embed_batch.call_args.kwargs["model"] == "ada"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_embedding_router.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `EmbeddingRouter`**

Create `panoramix_core/embedding_router.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_embedding_router.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add panoramix_core/embedding_router.py tests/test_embedding_router.py
git commit -m "feat(router): add EmbeddingRouter with batch-aware cache fallback"
```

---

## Phase 5 — Wire into existing classes

### Task 10: Extend `panoramix_core/config.py` with new env vars

**Files:**
- Modify: `panoramix_core/config.py`

- [ ] **Step 1: Add Path import near the top**

Add this immediately after the existing `import os` line at the top of `panoramix_core/config.py`:

```python
from pathlib import Path as _Path
_DEFAULT_MODEL_MAP = _Path(__file__).resolve().parent.parent / "config" / "openrouter_model_map.yaml"
```

- [ ] **Step 2: Append new config block at the end of the file**

Append to `panoramix_core/config.py`:

```python
# ── Backup chain providers ──
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
TUNNEL_BASE_URL = os.getenv("TUNNEL_BASE_URL", "")
TUNNEL_OPENAI_KEY = os.getenv("TUNNEL_OPENAI_KEY", "")

# ── Timeouts (seconds) ──
OPENAI_TIMEOUT_CONNECT = float(os.getenv("OPENAI_TIMEOUT_CONNECT", "5"))
OPENAI_TIMEOUT_READ = float(os.getenv("OPENAI_TIMEOUT_READ", "15"))
OPENROUTER_TIMEOUT_CONNECT = float(os.getenv("OPENROUTER_TIMEOUT_CONNECT", "5"))
OPENROUTER_TIMEOUT_READ = float(os.getenv("OPENROUTER_TIMEOUT_READ", "15"))
TUNNEL_TIMEOUT_CONNECT = float(os.getenv("TUNNEL_TIMEOUT_CONNECT", "5"))
TUNNEL_TIMEOUT_READ = float(os.getenv("TUNNEL_TIMEOUT_READ", "10"))

# ── Cache ──
LLM_CACHE_PATH = os.getenv("LLM_CACHE_PATH", "./storage/llm_cache.json")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "./storage/embedding_cache.json")
LLM_CACHE_READ_ONLY = os.getenv("LLM_CACHE_READ_ONLY", "").strip() == "1"

# ── OpenRouter model translation ──
OPENROUTER_MODEL_MAP_PATH = os.getenv("OPENROUTER_MODEL_MAP_PATH", str(_DEFAULT_MODEL_MAP))
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from panoramix_core import config; print(config.OPENROUTER_MODEL_MAP_PATH)"
```
Expected: ends with `config/openrouter_model_map.yaml`.

- [ ] **Step 3: Commit**

```bash
git add panoramix_core/config.py
git commit -m "config: add provider/timeout/cache env vars"
```

---

### Task 11: Rewrite `LLMClient.invoke()` to delegate to `LLMRouter`

**Files:**
- Modify: `panoramix_core/llm_client.py`

- [ ] **Step 1: Replace the file body**

Overwrite `panoramix_core/llm_client.py` with:

```python
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
```

- [ ] **Step 2: Smoke test the import**

Run:
```bash
OPENAI_API_KEY=dummy python -c "from panoramix_core.llm_client import LLMClient; c = LLMClient(); print(c.model)"
```
Expected: prints `gpt-4.1` (or your overridden model). No traceback.

- [ ] **Step 3: Run the existing test suite to confirm nothing else broke**

Run: `pytest tests/ -x -q --ignore=tests/test_kling_omnivideo.py --ignore=tests/test_storyboard_generation.py --ignore=tests/test_embedding_distance.py --ignore=tests/test_eps_sweep.py`
Expected: previous test counts still pass (any failure here is a regression to fix before continuing).

- [ ] **Step 4: Commit**

```bash
git add panoramix_core/llm_client.py
git commit -m "refactor(llm_client): delegate invoke() to LLMRouter chain"
```

---

### Task 12: Rewrite `Embedder` to delegate to `EmbeddingRouter`

**Files:**
- Modify: `panoramix_core/embedder.py`

- [ ] **Step 1: Overwrite file**

Replace `panoramix_core/embedder.py`:

```python
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
```

- [ ] **Step 2: Smoke test**

Run:
```bash
OPENAI_API_KEY=dummy python -c "from panoramix_core.embedder import Embedder; e = Embedder(); print(e.model)"
```
Expected: prints `text-embedding-ada-002`.

- [ ] **Step 3: Run wider test suite**

Run: `pytest tests/ -x -q --ignore=tests/test_kling_omnivideo.py --ignore=tests/test_storyboard_generation.py --ignore=tests/test_embedding_distance.py --ignore=tests/test_eps_sweep.py`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add panoramix_core/embedder.py
git commit -m "refactor(embedder): delegate to EmbeddingRouter chain"
```

---

## Phase 6 — Streamlit UI: emergency mode + indicator

### Task 13: Cache-status footer component

**Files:**
- Create: `simulator/components/cache_status.py`

- [ ] **Step 1: Create the component**

Create `simulator/components/cache_status.py`:

```python
"""Sidebar footer indicator: `live ✓` / `CACHE ⚠`."""
import streamlit as st
from panoramix_core.demo_mode import is_force_cache


def render_cache_status_footer() -> None:
    label = "CACHE ⚠" if is_force_cache() else "live ✓"
    color = "#cc8800" if is_force_cache() else "#888888"
    st.sidebar.markdown(
        f"<div style='position:fixed;bottom:8px;left:8px;"
        f"font-size:11px;color:{color};opacity:0.7'>{label}</div>",
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add simulator/components/cache_status.py
git commit -m "feat(ui): add hidden cache-status footer indicator"
```

---

### Task 14: Wire `?cache_only=1` and footer into `simulator/app.py`

**Files:**
- Modify: `simulator/app.py`

- [ ] **Step 1: Read the current state of `app.py`**

Run: `head -80 simulator/app.py`
Note where `st.set_page_config(...)` is and where the tabs are rendered.

- [ ] **Step 2: Insert query-param reader and footer after `set_page_config`**

After `st.set_page_config(...)` block, insert:

```python
# ── Demo emergency switch ──
from panoramix_core.demo_mode import set_force_cache_runtime
from simulator.components.cache_status import render_cache_status_footer

_qp = st.query_params
_force = _qp.get("cache_only") == "1"
set_force_cache_runtime(_force)
```

At the very end of the file (after all tabs render), insert:

```python
render_cache_status_footer()
```

- [ ] **Step 3: Manual smoke test**

Run:
```bash
streamlit run simulator/app.py --server.port=8501 --server.headless=true &
STPID=$!
sleep 4
curl -s http://localhost:8501/?cache_only=1 | grep -c "CACHE ⚠" || echo "indicator missing"
kill $STPID
```
Expected: prints a non-zero count or text `CACHE ⚠`.

(If running headless `curl` against Streamlit doesn't reveal HTML state, instead open the browser at `http://localhost:8501/?cache_only=1` and confirm the dim text in the lower-left.)

- [ ] **Step 4: Commit**

```bash
git add simulator/app.py
git commit -m "feat(ui): wire ?cache_only=1 emergency switch and indicator"
```

---

## Phase 7 — Home FastAPI proxy + cloudflared

### Task 15: `home_proxy/proxy.py`

**Files:**
- Create: `home_proxy/proxy.py`
- Create: `home_proxy/README.md`

- [ ] **Step 1: Implement the proxy**

Create `home_proxy/proxy.py`:

```python
"""Thin OpenAI-compatible proxy for the demo home tunnel.

Forwards `/v1/chat/completions` and `/v1/embeddings` to OpenAI using the
desktop's own OPENAI_API_KEY. Authenticates inbound requests by matching
their Authorization header against TUNNEL_SHARED_SECRET.
"""
from __future__ import annotations

import os
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TUNNEL_SHARED_SECRET = os.environ["TUNNEL_SHARED_SECRET"]
OPENAI_BASE_URL = os.environ.get("OPENAI_UPSTREAM", "https://api.openai.com/v1")

app = FastAPI(title="habit-memory-demo home proxy")
client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0))


def _check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != TUNNEL_SHARED_SECRET:
        raise HTTPException(403, "invalid bearer token")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request,
                  authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    upstream = f"{OPENAI_BASE_URL.rstrip('/')}/{path}"
    body = await request.body()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": request.headers.get("Content-Type", "application/json"),
    }
    resp = await client.request(request.method, upstream, content=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)
```

- [ ] **Step 2: Smoke test the import**

Run:
```bash
OPENAI_API_KEY=dummy TUNNEL_SHARED_SECRET=test python -c "import home_proxy.proxy; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Local self-test (optional but recommended)**

Run:
```bash
OPENAI_API_KEY=dummy TUNNEL_SHARED_SECRET=secret \
  python -m uvicorn home_proxy.proxy:app --port 8787 &
sleep 2
curl -s http://localhost:8787/healthz
kill %1
```
Expected: `{"ok":true}`.

- [ ] **Step 4: Commit**

```bash
git add home_proxy/proxy.py
git commit -m "feat(home_proxy): add FastAPI thin proxy with bearer-token gate"
```

---

### Task 16: `home_proxy/start.sh` — launch uvicorn + cloudflared

**Files:**
- Create: `home_proxy/start.sh`
- Modify: `home_proxy/README.md`

- [ ] **Step 1: Write the launcher**

Create `home_proxy/start.sh`:

```bash
#!/usr/bin/env bash
# Start the home proxy + cloudflared and print the public URL.
# Requires: cloudflared installed and on PATH; OPENAI_API_KEY and
# TUNNEL_SHARED_SECRET exported (e.g. from ~/.config/habit-memory-demo.env).
set -euo pipefail

PORT="${HOME_PROXY_PORT:-8787}"
LOG_DIR="${HOME_PROXY_LOG_DIR:-$HOME/.cache/habit-memory-demo}"
mkdir -p "$LOG_DIR"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be exported}"
: "${TUNNEL_SHARED_SECRET:?TUNNEL_SHARED_SECRET must be exported}"

# Start FastAPI in background
echo "▶ starting uvicorn on :$PORT"
python -m uvicorn home_proxy.proxy:app --host 127.0.0.1 --port "$PORT" \
  >"$LOG_DIR/proxy.log" 2>&1 &
UVI_PID=$!
trap 'kill $UVI_PID 2>/dev/null || true' EXIT

sleep 2
curl -fs "http://127.0.0.1:$PORT/healthz" >/dev/null \
  || { echo "✗ proxy did not come up; see $LOG_DIR/proxy.log"; exit 1; }

# Start cloudflared and capture URL
echo "▶ starting cloudflared tunnel"
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
  >"$LOG_DIR/cloudflared.log" 2>&1 &
CFD_PID=$!
trap 'kill $UVI_PID $CFD_PID 2>/dev/null || true' EXIT

# Wait for the public URL to appear in the log
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' \
        "$LOG_DIR/cloudflared.log" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "✗ failed to obtain cloudflared URL"; exit 1; }

echo
echo "════════════════════════════════════════════════════════"
echo "  TUNNEL_BASE_URL=${URL}/v1"
echo "  TUNNEL_OPENAI_KEY=${TUNNEL_SHARED_SECRET}"
echo "════════════════════════════════════════════════════════"
echo "Paste the two lines above into the demo laptop's .env"
echo "Logs: $LOG_DIR/{proxy,cloudflared}.log"
wait
```

- [ ] **Step 2: Make executable**

Run: `chmod +x home_proxy/start.sh`

- [ ] **Step 3: Write deployment README**

Create `home_proxy/README.md`:

```markdown
# Home Proxy (Renault demo backup)

This directory is **deployed on the local desktop**, not on the demo laptop.

## One-time setup (desktop)

1. Install cloudflared:
   ```
   sudo apt-get install -y cloudflared
   ```
2. Activate the project venv with FastAPI/uvicorn installed:
   ```
   cd /path/to/habit-memory-demo
   source .venv/bin/activate
   pip install fastapi uvicorn httpx
   ```
3. Create `~/.config/habit-memory-demo.env`:
   ```bash
   export OPENAI_API_KEY=sk-...
   export TUNNEL_SHARED_SECRET=$(openssl rand -hex 24)
   ```

## Demo-day startup (desktop)

```bash
source ~/.config/habit-memory-demo.env
bash home_proxy/start.sh
```

The script prints two lines to copy into the laptop's `.env`:

```
TUNNEL_BASE_URL=https://xxx.trycloudflare.com/v1
TUNNEL_OPENAI_KEY=<the shared secret>
```

Keep this terminal open for the duration of the demo. Re-running the script
yields a **new** trycloudflare URL.
```

- [ ] **Step 4: Commit**

```bash
git add home_proxy/start.sh home_proxy/README.md
git commit -m "feat(home_proxy): add start.sh launcher and deployment README"
```

---

## Phase 8 — Migration package files

### Task 17: `requirements.txt` and `requirements-lock.txt`

**Files:**
- Create: `requirements.txt`
- Create: `requirements-lock.txt`

- [ ] **Step 1: Write top-level requirements**

Create `requirements.txt`:

```
streamlit==1.36.0
openai==1.40.0
httpx==0.27.0
chromadb==0.5.5
pydantic==2.7.0
numpy==1.26.4
matplotlib==3.8.4
requests==2.32.3
PyJWT==2.8.0
PyYAML==6.0.1
fastapi==0.111.0
uvicorn==0.30.0
pytest==8.2.0
```

(Adjust each pin to match what is currently installed in the dev venv via `pip freeze | grep -iE 'streamlit|openai|chromadb|pydantic|numpy|matplotlib|requests|jwt|yaml|fastapi|uvicorn|pytest'` — these are placeholders, replace with the engineer's actual freeze output.)

- [ ] **Step 2: Generate lock file**

Run:
```bash
pip install pip-tools
pip-compile --output-file=requirements-lock.txt requirements.txt
```
Expected: `requirements-lock.txt` with full transitive pins.

- [ ] **Step 3: Smoke install in throwaway venv**

Run:
```bash
python -m venv /tmp/_demo_venv
/tmp/_demo_venv/bin/pip install -r requirements-lock.txt
/tmp/_demo_venv/bin/python -c "import streamlit, openai, chromadb; print('ok')"
rm -rf /tmp/_demo_venv
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt requirements-lock.txt
git commit -m "build: pin runtime dependencies"
```

---

### Task 18: `.env.example`

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create the file**

Create `.env.example`:

```bash
# Habit Memory Demo — environment variables
# Copy to `.env` and fill in. NEVER commit `.env`.

# === Primary LLM ===
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

# === Backup 1: OpenRouter ===
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# === Backup 2: Home cloudflared tunnel ===
# Filled in on demo day from `home_proxy/start.sh` output.
TUNNEL_BASE_URL=
TUNNEL_OPENAI_KEY=

# === Models ===
HABIT_DETECTION_LLM_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-ada-002

# === Timeouts (seconds) ===
OPENAI_TIMEOUT_CONNECT=5
OPENAI_TIMEOUT_READ=15
OPENROUTER_TIMEOUT_CONNECT=5
OPENROUTER_TIMEOUT_READ=15
TUNNEL_TIMEOUT_CONNECT=5
TUNNEL_TIMEOUT_READ=10

# === Cache ===
LLM_CACHE_PATH=./storage/llm_cache.json
EMBEDDING_CACHE_PATH=./storage/embedding_cache.json
LLM_CACHE_READ_ONLY=1     # demo runtime: read-only to prevent pollution

# === Emergency switch (leave blank in normal operation) ===
DEMO_FORCE_CACHE=

# === DBSCAN (rarely changed) ===
HYBRID_ALPHA=0.8
DBSCAN_EPS=0.10
DBSCAN_MIN_SAMPLES=3
```

- [ ] **Step 2: Verify `.env.example` is not in `.gitignore`**

Run: `grep -E "^\.env" .gitignore`
Expected: only `.env` and `.env.local`, **not** `.env.example`.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add annotated .env.example for demo packaging"
```

---

### Task 19: `scripts/setup.sh` — offline venv install + fixture unpack

**Files:**
- Create: `scripts/setup.sh`

- [ ] **Step 1: Write the script**

Create `scripts/setup.sh`:

```bash
#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu laptop. Assumes the demo pack folder layout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Python version check
PY_VER=$(python3 --version 2>&1 | grep -oE '3\.[0-9]+')
if [ "$PY_VER" != "3.12" ]; then
  echo "✗ Python 3.12 required, found $PY_VER"
  echo "  Install: sudo apt-get install -y python3.12 python3.12-venv"
  exit 1
fi

# 2. venv
if [ ! -d .venv ]; then
  echo "▶ creating venv"
  python3.12 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# 3. offline pip install
if [ ! -d wheels ]; then
  echo "✗ wheels/ directory missing — this pack was not built for offline install"
  exit 1
fi
echo "▶ installing from wheels/ (offline)"
pip install --no-index --find-links wheels/ -r requirements-lock.txt

# 4. unpack fixtures
mkdir -p storage
echo "▶ seeding storage/ from fixtures/"
cp -f fixtures/habit_memory.db storage/habit_memory.db
rm -rf storage/memories
cp -r fixtures/chroma_memories storage/memories
cp -f fixtures/llm_cache.json storage/llm_cache.json
cp -f fixtures/embedding_cache.json storage/embedding_cache.json

# 5. .env reminder
if [ ! -f .env ]; then
  echo "⚠ .env missing — copy from .env.example and fill in keys before running."
fi

echo "✓ setup complete. Next:"
echo "    cp .env.example .env && nano .env"
echo "    bash scripts/verify.sh"
echo "    bash scripts/run_demo.sh"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/setup.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.sh
git commit -m "build: add setup.sh for offline laptop install"
```

---

### Task 20: `scripts/verify.sh` — preflight checks

**Files:**
- Create: `scripts/verify.sh`

- [ ] **Step 1: Write the script**

Create `scripts/verify.sh`:

```bash
#!/usr/bin/env bash
# Demo-day preflight: keys reachable, fixtures present, cache populated, videos OK.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
[ -f .venv/bin/activate ] && source .venv/bin/activate
# shellcheck source=/dev/null
[ -f .env ] && set -a && source .env && set +a

PASS=0
FAIL=0
report() {
  local ok=$1; shift
  if [ "$ok" = "0" ]; then echo "  ✓ $*"; PASS=$((PASS+1))
  else echo "  ✗ $*"; FAIL=$((FAIL+1)); fi
}

echo "=== Keys ==="
curl -fsS -o /dev/null -m 8 \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-_missing}" \
  "${OPENAI_BASE_URL:-https://api.openai.com/v1}/models"
report $? "OpenAI direct"

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  curl -fsS -o /dev/null -m 8 \
    -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
    "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}/models"
  report $? "OpenRouter"
else
  echo "  - OpenRouter key not set"
fi

if [ -n "${TUNNEL_BASE_URL:-}" ] && [ -n "${TUNNEL_OPENAI_KEY:-}" ]; then
  curl -fsS -o /dev/null -m 8 \
    -H "Authorization: Bearer ${TUNNEL_OPENAI_KEY}" \
    "${TUNNEL_BASE_URL%/v1}/healthz"
  report $? "Tunnel /healthz"
else
  echo "  - Tunnel not configured"
fi

echo
echo "=== Fixtures ==="
[ -f storage/habit_memory.db ];           report $? "habit_memory.db"
[ -d storage/memories ];                  report $? "chroma memories/"
[ -s storage/llm_cache.json ];            report $? "llm_cache.json non-empty"
[ -s storage/embedding_cache.json ];      report $? "embedding_cache.json non-empty"

echo
echo "=== Videos ==="
shopt -s nullglob
mp4_count=$(ls demo_videos/*.mp4 2>/dev/null | wc -l)
[ "$mp4_count" -ge 1 ]
report $? "demo_videos/ has $mp4_count mp4 file(s)"

compgen -G "recordings/*.mp4" >/dev/null
report $? "recordings/ has fallback mp4"

echo
echo "── result: $PASS passed, $FAIL failed ──"
[ "$FAIL" = "0" ]
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/verify.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/verify.sh
git commit -m "build: add verify.sh preflight check"
```

---

### Task 21: `scripts/run_demo.sh`

**Files:**
- Create: `scripts/run_demo.sh`

- [ ] **Step 1: Write**

Create `scripts/run_demo.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate

PORT=8501
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=8502
fi

# Disable screen blank during demo
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true

(sleep 3 && xdg-open "http://localhost:$PORT/" >/dev/null 2>&1) &

streamlit run simulator/app.py \
  --server.port=$PORT \
  --server.headless=false \
  --browser.gatherUsageStats=false
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/run_demo.sh
git add scripts/run_demo.sh
git commit -m "build: add run_demo.sh launcher"
```

---

### Task 22: `scripts/reset_demo.sh` + idempotency test

**Files:**
- Create: `scripts/reset_demo.sh`
- Create: `tests/test_demo_reset_idempotent.py`

- [ ] **Step 1: Write the reset script**

Create `scripts/reset_demo.sh`:

```bash
#!/usr/bin/env bash
# Reset demo state to fixtures/. Idempotent: running twice yields same state.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Stop any running streamlit
pkill -f "streamlit run simulator/app.py" 2>/dev/null || true
sleep 1

# Wipe runtime state
rm -f storage/habit_memory.db \
      storage/habit_memory.db-wal \
      storage/habit_memory.db-shm \
      storage/habit_memory.db-journal
rm -rf storage/memories

# Restore fixtures
cp -f fixtures/habit_memory.db storage/habit_memory.db
cp -r fixtures/chroma_memories storage/memories
cp -f fixtures/llm_cache.json storage/llm_cache.json
cp -f fixtures/embedding_cache.json storage/embedding_cache.json

echo "✓ demo state reset. Restart with: bash scripts/run_demo.sh"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/reset_demo.sh`

- [ ] **Step 3: Write idempotency test**

Create `tests/test_demo_reset_idempotent.py`:

```python
"""Verify scripts/reset_demo.sh produces identical state across runs."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _hash_tree(p: Path) -> str:
    """Hash all files under p, sorted by relative path."""
    h = hashlib.sha256()
    if not p.exists():
        return "missing"
    for root, dirs, files in os.walk(p):
        dirs.sort()
        for name in sorted(files):
            full = Path(root) / name
            rel = full.relative_to(p)
            h.update(str(rel).encode())
            h.update(full.read_bytes())
    return h.hexdigest()


@pytest.fixture
def fixtures_present():
    fx = ROOT / "fixtures"
    if not (fx / "habit_memory.db").exists():
        pytest.skip("fixtures/habit_memory.db not present yet")
    return fx


def test_reset_is_idempotent(fixtures_present, tmp_path):
    # Run reset
    subprocess.check_call(["bash", str(ROOT / "scripts" / "reset_demo.sh")])
    state1_db = (ROOT / "storage" / "habit_memory.db").read_bytes()
    state1_mem = _hash_tree(ROOT / "storage" / "memories")
    state1_cache = (ROOT / "storage" / "llm_cache.json").read_bytes()

    # Mutate state to simulate a demo run
    (ROOT / "storage" / "habit_memory.db").write_bytes(b"corrupted")
    shutil.rmtree(ROOT / "storage" / "memories", ignore_errors=True)

    # Run reset again
    subprocess.check_call(["bash", str(ROOT / "scripts" / "reset_demo.sh")])
    state2_db = (ROOT / "storage" / "habit_memory.db").read_bytes()
    state2_mem = _hash_tree(ROOT / "storage" / "memories")
    state2_cache = (ROOT / "storage" / "llm_cache.json").read_bytes()

    assert state1_db == state2_db, "habit_memory.db differs between runs"
    assert state1_mem == state2_mem, "chroma memories tree differs"
    assert state1_cache == state2_cache, "llm_cache differs"
```

- [ ] **Step 4: Skip-locally execution**

The test will skip if fixtures aren't yet seeded (Task 25 builds fixtures). Run:

```bash
pytest tests/test_demo_reset_idempotent.py -v
```
Expected: skipped (or passing, once fixtures exist).

- [ ] **Step 5: Commit**

```bash
git add scripts/reset_demo.sh tests/test_demo_reset_idempotent.py
git commit -m "build: add reset_demo.sh and idempotency test"
```

---

### Task 23: `scripts/download_wheels.sh`

**Files:**
- Create: `scripts/download_wheels.sh`

- [ ] **Step 1: Write the script**

Create `scripts/download_wheels.sh`:

```bash
#!/usr/bin/env bash
# Run on the dev desktop to populate wheels/ for offline laptop install.
# Target laptop must run the same Python (3.12) and same OS family (Ubuntu x86_64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p wheels
pip download \
  -r requirements-lock.txt \
  -d wheels/ \
  --platform manylinux2014_x86_64 \
  --python-version 3.12 \
  --only-binary=:all:
echo "✓ wheels/ populated. Total: $(ls wheels/ | wc -l) files, $(du -sh wheels/ | cut -f1)"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/download_wheels.sh
git add scripts/download_wheels.sh
git commit -m "build: add download_wheels.sh for offline-install bundle"
```

---

## Phase 9 — Fixtures (frozen demo state) + cache warmup

### Task 24: `scripts/seed_demo_data.py` (helper invoked by setup/reset)

**Files:**
- Create: `scripts/seed_demo_data.py`

- [ ] **Step 1: Write the helper**

Create `scripts/seed_demo_data.py`:

```python
"""Idempotent helper to copy fixtures/ → storage/. Used by setup.sh / reset_demo.sh."""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "fixtures"
ST = ROOT / "storage"


def seed() -> None:
    ST.mkdir(exist_ok=True)
    shutil.copy2(FX / "habit_memory.db", ST / "habit_memory.db")
    if (ST / "memories").exists():
        shutil.rmtree(ST / "memories")
    shutil.copytree(FX / "chroma_memories", ST / "memories")
    shutil.copy2(FX / "llm_cache.json", ST / "llm_cache.json")
    shutil.copy2(FX / "embedding_cache.json", ST / "embedding_cache.json")
    print("✓ seeded storage/ from fixtures/")


if __name__ == "__main__":
    seed()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/seed_demo_data.py
git commit -m "build: add scripts/seed_demo_data.py helper"
```

---

### Task 25: Build the initial `fixtures/` from a live dry run

**Files:**
- Create: `fixtures/habit_memory.db` (binary, generated)
- Create: `fixtures/chroma_memories/` (directory, generated)
- Create: `fixtures/llm_cache.json` (generated)
- Create: `fixtures/embedding_cache.json` (generated)

- [ ] **Step 1: Wipe local state**

Run:
```bash
rm -rf storage/habit_memory.db storage/memories storage/llm_cache.json storage/embedding_cache.json
mkdir -p storage
```

- [ ] **Step 2: Set cache to read-write for warmup**

Run:
```bash
export LLM_CACHE_READ_ONLY=
```

- [ ] **Step 3: Run the canonical demo flow once with live LLM**

Run:
```bash
streamlit run simulator/app.py --server.port=8501
```

Then in the browser go through the **scripted demo path end-to-end** (Tab 1 inputs → habit detection → Tab 2 recommendations). Click every button the live demo will use. This populates `storage/habit_memory.db`, `storage/memories/`, and writes through to `storage/llm_cache.json` + `storage/embedding_cache.json`.

- [ ] **Step 4: Stop streamlit, copy state into `fixtures/`**

Run:
```bash
mkdir -p fixtures
cp storage/habit_memory.db fixtures/habit_memory.db
rm -rf fixtures/chroma_memories
cp -r storage/memories fixtures/chroma_memories
cp storage/llm_cache.json fixtures/llm_cache.json
cp storage/embedding_cache.json fixtures/embedding_cache.json
```

- [ ] **Step 5: Verify fixtures complete + idempotency test now passes**

Run:
```bash
ls -la fixtures/
pytest tests/test_demo_reset_idempotent.py -v
```
Expected: 4 fixture artifacts present; test passes (no longer skipped).

- [ ] **Step 6: Commit**

```bash
git add fixtures/
git commit -m "fixtures: seed frozen demo state captured from live dry run"
```

---

### Task 26: `scripts/warmup_llm_cache.py` — re-warm after prompt edits

**Files:**
- Create: `scripts/warmup_llm_cache.py`

- [ ] **Step 1: Write the runner**

Create `scripts/warmup_llm_cache.py`:

```python
"""Re-warm the LLM cache by replaying the canonical demo prompts.

Run this whenever prompt templates or signal scripts change, then re-snapshot
fixtures/ via:
    rm -rf fixtures/chroma_memories storage/memories
    bash scripts/reset_demo.sh   # not yet — first refresh fixtures
    cp -r storage/memories fixtures/chroma_memories  (etc)

Goal: every prompt the scripted demo will ever issue ends up in
storage/llm_cache.json so emergency cache mode covers 100% of demo path.
"""
from __future__ import annotations
import os
os.environ.setdefault("LLM_CACHE_READ_ONLY", "")  # ensure writes flow

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.habit_engine import HabitDemoEngine
from panoramix_core.llm_client import LLMClient
from scenarios.mock_data_generator import MOCK_DEMO_SIGNALS  # noqa: F401 — guard
# NOTE: replace the import above with the actual scripted-signal source used
# by the demo. The list of (signal, ts) tuples drives the warmup.

USERNAME = "demo_driver"


def main() -> None:
    llm = LLMClient()
    engine = HabitDemoEngine(username=USERNAME, llm_client=llm)

    # Replay scripted Tab 1 signal stream
    for sig, ts in MOCK_DEMO_SIGNALS:
        engine.ingest_signal(sig, timestamp=ts)
    engine.run_clustering_and_synthesize()

    # Replay scripted Tab 2 recommendation triggers
    for ctx in engine.scripted_recommendation_contexts():
        engine.recommend(ctx)

    print("✓ warmup complete; storage/llm_cache.json populated")


if __name__ == "__main__":
    main()
```

> **NOTE for engineer:** The actual symbol names `MOCK_DEMO_SIGNALS`,
> `engine.ingest_signal`, `engine.run_clustering_and_synthesize`,
> `engine.scripted_recommendation_contexts`, `engine.recommend` may differ in
> the current codebase — read `engine/habit_engine.py` and the existing
> Tab 1 / Tab 2 controllers, then adapt the calls so this script reproduces the
> exact end-to-end flow demoed in the UI. Keep the prompts byte-identical to
> the live UI run (same temperature, same ordering).

- [ ] **Step 2: Run the warmup**

Run:
```bash
LLM_CACHE_READ_ONLY= python scripts/warmup_llm_cache.py
```
Expected: cache file size grows; final line `✓ warmup complete`.

- [ ] **Step 3: Re-capture fixtures**

Run:
```bash
cp storage/llm_cache.json fixtures/llm_cache.json
cp storage/embedding_cache.json fixtures/embedding_cache.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/warmup_llm_cache.py fixtures/llm_cache.json fixtures/embedding_cache.json
git commit -m "build: add warmup script and refresh fixture caches"
```

---

## Phase 10 — Documentation

### Task 27: `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:

````markdown
# Habit Memory Demo — PC Simulator

Streamlit app demoing context-aware behavioral clustering + proactive
recommendation for an automotive cockpit. Tab 1 ingests signals and learns
habits via DBSCAN over hybrid (text + context) embeddings; Tab 2 surfaces
recommendation cards driven by a learned user profile.

## Prerequisites

- Ubuntu 22.04 LTS (or compatible)
- Python 3.12 and `python3.12-venv`
- A `wheels/` directory bundled with this distribution (offline pip install)

## 5-step open-box install

```bash
# 1. Copy the demo pack from USB
cp -r /media/usb/habit-memory-demo-2026-05-XX ~/

# 2. Install offline (no pypi)
cd ~/habit-memory-demo-2026-05-XX
bash scripts/setup.sh

# 3. Configure keys
cp .env.example .env
nano .env

# 4. Preflight
bash scripts/verify.sh

# 5. Run
bash scripts/run_demo.sh
```

## Backup chain

LLM and embedding calls are routed through a fallback chain:

```
OpenAI (5s connect / 15s read)
  └─→ OpenRouter (5s/15s)
        └─→ Home cloudflared tunnel (5s/10s)
              └─→ local cache (instant)
```

Successful calls write through to `storage/llm_cache.json` opportunistically.
Hidden emergency switch: append `?cache_only=1` to the Streamlit URL to skip
all providers and serve from cache only. Sidebar footer shows `live ✓` or
`CACHE ⚠` accordingly.

## Project layout

- `panoramix_core/` — habit detection / embedding / LLM core
- `engine/` — habit lifecycle and proactive executor
- `simulator/` — Streamlit UI
- `home_proxy/` — FastAPI proxy deployed on the dev desktop (NOT on demo laptop)
- `fixtures/` — frozen demo state, restored by `scripts/reset_demo.sh`
- `demo_videos/` — pre-rendered storyboard mp4s
- `recordings/` — full-flow demo recording (final fallback)
- `scripts/` — setup / verify / run / reset / warmup utilities

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3.12: command not found` | `sudo apt-get install python3.12 python3.12-venv` |
| `setup.sh` says wheels/ missing | This pack was not built for offline install. Use a complete pack. |
| Streamlit says port 8501 in use | `run_demo.sh` auto-falls back to 8502; open the URL it prints. |
| `verify.sh` fails on OpenAI | check `OPENAI_API_KEY` in `.env`; on slow Wi-Fi, retry once |
| `verify.sh` fails on Tunnel /healthz | desktop home proxy not running; restart `home_proxy/start.sh` and update `.env` |
| Demo state looks corrupted | `bash scripts/reset_demo.sh` |
| All LLM providers down | indicator turns `CACHE ⚠`; scripted path still works from cache |
| Total disaster | open `recordings/full_demo_<date>.mp4` and narrate |

See `README_DEMO_RUNBOOK.md` for day-of operational checklist.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with 5-step install and troubleshooting"
```

---

### Task 28: `README_DEMO_RUNBOOK.md`

**Files:**
- Create: `README_DEMO_RUNBOOK.md`

- [ ] **Step 1: Write runbook**

Create `README_DEMO_RUNBOOK.md`:

```markdown
# Demo Day Runbook — Renault on-site

> **Demo length target:** 5 minutes. Multi-round, sequential audiences.
>
> **Carry list:** Demo laptop · 2× USB stick (full pack) · Power adapter +
> EU plug converter · USB-C ↔ HDMI / VGA / DP adapters · Phone with 4G
> hotspot · Charged earphones (audio for video clips, if needed).

## T-24h (hotel, evening before)

- [ ] `bash scripts/verify.sh` — all rows ✓
- [ ] `curl -fsS -m 8 -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_BASE_URL/models` — 200
- [ ] WeChat the desktop owner: start `home_proxy/start.sh`, copy two lines into laptop `.env`
- [ ] `curl -fsS -m 8 -H "Authorization: Bearer $TUNNEL_OPENAI_KEY" $TUNNEL_BASE_URL/healthz` — `{"ok":true}`
- [ ] Full dry-run end-to-end (≤5 min) — confirm `live ✓` indicator
- [ ] **Recording present**: `ls -lh recordings/full_demo_*.mp4` — file size > 50 MB. **If missing, record tonight.**
- [ ] Spare USB also has full pack
- [ ] Laptop charged + charger packed

## T-1h (venue)

- [ ] Plug in power. Disable sleep:
      `gsettings set org.gnome.desktop.session idle-delay 0`
- [ ] Connect to venue Wi-Fi. **Open browser to any HTTP site to clear captive portal.**
- [ ] Re-run `bash scripts/verify.sh` — all rows ✓
- [ ] If tunnel row failed, WeChat the desktop owner to restart `home_proxy/start.sh`
      and paste the new URL into `.env`
- [ ] Open two browser tabs:
      - `http://localhost:8501/`
      - `http://localhost:8501/?cache_only=1`
- [ ] Confirm `recordings/full_demo_*.mp4` plays (single click → opens VLC/mpv)
- [ ] Quit Slack / WeChat / mail clients. Disable system notifications.
- [ ] Project to screen. Confirm resolution.

## During each presentation (~5 min)

1. `bash scripts/reset_demo.sh` (≤30 s)
2. Browser: hard reload Tab 1 of the live URL (Ctrl+Shift+R)
3. Sidebar should show `live ✓` (lower left, dim grey)
4. Walk the scripted path: Tab 1 → habits emerge → Tab 2 → recommendations → click `morning_commute.mp4` → arrive home video
5. **If any LLM call shows >8 s spinner OR sidebar flips to `CACHE ⚠`:**
   click address bar → append `?cache_only=1` → Enter. Indicator goes amber.
   Continue script. Audience will not notice.
6. **If browser/streamlit crashes:** Ctrl+C in terminal, `bash scripts/run_demo.sh`
   to relaunch (≤10 s), reset, continue.
7. **If everything fails:** double-click `recordings/full_demo_<date>.mp4`,
   narrate live.

## Between presentations

- `bash scripts/reset_demo.sh` (kills streamlit, restores fixtures)
- `bash scripts/run_demo.sh` (relaunch)
- 30-second delta between back-to-back groups.

## Recording fallback (FINAL SAFETY NET)

> ⚠️ The mp4 in `recordings/` is the last line of defense.
>
> - File path: `recordings/full_demo_<date>.mp4`
> - Must contain the complete 5-minute scripted flow with key visual pauses.
> - Stored in **two places**: laptop disk **and** USB stick.
> - **T-24h checklist row 6 is mandatory** — confirm the file exists, size >50 MB, plays.
> - If on arrival in France the recording is missing, **record it that evening**
>   in the hotel — do not push it to demo morning.

## Failure mode quick-card

| Symptom | Action |
|---|---|
| OpenAI single timeout | Wait ~5 s — router auto-fails over. No action. |
| OpenAI key locked | Auto-falls to OpenRouter. Check dashboard after demo. |
| All 3 providers down | Indicator goes `CACHE ⚠`. Scripted path keeps working. |
| Streamlit crash | Ctrl+C → `bash scripts/run_demo.sh` (≤10 s). |
| Laptop hangs | Reboot. State persists in `fixtures/`; relaunch to recover. |
| Total failure | Play `recordings/full_demo_*.mp4`, narrate. |
```

- [ ] **Step 2: Commit**

```bash
git add README_DEMO_RUNBOOK.md
git commit -m "docs: add Demo Day runbook with multi-round reset procedure"
```

---

### Task 29: `docs/DEMO_BACKUP_PLAN.md` (stakeholder version)

**Files:**
- Create: `docs/DEMO_BACKUP_PLAN.md`

- [ ] **Step 1: Write the doc**

Create `docs/DEMO_BACKUP_PLAN.md`:

```markdown
# Demo Backup Plan — Stakeholder Summary

**For:** Internal demo dry-run reviewers and the on-site team.

## What can go wrong

The demo depends on calling OpenAI for habit detection and embeddings, on
public Wi-Fi at the Renault venue, and on a laptop transported from China to
France. Three failure surfaces matter:

1. **LLM provider:** OpenAI key throttled, blocked, or unreachable.
2. **Network:** captive portal, slow Wi-Fi, blocked outbound HTTPS.
3. **Hardware/state:** crash, projector mismatch, between-show state corruption.

## How we handle each

### LLM provider failure

LLM and embedding calls walk a 4-tier fallback chain:

1. **OpenAI direct** (primary) — 5 s connect / 15 s read.
2. **OpenRouter** — same model via translation map, 5 s / 15 s.
3. **Home tunnel** — cloudflared tunnel back to the dev desktop in China,
   which proxies to OpenAI from there. 5 s / 10 s.
4. **Local cache** — every successful prior call is replayed instantly.

Audience sees no UI difference; the sidebar footer shows a small dim
`live ✓` (normal) or `CACHE ⚠` (degraded). The presenter can force cache-only
mode by appending `?cache_only=1` to the URL — invisible to the audience.

### Network failure

- Per-call timeouts cap any single hang at ~20 s before falling over.
- Demo videos are pre-rendered offline (`demo_videos/*.mp4`), no live calls.
- Cache covers the entire scripted demo path: even with full network outage
  the scripted demo runs end-to-end from local replay.

### Hardware/state failure

- `scripts/reset_demo.sh` restores the demo state from `fixtures/` in 30 s,
  so back-to-back audiences see identical demos.
- A complete recording (`recordings/full_demo_<date>.mp4`) is the final
  fallback if everything fails. Stored on laptop disk and USB stick.

## What's portable

The entire demo pack is a self-contained folder copied via USB:

- Source code
- Pinned dependency wheels for offline install (no pypi)
- Frozen demo state (DB + ChromaDB + LLM cache)
- Pre-rendered videos
- Backup recording

A fresh Ubuntu 22.04 laptop runs the demo in 5 commands and ~15 minutes.

## Three-week schedule

- **Week 1 (5/3 – 5/9):** backup chain implementation
- **Week 2 (5/10 – 5/16):** packaging + dry-run on a borrowed laptop
- **Week 3 (5/17 – 5/23):** travel to France, on-site adaptation, demo
```

- [ ] **Step 2: Commit**

```bash
git add docs/DEMO_BACKUP_PLAN.md
git commit -m "docs: add stakeholder-readable backup plan summary"
```

---

## Phase 11 — Final integration & dry-run

### Task 30: `Makefile` convenience wrapper

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write Makefile**

Create `Makefile`:

```makefile
.PHONY: install verify run reset wheels warmup test clean

install:
	bash scripts/setup.sh

verify:
	bash scripts/verify.sh

run:
	bash scripts/run_demo.sh

reset:
	bash scripts/reset_demo.sh

wheels:
	bash scripts/download_wheels.sh

warmup:
	LLM_CACHE_READ_ONLY= python scripts/warmup_llm_cache.py

test:
	pytest tests/ -x -q --ignore=tests/test_kling_omnivideo.py --ignore=tests/test_storyboard_generation.py --ignore=tests/test_embedding_distance.py --ignore=tests/test_eps_sweep.py

clean:
	rm -rf .venv storage/habit_memory.db* storage/memories storage/llm_cache.json storage/embedding_cache.json
```

- [ ] **Step 2: Smoke test**

Run: `make test`
Expected: green (or skipped where fixtures absent).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile convenience wrapper"
```

---

### Task 31: Build `wheels/` on dev desktop

**Files:**
- Create: `wheels/` (directory of `.whl` files, ~200–500 MB total)

- [ ] **Step 1: Run downloader**

Run:
```bash
bash scripts/download_wheels.sh
```
Expected: `wheels/` directory populated; final line shows file count and total size.

- [ ] **Step 2: Verify offline install in throwaway venv**

Run:
```bash
python3.12 -m venv /tmp/_offline_test
/tmp/_offline_test/bin/pip install --no-index --find-links wheels/ -r requirements-lock.txt
/tmp/_offline_test/bin/python -c "import streamlit, openai, chromadb, fastapi; print('ok')"
rm -rf /tmp/_offline_test
```
Expected: `ok`.

- [ ] **Step 3: Decide whether to commit wheels/**

`wheels/` is large. Two options:

- **Option A:** add it to `.gitignore`; rebuild on each pack via `make wheels`.
- **Option B:** commit it (rejected — bloats repo).

Choose **A**:

```bash
echo "" >> .gitignore
echo "# Offline pip install wheels (rebuilt by scripts/download_wheels.sh)" >> .gitignore
echo "wheels/" >> .gitignore
git add .gitignore
git commit -m "build: ignore generated wheels/ directory"
```

---

### Task 32: Borrowed-laptop dry-run

**Files:** none (verification only)

- [ ] **Step 1: Build the demo pack on dev desktop**

Run:
```bash
PACK_NAME="habit-memory-demo-$(date +%Y-%m-%d)"
PACK_DIR="/tmp/$PACK_NAME"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"
rsync -a --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' \
      --exclude=.pytest_cache ./ "$PACK_DIR/"
# Confirm wheels/ is included
ls -d "$PACK_DIR/wheels" || { echo "wheels/ missing — run make wheels first"; exit 1; }
# Confirm fixtures present
ls "$PACK_DIR/fixtures/habit_memory.db"
# Confirm recordings present (warn but allow if missing yet)
ls "$PACK_DIR"/recordings/*.mp4 2>/dev/null || echo "⚠ no recording yet"
echo "Pack ready at $PACK_DIR; size: $(du -sh $PACK_DIR | cut -f1)"
```

- [ ] **Step 2: Copy to USB and to borrowed laptop**

Manually transfer `$PACK_DIR` to USB stick, then to the borrowed Ubuntu 22.04 laptop's home directory.

- [ ] **Step 3: On the borrowed laptop, run install + verify**

```bash
cd ~/habit-memory-demo-<date>
bash scripts/setup.sh
cp .env.example .env
# fill in real keys
nano .env
bash scripts/verify.sh
```
Expected: all `verify.sh` rows ✓.

- [ ] **Step 4: Run the demo end-to-end**

```bash
bash scripts/run_demo.sh
```

Walk the scripted path. Then test the three failure modes:

- **Test 1 — emergency switch:** append `?cache_only=1` to URL, repeat scripted path. Sidebar should show `CACHE ⚠`. All scripted prompts return from cache.
- **Test 2 — OpenAI down:** `unset OPENAI_API_KEY` (or `export OPENAI_API_KEY=invalid`), restart, walk path. Should see fallback to OpenRouter (logs in terminal).
- **Test 3 — full network down:** disable Wi-Fi on the borrowed laptop. Indicator stays `live ✓` until first call fails over to cache (eventually). Scripted path completes from cache.

- [ ] **Step 5: Test reset between rounds**

```bash
bash scripts/reset_demo.sh
bash scripts/run_demo.sh
# Walk path again — should be byte-identical demo state
```

- [ ] **Step 6: Note any failures, fix, repeat dry-run.**

(no commit step — this task is verification)

---

## Self-review notes

This plan covers spec sections 1–9 (architecture, packaging, scripts, runbook, risks, schedule, testing, deliverables) with the following caveats:

- **Risk Register coverage:** Tasks address risks 1, 2, 6 (router/cache/timeout); 9, 11, 12 (carry list in runbook); 15, 17 (Python/Chroma pinning); 21, 22 (reset idempotency); 30 (`.env` not in pack); 33 (USB strategy lives in runbook). Risks 3, 4', 5, 13, 19, 20, 24, 25–29, 34–36 are operational/logistic — they live in the runbook and stakeholder doc rather than code.
- **Time-budget alignment:** Phase 1–9 ≈ Week 1 (engineering main body); Phase 10–11 ≈ Week 2 (docs + dry-run + wheels). Week 3 in France is operational only.
- **Recording mp4** is intentionally the user's responsibility (per design spec); the runbook makes it explicit and the verify script checks for the file's presence.
