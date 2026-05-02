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
