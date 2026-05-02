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
