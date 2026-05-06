"""Re-warm storage/llm_cache.json + storage/embedding_cache.json by replaying
the canonical demo signal stream through the real Tab 1 engine.

When to run:
- After prompt templates change (any edit to `panoramix_core/clustering/habits_detector.py`
  prompts or signal-rules YAML)
- After the mock dataset (`scenarios/mock_data_generator.py`) changes
- Before re-snapshotting `fixtures/` for a fresh demo pack

What it does:
1. Ensures cache files are writable (clears LLM_CACHE_READ_ONLY)
2. Wipes runtime state in `storage/` so the engine starts cold
3. Generates the 5-day Mary signal stream via `generate_full_dataset()`
4. Feeds the events to `HabitDemoEngine.ingest_signal_batch()`, which triggers
   the LLM rewording + embedding calls; each call writes through to the cache
   on success
5. Prints final cache file sizes

After running, copy the populated caches into `fixtures/`:
    cp storage/llm_cache.json fixtures/llm_cache.json
    cp storage/embedding_cache.json fixtures/embedding_cache.json

Tab 2 recommendation prompts are NOT exercised here. Walk the Tab 2 UI once
in Streamlit during the initial fixtures build (Task 25) to populate those.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force cache to be writable before any panoramix_core import that reads the env var.
os.environ["LLM_CACHE_READ_ONLY"] = ""

from datetime import datetime  # noqa: E402

from engine.habit_engine import HabitDemoEngine  # noqa: E402
from panoramix_core.llm_client import LLMClient  # noqa: E402
from scenarios.mock_data_generator import generate_full_dataset  # noqa: E402

USERNAME = "demo_driver"


def _wipe_runtime_state() -> None:
    storage = ROOT / "storage"
    storage.mkdir(exist_ok=True)
    for f in [
        storage / "habit_memory.db",
        storage / "habit_memory.db-wal",
        storage / "habit_memory.db-shm",
        storage / "habit_memory.db-journal",
    ]:
        if f.exists():
            f.unlink()
    mem = storage / "memories"
    if mem.exists():
        shutil.rmtree(mem)


def main() -> None:
    print("▶ wiping storage/ for cold-start ingestion")
    _wipe_runtime_state()

    print("▶ generating canonical 5-day demo signal stream")
    dataset = generate_full_dataset(base_date=datetime(2026, 4, 1))
    events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset \
        else dataset  # tolerate either shape

    if not isinstance(events, list):
        raise SystemExit(
            f"unexpected dataset shape from generate_full_dataset(): {type(events)!r}"
        )

    print(f"▶ ingesting {len(events)} events through HabitDemoEngine "
          "(this calls live LLM + embedding APIs)")
    llm = LLMClient()
    engine = HabitDemoEngine(username=USERNAME, llm_client=llm)
    result = engine.ingest_signal_batch(events)
    print(f"  ingest result: {result}")

    llm_cache = ROOT / "storage" / "llm_cache.json"
    emb_cache = ROOT / "storage" / "embedding_cache.json"
    print()
    print("✓ warmup complete")
    print(f"  llm_cache.json:       {llm_cache.stat().st_size:>10,} bytes")
    print(f"  embedding_cache.json: {emb_cache.stat().st_size:>10,} bytes")
    print()
    print("Next: snapshot the populated caches into fixtures/")
    print("    cp storage/llm_cache.json fixtures/llm_cache.json")
    print("    cp storage/embedding_cache.json fixtures/embedding_cache.json")


if __name__ == "__main__":
    main()
