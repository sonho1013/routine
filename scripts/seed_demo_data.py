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
