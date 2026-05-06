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
