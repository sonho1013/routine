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
