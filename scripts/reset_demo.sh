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
