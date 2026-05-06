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
