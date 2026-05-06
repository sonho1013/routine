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
