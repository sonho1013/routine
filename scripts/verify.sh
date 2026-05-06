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
