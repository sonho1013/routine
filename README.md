# Habit Memory Demo — PC Simulator

Streamlit app demoing context-aware behavioral clustering + proactive
recommendation for an automotive cockpit. Tab 1 ingests signals and learns
habits via DBSCAN over hybrid (text + context) embeddings; Tab 2 surfaces
recommendation cards driven by a learned user profile.

## Prerequisites

- Ubuntu 22.04 LTS (or compatible)
- Python 3.12 and `python3.12-venv`
- A `wheels/` directory bundled with this distribution (offline pip install)

## 5-step open-box install

```bash
# 1. Copy the demo pack from USB
cp -r /media/usb/habit-memory-demo-2026-05-XX ~/

# 2. Install offline (no pypi)
cd ~/habit-memory-demo-2026-05-XX
bash scripts/setup.sh

# 3. Configure keys
cp .env.example .env
nano .env

# 4. Preflight
bash scripts/verify.sh

# 5. Run
bash scripts/run_demo.sh
```

## Backup chain

LLM and embedding calls are routed through a fallback chain:

```
OpenAI (5s connect / 15s read)
  └─→ OpenRouter (5s/15s)
        └─→ Home cloudflared tunnel (5s/10s)
              └─→ local cache (instant)
```

Successful calls write through to `storage/llm_cache.json` opportunistically.
Hidden emergency switch: append `?cache_only=1` to the Streamlit URL to skip
all providers and serve from cache only. Sidebar footer shows `live ✓` or
`CACHE ⚠` accordingly.

## Project layout

- `panoramix_core/` — habit detection / embedding / LLM core
- `engine/` — habit lifecycle and proactive executor
- `simulator/` — Streamlit UI
- `home_proxy/` — FastAPI proxy deployed on the dev desktop (NOT on demo laptop)
- `fixtures/` — frozen demo state, restored by `scripts/reset_demo.sh`
- `demo_videos/` — pre-rendered storyboard mp4s
- `recordings/` — full-flow demo recording (final fallback)
- `scripts/` — setup / verify / run / reset / warmup utilities

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3.12: command not found` | `sudo apt-get install python3.12 python3.12-venv` |
| `setup.sh` says wheels/ missing | This pack was not built for offline install. Use a complete pack. |
| Streamlit says port 8501 in use | `run_demo.sh` auto-falls back to 8502; open the URL it prints. |
| `verify.sh` fails on OpenAI | check `OPENAI_API_KEY` in `.env`; on slow Wi-Fi, retry once |
| `verify.sh` fails on Tunnel /healthz | desktop home proxy not running; restart `home_proxy/start.sh` and update `.env` |
| Demo state looks corrupted | `bash scripts/reset_demo.sh` |
| All LLM providers down | indicator turns `CACHE ⚠`; scripted path still works from cache |
| Total disaster | open `recordings/full_demo_<date>.mp4` and narrate |

See `README_DEMO_RUNBOOK.md` for day-of operational checklist.
