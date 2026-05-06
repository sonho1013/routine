# Deployment Guide — Habit Memory Demo

> From a bare Ubuntu 22.04+ laptop to a running demo in **5 commands /
> ~15 minutes**, with zero pypi access required.

---

## 1. Prerequisites on the target laptop

| Requirement | Check command | Install if missing |
|---|---|---|
| Ubuntu 22.04+ (or any x86_64 Linux with glibc ≥ 2.35) | `cat /etc/os-release` | — |
| Python 3.12 | `python3.12 --version` | `sudo apt install python3.12 python3.12-venv` |
| `curl`, `git`, `lsof`, `xset` | `which curl git lsof xset` | `sudo apt install curl git lsof x11-xserver-utils` |
| Chrome or Firefox | — | any browser works |

**Nothing else is required at prerequisite time.** No Anthropic/OpenAI CLI,
no docker, no node, no conda.

---

## 2. Transfer the pack from USB

```bash
# Copy the folder off the USB stick to your home directory
cp -r /media/$USER/<USB_LABEL>/habit-memory-demo-pack-<date> ~/habit-memory-demo
cd ~/habit-memory-demo
```

Expected contents (verify with `ls`):

```
habit-memory-demo/
├── README.md                         short overview
├── README_DEMO_RUNBOOK.md            day-of demo runbook (T-24h/T-1h/during)
├── DEPLOYMENT.md                     THIS FILE
├── BACKUP_OPERATIONS.md              which backup tiers are auto vs manual
├── .env.example                      key template (copy to .env and fill)
├── requirements.txt + requirements-lock.txt
├── wheels/                           ~189 MB offline pip wheels
├── fixtures/                         seeded demo state (db + chroma + cache)
├── output/videos/                    pre-rendered mp4s for the storyboard
├── recordings/                       ⚠ add full-demo mp4 fallback HERE
├── scripts/                          setup/verify/run/reset/warmup
├── home_proxy/                       FastAPI proxy (stays on dev desktop)
├── docs/                             extended docs
├── panoramix_core/                   source
├── engine/                           source
├── simulator/                        source (Streamlit UI)
├── tests/                            test suite
├── scenarios/ · data/ · signals/ · config/   auxiliary modules & mock data
└── Makefile
```

---

## 3. Install (offline)

```bash
bash scripts/setup.sh
```

What this does:

1. Asserts `python3.12` is available
2. Creates `.venv/` using `python3.12 -m venv`
3. Runs `pip install --no-index --find-links wheels/ -r requirements-lock.txt` — **never touches pypi**
4. Restores `fixtures/habit_memory.db`, `fixtures/chroma_memories/`, and
   both cache JSONs into `storage/`

If `setup.sh` fails with "wheels/ missing", the pack is incomplete — ask
for a full re-export.

---

## 4. Configure API keys

```bash
cp .env.example .env
nano .env   # or your editor of choice
```

Fill these four fields. **Keep the rest of the file unchanged.**

```bash
# ── Primary key (required) ──
OPENAI_API_KEY=sk-...

# ── Backup 1: OpenRouter (required for demo-day redundancy) ──
# Get one at https://openrouter.ai/keys — ~5 minutes, credit-card optional,
# $5 free tier covers our demo workload easily.
OPENROUTER_API_KEY=sk-or-...

# ── Backup 2: Home tunnel (filled on demo day, leave blank here) ──
# These two values are printed by `home_proxy/start.sh` running on the
# dev desktop in China. Paste them on the evening before demo.
TUNNEL_BASE_URL=
TUNNEL_OPENAI_KEY=
```

**Do NOT commit `.env` anywhere.** It's in `.gitignore`; keep it that way.

All other variables in `.env.example` are safe defaults — leave them alone
unless you need to tune timeouts.

---

## 5. Preflight

```bash
bash scripts/verify.sh
```

You want to see `✓` on every row:

```
=== Keys ===
  ✓ OpenAI direct
  ✓ OpenRouter
  - Tunnel not configured                ← expected pre-demo-day
=== Fixtures ===
  ✓ habit_memory.db
  ✓ chroma memories/
  ✓ llm_cache.json non-empty
  ✓ embedding_cache.json non-empty
=== Videos ===
  ✓ demo_videos/ has 15 mp4 file(s)
  ✗ recordings/ has fallback mp4          ← add your recording!
── result: X passed, Y failed ──
```

The only row you're expected to see ✗ on is the recording — that's your
responsibility to produce before demo day.

---

## 6. Run the demo

```bash
bash scripts/run_demo.sh
```

Browser auto-opens at `http://localhost:8501/`. Walk Tab 1 → Tab 2.

To reset between back-to-back presentations:

```bash
bash scripts/reset_demo.sh      # 30s, restores fixtures/
bash scripts/run_demo.sh
```

---

## 7. Where things live in the running demo

| Artifact | Path | Regenerable? |
|---|---|---|
| SQLite scene-card DB | `storage/habit_memory.db` | Yes — restored by reset_demo |
| ChromaDB facts + embeddings | `storage/memories/` | Yes — restored by reset_demo |
| LLM response cache | `storage/llm_cache.json` | Yes — restored by reset_demo |
| Embedding vector cache | `storage/embedding_cache.json` | Yes — restored by reset_demo |
| Demo videos | `output/videos/mary/day*/` | No — pre-rendered once |
| Fallback recording | `recordings/full_demo_*.mp4` | No — you record this |
| Logs | terminal stdout + stderr | — |

---

## 8. Quick smoke test of the 4-tier LLM backup chain

```bash
# Force cache-only mode: three providers are skipped, only cache is used
curl -s "http://localhost:8501/?cache_only=1"
# Look at the sidebar footer — should read  CACHE ⚠
# Walk the scripted demo path; cache covers 100% of it.

# Restore normal behavior
# (just remove ?cache_only from the URL and hard-reload)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3.12: command not found` | `sudo apt install python3.12 python3.12-venv` |
| `setup.sh` says "wheels/ missing" | Pack is incomplete — ask for full re-export |
| Streamlit port 8501 busy | `run_demo.sh` auto-falls back to 8502 |
| `verify.sh` OpenAI row ✗ on slow Wi-Fi | retry once; if still ✗, check the key and the `OPENAI_BASE_URL` |
| `verify.sh` Tunnel row ✗ | Tunnel URL only valid while `home_proxy/start.sh` is running on the dev desktop; have the operator restart it and re-paste the printed URL |
| Browser shows `CACHE ⚠` unexpectedly | something in the URL or env turned on force-cache — hard-reload to a clean URL and restart streamlit |
| Multiple presentations in a row | `bash scripts/reset_demo.sh` between each one |
| Total disaster | play `recordings/full_demo_*.mp4` |

See `README_DEMO_RUNBOOK.md` for the full day-of operational checklist.
