# Backup Operations — What's Automatic, What's Manual

Quick reference for the 4-tier LLM/embedding fallback chain and the other
safety nets. Use this doc when something seems wrong during the demo and
you need to know whether to intervene or let the system self-heal.

---

## Tier 1 — OpenAI direct (primary)

- **When**: every LLM / embedding call first tries OpenAI with a 5 s connect
  + 15 s read timeout.
- **Failure modes covered**: transient network glitch, OpenAI 5xx, OpenAI
  timeout, OpenAI auth error (bad/expired key), rate-limit.
- **Action on failure**: **automatic** — router silently moves to Tier 2.
  Logs a warning line in the terminal running Streamlit.

**You do nothing.**

---

## Tier 2 — OpenRouter (backup 1)

- **When**: OpenAI failed (any reason above).
- **Model translation**: `gpt-4.1` → `openai/gpt-4-turbo` via
  `config/openrouter_model_map.yaml`. Embeddings not supported on
  OpenRouter — embedding requests skip this tier directly to Tier 3.
- **Requires**: `OPENROUTER_API_KEY` in `.env`.
- **Action on failure**: **automatic** — router moves to Tier 3.

**You do nothing.**

---

## Tier 3 — Home tunnel (backup 2)

- **When**: OpenAI + OpenRouter both failed.
- **How it works**: a FastAPI proxy runs on the dev desktop in China. A
  cloudflared tunnel exposes it as `https://<random>.trycloudflare.com`.
  The demo laptop's `.env` points `TUNNEL_BASE_URL` at that URL, and the
  proxy forwards to OpenAI using the desktop's own key.
- **Requires**:
  - Desktop operator ran `home_proxy/start.sh` that morning
  - Trycloudflare URL + shared secret pasted into laptop's `.env`
  - `curl $TUNNEL_BASE_URL/../healthz` returns `{"ok":true}` during
    `verify.sh`
- **Action on failure**: **automatic** — router moves to Tier 4.

**You do nothing during the demo.** If the whole tier is wrong
(e.g. wrong URL in `.env`) the daily `verify.sh` row will be ✗ —
that's your T-24h / T-1h trigger to WeChat the desktop operator.

---

## Tier 4 — Local cache (final fallback)

- **When**: all three live providers failed.
- **What it serves**: previous successful responses keyed by
  `sha256(model + prompt + temperature)`. `fixtures/llm_cache.json` +
  `fixtures/embedding_cache.json` are pre-seeded with every call the
  scripted demo path issues, so the scripted path is 100% reproducible
  from cache.
- **When it can't serve**: if someone types a **novel prompt** during Q&A
  (not in the scripted path) and all three live tiers are down, the cache
  will miss and raise KeyError. That's the signal to go to the manual
  fallback.
- **Action on failure**: **not automatic** — see next section.

**You do nothing unless the sidebar footer flips to `CACHE ⚠` and a
novel prompt fails.**

---

## Emergency cache-only mode (preemptive)

This is **manual**, activated by you when you anticipate trouble.

- **How to enable**: append `?cache_only=1` to the browser URL and press
  Enter. No UI button — the audience won't see you do it.
- **What changes**: router skips all three live tiers and serves straight
  from cache. Fast (< 1 ms), deterministic, zero external calls.
- **When to use**: any time you see a long spinner (> 8 s), see the
  sidebar footer turn amber (`CACHE ⚠`), or you know the venue Wi-Fi just
  went bad.
- **How to exit**: remove `?cache_only=1` from the URL, hard-reload.

**You flip this proactively the moment you smell trouble.**

---

## Tier 5 — Recording mp4 (absolute final fallback)

- **When**: everything above failed, browser won't load, laptop hangs,
  projector fails mid-demo, etc.
- **Path**: `recordings/full_demo_<date>.mp4` — you record this before
  departure.
- **Action**: double-click the mp4 (opens in your default video player),
  narrate over it.

**Entirely manual.** `verify.sh` checks the file exists but nothing else.

---

## Other automatic self-healing

| What | Covered automatically? | Notes |
|---|---|---|
| Socks proxy scheme translation (`socks://` → `socks5://`) | Yes | At module import time in OpenAIProvider / home_proxy |
| Streamlit port 8501 busy | Yes | `run_demo.sh` falls back to 8502 |
| Screen blank during demo | Yes | `run_demo.sh` calls `xset s off` |
| SQLite WAL leftover between runs | Yes | `reset_demo.sh` deletes it |
| Stale embedding cache after prompt template changes | **No** | You rerun `scripts/warmup_llm_cache.py` + re-snapshot fixtures |
| New scripted question appearing in cache | **No** | Run the new question once with live LLM, cache captures the response, re-snapshot fixtures |

---

## Decision cheatsheet (during demo)

```
                 ┌── single slow LLM call? ─ wait ≤ 20 s, router self-heals
                 │
observed:  ──────┼── sidebar  CACHE ⚠  visible?
                 │        └─ scripted path still works from cache,
                 │           only novel Q&A questions will fail
                 │
                 ├── about to hit novel Q&A on shaky network?
                 │        └─ pre-emptively add  ?cache_only=1  to URL
                 │
                 └── everything broken / browser dead?
                          └─ play  recordings/full_demo_*.mp4
```

## Decision cheatsheet (T-24h)

```
verify.sh rows all ✓?                      → nothing to do
OpenAI row ✗, slow Wi-Fi?                  → retry once
OpenAI row ✗ persistently?                 → check OPENAI_API_KEY validity
                                              + OPENAI_BASE_URL
OpenRouter row ✗?                          → check OPENROUTER_API_KEY
                                              validity at openrouter.ai
Tunnel row ✗?                              → WeChat desktop operator
                                              to restart home_proxy/start.sh
                                              paste new URL into .env
recordings row ✗?                          → record demo tonight
```
