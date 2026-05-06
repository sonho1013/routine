# Demo Backup Plan — Stakeholder Summary

**For:** Internal demo dry-run reviewers and the on-site team.

## What can go wrong

The demo depends on calling OpenAI for habit detection and embeddings, on
public Wi-Fi at the Renault venue, and on a laptop transported from China to
France. Three failure surfaces matter:

1. **LLM provider:** OpenAI key throttled, blocked, or unreachable.
2. **Network:** captive portal, slow Wi-Fi, blocked outbound HTTPS.
3. **Hardware/state:** crash, projector mismatch, between-show state corruption.

## How we handle each

### LLM provider failure

LLM and embedding calls walk a 4-tier fallback chain:

1. **OpenAI direct** (primary) — 5 s connect / 15 s read.
2. **OpenRouter** — same model via translation map, 5 s / 15 s.
3. **Home tunnel** — cloudflared tunnel back to the dev desktop in China,
   which proxies to OpenAI from there. 5 s / 10 s.
4. **Local cache** — every successful prior call is replayed instantly.

Audience sees no UI difference; the sidebar footer shows a small dim
`live ✓` (normal) or `CACHE ⚠` (degraded). The presenter can force cache-only
mode by appending `?cache_only=1` to the URL — invisible to the audience.

### Network failure

- Per-call timeouts cap any single hang at ~20 s before falling over.
- Demo videos are pre-rendered offline (`demo_videos/*.mp4`), no live calls.
- Cache covers the entire scripted demo path: even with full network outage
  the scripted demo runs end-to-end from local replay.

### Hardware/state failure

- `scripts/reset_demo.sh` restores the demo state from `fixtures/` in 30 s,
  so back-to-back audiences see identical demos.
- A complete recording (`recordings/full_demo_<date>.mp4`) is the final
  fallback if everything fails. Stored on laptop disk and USB stick.

## What's portable

The entire demo pack is a self-contained folder copied via USB:

- Source code
- Pinned dependency wheels for offline install (no pypi)
- Frozen demo state (DB + ChromaDB + LLM cache)
- Pre-rendered videos
- Backup recording

A fresh Ubuntu 22.04 laptop runs the demo in 5 commands and ~15 minutes.

## Three-week schedule

- **Week 1 (5/3 – 5/9):** backup chain implementation
- **Week 2 (5/10 – 5/16):** packaging + dry-run on a borrowed laptop
- **Week 3 (5/17 – 5/23):** travel to France, on-site adaptation, demo
