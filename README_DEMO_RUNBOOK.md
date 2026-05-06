# Demo Day Runbook — Renault on-site

> **Demo length target:** 5 minutes. Multi-round, sequential audiences.
>
> **Carry list:** Demo laptop · 2× USB stick (full pack) · Power adapter +
> EU plug converter · USB-C ↔ HDMI / VGA / DP adapters · Phone with 4G
> hotspot · Charged earphones (audio for video clips, if needed).

## T-24h (hotel, evening before)

- [ ] `bash scripts/verify.sh` — all rows ✓
- [ ] `curl -fsS -m 8 -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_BASE_URL/models` — 200
- [ ] WeChat the desktop owner: start `home_proxy/start.sh`, copy two lines into laptop `.env`
- [ ] `curl -fsS -m 8 -H "Authorization: Bearer $TUNNEL_OPENAI_KEY" $TUNNEL_BASE_URL/healthz` — `{"ok":true}`
- [ ] Full dry-run end-to-end (≤5 min) — confirm `live ✓` indicator
- [ ] **Recording present**: `ls -lh recordings/full_demo_*.mp4` — file size > 50 MB. **If missing, record tonight.**
- [ ] Spare USB also has full pack
- [ ] Laptop charged + charger packed

## T-1h (venue)

- [ ] Plug in power. Disable sleep:
      `gsettings set org.gnome.desktop.session idle-delay 0`
- [ ] Connect to venue Wi-Fi. **Open browser to any HTTP site to clear captive portal.**
- [ ] Re-run `bash scripts/verify.sh` — all rows ✓
- [ ] If tunnel row failed, WeChat the desktop owner to restart `home_proxy/start.sh`
      and paste the new URL into `.env`
- [ ] Open two browser tabs:
      - `http://localhost:8501/`
      - `http://localhost:8501/?cache_only=1`
- [ ] Confirm `recordings/full_demo_*.mp4` plays (single click → opens VLC/mpv)
- [ ] Quit Slack / WeChat / mail clients. Disable system notifications.
- [ ] Project to screen. Confirm resolution.

## During each presentation (~5 min)

1. `bash scripts/reset_demo.sh` (≤30 s)
2. Browser: hard reload Tab 1 of the live URL (Ctrl+Shift+R)
3. Sidebar should show `live ✓` (lower left, dim grey)
4. Walk the scripted path: Tab 1 → habits emerge → Tab 2 → recommendations → click `morning_commute.mp4` → arrive home video
5. **If any LLM call shows >8 s spinner OR sidebar flips to `CACHE ⚠`:**
   click address bar → append `?cache_only=1` → Enter. Indicator goes amber.
   Continue script. Audience will not notice.
6. **If browser/streamlit crashes:** Ctrl+C in terminal, `bash scripts/run_demo.sh`
   to relaunch (≤10 s), reset, continue.
7. **If everything fails:** double-click `recordings/full_demo_<date>.mp4`,
   narrate live.

## Between presentations

- `bash scripts/reset_demo.sh` (kills streamlit, restores fixtures)
- `bash scripts/run_demo.sh` (relaunch)
- 30-second delta between back-to-back groups.

## Recording fallback (FINAL SAFETY NET)

> ⚠️ The mp4 in `recordings/` is the last line of defense.
>
> - File path: `recordings/full_demo_<date>.mp4`
> - Must contain the complete 5-minute scripted flow with key visual pauses.
> - Stored in **two places**: laptop disk **and** USB stick.
> - **T-24h checklist row 6 is mandatory** — confirm the file exists, size >50 MB, plays.
> - If on arrival in France the recording is missing, **record it that evening**
>   in the hotel — do not push it to demo morning.

## Failure mode quick-card

| Symptom | Action |
|---|---|
| OpenAI single timeout | Wait ~5 s — router auto-fails over. No action. |
| OpenAI key locked | Auto-falls to OpenRouter. Check dashboard after demo. |
| All 3 providers down | Indicator goes `CACHE ⚠`. Scripted path keeps working. |
| Streamlit crash | Ctrl+C → `bash scripts/run_demo.sh` (≤10 s). |
| Laptop hangs | Reboot. State persists in `fixtures/`; relaunch to recover. |
| Total failure | Play `recordings/full_demo_*.mp4`, narrate. |
