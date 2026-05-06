# Home Proxy (Renault demo backup)

This directory is **deployed on the local desktop**, not on the demo laptop.

## One-time setup (desktop)

1. Install cloudflared:
   ```
   sudo apt-get install -y cloudflared
   ```
2. Activate the project venv with FastAPI/uvicorn installed:
   ```
   cd /path/to/habit-memory-demo
   source .venv/bin/activate
   pip install fastapi uvicorn httpx
   ```
3. Create `~/.config/habit-memory-demo.env`:
   ```bash
   export OPENAI_API_KEY=sk-...
   export TUNNEL_SHARED_SECRET=$(openssl rand -hex 24)
   ```

## Demo-day startup (desktop)

```bash
source ~/.config/habit-memory-demo.env
bash home_proxy/start.sh
```

The script prints two lines to copy into the laptop's `.env`:

```
TUNNEL_BASE_URL=https://xxx.trycloudflare.com/v1
TUNNEL_OPENAI_KEY=<the shared secret>
```

Keep this terminal open for the duration of the demo. Re-running the script
yields a **new** trycloudflare URL.
