# Hosting the PC Simulator on the Public Internet — Feasibility

**Question:** Can we expose this Streamlit app (the Renault demo) at a URL that
outside parties can browse to, rather than only via the demo laptop?

**Short answer:** Yes, easily. Four realistic options, ranked by effort.

---

## Option A — cloudflared quick tunnel (0 effort, unstable URL)

Same tool we already use for the home LLM proxy.

```bash
# on the machine where streamlit is running
streamlit run simulator/app.py --server.port=8501 &
cloudflared tunnel --url http://localhost:8501
# prints  https://<random>.trycloudflare.com
```

**Pros:**
- Free, zero setup, HTTPS included.
- Works from behind NAT without port forwarding.
- Same cloudflared binary we already install for the tunnel backup.

**Cons:**
- URL is random and changes every time you restart.
- No auth — anyone with the URL can use it (and burn your OpenAI quota).
- Trycloudflare free tier is rate-limited, not for production traffic.

**Fit:** one-off sharing with a colleague, or as a stopgap during demo
rehearsal. **Not** for public audience.

---

## Option B — Streamlit Community Cloud (low effort, stable URL, free)

Free hosting for public Streamlit apps at `<app-name>.streamlit.app`.

**How:**

1. Push the repo to a public or private GitHub repo.
2. Sign in at https://streamlit.io/cloud with GitHub.
3. Point it at `simulator/app.py`.
4. Add secrets (OpenAI / OpenRouter keys) in the Streamlit Cloud UI.

**Pros:**
- Stable URL, TLS, CI rebuilds on push.
- Zero cost up to 1 GB memory / 1 vCPU per app.
- Secrets are encrypted at rest.

**Cons & gotchas:**
- 1 GB memory ceiling — our ChromaDB + fixtures + embedding cache are
  ~300 MB. Fits, but tight; heavy Q&A that expands cache could OOM.
- Cold-start is ~20 s.
- Apps shut down after 7 days of no traffic (wakeable).
- Requires the repo to be accessible to Streamlit Cloud. If the repo
  contains anything proprietary (it does — internal signal definitions,
  scenario data), you want the **private** repo flow.
- Fixtures are committed to git (current state), so cloud build restores
  state at deploy time. Users interacting with the cloud app mutate state
  that persists until restart.

**Fit:** if you want "hand Renault a URL to browse at their leisure"
without running our own infra. Probably the right answer for a
post-demo follow-up.

---

## Option C — VPS + Nginx + systemd (medium effort, full control)

Rent a $5–10/mo Linux VPS (DigitalOcean, Vultr, Hetzner). Run Streamlit
behind Nginx with basic auth or an OAuth proxy.

**Skeleton:**

```bash
# on the VPS
sudo apt install python3.12 python3.12-venv nginx certbot
# rsync the pack
rsync -av --delete habit-memory-demo/ root@vps:/srv/habit-memory-demo/

# on the VPS
cd /srv/habit-memory-demo
bash scripts/setup.sh
# write /etc/systemd/system/habit-memory.service  (runs streamlit as a daemon)
# configure nginx reverse proxy to localhost:8501
# certbot --nginx -d demo.yourdomain.com  (Let's Encrypt TLS)
```

**Pros:**
- Your URL (`demo.yourdomain.com`), stable forever.
- No memory ceiling, no sleep-after-7-days.
- Can put basic auth in front for Renault-only access.

**Cons:**
- You own uptime. Disk, memory, TLS renewal, OS patches are yours.
- Renault's corporate network may block random VPS endpoints.
- ~2 hours to set up the first time.

**Fit:** if Renault asks for "a URL we can give to people internally, 24/7".
The Stream Cloud option is easier; VPS is better if you hit memory limits
or want private-network-friendly IP ranges.

---

## Option D — Docker image + managed service (higher effort, portable)

Dockerize the pack and deploy to Fly.io / Railway / Google Cloud Run.

**Sketch:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements-lock.txt wheels/ ./
RUN pip install --no-index --find-links wheels/ -r requirements-lock.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "simulator/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Pros:**
- Clean separation from host OS; reproducible.
- Fly.io / Cloud Run auto-TLS + custom domain built in.
- Pay-per-use pricing; demo traffic is trivial.

**Cons:**
- Docker adds a layer you haven't dealt with yet.
- First-time setup of Fly / Cloud Run IAM is 1–2 hours.
- ChromaDB persistence needs a volume mount — the pattern differs per
  platform.

**Fit:** if this demo evolves into something semi-productized and you
want it alongside other services. Overkill for a one-off Renault demo.

---

## Security considerations (regardless of option)

The app currently has **no authentication** — anyone with the URL can:
- Trigger LLM calls that burn your OpenAI / OpenRouter quota
- Observe cached prompts / responses (no PII today, but stay aware)
- See Mary's synthetic driving data (not sensitive)

Minimum recommended hardening before public exposure:

1. **Nginx basic auth** (Option C) or **Streamlit Cloud private app**
   (Option B) — cheapest gate.
2. Set `LLM_CACHE_READ_ONLY=1` on the public instance so random visitors
   can't poison the cache.
3. Cap provider budgets at the vendor dashboard (OpenAI has per-key spend
   limits, OpenRouter has per-key caps).
4. Do **not** expose the home tunnel — keep `TUNNEL_BASE_URL` blank on
   the public instance. The three-tier fallback collapses to
   OpenAI → OpenRouter → cache, which is fine for web traffic.

---

## My recommendation

For "give Renault a URL they can browse at their own pace": **Option B
(Streamlit Cloud)**. Stable URL, zero-effort secret management, free, and
it forces you to put the source in git properly (which you already have).

For the demo itself (one-shot, you standing next to it): keep the local
laptop plan. Remote-serving adds a failure mode (web network) without
buying anything over the local demo.

If Renault later says "can we put this in our intranet": pivot to
**Option C (VPS)** or let their IT team containerize it (**Option D**).
