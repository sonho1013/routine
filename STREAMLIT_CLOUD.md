# Streamlit Community Cloud Deployment

Public-internet hosting for the Habit Memory Demo — a follow-up channel for
Renault (or anyone with the URL) to browse the same Tab 1 + Tab 2 walkthrough
without us running the laptop.

This is **complementary** to the on-site laptop demo, not a replacement. The
laptop path stays primary; cloud is the leave-behind.

---

## What's different from the laptop deployment

| Concern | Laptop (`DEPLOYMENT.md`) | Streamlit Cloud |
|---|---|---|
| Install | `bash scripts/setup.sh` (offline wheels) | `pip install -r requirements.txt` (Cloud auto) |
| Storage seed | `setup.sh` copies `fixtures/` → `storage/` | `simulator/app.py` does it on first import |
| Secrets | `.env` file | Streamlit Cloud "Secrets" UI (TOML) |
| Tier 3 (home tunnel) | Active | **Disabled** — leave `TUNNEL_BASE_URL` blank |
| LLM cache writes | Live writes allowed | `LLM_CACHE_READ_ONLY=1` recommended |
| Recording fallback | `recordings/full_demo_*.mp4` | N/A |
| Python | 3.12 (apt) | 3.12 (pinned via `runtime.txt`) |

---

## One-time setup

### 1. Create a public GitHub repo

Free tier requires a public repo. The codebase contains no proprietary data
(synthetic Mary scenarios only) — public is OK.

Either install `gh` and run:

```bash
sudo apt install gh        # if not installed
gh auth login              # web flow
gh repo create paranomix/habit-memory-demo --public --source=. --remote=origin
```

Or do it manually via https://github.com/new — create the empty repo, then:

```bash
git remote add origin https://github.com/<you>/habit-memory-demo.git
```

### 2. Push the branch

```bash
# clean up workspace first
git status
# stage what's needed (see "What gets pushed" below)
git push -u origin feature/trigger-action-alignment
# or push as main if you want Streamlit Cloud to default-deploy main
```

### 3. Connect Streamlit Cloud

1. Sign in at https://streamlit.io/cloud with the same GitHub account.
2. **New app** → pick repo → branch → main file: `simulator/app.py`.
3. **Advanced settings → Python version**: `3.12`.
4. **Secrets**: paste the contents of `.streamlit/secrets.toml.example`,
   replacing `sk-...` with real keys. Save.
5. **Deploy**. First boot takes ~5 min (pip install + chromadb wheel build).
6. App URL: `https://<your-app-name>.streamlit.app`.

Pushes to the branch automatically trigger redeploys.

---

## What gets pushed

Repo is ~540 MB tracked. Within Streamlit Cloud's tolerance, but big enough
that the initial clone is slow.

| Path | Size | Why |
|---|---|---|
| `panoramix_core/` `engine/` `simulator/` `scripts/` | small | source |
| `data/four_week_week_case_01/` | 203 MB | Tab 1 four-week signal stream |
| `output/videos/mary/day*/*.mp4` (16 files) | 177 MB | day-level assembled videos |
| `output/videos/mary/day1/<scene>/scene.mp4` (3 files) | 157 MB | Tab 1 day-1 scene clips (the ones the simulator actually loads) |
| `fixtures/` | 5 MB | seeded SQLite + chroma + caches |
| `requirements.txt` `runtime.txt` `.streamlit/` | tiny | Cloud config |

Excluded by `.gitignore`:

- `wheels/` (189 MB) — Cloud uses PyPI, not offline wheels
- `output/videos/*/day*/*/beats/` — rendering intermediates
- `storage/` — runtime, bootstrapped from fixtures
- `recordings/` — laptop-only fallback

---

## Operational notes

- **Sleep after 7 days idle.** Cloud apps spin down without traffic; first
  request after sleep takes ~30 s to wake.
- **1 GB RAM ceiling.** ChromaDB + embedding cache fit comfortably (~300 MB
  resident). Heavy Q&A from random visitors could OOM the cache layer, which
  is why `LLM_CACHE_READ_ONLY=1` is in the secrets template.
- **Provider quota burn.** Anyone with the URL can trigger LLM calls. Cap the
  `OPENAI_API_KEY` and `OPENROUTER_API_KEY` spend at the vendor dashboard
  before sharing the URL.
- **Tunnel tier is unreachable from Cloud.** Don't fill `TUNNEL_BASE_URL`.
  Cloud collapses to OpenAI → OpenRouter → cache.
- **Logs.** Streamlit Cloud surfaces stdout/stderr in the "Manage app" panel.
  Watch for repeated `provider X failed → falling back` lines on first deploy.

---

## Updating after deploy

```bash
git add <changed files>
git commit -m "..."
git push
```

Streamlit Cloud picks up the push within ~30 s and rebuilds. Secrets persist
across rebuilds. To roll back, point Cloud at a previous commit hash via the
"Manage app" UI.
