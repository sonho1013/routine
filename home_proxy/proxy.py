"""Thin OpenAI-compatible proxy for the demo home tunnel.

Forwards `/v1/chat/completions` and `/v1/embeddings` to OpenAI using the
desktop's own OPENAI_API_KEY. Authenticates inbound requests by matching
their Authorization header against TUNNEL_SHARED_SECRET.
"""
from __future__ import annotations

import os
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TUNNEL_SHARED_SECRET = os.environ["TUNNEL_SHARED_SECRET"]
OPENAI_BASE_URL = os.environ.get("OPENAI_UPSTREAM", "https://api.openai.com/v1")

app = FastAPI(title="habit-memory-demo home proxy")
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), trust_env=False)


def _check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != TUNNEL_SHARED_SECRET:
        raise HTTPException(403, "invalid bearer token")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request,
                  authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    upstream = f"{OPENAI_BASE_URL.rstrip('/')}/{path}"
    body = await request.body()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": request.headers.get("Content-Type", "application/json"),
    }
    resp = await client.request(request.method, upstream, content=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)
