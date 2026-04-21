"""
Kling REST API client — direct, no ComfyUI dependency.

Implements JWT (HS256) auth, image generation (T2I), image-to-video (I2V),
task polling, and result download. Mirrors the request shapes used by
ComfyUI's `nodes_kling.py` so behavior matches the existing workflows.

Endpoints (Kling China):
    POST /v1/images/generations              — submit T2I
    GET  /v1/images/generations/{task_id}    — poll T2I
    POST /v1/videos/image2video              — submit I2V
    GET  /v1/videos/image2video/{task_id}    — poll I2V
"""
import base64
import logging
import os
import time

import jwt
import requests

log = logging.getLogger(__name__)


class KlingError(Exception):
    """Any Kling API failure (auth, validation, task failure, timeout)."""
    pass


class KlingClient:
    """Minimal direct client for Kling REST API."""

    def __init__(self, access_key: str, secret_key: str, base_url: str):
        if not access_key or not secret_key:
            raise KlingError("KLING_ACCESS_KEY / KLING_SECRET_KEY are required")
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # Kling API + CDN are domestic (China) — never route through HTTP_PROXY
        # (a SOCKS/HTTP proxy meant for foreign traffic mangles Chinese-CDN TLS).
        self.session.trust_env = False

    # ── Auth ──

    def _make_token(self) -> str:
        """Build a fresh JWT (valid for 30 minutes)."""
        now = int(time.time())
        payload = {
            "iss": self.access_key,
            "exp": now + 1800,
            "nbf": now - 5,
        }
        return jwt.encode(
            payload,
            self.secret_key,
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT"},
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._make_token()}",
            "Content-Type": "application/json",
        }

    # ── HTTP wrappers ──

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=body, headers=self._headers(), timeout=60)
        if resp.status_code >= 400:
            raise KlingError(f"POST {path} → HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        if data.get("code", 0) != 0:
            raise KlingError(
                f"Kling API error on POST {path}: code={data.get('code')} "
                f"message={data.get('message')}"
            )
        return data

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, headers=self._headers(), timeout=60)
        if resp.status_code >= 400:
            raise KlingError(f"GET {path} → HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        if data.get("code", 0) != 0:
            raise KlingError(
                f"Kling API error on GET {path}: code={data.get('code')} "
                f"message={data.get('message')}"
            )
        return data

    # ── Text-to-Image ──

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: str = "kling-v2",
        aspect_ratio: str = "1:1",
        n: int = 1,
        image_fidelity: float = 0.5,
        human_fidelity: float = 0.45,
        image_reference_b64: str | None = None,
        reference_type: str = "subject",
    ) -> str:
        """Submit a T2I task, poll until done, return the first image URL.

        When `image_reference_b64` is provided, Kling uses it as a subject/style
        anchor. `reference_type` is "subject" (lock face/identity) or "face".
        """
        if len(prompt) > 500:
            raise KlingError(
                f"T2I prompt is {len(prompt)} chars; Kling limit is 500"
            )
        if negative_prompt and len(negative_prompt) > 200:
            raise KlingError(
                f"T2I negative_prompt is {len(negative_prompt)} chars; "
                f"Kling limit is 200"
            )
        body = {
            "model_name": model_name,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "n": n,
            "image_fidelity": image_fidelity,
            "human_fidelity": human_fidelity,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if image_reference_b64:
            body["image"] = image_reference_b64
            body["image_reference"] = reference_type

        log.info(f"  Kling T2I submit ({model_name}): {prompt[:60]}...")
        resp = self._post("/v1/images/generations", body)
        task_id = resp["data"]["task_id"]
        log.info(f"  T2I task {task_id} submitted, polling...")

        result = self._poll_task("/v1/images/generations", task_id, poll_interval=8, timeout=600)
        images = result["data"]["task_result"]["images"]
        if not images:
            raise KlingError("T2I succeeded but returned no images")
        return images[0]["url"]

    # ── Image-to-Video ──

    def image_to_video(
        self,
        prompt: str,
        image_b64: str,
        image_tail_b64: str | None = None,
        negative_prompt: str = "",
        model_name: str = "kling-v2-1-master",
        cfg_scale: float = 0.5,
        mode: str = "pro",
        aspect_ratio: str = "16:9",
        duration: str = "5",
    ) -> str:
        """Submit an I2V task, poll until done, return the first video URL.

        If `image_tail_b64` is provided, Kling interpolates motion between
        the two keyframes. The caller is responsible for choosing a model
        that supports `image_tail` (e.g. kling-v1-6); this method does not
        validate model capability.
        """
        if len(prompt) > 2500:
            raise KlingError(
                f"I2V prompt is {len(prompt)} chars; Kling limit is 2500"
            )
        if negative_prompt and len(negative_prompt) > 2500:
            raise KlingError(
                f"I2V negative_prompt is {len(negative_prompt)} chars; "
                f"Kling limit is 2500"
            )
        body = {
            "model_name": model_name,
            "image": image_b64,
            "prompt": prompt,
            "cfg_scale": cfg_scale,
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }
        if image_tail_b64:
            body["image_tail"] = image_tail_b64
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        log.info(f"  Kling I2V submit ({model_name}, {mode}, {duration}s)")
        resp = self._post("/v1/videos/image2video", body)
        task_id = resp["data"]["task_id"]
        log.info(f"  I2V task {task_id} submitted, polling...")

        result = self._poll_task("/v1/videos/image2video", task_id, poll_interval=16, timeout=900)
        videos = result["data"]["task_result"]["videos"]
        if not videos:
            raise KlingError("I2V succeeded but returned no videos")
        return videos[0]["url"]

    # ── OmniVideo (multi-image storytelling) ──

    def omni_video(
        self,
        prompt: str,
        image_list: list[str],
        duration: str = "5",
        mode: str = "pro",
        aspect_ratio: str = "16:9",
        model_name: str = "kling-video-o1",
    ) -> str:
        """Submit an OmniVideo multi-image task, poll until done, return
        the first video URL.

        `image_list` holds base64-encoded reference images in the order
        the prompt binds them via <<<image_1>>>, <<<image_2>>>, ...
        """
        if not image_list:
            raise KlingError("omni_video requires a non-empty image_list")
        if len(prompt) > 2400:
            raise KlingError(
                f"OmniVideo prompt is {len(prompt)} chars; cap is 2400"
            )
        body = {
            "model_name": model_name,
            "mode": mode,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "prompt": prompt,
            "image_list": [{"image_url": b64} for b64 in image_list],
        }
        log.info(
            f"  Kling OmniVideo submit ({model_name}, {mode}, "
            f"{duration}s, {len(image_list)} refs)"
        )
        resp = self._post("/v1/videos/omni-video", body)
        task_id = resp["data"]["task_id"]
        log.info(f"  OmniVideo task {task_id} submitted, polling...")

        result = self._poll_task(
            "/v1/videos/omni-video", task_id,
            poll_interval=16, timeout=900,
        )
        videos = result["data"]["task_result"]["videos"]
        if not videos:
            raise KlingError("OmniVideo succeeded but returned no videos")
        return videos[0]["url"]

    # ── Task polling ──

    def _poll_task(
        self,
        path_prefix: str,
        task_id: str,
        poll_interval: int,
        timeout: int,
    ) -> dict:
        """Poll a task endpoint until status is succeed/failed or timeout."""
        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            data = self._get(f"{path_prefix}/{task_id}")
            status = data["data"]["task_status"]
            if status != last_status:
                log.info(f"  ... task {task_id[-8:]} status: {status}")
                last_status = status
            if status == "succeed":
                return data
            if status == "failed":
                msg = data["data"].get("task_status_msg", "no detail")
                raise KlingError(f"Task {task_id} failed: {msg}")
            time.sleep(poll_interval)
        raise KlingError(f"Task {task_id} did not complete within {timeout}s")

    # ── Download ──

    def download(self, url: str, dest_path: str) -> None:
        """Stream a result URL to disk.

        Writes to `<dest_path>.part` first, then atomically renames on success.
        This guarantees that a partial/interrupted download cannot be mistaken
        for a finished file by the pipeline's resume-on-restart logic.
        """
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tmp_path = dest_path + ".part"
        try:
            with self.session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, dest_path)
        except BaseException:
            # Clean up partial file on any failure (network error, KeyboardInterrupt, etc.)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        log.info(f"  Saved → {dest_path} ({size_mb:.2f} MB)")


def encode_image_b64(path: str) -> str:
    """Base64-encode a local image file. Kling expects no `data:image/...` prefix."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")
