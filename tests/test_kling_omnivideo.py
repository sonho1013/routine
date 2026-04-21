"""Unit tests for KlingClient.omni_video."""
from unittest.mock import MagicMock, patch

import pytest

from scripts.video_gen.kling_client import KlingClient, KlingError


def _make_client() -> KlingClient:
    return KlingClient(
        access_key="ak", secret_key="sk",
        base_url="https://api.klingai.com",
    )


def _mock_response(payload: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = ""
    return resp


def test_omni_video_submits_multi_image_body_and_returns_url():
    client = _make_client()
    submit_resp = _mock_response({"code": 0, "data": {"task_id": "T1"}})
    poll_resp = _mock_response({"code": 0, "data": {
        "task_status": "succeed",
        "task_result": {"videos": [{"url": "https://cdn/video.mp4"}]},
    }})
    with patch.object(client.session, "post", return_value=submit_resp) as mpost, \
         patch.object(client.session, "get",  return_value=poll_resp) as mget:
        url = client.omni_video(
            prompt="Mary <<<image_1>>> near <<<image_2>>>",
            image_list=["B64A", "B64B"],
            duration="5",
            mode="pro",
            aspect_ratio="16:9",
            model_name="kling-video-o1",
        )
    assert url == "https://cdn/video.mp4"

    # POST went to the OmniVideo endpoint
    (post_url,), post_kwargs = mpost.call_args
    assert post_url.endswith("/v1/videos/omni-video")
    body = post_kwargs["json"]
    assert body["model_name"] == "kling-video-o1"
    assert body["mode"] == "pro"
    assert body["duration"] == "5"
    assert body["aspect_ratio"] == "16:9"
    assert body["prompt"] == "Mary <<<image_1>>> near <<<image_2>>>"
    assert body["image_list"] == [
        {"image_url": "B64A"},
        {"image_url": "B64B"},
    ]

    # GET polled the OmniVideo task endpoint with the returned task_id
    (get_url,), _ = mget.call_args
    assert get_url.endswith("/v1/videos/omni-video/T1")


def test_omni_video_raises_on_failed_task():
    client = _make_client()
    submit_resp = _mock_response({"code": 0, "data": {"task_id": "T2"}})
    poll_resp = _mock_response({"code": 0, "data": {
        "task_status": "failed",
        "task_status_msg": "prompt rejected",
    }})
    with patch.object(client.session, "post", return_value=submit_resp), \
         patch.object(client.session, "get",  return_value=poll_resp):
        with pytest.raises(KlingError, match="prompt rejected"):
            client.omni_video(
                prompt="x", image_list=["B64"], duration="5",
            )


def test_omni_video_rejects_prompt_over_2400_chars():
    client = _make_client()
    with pytest.raises(KlingError, match="2400"):
        client.omni_video(prompt="x" * 2401, image_list=["B64"])


def test_omni_video_rejects_empty_image_list():
    client = _make_client()
    with pytest.raises(KlingError, match="image_list"):
        client.omni_video(prompt="p", image_list=[])
