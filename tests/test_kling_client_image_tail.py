from unittest.mock import MagicMock, patch

from scripts.video_gen.kling_client import KlingClient


def test_image_to_video_sends_image_tail_when_provided():
    client = KlingClient("ak", "sk", "https://example.test")
    fake_post_resp = {"data": {"task_id": "t1"}}
    fake_poll_resp = {"data": {"task_status": "succeed",
                                "task_result": {"videos": [{"url": "http://v"}]}}}

    with patch.object(client, "_post", return_value=fake_post_resp) as mpost, \
         patch.object(client, "_get", return_value=fake_poll_resp):
        client.image_to_video(
            prompt="move camera",
            image_b64="AAAA",
            image_tail_b64="BBBB",
            model_name="kling-v1-6",
            mode="pro",
            duration="5",
        )

    _, kwargs = mpost.call_args
    body = kwargs["body"] if "body" in kwargs else mpost.call_args.args[1]
    assert body["image"] == "AAAA"
    assert body["image_tail"] == "BBBB"


def test_image_to_video_omits_image_tail_when_none():
    client = KlingClient("ak", "sk", "https://example.test")
    fake_post_resp = {"data": {"task_id": "t1"}}
    fake_poll_resp = {"data": {"task_status": "succeed",
                                "task_result": {"videos": [{"url": "http://v"}]}}}

    with patch.object(client, "_post", return_value=fake_post_resp) as mpost, \
         patch.object(client, "_get", return_value=fake_poll_resp):
        client.image_to_video(prompt="p", image_b64="AAAA", model_name="kling-v3")

    body = mpost.call_args.args[1]
    assert "image_tail" not in body
