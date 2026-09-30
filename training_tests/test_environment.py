from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


environment = load_module("environment", ROOT / "rl/environment.py")


def test_single_rollout_image_resolves_model_alias() -> None:
    env = environment.SearchEnvironment()
    env.reset(images=[Image.new("RGB", (4, 4), "blue")])
    uri = env._resolve_image_reference("image.png")
    assert uri.startswith("data:image/png;base64,")


def test_multiple_rollout_images_require_an_index() -> None:
    env = environment.SearchEnvironment()
    env.reset(images=[Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))])
    assert env._resolve_image_reference("img_1") == env._image_uri(env.images[0])
    assert env._resolve_image_reference("img_2") == env._image_uri(env.images[1])
    with pytest.raises(environment.ToolGatewayError, match="ambiguous"):
        env._resolve_image_reference("the photo")


def test_layout_result_preserves_gateway_observation(monkeypatch) -> None:
    env = environment.SearchEnvironment()
    env.reset(images=[Image.new("RGB", (4, 4), "white")])
    calls = []

    def fake_call(name, arguments):
        env.calls += 1
        calls.append((name, arguments))
        return '{"text":"SIGHTLINE 2026","blocks":[{"text":"SIGHTLINE 2026","box":[[0,0]]}]}'

    monkeypatch.setattr(env, "_call", fake_call)
    first = env.layout_parsing("img_1")
    second = env.layout_parsing("img_1")
    assert first == '{"text":"SIGHTLINE 2026","blocks":[{"text":"SIGHTLINE 2026","box":[[0,0]]}]}'
    assert second == first
    assert len(calls) == 2


def test_gateway_retries_an_empty_response(monkeypatch) -> None:
    env = environment.SearchEnvironment()
    env.reset()
    monkeypatch.setenv("TOOL_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(environment.time, "sleep", lambda _seconds: None)

    class Response:
        def __init__(self, body, *, status=200, text=""):
            self.body = body
            self.status_code = status
            self.ok = status < 400
            self.text = text

        def json(self):
            if self.body is None:
                raise ValueError("empty response")
            return self.body

    responses = iter(
        [
            Response(None),
            Response({"ok": True, "content": [{"type": "text", "text": "evidence"}]}),
        ]
    )
    monkeypatch.setattr(env.http, "post", lambda *args, **kwargs: next(responses))

    assert env.text_search("Sydney") == "evidence"
    assert env.calls == 1
    assert env.failures == 0


def test_gateway_client_does_not_inherit_proxy_environment() -> None:
    env = environment.SearchEnvironment()
    assert env.http.trust_env is False


@pytest.mark.parametrize("method_name", ["crop", "sharpen", "super_resolution", "perspective_correct"])
def test_image_transform_is_available_to_the_next_tool_call(monkeypatch, method_name: str) -> None:
    env = environment.SearchEnvironment()
    original = Image.new("RGB", (8, 8), "white")
    transformed = Image.new("RGB", (4, 4), "black")
    env.reset(images=[original])
    monkeypatch.setattr(
        env,
        "_call",
        lambda _name, _arguments: [
            {"type": "text", "text": "done"},
            {"type": "image", "image": transformed},
        ],
    )

    method = getattr(env, method_name)
    if method_name == "crop":
        method("sandbox:/mnt/data/image.png", 0, 0, 4, 4)
    else:
        method("sandbox:/mnt/data/image.png")

    assert len(env.images) == 2
    assert env._resolve_image_reference("img_2") == env._image_uri(transformed)
