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
    assert env._resolve_image_reference("image_1").startswith("data:image/png;base64,")
    with pytest.raises(environment.ToolGatewayError, match="ambiguous"):
        env._resolve_image_reference("the photo")


def test_layout_result_is_compact_and_cached(monkeypatch) -> None:
    env = environment.SearchEnvironment()
    env.reset(images=[Image.new("RGB", (4, 4), "white")])
    calls = []

    def fake_call(name, arguments):
        env.calls += 1
        calls.append((name, arguments))
        return '{"text":"SIGHTLINE 2026","blocks":[{"text":"SIGHTLINE 2026","box":[[0,0]]}]}'

    monkeypatch.setattr(env, "_call", fake_call)
    first = env.layout_parsing("image.png")
    second = env.layout_parsing("image.png")
    assert first == "Extracted text in reading order:\nSIGHTLINE 2026"
    assert "already parsed" in second
    assert len(calls) == 1
    assert env.get_reward() == pytest.approx(-0.1)
