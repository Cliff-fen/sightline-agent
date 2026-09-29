from __future__ import annotations

import io

import requests
from PIL import Image

from data.generation.clip_score import ClipScorer
from data.generation.pipeline import _download_image, _rewrite_visual_anchor
from data.generation.quality import deterministic_leak_check


class _Output:
    def __init__(self, value: object) -> None:
        self.pooler_output = value


def test_clip_features_accepts_transformers_4_tensor_shape() -> None:
    marker = object()
    assert ClipScorer._features(marker) is marker


def test_clip_features_accepts_transformers_5_model_output() -> None:
    marker = object()
    assert ClipScorer._features(_Output(marker)) is marker


def test_clip_features_accepts_tuple_output() -> None:
    marker = object()
    assert ClipScorer._features((marker, object())) is marker


class _JsonModel:
    def __init__(self, question: str) -> None:
        self.question = question

    def json(self, *args, **kwargs):
        return {"question": self.question}


def test_visual_anchor_repairs_token_boundaries_without_dropping_clues() -> None:
    source = type("Source", (), {"title": "Example", "extract": "Example extract"})()
    question = "The the entity shown in the imagefrom the northern station, what year did it close?"
    rewritten = _rewrite_visual_anchor(_JsonModel(question), question=question, source=source)
    assert rewritten == "The entity shown in the image, from the northern station, what year did it close?"


def test_leak_check_rejects_exact_answer_phrase() -> None:
    assert not deterministic_leak_check("It was found in Australian waters.", ["Australian waters"])
    assert deterministic_leak_check("In what waters was it found?", ["Australian waters"])


def test_image_download_retries_transient_tls_failure(monkeypatch, tmp_path) -> None:
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(payload, format="PNG")
    calls = 0

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        content = payload.getvalue()

        def raise_for_status(self) -> None:
            return None

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.exceptions.SSLError("temporary EOF")
        return Response()

    monkeypatch.setattr("data.generation.pipeline.requests.get", get)
    monkeypatch.setattr("data.generation.pipeline.time.sleep", lambda _: None)
    destination = tmp_path / "image.jpg"
    assert _download_image("https://example.test/image.png", destination) == destination
    assert destination.is_file()
    assert calls == 2
