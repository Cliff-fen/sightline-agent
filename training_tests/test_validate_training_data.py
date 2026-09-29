from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("validate_training_data", ROOT / "data/validate_training_data.py")
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


def test_rejects_unmatched_sft_tool_result(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded by structural validation")
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]},
            {"role": "tool", "tool_call_id": "missing", "name": "web_search", "content": [{"type": "text", "text": "result"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
        "images": [str(image)],
        "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
    }
    with pytest.raises(validator.ValidationError, match="no matching pending tool call"):
        validator.validate_sft(row, base_dir=tmp_path)


def test_rejects_rl_image_count_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded by structural validation")
    row = {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
        "answer": ["answer"],
        "images": [str(image)],
    }
    with pytest.raises(validator.ValidationError, match="placeholder count"):
        validator.validate_rl(row, base_dir=tmp_path)


def test_decode_images_rejects_corrupt_file(tmp_path: Path) -> None:
    image = tmp_path / "broken.png"
    image.write_bytes(b"not an image")
    row = {
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]}],
        "answer": ["answer"],
        "images": [str(image)],
    }
    with pytest.raises(validator.ValidationError, match="cannot be decoded"):
        validator.validate_rl(row, base_dir=tmp_path, decode_images=True)


def test_validate_file_reports_training_statistics(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image)
    data = tmp_path / "train.jsonl"
    row = {
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]}],
        "answer": ["two word"],
        "images": [str(image)],
    }
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validator.validate_file(data, "rl", decode_images=True)

    assert report["records"] == 1
    assert report["images"] == 1
    assert report["imagesDecoded"] is True
    assert report["answerWords"]["median"] == 2
