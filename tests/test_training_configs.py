from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def _config(stage: str) -> dict:
    path = ROOT / stage / "configs" / "production.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_public_profiles_are_epoch_driven() -> None:
    assert "max_steps" not in _config("sft")
    assert "max_steps" not in _config("rl")


def test_public_profiles_use_global_batch_32() -> None:
    for stage in ("sft", "rl"):
        config = _config(stage)
        assert config["expected_world_size"] == 8
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 4
        assert config["expected_global_batch_size"] == 32
