from __future__ import annotations

from pathlib import Path

import pytest

from scripts.preflight import PreflightError, check_training_run


def _write_run(tmp_path: Path, *, calls: float, failures: float, checkpoint: bool = True) -> Path:
    log = tmp_path / "train.log"
    log.write_text(
        str({"tools/call_frequency": str(calls), "tools/failure_frequency": str(failures), "reward": "0.2"})
        + "\n",
        encoding="utf-8",
    )
    if checkpoint:
        directory = tmp_path / "checkpoint-1"
        directory.mkdir()
        (directory / "trainer_state.json").write_text("{}", encoding="utf-8")
    return log


def test_training_run_requires_real_tool_calls_and_checkpoint(tmp_path: Path) -> None:
    report = check_training_run(_write_run(tmp_path, calls=1, failures=0), 0)
    assert report["toolCallFrequency"] == 1
    assert report["toolFailureFrequency"] == 0
    assert report["checkpoint"].endswith("checkpoint-1")


def test_training_run_rejects_tool_failures(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="failure frequency"):
        check_training_run(_write_run(tmp_path, calls=1, failures=0.5), 0)


def test_training_run_rejects_no_calls(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="did not execute"):
        check_training_run(_write_run(tmp_path, calls=0, failures=0), 0)
