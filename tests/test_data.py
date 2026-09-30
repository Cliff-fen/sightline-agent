from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


data = load_module("build_datasets", ROOT / "data/build_datasets.py")
rewards = load_module("rewards", ROOT / "rl/rewards.py")


def test_sft_converter_structures_tool_calls() -> None:
    row = {
        "system": "Use evidence.",
        "images": ["image.png"],
        "tools": [],
        "conversations": [
            {"from": "human", "value": "<image>Identify this."},
            {"from": "gpt", "value": '<tool_call>{"name":"web_search","arguments":{"query":"landmark"}}</tool_call>'},
            {"from": "observation", "value": "A source result."},
            {"from": "gpt", "value": "It is the landmark."},
        ],
    }
    converted = data.convert_sft(row)
    assert converted["images"] == ["image.png"]
    call = converted["messages"][2]["tool_calls"][0]
    result = converted["messages"][3]
    assert call["function"]["name"] == "web_search"
    assert result["tool_call_id"] == call["id"]


def test_rl_converter_preserves_image_order() -> None:
    converted = data.convert_rl({"question": "What?", "answer": ["A"], "images": ["a.png", "b.png"]})
    assert converted["prompt"][0]["role"] == "system"
    assert [part["type"] for part in converted["prompt"][1]["content"]] == ["image", "image", "text"]
    assert converted["images"] == ["a.png", "b.png"]


def test_trajectory_reward_uses_both_judges_and_format_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        rewards,
        "_call_judge",
        lambda kind, *args, **kwargs: "correct: yes" if kind == "accuracy" else "score: 0.5",
    )
    completion = [
        {
            "role": "assistant",
            "content": "<think>search first</think>",
            "tool_calls": [{"function": {"name": "text_search", "arguments": {"q": "lake"}}}],
        },
        {"role": "tool", "name": "text_search", "content": "evidence"},
        {"role": "assistant", "content": "<think>enough evidence</think><response>The location is Attabad Lake.</response>"},
    ]
    score = asyncio.run(rewards.trajectory_reward(
        completions=[completion],
        answer=[["Attabad Lake"]],
        prompts=[[{"role": "user", "content": "Which lake?"}]],
    ))
    assert score == [0.9]


def test_format_failure_gates_both_judge_rewards(monkeypatch) -> None:
    monkeypatch.setattr(rewards, "_call_judge", lambda kind, *args, **kwargs: "correct: yes" if kind == "accuracy" else "score: 1.0")
    score = asyncio.run(rewards.trajectory_reward(
        completions=[[{"role": "assistant", "content": "Attabad Lake"}]],
        answer=[["Attabad Lake"]],
        prompts=[[{"role": "user", "content": "Which lake?"}]],
    ))
    assert score == [0.0]
