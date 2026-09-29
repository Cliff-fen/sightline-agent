from __future__ import annotations

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
    assert [part["type"] for part in converted["prompt"][0]["content"]] == ["image", "image", "text"]
    assert converted["images"] == ["a.png", "b.png"]


def test_answer_reward_is_rule_based() -> None:
    assert rewards.answer_reward(["The answer is Attabad Lake."], [["Attabad Lake"]]) == [1.0]
    assert rewards.answer_reward(["No evidence."], [["Attabad Lake"]])[0] == 0.0


def test_final_answer_rejects_raw_tool_tags() -> None:
    assert rewards.final_answer_reward(["A grounded answer."]) == [0.2]
    assert rewards.final_answer_reward(["<tool_call>{}</tool_call>"]) == [-0.5]
