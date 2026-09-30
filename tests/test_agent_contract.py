from __future__ import annotations

from data.prompts import AGENT_TOOL_NAMES, MAX_AGENT_TURNS
from rl.environment import SearchEnvironment


def test_training_environment_matches_shared_tool_contract() -> None:
    environment = SearchEnvironment()
    exposed = tuple(
        name for name in AGENT_TOOL_NAMES if callable(getattr(environment, name, None))
    )
    assert exposed == AGENT_TOOL_NAMES


def test_shared_turn_limit_is_positive() -> None:
    assert MAX_AGENT_TURNS > 0
