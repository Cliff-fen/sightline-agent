from __future__ import annotations

import json
from pathlib import Path


_CONTRACT_PATH = Path(__file__).parents[1] / "shared" / "agent-contract.json"
AGENT_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
AGENT_SYSTEM_PROMPT = str(AGENT_CONTRACT["systemPrompt"])
MAX_AGENT_TURNS = int(AGENT_CONTRACT["maxTurns"])
AGENT_TOOL_NAMES = tuple(str(name) for name in AGENT_CONTRACT["tools"])

_TRAINING_ACTION_PROTOCOL = """Each assistant turn must contain exactly one <think>...</think> block followed by exactly one action: either one <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>, or a terminal <response>...</response>. Execute only one tool per turn and wait for its observation. Refer to the original image as img_1 and to later image artifacts as img_2, img_3, and so on."""

DEEP_RESEARCH_SYSTEM_PROMPT = f"{AGENT_SYSTEM_PROMPT}\n\n{_TRAINING_ACTION_PROTOCOL}"
