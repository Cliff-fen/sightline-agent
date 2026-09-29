from __future__ import annotations

import json
import os
import base64
import mimetypes
from dataclasses import dataclass
from typing import Any

import requests

from .model_api import RelayModel
from .quality import final_answer_check, process_check


@dataclass(frozen=True)
class GeneratedTrajectory:
    messages: list[dict[str, Any]]
    final_answer: str
    events: list[dict[str, Any]]
    process: dict[str, Any]


def _event_trace(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "message_delta":
            lines.append(f"ASSISTANT: {event.get('delta', '')}")
        elif kind == "tool_started":
            call = event.get("call") or {}
            lines.append(f"TOOL_CALL {call.get('name')}: {json.dumps(call.get('arguments') or {}, ensure_ascii=False)}")
        elif kind == "tool_finished":
            result = event.get("result") or {}
            content = result.get("content") or []
            text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            lines.append(f"TOOL_RESULT {result.get('name')}: {text[:3000]}")
    return "\n".join(lines)


def events_to_messages(events: list[dict[str, Any]], *, image_url: str, question: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    pending_text: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "message_delta":
            pending_text.append(str(event.get("delta", "")))
        elif kind == "tool_started":
            call = event.get("call") or {}
            message: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": "".join(pending_text)}]}
            message["tool_calls"] = [{"type": "function", "id": call.get("id", "call-generated"), "function": {"name": call.get("name"), "arguments": call.get("arguments") or {}}}]
            messages.append(message)
            pending_text = []
        elif kind == "tool_finished":
            result = event.get("result") or {}
            content = result.get("content") or []
            messages.append({"role": "tool", "tool_call_id": result.get("callId"), "name": result.get("name"), "content": content})
    final_text = "".join(pending_text).strip()
    if final_text:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": final_text}]})
    return messages


class AgentTrajectoryClient:
    def __init__(self, *, agent_url: str | None = None, timeout: float = 600):
        self.agent_url = (agent_url or os.environ.get("SIGHTLINE_AGENT_URL", "http://127.0.0.1:18080")).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # The local Agent service must not be sent through a desktop HTTP proxy.
        self.session.trust_env = False

    def generate(self, *, image_url: str, question: str, answer: str, judge: RelayModel) -> GeneratedTrajectory | None:
        if image_url.startswith("file://"):
            path = image_url.removeprefix("file://")
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as handle:
                image_url = f"data:{mime};base64,{base64.b64encode(handle.read()).decode('ascii')}"
        system = (
            "You are a visual investigation agent. Verify, do not guess. "
            "Use crop/layout_parsing for small or textual regions, sharpen/super_resolution/perspective_correct "
            "when image quality requires it, and use external search for facts not visible in pixels. "
            "After image_search, always use text_search to verify the requested fact. "
            "Take one action per turn, wait for observations, avoid repeated queries, and finish with a concise supported answer."
        )
        response = self.session.post(
            f"{self.agent_url}/v1/runs",
            json={"systemPrompt": system, "input": [{"type": "text", "text": question}, {"type": "image", "image": {"kind": "image", "id": "source", "uri": image_url}}]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events") or []
        final = str(payload.get("output") or "").strip()
        if not final or not final_answer_check(judge, question=question, answer=answer, response=final):
            return None
        trace = _event_trace(events)
        process = process_check(judge, question=question, answer=answer, trace=trace)
        if not process.get("accepted"):
            return None
        return GeneratedTrajectory(events_to_messages(events, image_url=image_url, question=question), final, events, process)
