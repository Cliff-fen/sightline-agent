from __future__ import annotations

import json
import os
import time
from types import MethodType
from typing import Any


TOOL_RESPONSE_TEMPLATE = {
    "version": 1,
    "start_anchor": "<|im_start|>assistant\n",
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {"close": ["<tool_call>", "<|im_end|>"], "content": "text", "optional": True},
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "content": "json",
            "repeats": False,
            "transform": {"type": "function", "function": "{content}"},
        },
    },
}


def _tool_call(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function", payload)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    name = function["name"]
    aliases = {
        "local_search": "text_search",
        "local_image_search": "image_search",
        "lens_scan": "image_search",
        "ocr": "layout_parsing",
    }
    name = aliases.get(name, name)
    if name in {"text_search", "web_search"}:
        query = arguments.get("query") or arguments.get("q")
        normalized = {"q": query}
        if arguments.get("hl") or arguments.get("lang"):
            normalized["hl"] = arguments.get("hl") or arguments.get("lang")
        if name == "text_search" and ("topK" in arguments or "top_k" in arguments):
            normalized["top_k"] = arguments.get("top_k", arguments.get("topK"))
        arguments = normalized
    elif name == "image_search":
        image = arguments.get("image") or arguments.get("url")
        arguments = {"url": image}
    elif name in {"crop", "layout_parsing", "sharpen", "super_resolution", "perspective_correct"}:
        arguments = dict(arguments)
        if "image" not in arguments and "url" in arguments:
            arguments["image"] = arguments.pop("url")
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def parse_tool_response_text(text: str) -> dict[str, Any]:
    """Parse tool markup and recover one call when a model omits its closing tag."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    first_marker = text.find("<tool_call>")
    cursor = first_marker

    while cursor >= 0:
        json_start = cursor + len("<tool_call>")
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1
        try:
            payload, json_end = decoder.raw_decode(text, json_start)
        except json.JSONDecodeError:
            break
        call = _tool_call(payload)
        if call is None:
            break
        trailing = json_end
        while trailing < len(text) and text[trailing].isspace():
            trailing += 1
        has_close = text.startswith("</tool_call>", trailing)
        calls.append(call)
        if not has_close:
            # Execute one recoverable call, then return its observation to the policy.
            break
        # The policy contract permits one action per model turn. Additional
        # calls are ignored and therefore receive no observation this turn.
        break

    content_end = first_marker if calls and first_marker >= 0 else len(text)
    content = text[:content_end].replace("<|im_end|>", "").strip()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    _trace_parse(text, message)
    return message


def _trace_parse(text: str, message: dict[str, Any]) -> None:
    path = os.getenv("TOOL_PARSE_TRACE_FILE", "").strip()
    if not path:
        return
    calls = [
        call.get("function", {})
        for call in message.get("tool_calls", [])
        if isinstance(call, dict)
    ]
    record = {
        "time": time.time(),
        "calls": calls,
        "containsToolMarker": "<tool_call>" in text,
        "raw": text[:4000],
    }
    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _decode_responses(tokenizer: Any, response: Any) -> tuple[list[str], bool]:
    if isinstance(response, str):
        return [response], False
    if isinstance(response, (list, tuple)) and (not response or isinstance(response[0], str)):
        return list(response), True
    decoded = tokenizer.decode(response)
    return (list(decoded), True) if isinstance(decoded, (list, tuple)) else ([decoded], False)


def _parse_response(self: Any, response: Any, schema: Any = None, *, prefix: Any = None, tools: Any = None):
    del schema, prefix, tools
    responses, batched = _decode_responses(self, response)
    parsed = [parse_tool_response_text(text) for text in responses]
    return parsed if batched else parsed[0]


def configure_tool_response_parser(processor: Any) -> Any:
    """Attach a tolerant parser matching the policy's tool-call markup."""
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.response_template = TOOL_RESPONSE_TEMPLATE
    tokenizer.parse_response = MethodType(_parse_response, tokenizer)
    return processor
