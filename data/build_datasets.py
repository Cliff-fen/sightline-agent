from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _content_with_images(text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    chunks = text.split("<image>")
    for index, chunk in enumerate(chunks):
        if index:
            parts.append({"type": "image"})
        if chunk:
            parts.append({"type": "text", "text": chunk})
    return parts or [{"type": "text", "text": ""}]


def _tool_call(text: str) -> tuple[str, dict[str, Any] | None]:
    match = TOOL_CALL_RE.search(text)
    if not match:
        return text, None
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text, None
    return (text[: match.start()] + text[match.end() :]).strip(), call


def convert_sft(record: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if record.get("system"):
        messages.append({"role": "system", "content": [{"type": "text", "text": str(record["system"])}]})
    pending_call: tuple[str, str] | None = None
    for turn_index, turn in enumerate(record.get("conversations") or []):
        role = turn.get("from")
        text = str(turn.get("value", ""))
        if role == "human":
            messages.append({"role": "user", "content": _content_with_images(text)})
        elif role == "gpt":
            clean, call = _tool_call(text)
            message: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": clean}] if clean else []}
            if call:
                call_id = f"call-{turn_index}"
                function = call.get("function") or call
                name = function.get("name")
                arguments = function.get("arguments") or {}
                message["tool_calls"] = [{"type": "function", "id": call_id, "function": {"name": name, "arguments": arguments}}]
                pending_call = (call_id, str(name))
            messages.append(message)
        elif role in {"observation", "tool"}:
            if pending_call is None:
                raise ValueError("tool result has no preceding tool call")
            call_id, name = pending_call
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": [{"type": "text", "text": text}]})
            pending_call = None
    if pending_call is not None:
        raise ValueError("trajectory ends before its tool result")
    tools = record.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    return {"messages": messages, "images": record.get("images") or [], "tools": tools}


def convert_rl(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record["question"])
    images = record.get("images") or []
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": question})
    return {
        "prompt": [{"role": "user", "content": content}],
        "answer": record["answer"],
        "images": images,
        "dataset": record.get("dataset", "unknown"),
    }


def records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON input must contain a list")
    yield from payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated SFT or GRPO JSONL datasets.")
    parser.add_argument("--stage", choices=("sft", "rl"), required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("validation-ratio must be in [0, 1)")
    converter = convert_sft if args.stage == "sft" else convert_rl
    converted: list[dict[str, Any]] = []
    rejected = 0
    for source in args.input:
        for record in records(Path(source)):
            try:
                converted.append(converter(record))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                rejected += 1
    if not converted:
        raise ValueError("no valid records were produced")
    random.Random(args.seed).shuffle(converted)
    validation_count = int(len(converted) * args.validation_ratio)
    validation = converted[:validation_count]
    train = converted[validation_count:]
    output = Path(args.output_dir)
    train_count = write_jsonl(output / "train.jsonl", train)
    validation_written = write_jsonl(output / "validation.jsonl", validation)
    print(json.dumps({"train": train_count, "validation": validation_written, "rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
