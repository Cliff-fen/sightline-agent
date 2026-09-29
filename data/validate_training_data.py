from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError


class ValidationError(ValueError):
    pass


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: each row must be an object")
            yield line_number, row


def content_parts(content: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        raise ValidationError(f"{field} must be a content-part list")
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in {"text", "image"}:
            raise ValidationError(f"{field}[{index}] must be a text or image part")
        if part["type"] == "text" and not isinstance(part.get("text"), str):
            raise ValidationError(f"{field}[{index}].text must be a string")
    return content


def _local_image_path(value: str, base_dir: Path) -> Path | None:
    if value.startswith(("http://", "https://", "data:")):
        return None
    path = Path(urlparse(value).path) if value.startswith("file://") else Path(value)
    return path if path.is_absolute() else base_dir / path


def _decode_image(image: Any, *, field: str, base_dir: Path) -> None:
    try:
        if isinstance(image, dict) and image.get("bytes") is not None:
            with Image.open(io.BytesIO(image["bytes"])) as decoded:
                decoded.verify()
            return
        value = image.get("path") if isinstance(image, dict) else image
        if not isinstance(value, str):
            return
        path = _local_image_path(value, base_dir)
        if path is None:
            return
        with Image.open(path) as decoded:
            decoded.verify()
    except (OSError, UnidentifiedImageError, TypeError, ValueError) as exc:
        raise ValidationError(f"{field} cannot be decoded: {exc}") from exc


def validate_images(images: Any, *, field: str, base_dir: Path, decode: bool = False) -> list[Any]:
    if not isinstance(images, list) or not images:
        raise ValidationError(f"{field} must be a non-empty list")
    for index, image in enumerate(images):
        value = image.get("path") if isinstance(image, dict) else image
        if not isinstance(value, str) or not value:
            raise ValidationError(f"{field}[{index}] must be a path or supported image object")
        if value.startswith(("http://", "https://", "data:")):
            continue
        if value.startswith("artifact://"):
            raise ValidationError(f"{field}[{index}] contains a runtime-only artifact URI")
        path = _local_image_path(value, base_dir)
        assert path is not None
        if not path.is_file():
            raise ValidationError(f"{field}[{index}] does not exist: {path}")
        if decode:
            _decode_image(image, field=f"{field}[{index}]", base_dir=base_dir)
    return images


def image_part_count(messages: list[dict[str, Any]], *, field: str) -> int:
    count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"{field}[{index}] must be an object")
        parts = content_parts(message.get("content"), field=f"{field}[{index}].content")
        count += sum(part.get("type") == "image" for part in parts)
    return count


def tool_names(tools: Any, *, field: str) -> set[str]:
    if not isinstance(tools, list) or not tools:
        raise ValidationError(f"{field} must be a non-empty list")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{field}[{index}] has no function name")
        if name in names:
            raise ValidationError(f"{field}[{index}] duplicates tool {name}")
        names.add(name)
    return names


def validate_sft(row: dict[str, Any], *, base_dir: Path, decode_images: bool = False) -> dict[str, int]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list")
    images = validate_images(row.get("images"), field="images", base_dir=base_dir, decode=decode_images)
    defined_tools = tool_names(row.get("tools"), field="tools")
    pending: dict[str, str] = {}
    final_answer = False
    call_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValidationError(f"messages[{index}].role is invalid")
        parts = content_parts(message.get("content"), field=f"messages[{index}].content")
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            raise ValidationError(f"messages[{index}] has tool_calls outside an assistant turn")
        for call_index, call in enumerate(calls):
            function = call.get("function") if isinstance(call, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(call_id, str) or not call_id or call_id in pending:
                raise ValidationError(f"messages[{index}].tool_calls[{call_index}] has an invalid id")
            if name not in defined_tools:
                raise ValidationError(f"messages[{index}] calls undefined tool {name!r}")
            if not isinstance(arguments, (dict, str)):
                raise ValidationError(f"messages[{index}].tool_calls[{call_index}] has invalid arguments")
            pending[call_id] = str(name)
            call_count += 1
        if role == "tool":
            call_id = message.get("tool_call_id")
            name = message.get("name")
            if call_id not in pending:
                raise ValidationError(f"messages[{index}] has no matching pending tool call")
            if name != pending[call_id]:
                raise ValidationError(f"messages[{index}] tool name does not match {call_id}")
            pending.pop(call_id)
        if role == "assistant" and not calls and any(part.get("type") == "text" and part.get("text", "").strip() for part in parts):
            final_answer = True
    if pending:
        raise ValidationError(f"trajectory ends with unmatched tool calls: {sorted(pending)}")
    if not final_answer:
        raise ValidationError("trajectory has no final assistant answer")
    placeholders = image_part_count(messages, field="messages")
    if placeholders != len(images):
        raise ValidationError(f"image placeholder count {placeholders} does not match images count {len(images)}")
    return {"images": len(images), "tool_calls": call_count}


def validate_rl(row: dict[str, Any], *, base_dir: Path, decode_images: bool = False) -> dict[str, int]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValidationError("prompt must be a non-empty message list")
    images = validate_images(row.get("images"), field="images", base_dir=base_dir, decode=decode_images)
    answers = row.get("answer")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or not answers or not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValidationError("answer must contain at least one non-empty string")
    placeholders = image_part_count(prompt, field="prompt")
    if placeholders != len(images):
        raise ValidationError(f"image placeholder count {placeholders} does not match images count {len(images)}")
    return {
        "images": len(images),
        "answers": len(answers),
        "answer_words": max(len(answer.split()) for answer in answers),
    }


def validate_file(path: Path, stage: str, *, decode_images: bool = False) -> dict[str, Any]:
    count = 0
    image_count = 0
    tool_call_count = 0
    answer_words: list[int] = []
    for line_number, row in read_jsonl(path):
        try:
            details = (validate_sft if stage == "sft" else validate_rl)(
                row,
                base_dir=path.parent,
                decode_images=decode_images,
            )
        except ValidationError as exc:
            raise ValidationError(f"{path}:{line_number}: {exc}") from exc
        count += 1
        image_count += details["images"]
        tool_call_count += details.get("tool_calls", 0)
        if "answer_words" in details:
            answer_words.append(details["answer_words"])
    if count == 0:
        raise ValidationError(f"{path}: file contains no records")
    report: dict[str, Any] = {
        "stage": stage,
        "path": str(path),
        "records": count,
        "images": image_count,
        "valid": True,
        "imagesDecoded": decode_images,
    }
    if stage == "sft":
        report["toolCalls"] = tool_call_count
    if answer_words:
        ordered = sorted(answer_words)
        report["answerWords"] = {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": ordered[-1],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SFT and GRPO JSONL before training.")
    parser.add_argument("--sft", action="append", default=[])
    parser.add_argument("--rl", action="append", default=[])
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Open every local image to detect corrupt files before training.",
    )
    args = parser.parse_args()
    if not args.sft and not args.rl:
        parser.error("provide at least one --sft or --rl file")
    reports = [validate_file(Path(path), "sft", decode_images=args.decode_images) for path in args.sft]
    reports.extend(validate_file(Path(path), "rl", decode_images=args.decode_images) for path in args.rl)
    print(json.dumps({"valid": True, "files": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
