from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    root = Path("data/processed")
    image_path = (root / "smoke-document.png").resolve()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
    except OSError:
        font = ImageFont.load_default()
    draw.text((28, 45), "SIGHTLINE 2026", fill="black", font=font)
    image.save(image_path)
    tool = {
        "type": "function",
        "function": {
            "name": "layout_parsing",
            "description": "Extract text and reading order from the current image.",
            "parameters": {
                "type": "object",
                "properties": {"image": {"type": "string"}},
                "required": ["image"],
            },
        },
    }
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the large text. Call layout_parsing exactly once, then answer directly without another tool."}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [{"type": "function", "id": "call-smoke-layout", "function": {"name": "layout_parsing", "arguments": {"image": "image.png"}}}],
        },
        {"role": "tool", "tool_call_id": "call-smoke-layout", "name": "layout_parsing", "content": [{"type": "text", "text": '{"text":"SIGHTLINE 2026"}'}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The text reads SIGHTLINE 2026."}]},
    ]
    sft_rows = [{"messages": messages, "images": [str(image_path)], "tools": [tool]} for _ in range(4)]
    rl_rows = [
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the large text. Call layout_parsing exactly once on image.png, then answer directly without another tool."}]}],
            "answer": ["SIGHTLINE 2026"],
            "images": [str(image_path)],
            "dataset": "smoke",
        }
        for _ in range(4)
    ]
    write(root / "sft/smoke.jsonl", sft_rows)
    write(root / "rl/smoke.jsonl", rl_rows)
    print(f"wrote smoke data under {root}")


if __name__ == "__main__":
    main()
