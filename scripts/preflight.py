from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.validate_training_data import validate_file  # noqa: E402
from rl.environment import SearchEnvironment  # noqa: E402


class PreflightError(RuntimeError):
    pass


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def build_probe_image(path: Path) -> None:
    image = Image.new("RGB", (800, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 50, 760, 430), outline="black", width=8)
    draw.text((105, 175), "SIGHTLINE 2026", fill="black", font=_font(64))
    image.save(path, format="PNG")


def _image_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [part for part in value if isinstance(part, dict) and part.get("type") == "image"]


def check_gateway_health(base_url: str, timeout: float) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise PreflightError("tool gateway health endpoint returned ok=false")
    return payload


def check_tools(
    probe_image: str | None = None,
    environment_factory: Callable[[], SearchEnvironment] = SearchEnvironment,
) -> dict[str, Any]:
    timings: dict[str, int] = {}
    details: dict[str, Any] = {}

    def run(name: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = function()
        timings[name] = round((time.perf_counter() - started) * 1000)
        return value

    with tempfile.TemporaryDirectory(prefix="sightline-preflight-") as directory:
        image_path = Path(directory) / "probe.png"
        build_probe_image(image_path)
        env = environment_factory()
        env.reset(images=[str(image_path)])

        web = run("web_search", lambda: env.web_search("Sydney Opera House official site", 2))
        text = run("text_search", lambda: env.text_search("Sydney Opera House location", 2))
        page = run("visit", lambda: env.visit("https://example.com"))
        lens: str | None = None
        lens_calls = 0
        lens_failures = 0
        if probe_image:
            lens_env = environment_factory()
            lens_env.reset(images=[probe_image])
            lens = run("image_search", lambda: lens_env.image_search("image_0"))
            lens_calls = lens_env.calls
            lens_failures = lens_env.failures
        crop = run("crop", lambda: env.crop("image_0", 70, 130, 660, 190))
        layout = run("layout_parsing", lambda: env.layout_parsing("image_0"))
        sharpen = run("sharpen", lambda: env.sharpen("image_0", 1.5))
        upscale = run("super_resolution", lambda: env.super_resolution("image_0", 1.5))
        corrected = run("perspective_correct", lambda: env.perspective_correct("image_0", 1.0))

        for name, value in {"web_search": web, "text_search": text}.items():
            if not isinstance(value, str) or "URL:" not in value:
                raise PreflightError(f"{name} returned no ranked evidence")
        for name, value in {"visit": page, "layout_parsing": layout}.items():
            if not isinstance(value, str) or not value.strip():
                raise PreflightError(f"{name} returned empty content")
        if probe_image and (not isinstance(lens, str) or not lens.strip()):
            raise PreflightError("image_search returned empty content")
        for name, value in {
            "crop": crop,
            "sharpen": sharpen,
            "super_resolution": upscale,
            "perspective_correct": corrected,
        }.items():
            blocks = _image_blocks(value)
            if not blocks or not isinstance(blocks[0].get("image"), Image.Image):
                raise PreflightError(f"{name} did not return a decoded image")

        details = {
            "calls": env.calls + lens_calls,
            "failures": env.failures + lens_failures,
            "trackedImages": len(env.images),
            "searchEvidence": {
                "web": len(web),
                "text": len(text),
                "lens": len(lens) if lens is not None else None,
            },
            "imageSearch": "passed" if probe_image else "skipped (provide --probe-image or --rl)",
            "layoutChars": len(layout),
            "timingsMs": timings,
        }
        failures = env.failures + lens_failures
        if failures:
            raise PreflightError(f"tool environment recorded {failures} failed call(s)")
    return details


def _training_metrics(path: Path) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "'tools/call_frequency'" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            item = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(item, dict):
            metrics.append(item)
    if not metrics:
        raise PreflightError(f"{path} contains no GRPO tool metrics")
    return metrics[-1]


def check_training_run(log_path: Path, max_failure_rate: float) -> dict[str, Any]:
    metrics = _training_metrics(log_path)
    call_rate = float(metrics.get("tools/call_frequency", 0))
    failure_rate = float(metrics.get("tools/failure_frequency", 1))
    if call_rate <= 0:
        raise PreflightError("GRPO smoke run did not execute any tools")
    if failure_rate > max_failure_rate:
        raise PreflightError(
            f"GRPO tool failure frequency {failure_rate:.3f} exceeds {max_failure_rate:.3f}"
        )
    checkpoints = sorted(log_path.parent.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise PreflightError(f"{log_path.parent} contains no complete checkpoint")
    return {
        "log": str(log_path),
        "toolCallFrequency": call_rate,
        "toolFailureFrequency": failure_rate,
        "reward": metrics.get("reward"),
        "checkpoint": str(checkpoints[-1].parent),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end preflight checks before agent training.")
    parser.add_argument("--gateway", default=os.getenv("TOOL_GATEWAY_URL", "http://127.0.0.1:8090"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("TOOL_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sft", action="append", default=[], help="SFT JSONL to validate; repeatable.")
    parser.add_argument("--rl", action="append", default=[], help="GRPO JSONL to validate; repeatable.")
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument(
        "--probe-image",
        required=True,
        help="Known searchable real image path/URL used to verify reverse-image search.",
    )
    parser.add_argument("--training-log", type=Path, help="Completed smoke-run log to validate.")
    parser.add_argument("--max-tool-failure-rate", type=float, default=0.0)
    args = parser.parse_args()

    os.environ["TOOL_GATEWAY_URL"] = args.gateway
    os.environ["TOOL_TIMEOUT_SECONDS"] = str(args.timeout)
    report: dict[str, Any] = {
        "valid": True,
        "gateway": check_gateway_health(args.gateway, args.timeout),
        "tools": check_tools(args.probe_image),
        "data": [],
    }
    for stage, paths in (("sft", args.sft), ("rl", args.rl)):
        report["data"].extend(
            validate_file(Path(path), stage, decode_images=args.decode_images) for path in paths
        )
    if args.training_log:
        report["training"] = check_training_run(args.training_log, args.max_tool_failure_rate)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
