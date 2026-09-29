from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from PIL import Image


class ToolGatewayError(RuntimeError):
    pass


class SearchEnvironment:
    """Stateful tool environment backed by the same HTTP contract as the online agent."""

    def __init__(self) -> None:
        self.base_url = os.getenv("TOOL_GATEWAY_URL", "http://127.0.0.1:8090").rstrip("/")
        self.timeout = float(os.getenv("TOOL_TIMEOUT_SECONDS", "60"))
        self.calls = 0
        self.failures = 0
        self.images: list[Any] = []
        self.layout_cache: dict[str, str] = {}
        self.duplicate_calls = 0

    def reset(self, images: list[Any] | None = None, image: Any | None = None, **_: Any) -> None:
        self.calls = 0
        self.failures = 0
        self.images = list(images or ([] if image is None else [image]))
        self.layout_cache = {}
        self.duplicate_calls = 0
        return None

    @staticmethod
    def _image_uri(image: Any) -> str:
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "file://", "artifact://", "data:")):
                return image
            path = Path(image).expanduser()
            if path.is_file():
                return path.resolve().as_uri()
        if isinstance(image, bytes):
            return "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        if isinstance(image, Image.Image):
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        if isinstance(image, dict):
            if image.get("bytes"):
                return SearchEnvironment._image_uri(image["bytes"])
            for key in ("url", "uri", "path", "image_url"):
                if image.get(key):
                    return SearchEnvironment._image_uri(image[key])
        raise ToolGatewayError(f"unsupported rollout image type: {type(image).__name__}")

    def _resolve_image_reference(self, reference: str) -> str:
        if reference.startswith(("http://", "https://", "file://", "artifact://", "data:")):
            return reference
        if not self.images:
            raise ToolGatewayError("the current rollout has no image to resolve")

        lowered = reference.casefold().strip()
        for candidate in self.images:
            if isinstance(candidate, str):
                names = {candidate.casefold(), Path(candidate).name.casefold(), Path(candidate).stem.casefold()}
                if lowered in names:
                    return self._image_uri(candidate)

        match = re.fullmatch(r"(?:input[_ -]?)?image(?:[_ -]?(\d+))?(?:\.[a-z0-9]+)?", lowered)
        if match and match.group(1) is not None:
            index = int(match.group(1))
            if index >= len(self.images) and 1 <= index <= len(self.images):
                index -= 1
            if 0 <= index < len(self.images):
                return self._image_uri(self.images[index])
        if len(self.images) == 1:
            return self._image_uri(self.images[0])
        raise ToolGatewayError(
            f"ambiguous image reference {reference!r}; use image_0 through image_{len(self.images) - 1}"
        )

    def _trace(self, event: dict[str, Any]) -> None:
        path = os.getenv("TOOL_TRACE_FILE", "").strip()
        if not path:
            return

        def scrub(value: Any) -> Any:
            if isinstance(value, str):
                if value.startswith("data:"):
                    return f"<data-uri:{len(value)} chars>"
                return value[:500]
            if isinstance(value, dict):
                return {str(key): scrub(item) for key, item in value.items()}
            if isinstance(value, list):
                return [scrub(item) for item in value[:20]]
            return value

        record = {"time": time.time(), **scrub(event)}
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)

    def _call(self, name: str, arguments: dict[str, Any]) -> str | list[dict[str, Any]]:
        self.calls += 1
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/v1/tools/{name}",
                json={"callId": f"train-{uuid.uuid4()}", "arguments": arguments},
                timeout=self.timeout,
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            self.failures += 1
            self._trace({"tool": name, "ok": False, "arguments": arguments, "error": str(exc), "durationMs": round((time.perf_counter() - started) * 1000)})
            raise ToolGatewayError(f"{name} transport failure: {exc}") from exc
        if not response.ok or not payload.get("ok"):
            self.failures += 1
            error = payload.get("error") or {}
            self._trace({"tool": name, "ok": False, "arguments": arguments, "status": response.status_code, "error": error, "durationMs": round((time.perf_counter() - started) * 1000)})
            raise ToolGatewayError(str(error.get("message") or f"{name} failed"))
        content = payload.get("content") or []
        converted: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image" and part.get("data"):
                converted.append({"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(part["data"]))).copy()})
            elif part.get("type") == "text":
                converted.append({"type": "text", "text": str(part.get("text", ""))})
        if len(converted) == 1 and converted[0]["type"] == "text":
            self._trace({"tool": name, "ok": True, "arguments": arguments, "durationMs": round((time.perf_counter() - started) * 1000)})
            return converted[0]["text"]
        self._trace({"tool": name, "ok": True, "arguments": arguments, "durationMs": round((time.perf_counter() - started) * 1000)})
        return converted

    def web_search(self, query: str, topK: int = 5) -> str:
        """Search the public web for ranked evidence.

        Args:
            query: A focused textual search query.
            topK: Maximum number of results to return.

        Returns:
            Titles, URLs, and evidence snippets.
        """
        return str(self._call("web_search", {"query": query, "topK": topK}))

    def text_search(self, query: str, topK: int = 5) -> str:
        """Search for textual evidence related to a question or entity.

        Args:
            query: A focused textual search query.
            topK: Maximum number of results to return.

        Returns:
            Ranked textual evidence.
        """
        return str(self._call("text_search", {"query": query, "topK": topK}))

    def visit(self, url: str) -> str:
        """Read a public webpage as clean text.

        Args:
            url: An absolute HTTP or HTTPS URL.

        Returns:
            Extracted page content.
        """
        return str(self._call("visit", {"url": url}))

    def image_search(self, image: str) -> str:
        """Find visually similar images and source pages.

        Args:
            image: A public image URL or an artifact URI returned by another tool.

        Returns:
            Visual matches and their source metadata.
        """
        return str(self._call("image_search", {"image": self._resolve_image_reference(image)}))

    def crop(self, image: str, x: int, y: int, width: int, height: int) -> list[dict[str, Any]]:
        """Crop an image and return the new image to the model.

        Args:
            image: A public image URL or artifact URI.
            x: Left coordinate in pixels.
            y: Top coordinate in pixels.
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            A text observation and the cropped image.
        """
        result = self._call("crop", {"image": self._resolve_image_reference(image), "x": x, "y": y, "width": width, "height": height})
        return result if isinstance(result, list) else [{"type": "text", "text": result}]

    def layout_parsing(self, image: str) -> str:
        """Extract text and reading order from a document image once.

        Args:
            image: A public image URL or artifact URI.

        Returns:
            Concise OCR text. Use it to answer instead of repeating the call.
        """
        resolved = self._resolve_image_reference(image)
        if resolved in self.layout_cache:
            self.duplicate_calls += 1
            return self.layout_cache[resolved] + "\nThe image is already parsed; answer from this observation."
        raw = str(self._call("layout_parsing", {"image": resolved}))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            concise = raw[:4000]
        else:
            text = str(payload.get("text") or "").strip()
            if text:
                concise = f"Extracted text in reading order:\n{text[:4000]}"
            else:
                blocks = payload.get("blocks") or []
                lines = [str(block.get("text", "")).strip() for block in blocks if isinstance(block, dict)]
                concise = "Extracted text in reading order:\n" + "\n".join(line for line in lines if line)[:4000]
        self.layout_cache[resolved] = concise
        return concise

    def get_reward(self) -> float:
        if self.failures:
            return -1.0
        if not self.calls:
            return 0.0
        return max(-1.0, 0.1 - 0.2 * self.duplicate_calls)
