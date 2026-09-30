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
        self.http = requests.Session()
        # The gateway is an internal service. Proxying loopback requests commonly
        # produces empty 502 responses in cluster environments.
        self.http.trust_env = False
        self.calls = 0
        self.failures = 0
        self.images: list[Any] = []

    def reset(self, images: list[Any] | None = None, image: Any | None = None, **_: Any) -> None:
        self.calls = 0
        self.failures = 0
        self.images = list(images or ([] if image is None else [image]))
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
        if lowered.startswith("sandbox:"):
            return self._image_uri(self.images[0])
        for candidate in self.images:
            if isinstance(candidate, str):
                names = {candidate.casefold(), Path(candidate).name.casefold(), Path(candidate).stem.casefold()}
                if lowered in names:
                    return self._image_uri(candidate)

        match = re.fullmatch(r"(?:(?:input[_ -]?)?image|img)(?:[_ -]?(\d+))?(?:\.[a-z0-9]+)?", lowered)
        if match and match.group(1) is not None:
            number = int(match.group(1))
            index = 0 if number == 0 else number - 1
            if 0 <= index < len(self.images):
                return self._image_uri(self.images[index])
        if len(self.images) == 1:
            return self._image_uri(self.images[0])
        raise ToolGatewayError(
            f"ambiguous image reference {reference!r}; use img_1 through img_{len(self.images)}"
        )

    def _remember_image_results(self, result: str | list[dict[str, Any]]) -> None:
        if not isinstance(result, list):
            return
        for part in result:
            if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None:
                self.images.append(part["image"])

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
        max_attempts = max(1, int(os.getenv("TOOL_MAX_ATTEMPTS", "2")))
        payload: dict[str, Any] | None = None
        last_error = "unknown tool gateway failure"
        last_status: int | None = None

        for attempt in range(1, max_attempts + 1):
            retryable = True
            try:
                response = self.http.post(
                    f"{self.base_url}/v1/tools/{name}",
                    json={"callId": f"train-{uuid.uuid4()}", "arguments": arguments},
                    timeout=self.timeout,
                )
                last_status = response.status_code
                try:
                    decoded = response.json()
                except ValueError:
                    last_error = (
                        f"invalid JSON from gateway (HTTP {response.status_code}): "
                        f"{response.text[:300]!r}"
                    )
                else:
                    if not isinstance(decoded, dict):
                        last_error = f"invalid gateway payload type: {type(decoded).__name__}"
                    elif response.ok and decoded.get("ok"):
                        payload = decoded
                        break
                    else:
                        error = decoded.get("error") or {}
                        last_error = str(error.get("message") or f"{name} failed")
                        retryable = bool(error.get("retryable", response.status_code >= 500))
            except requests.RequestException as exc:
                last_error = f"transport failure: {exc}"

            if attempt < max_attempts and retryable:
                self._trace({
                    "tool": name,
                    "ok": False,
                    "retry": True,
                    "attempt": attempt,
                    "arguments": arguments,
                    "status": last_status,
                    "error": last_error,
                })
                time.sleep(0.25 * attempt)
                continue
            break

        if payload is None:
            self.failures += 1
            self._trace({
                "tool": name,
                "ok": False,
                "arguments": arguments,
                "status": last_status,
                "error": last_error,
                "durationMs": round((time.perf_counter() - started) * 1000),
            })
            raise ToolGatewayError(f"{name} failed: {last_error}")
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

    def web_search(self, q: str, hl: str = "en") -> str:
        """Search the public web for ranked evidence.

        Args:
            q: A focused textual search query.
            hl: Search language code.

        Returns:
            Titles, URLs, and evidence snippets.
        """
        return str(self._call("web_search", {"q": q, "hl": hl}))

    def text_search(
        self,
        q: str = "",
        query: str = "",
        hl: str = "en",
        top_k: int = 5,
        lang: str = "",
    ) -> str:
        """Search for textual evidence related to a question or entity.

        Args:
            q: A focused textual search query.
            query: Alternative spelling for q.
            hl: Search language code.
            top_k: Maximum number of pages to summarize.
            lang: Alternative spelling for hl.

        Returns:
            Ranked textual evidence.
        """
        return str(self._call("text_search", {"q": q or query, "hl": hl or lang or "en", "top_k": top_k}))

    def image_search(self, url: str) -> str:
        """Find visually similar images and source pages.

        Args:
            url: A public image URL, image reference, or artifact URI.

        Returns:
            Visual matches and their source metadata.
        """
        return str(self._call("image_search", {"url": self._resolve_image_reference(url)}))

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
        self._remember_image_results(result)
        return result if isinstance(result, list) else [{"type": "text", "text": result}]

    def sharpen(self, image: str, amount: float = 1.5) -> list[dict[str, Any]]:
        """Sharpen a blurry image and return the enhanced image to the model.

        Args:
            image: A current input image or transformed image reference.
            amount: Sharpening strength from zero to five.

        Returns:
            A text observation and the sharpened image.
        """
        result = self._call("sharpen", {"image": self._resolve_image_reference(image), "amount": amount})
        self._remember_image_results(result)
        return result if isinstance(result, list) else [{"type": "text", "text": result}]

    def super_resolution(self, image: str, scale: int = 4) -> list[dict[str, Any]]:
        """Upscale a low-resolution image and return it to the model.

        Args:
            image: A current input image or transformed image reference.
            scale: Upscaling factor from 1.1 to four.

        Returns:
            A text observation and the upscaled image.
        """
        result = self._call("super_resolution", {"image": self._resolve_image_reference(image), "scale": scale})
        self._remember_image_results(result)
        return result if isinstance(result, list) else [{"type": "text", "text": result}]

    def perspective_correct(self, image: str) -> list[dict[str, Any]]:
        """Correct mild rotation or perspective distortion and return the result.

        Args:
            image: A current input image or transformed image reference.
        Returns:
            A text observation and the corrected image.
        """
        result = self._call("perspective_correct", {"image": self._resolve_image_reference(image)})
        self._remember_image_results(result)
        return result if isinstance(result, list) else [{"type": "text", "text": result}]

    def layout_parsing(
        self,
        image: str = "",
        file_path: str = "",
        use_chart_recognition: bool = False,
        use_doc_orientation_classify: bool = False,
    ) -> str:
        """Extract structured text and reading order from a document image.

        Args:
            image: A public image URL, image reference, or artifact URI.
            file_path: Optional absolute image path.
            use_chart_recognition: Enable chart recognition.
            use_doc_orientation_classify: Enable orientation classification.

        Returns:
            Structured OCR text.
        """
        source = file_path or image
        resolved = self._resolve_image_reference(source)
        return str(self._call("layout_parsing", {
            "image": resolved,
            "use_chart_recognition": use_chart_recognition,
            "use_doc_orientation_classify": use_doc_orientation_classify,
        }))
