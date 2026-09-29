"""Local tool gateway for the Pi agent.

The language model only chooses tools. This process performs network requests,
image downloads, image transforms and OCR. Tool failures are returned as
structured errors so the agent cannot mistake a placeholder for evidence.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - installation error is reported by health
    requests = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - installation error is reported by health
    Image = None  # type: ignore[assignment]


ROOT = Path(os.getenv("TOOL_ARTIFACT_DIR", ".artifacts")).expanduser().resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = int(os.getenv("TOOL_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
MAX_RESULTS = 10
SERPER_URL = os.getenv("SERPER_SEARCH_URL", "https://google.serper.dev/search")
SERPER_LENS_URL = os.getenv("SERPER_LENS_URL", "https://google.serper.dev/lens")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_READER_URL = os.getenv("JINA_READER_URL", "https://r.jina.ai").rstrip("/")
JINA_ALLOW_ANONYMOUS = os.getenv("JINA_ALLOW_ANONYMOUS", "true").lower() in {"1", "true", "yes"}
LAYOUT_URL = os.getenv("LAYOUT_PARSING_URL", "").strip()
IMAGE_CAPTION_BASE_URL = os.getenv("IMAGE_CAPTION_BASE_URL", "").rstrip("/")
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "").strip()
IMAGE_CAPTION_API_KEY = os.getenv("IMAGE_CAPTION_API_KEY", "local").strip()
COS_SECRET_ID = os.getenv("COS_SECRET_ID", "").strip()
COS_SECRET_KEY = os.getenv("COS_SECRET_KEY", "").strip()
COS_REGION = os.getenv("COS_REGION", "").strip()
COS_BUCKET = os.getenv("COS_BUCKET", "").strip()
COS_OBJECT_PREFIX = os.getenv("COS_OBJECT_PREFIX", "sightline-agent").strip("/")
COS_SIGNED_URL_TTL_SECONDS = int(os.getenv("COS_SIGNED_URL_TTL_SECONDS", "600"))
COS_DELETE_AFTER_SEARCH = os.getenv("COS_DELETE_AFTER_SEARCH", "true").lower() in {"1", "true", "yes"}


class ToolFailure(RuntimeError):
    """A user-visible tool failure with a stable error code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _query(arguments: dict[str, Any]) -> str:
    value = arguments.get("query") or arguments.get("q")
    query = " ".join(_text(value).split()).strip("\\/|,;:!?()[]{}<>\"'`~")
    if len(query) < 2 or not any(char.isalnum() for char in query):
        raise ToolFailure("INVALID_QUERY", "query must contain meaningful letters or numbers")
    return query[:512]


def _require_serper() -> None:
    if not SERPER_API_KEY:
        raise ToolFailure("MISSING_SERPER_API_KEY", "SERPER_API_KEY is not configured")
    if requests is None:
        raise ToolFailure("MISSING_DEPENDENCY", "requests is required")


def _cos_configured() -> bool:
    return all((COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET))


def _cos_client():
    if not _cos_configured():
        raise ToolFailure(
            "COS_NOT_CONFIGURED",
            "local reverse-image search requires COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, and COS_BUCKET",
        )
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as exc:
        raise ToolFailure("MISSING_DEPENDENCY", "cos-python-sdk-v5 is required for local reverse-image search") from exc
    config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY, Scheme="https")
    return CosS3Client(config)


def _upload_private_image(data: bytes, mime: str) -> tuple[Any, str, str]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolFailure("IMAGE_TOO_LARGE", f"image exceeds {MAX_IMAGE_BYTES} bytes")
    client = _cos_client()
    suffix = mimetypes.guess_extension(mime) or ".png"
    key = f"{COS_OBJECT_PREFIX}/{time.strftime('%Y/%m/%d')}/{uuid.uuid4().hex}{suffix}"
    try:
        client.put_object(Bucket=COS_BUCKET, Body=data, Key=key, ContentType=mime)
        signed_url = client.get_presigned_url(
            Method="GET",
            Bucket=COS_BUCKET,
            Key=key,
            Expired=max(60, min(COS_SIGNED_URL_TTL_SECONDS, 3600)),
        )
    except Exception as exc:
        raise ToolFailure("COS_UPLOAD_ERROR", f"failed to stage image for reverse search: {exc}", retryable=True) from exc
    return client, key, signed_url


def _delete_private_image(client: Any, key: str) -> None:
    if not COS_DELETE_AFTER_SEARCH:
        return
    try:
        client.delete_object(Bucket=COS_BUCKET, Key=key)
    except Exception as exc:
        print(f"temporary COS cleanup failed for {key}: {exc}")


def _serper_search(query: str, top_k: int) -> list[dict[str, Any]]:
    _require_serper()
    response = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "location": "United States", "hl": "en", "num": min(max(top_k, 1), MAX_RESULTS)},
        timeout=30,
    )
    if not response.ok:
        raise ToolFailure("SERPER_HTTP_ERROR", f"Serper returned HTTP {response.status_code}: {response.text[:300]}", retryable=response.status_code >= 500)
    return list(response.json().get("organic") or [])


def _format_search(query: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"No search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, item in enumerate(items, 1):
        title = _text(item.get("title")) or "Untitled"
        link = _text(item.get("link"))
        snippet = _text(item.get("snippet"))
        lines.append(f"\n{index}. {title}\nURL: {link}\nSnippet: {snippet}")
    return "\n".join(lines)


def _serper_lens(public_url: str, *, include_source_url: bool) -> str:
    _require_serper()
    response = requests.post(
        SERPER_LENS_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"url": public_url},
        timeout=45,
    )
    if not response.ok:
        raise ToolFailure(
            "SERPER_LENS_ERROR",
            f"Serper Lens returned HTTP {response.status_code}: {response.text[:300]}",
            retryable=response.status_code >= 500,
        )
    payload = response.json()
    matches = (
        payload.get("organic")
        or payload.get("visual_matches")
        or payload.get("visualMatches")
        or payload.get("image_results")
        or payload.get("images")
        or []
    )
    if not isinstance(matches, list) or not matches:
        message = payload.get("message") or payload.get("error") or "Serper Lens returned no image matches"
        raise ToolFailure("SERPER_LENS_EMPTY", str(message))
    result: dict[str, Any] = {
        "provider": "serper_lens",
        "mode": "reverse_image",
        "matches": matches[:MAX_RESULTS],
        "credits": payload.get("credits"),
    }
    if include_source_url:
        result["url"] = public_url
    else:
        result["source"] = "private_temporary_artifact"
    return json.dumps(result, ensure_ascii=False)


def _jina_read(url: str) -> str:
    if not JINA_API_KEY and not JINA_ALLOW_ANONYMOUS:
        raise ToolFailure("MISSING_JINA_API_KEY", "JINA_API_KEY is not configured")
    if requests is None:
        raise ToolFailure("MISSING_DEPENDENCY", "requests is required")
    headers = {"Accept": "text/plain"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        response = requests.get(f"{JINA_READER_URL}/{url}", headers=headers, timeout=20)
    except requests.RequestException:
        return _direct_read(url)
    # A newly created/free key can legitimately have zero paid balance. The
    # Reader's anonymous tier is slower but still useful for local smoke tests.
    if response.status_code in {401, 402} and JINA_ALLOW_ANONYMOUS and JINA_API_KEY:
        try:
            response = requests.get(f"{JINA_READER_URL}/{url}", headers={"Accept": "text/plain"}, timeout=20)
        except requests.RequestException:
            return _direct_read(url)
    if not response.ok:
        return _direct_read(url)
    if not response.text.strip():
        raise ToolFailure("JINA_EMPTY_RESPONSE", "Jina Reader returned empty content")
    return response.text[:12000]


def _direct_read(url: str) -> str:
    if requests is None:
        raise ToolFailure("MISSING_DEPENDENCY", "requests is required")
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SightlineAgent/0.2)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ToolFailure("PAGE_READ_ERROR", f"page reader and direct fetch failed: {exc}", retryable=True) from exc
    content_type = response.headers.get("content-type", "")
    if "text/" not in content_type and "json" not in content_type and "xml" not in content_type:
        raise ToolFailure("UNSUPPORTED_PAGE_TYPE", f"cannot extract page content type: {content_type}")
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ToolFailure("MISSING_DEPENDENCY", "beautifulsoup4 is required for direct page fallback") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if not text:
        raise ToolFailure("PAGE_EMPTY_RESPONSE", "direct page fetch returned no readable text")
    return text[:12000]


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    match = re.match(r"^data:([^;,]+);base64,(.+)$", uri, re.DOTALL)
    if not match:
        raise ToolFailure("INVALID_IMAGE_URI", "invalid data URI")
    return base64.b64decode(match.group(2)), match.group(1)


def _download(uri: str) -> tuple[bytes, str]:
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    if uri.startswith("artifact://"):
        path = ROOT / uri.removeprefix("artifact://")
        if not path.exists() or not path.is_file():
            raise ToolFailure("IMAGE_NOT_FOUND", f"artifact does not exist: {uri}")
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png"
    if uri.startswith("file://"):
        path = Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path)).resolve()
        if not path.is_file():
            raise ToolFailure("IMAGE_NOT_FOUND", f"file does not exist: {uri}")
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png"
    if uri.startswith(("http://", "https://")):
        request = urllib.request.Request(uri, headers={"User-Agent": "sightline-agent/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
                mime = response.headers.get_content_type() or "image/png"
        except Exception as exc:
            raise ToolFailure("IMAGE_DOWNLOAD_ERROR", f"failed to download image: {exc}", retryable=True) from exc
        return data, mime
    raise ToolFailure("INVALID_IMAGE_URI", "image must be an http(s), file, artifact, or data URI")


def _save_image(data: bytes, mime: str, *, prefix: str) -> tuple[str, str]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolFailure("IMAGE_TOO_LARGE", f"image exceeds {MAX_IMAGE_BYTES} bytes")
    suffix = mimetypes.guess_extension(mime) or ".png"
    name = f"{prefix}-{uuid.uuid4().hex}{suffix}"
    path = ROOT / name
    path.write_bytes(data)
    return f"artifact://{name}", str(path)


def _caption_image(data: bytes, mime: str) -> str:
    if not IMAGE_CAPTION_BASE_URL or not IMAGE_CAPTION_MODEL:
        raise ToolFailure(
            "LOCAL_IMAGE_SEARCH_NOT_CONFIGURED",
            "local image_search requires IMAGE_CAPTION_BASE_URL and IMAGE_CAPTION_MODEL",
        )
    if requests is None:
        raise ToolFailure("MISSING_DEPENDENCY", "requests is required")
    encoded = base64.b64encode(data).decode("ascii")
    response = requests.post(
        f"{IMAGE_CAPTION_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {IMAGE_CAPTION_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": IMAGE_CAPTION_MODEL,
            "temperature": 0,
            "max_tokens": 160,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify the main entity, place, artwork, text, and distinctive visual details. Return one concise web-search query only."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
        },
        timeout=60,
    )
    if not response.ok:
        raise ToolFailure("CAPTION_HTTP_ERROR", f"caption service returned HTTP {response.status_code}: {response.text[:300]}", retryable=response.status_code >= 500)
    try:
        caption = _query({"query": response.json()["choices"][0]["message"]["content"]})
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ToolFailure("CAPTION_INVALID_RESPONSE", "caption service returned an invalid response") from exc
    return caption


def _require_pillow() -> None:
    if Image is None:
        raise ToolFailure("MISSING_PILLOW", "Pillow is required for image tools")


def _enhance_image(name: str, arguments: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Apply a real, deterministic image enhancement and return an artifact URI."""
    _require_pillow()
    data, _mime = _resolve_image_arg(arguments)
    try:
        from PIL import ImageEnhance, ImageFilter
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if name == "sharpen":
            amount = max(0.0, min(float(arguments.get("amount", 1.5)), 5.0))
            image = ImageEnhance.Sharpness(image).enhance(amount).filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        elif name == "super_resolution":
            scale = max(1.1, min(float(arguments.get("scale", 2.0)), 4.0))
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        elif name == "perspective_correct":
            # A conservative deskew fallback for mildly tilted inputs. The operation is
            # intentionally non-destructive when no quadrilateral is supplied.
            angle = max(-20.0, min(float(arguments.get("angle", 0.0)), 20.0))
            image = image.rotate(angle, expand=True, fillcolor="white")
        else:
            raise ToolFailure("UNKNOWN_TOOL", f"unknown enhancement tool: {name}")
        output = io.BytesIO()
        image.save(output, format="PNG")
    except ToolFailure:
        raise
    except Exception as exc:
        raise ToolFailure("IMAGE_ENHANCEMENT_ERROR", f"{name} failed: {exc}") from exc
    uri, path = _save_image(output.getvalue(), "image/png", prefix=name)
    artifact = {"kind": "image", "id": uri.removeprefix("artifact://"), "uri": uri, "mimeType": "image/png"}
    return f"{name} result saved as {uri} ({image.width}x{image.height}).", [artifact]


def _local_layout_parse(data: bytes) -> dict[str, Any]:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ToolFailure(
            "LAYOUT_NOT_CONFIGURED",
            "set LAYOUT_PARSING_URL or install rapidocr-onnxruntime for local OCR",
        ) from exc
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        raw, elapsed = RapidOCR()(data)
    except Exception as exc:
        raise ToolFailure("LAYOUT_LOCAL_ERROR", f"local OCR failed: {exc}") from exc
    blocks: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, score = item[0], _text(item[1]), float(item[2])
        points = [[round(float(point[0]), 2), round(float(point[1]), 2)] for point in box]
        blocks.append({"type": "text", "text": text, "score": round(score, 4), "box": points})
    blocks.sort(key=lambda block: (min(point[1] for point in block["box"]), min(point[0] for point in block["box"])))
    for index, block in enumerate(blocks):
        block["readingOrder"] = index
    return {
        "provider": "rapidocr_onnxruntime",
        "width": image.width,
        "height": image.height,
        "blocks": blocks,
        "text": "\n".join(block["text"] for block in blocks),
        "elapsedSeconds": elapsed,
    }


def _resolve_image_arg(arguments: dict[str, Any]) -> tuple[bytes, str]:
    value = _text(arguments.get("image") or arguments.get("url"))
    if not value:
        raise ToolFailure("MISSING_IMAGE", "image or url is required")
    return _download(value)


def execute_tool(name: str, arguments: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if name in {"web_search", "text_search"}:
        query = _query(arguments)
        items = _serper_search(query, int(arguments.get("topK") or arguments.get("top_k") or 5))
        return _format_search(query, items), []
    if name == "visit":
        url = _text(arguments.get("url"))
        if not urllib.parse.urlparse(url).scheme:
            raise ToolFailure("INVALID_URL", "url must include http:// or https://")
        return _jina_read(url), []
    if name == "image_search":
        value = _text(arguments.get("image") or arguments.get("url"))
        if not value:
            raise ToolFailure("MISSING_IMAGE", "image or url is required")
        if value.startswith(("http://", "https://")):
            return _serper_lens(value, include_source_url=True), []
        data, mime = _download(value)
        if _cos_configured():
            client, key, signed_url = _upload_private_image(data, mime)
            try:
                return _serper_lens(signed_url, include_source_url=False), []
            finally:
                _delete_private_image(client, key)
        caption = _caption_image(data, mime)
        items = _serper_search(caption, MAX_RESULTS)
        return json.dumps({"provider": "local_vlm_and_serper", "mode": "semantic_image", "query": caption, "matches": items}, ensure_ascii=False), []
    if name == "crop":
        _require_pillow()
        data, _mime = _resolve_image_arg(arguments)
        try:
            source = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ToolFailure("INVALID_IMAGE", f"cannot decode image: {exc}") from exc
        x, y = int(arguments.get("x", 0)), int(arguments.get("y", 0))
        width, height = int(arguments.get("width", 0)), int(arguments.get("height", 0))
        if width <= 0 or height <= 0:
            raise ToolFailure("INVALID_CROP", "width and height must be positive")
        left, top = max(0, x), max(0, y)
        right, bottom = min(source.width, left + width), min(source.height, top + height)
        if right <= left or bottom <= top:
            raise ToolFailure("INVALID_CROP", "crop is outside image bounds")
        output = io.BytesIO()
        source.crop((left, top, right, bottom)).save(output, format="PNG")
        uri, _ = _save_image(output.getvalue(), "image/png", prefix="crop")
        artifact = {"kind": "image", "id": uri.removeprefix("artifact://"), "uri": uri, "mimeType": "image/png"}
        return f"Cropped image saved as {uri} ({right-left}x{bottom-top}).", [artifact]
    if name == "layout_parsing":
        _require_pillow()
        data, mime = _resolve_image_arg(arguments)
        if LAYOUT_URL:
            if requests is None:
                raise ToolFailure("MISSING_DEPENDENCY", "requests is required")
            response = requests.post(LAYOUT_URL, files={"file": ("image", data, mime)}, timeout=90)
            if not response.ok:
                raise ToolFailure("LAYOUT_HTTP_ERROR", f"layout service returned HTTP {response.status_code}: {response.text[:300]}", retryable=response.status_code >= 500)
            return json.dumps(response.json(), ensure_ascii=False), []
        return json.dumps(_local_layout_parse(data), ensure_ascii=False), []
    if name in {"sharpen", "perspective_correct", "super_resolution"}:
        return _enhance_image(name, arguments)
    raise ToolFailure("UNKNOWN_TOOL", f"unknown tool: {name}")


def result(call_id: str, name: str, text: str, *, ok: bool = True, error: dict[str, Any] | None = None, artifacts: list[dict[str, str]] | None = None, duration_ms: int = 0) -> dict[str, Any]:
    body: dict[str, Any] = {"callId": call_id, "name": name, "ok": ok, "content": [{"type": "text", "text": text}], "durationMs": duration_ms}
    if error:
        body["error"] = error
    if artifacts:
        body["artifacts"] = artifacts
        body["content"].extend({"type": "image", "data": _artifact_data(artifact), "mimeType": artifact.get("mimeType", "image/png")} for artifact in artifacts)
    return body


def _artifact_data(artifact: dict[str, str]) -> str:
    data, _mime = _download(artifact["uri"])
    return base64.b64encode(data).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    server_version = "sightline-agent-tools/0.2"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            try:
                import rapidocr_onnxruntime  # noqa: F401
                local_layout = True
            except ImportError:
                local_layout = False
            self._send(200, {"ok": True, "serper": bool(SERPER_API_KEY), "jina": bool(JINA_API_KEY), "cos": _cos_configured(), "layout": bool(LAYOUT_URL) or local_layout, "layoutProvider": "remote" if LAYOUT_URL else ("rapidocr" if local_layout else None), "imageCaptionFallback": bool(IMAGE_CAPTION_BASE_URL and IMAGE_CAPTION_MODEL), "pillow": Image is not None})
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/tools/"):
            self._send(404, {"error": "not_found"})
            return
        name = self.path.rsplit("/", 1)[-1]
        call_id = "unknown"
        try:
            length = int(self.headers.get("content-length", "0"))
            if length > 32 * 1024 * 1024:
                raise ToolFailure("REQUEST_TOO_LARGE", "request exceeds 32 MiB")
            payload = json.loads(self.rfile.read(length) or b"{}")
            call_id = str(payload.get("callId", "unknown"))
            arguments = payload.get("arguments") or {}
            started = time.perf_counter()
            text, artifacts = execute_tool(name, arguments)
            body = result(call_id, name, text, artifacts=artifacts, duration_ms=round((time.perf_counter() - started) * 1000))
            self._send(200, body)
        except ToolFailure as exc:
            self._send(422, result(call_id, name, str(exc), ok=False, error={"code": exc.code, "message": str(exc), "retryable": exc.retryable}))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send(500, result(call_id, name, "Tool execution failed.", ok=False, error={"code": "TOOL_EXECUTION_ERROR", "message": str(exc), "retryable": True}))

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


if __name__ == "__main__":
    host = os.getenv("TOOL_HOST", "127.0.0.1")
    port = int(os.getenv("TOOL_PORT", "8090"))
    print(f"sightline-agent tool gateway listening on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
