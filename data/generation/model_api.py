from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class RelayResponse:
    text: str
    raw: dict[str, Any]


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"model did not return a JSON object: {text[:300]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class RelayModel:
    """Small provider-neutral client for the configured OpenAI-compatible relay."""

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("MODEL_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("MODEL_NAME", "")
        self.style = os.environ.get("MODEL_API_STYLE", "responses").lower()
        self.timeout = float(os.environ.get("MODEL_API_TIMEOUT", "180"))
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("MODEL_BASE_URL, OPENAI_API_KEY, and MODEL_NAME are required")
        self.session = requests.Session()

    def _endpoint(self) -> str:
        suffix = "/chat/completions" if self.style == "chat" else "/responses"
        return self.base_url + suffix

    @staticmethod
    def _content(text: str, image_url: str | None, *, style: str) -> list[dict[str, Any]]:
        text_type = "text" if style == "chat" else "input_text"
        content: list[dict[str, Any]] = [{"type": text_type, "text": text}]
        if image_url:
            if style == "chat":
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                content.append({"type": "input_image", "image_url": image_url})
        return content

    def complete(
        self,
        prompt: str,
        *,
        image_url: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> RelayResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.style == "chat":
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": self._content(prompt, image_url, style="chat")})
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_output_tokens}
        else:
            input_messages: list[dict[str, Any]] = []
            if system:
                input_messages.append({"role": "system", "content": [{"type": "input_text", "text": system}]})
            input_messages.append({"role": "user", "content": self._content(prompt, image_url, style="responses")})
            payload = {"model": self.model, "input": input_messages, "temperature": temperature, "max_output_tokens": max_output_tokens, "store": False}
        response = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.post(self._endpoint(), headers=headers, json=payload, timeout=self.timeout)
                if response.status_code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError(f"relay request failed: {exc}") from exc
                time.sleep(2 ** attempt)
        if response is None:
            raise RuntimeError(f"relay request failed: {last_error}")
        if not response.ok:
            raise RuntimeError(f"relay returned HTTP {response.status_code}: {response.text[:500]}")
        raw = response.json()
        text = self._extract_text(raw)
        if not text:
            raise RuntimeError("relay returned an empty response")
        return RelayResponse(text=text, raw=raw)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            content = (choices[0].get("message") or {}).get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        chunks: list[str] = []
        for item in payload.get("output") or []:
            for content in item.get("content") or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "".join(chunks)

    def json(self, prompt: str, *, image_url: str | None = None, system: str | None = None, max_output_tokens: int = 2048) -> dict[str, Any]:
        return _json_object(self.complete(prompt, image_url=image_url, system=system, max_output_tokens=max_output_tokens).text)
