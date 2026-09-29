from __future__ import annotations

import json
import os
import re
import unicodedata
import time
from collections import Counter
from typing import Any

import requests


def _trace_reward(kind: str, completion: Any, score: float, **fields: Any) -> None:
    path = os.getenv("REWARD_TRACE_FILE", "").strip()
    if not path:
        return
    record = {
        "time": time.time(),
        "kind": kind,
        "completion": _completion_text(completion)[:4000],
        "rawCompletion": completion,
        "score": score,
        **fields,
    }
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _completion_text(completion.get("content", ""))
    if isinstance(completion, list):
        chunks: list[str] = []
        for item in completion:
            if isinstance(item, dict) and item.get("role") == "assistant":
                chunks.append(_completion_text(item.get("content", "")))
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(completion or "")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    scores: list[float] = []
    for completion, expected in zip(completions, answer, strict=True):
        prediction = _completion_text(completion)
        references = expected if isinstance(expected, list) else [expected]
        normalized_prediction = _normalize(prediction)
        exact = any(_normalize(str(reference)) in normalized_prediction for reference in references if _normalize(str(reference)))
        f1 = max((_token_f1(prediction, str(reference)) for reference in references), default=0.0)
        score = 1.0 if exact else f1
        scores.append(score)
        _trace_reward("answer", completion, score, references=references)
    return scores


def final_answer_reward(completions: list[Any], **_: Any) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        text = _completion_text(completion).strip()
        malformed = any(marker in text for marker in ("<tool_call>", "Traceback", "TOOL_GATEWAY_UNAVAILABLE"))
        score = 0.2 if text and not malformed else -0.5
        scores.append(score)
        _trace_reward("final_answer", completion, score)
    return scores


def fatal_review_reward(completions: list[Any], **_: Any) -> list[float]:
    """Optionally penalize only suspicious final answers; correctness stays rule-based."""
    base_url = os.getenv("FATAL_REVIEW_BASE_URL", "").rstrip("/")
    model = os.getenv("FATAL_REVIEW_MODEL", "")
    if not base_url or not model:
        return [0.0] * len(completions)
    scores: list[float] = []
    for completion in completions:
        text = _completion_text(completion)
        if text.strip() and not any(marker in text for marker in ("Traceback", "tool failed", "TOOL_")):
            scores.append(0.0)
            continue
        prompt = (
            "Classify only whether this answer has a fatal execution failure. "
            "Do not judge factual correctness. Return JSON {\"fatal\": true|false}.\n\n" + text[:6000]
        )
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('FATAL_REVIEW_API_KEY', 'local')}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 32},
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0))
            scores.append(-1.0 if parsed.get("fatal") is True else 0.0)
        except Exception:
            scores.append(0.0)
    return scores
