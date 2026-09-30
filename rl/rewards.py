from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import requests


ACCURACY_WEIGHT = 0.8
QUERY_WEIGHT = 0.2

_FORMAT_PATTERN = re.compile(
    r"^<think>[\s\S]*?</think>\s*(?:<tool_call>[\s\S]*?</tool_call>|<response>[\s\S]*?</response>)\s*$"
)
_TOOL_ERROR_MARKERS = (
    "traceback",
    "tool execution failed",
    "tool_gateway_unavailable",
    "tool_gateway_error",
    "[json parse error]",
    "error executing",
)


class JudgeConfigurationError(RuntimeError):
    pass


def validate_judge_config() -> None:
    if not _judge_endpoint("accuracy") or not _judge_model("accuracy"):
        raise JudgeConfigurationError(
            "semantic correctness Judge is required; set JUDGE_API_BASE_URL (or JUDGE_URL), "
            "JUDGE_API_KEY, and JUDGE_MODEL"
        )
    if not _judge_endpoint("query") or not _judge_model("query"):
        raise JudgeConfigurationError(
            "query-quality Judge is required; set QUERY_JUDGE_API_BASE_URL and QUERY_JUDGE_MODEL "
            "or let them inherit the JUDGE_* settings"
        )


def _judge_endpoint(kind: str) -> str:
    base = os.getenv("QUERY_JUDGE_API_BASE_URL", "").strip() if kind == "query" else ""
    base = base or os.getenv("JUDGE_API_BASE_URL", "").strip() or os.getenv("JUDGE_URL", "").strip()
    if not base:
        return ""
    if re.search(r"/chat/completions/?$", base):
        return base.rstrip("/")
    return f"{base.rstrip('/')}/chat/completions"


def _judge_model(kind: str) -> str:
    if kind == "query":
        return os.getenv("QUERY_JUDGE_MODEL", "").strip() or os.getenv("JUDGE_MODEL", "").strip()
    return os.getenv("JUDGE_MODEL", "").strip()


def _judge_key(kind: str) -> str:
    if kind == "query":
        return os.getenv("QUERY_JUDGE_API_KEY", "").strip() or os.getenv("JUDGE_API_KEY", "").strip()
    return os.getenv("JUDGE_API_KEY", "").strip()


def _call_judge(kind: str, prompt: str, *, max_tokens: int) -> str:
    endpoint = _judge_endpoint(kind)
    model = _judge_model(kind)
    if not endpoint or not model:
        raise JudgeConfigurationError(f"{kind} Judge is not configured")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    key = _judge_key(kind)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    attempts = max(1, int(os.getenv("JUDGE_RETRY_ATTEMPTS", "3")))
    timeout = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120"))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{kind} Judge failed after {attempts} attempts: {last_error}")


def _messages(completion: Any) -> list[dict[str, Any]]:
    if isinstance(completion, dict):
        return [completion]
    if isinstance(completion, list):
        return [item for item in completion if isinstance(item, dict)]
    return [{"role": "assistant", "content": str(completion or "")}]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _assistant_text(message: dict[str, Any]) -> str:
    return _content_text(message.get("content", "")).strip()


def _final_answer(completion: Any) -> str:
    for message in reversed(_messages(completion)):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        text = _assistant_text(message)
        match = re.search(r"<response>([\s\S]*?)</response>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _fatal_prefix(completion: Any) -> tuple[list[dict[str, Any]], bool]:
    messages = _messages(completion)
    consecutive_errors = 0
    error_run_start = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        is_error = False
        if index + 1 < len(messages) and messages[index + 1].get("role") == "tool":
            observation = _content_text(messages[index + 1].get("content", "")).casefold()
            is_error = any(marker in observation for marker in _TOOL_ERROR_MARKERS)
        if is_error:
            if consecutive_errors == 0:
                error_run_start = index
            consecutive_errors += 1
            if consecutive_errors >= 3:
                return messages[:error_run_start], True
        else:
            consecutive_errors = 0
    return messages, not bool(_final_answer(messages))


def _question(prompt: Any, explicit: Any = None) -> str:
    if explicit:
        return str(explicit)
    messages = prompt if isinstance(prompt, list) else [prompt]
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            chunks.append(_content_text(message.get("content")))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _reference(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return " | ".join(str(item) for item in answer)
    return str(answer or "")


def _normalized_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _format_score(completion: Any) -> float:
    messages = _messages(completion)
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        return 0.0
    scores: list[float] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        raw = str(message.get("raw_content") or _assistant_text(message)).strip()
        tool_calls = message.get("tool_calls") or []
        if raw and _FORMAT_PATTERN.fullmatch(raw):
            score = 1.0
        elif len(tool_calls) == 1 and re.fullmatch(
            r"<think>[\s\S]*?</think>", _assistant_text(message)
        ):
            score = 1.0
        else:
            score = 0.0
        if index + 1 < len(messages) and messages[index + 1].get("role") == "tool":
            text = _content_text(messages[index + 1].get("content", "")).casefold()
            if any(marker in text for marker in _TOOL_ERROR_MARKERS):
                score = 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def _trajectory_summary(completion: Any, max_steps: int = 20) -> str:
    lines: list[str] = []
    messages = _messages(completion)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in (message.get("tool_calls") or [])[:1]:
            function = call.get("function", call) if isinstance(call, dict) else {}
            name = str(function.get("name", "unknown"))
            arguments = function.get("arguments", {})
            result = ""
            if index + 1 < len(messages) and messages[index + 1].get("role") == "tool":
                result = _content_text(messages[index + 1].get("content", ""))[:400]
            lines.append(
                f"Step {len(lines) + 1}: {name}({json.dumps(arguments, ensure_ascii=False)[:300]})\n"
                f"  Result: {result}"
            )
            if len(lines) >= max_steps:
                break
        if len(lines) >= max_steps:
            break
    return "\n".join(lines)


def _accuracy(question: str, answer: Any, prediction: str) -> float:
    references = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_prediction = _normalized_answer(prediction)
    if any(normalized_prediction == _normalized_answer(item) for item in references):
        return 1.0
    prompt = (
        "You are an impartial judge evaluating whether a deep research report correctly answers the question.\n\n"
        f"[Question]\n{question}\n\n[Correct Answer]\n{_reference(answer)}\n\n"
        f"[Deep Research Report]\n{prediction}\n\n"
        "Determine whether the report gives the correct answer to the specific information requested.\n\n"
        "Instructions:\n"
        "1. Identify the central field requested by the question, such as a name, number, date, decision, object type, or scene type.\n"
        "2. Judge that central field, not whether the report repeats every explanatory detail in the reference.\n"
        "3. Accept semantic equivalents, paraphrases, compatible broader or narrower descriptions, and logically equivalent yes/no statements.\n"
        "4. A short answer is sufficient. Unasked-for side details do not invalidate an otherwise correct central answer.\n"
        "5. Reject answers that omit the requested field, contradict the reference, or give an incompatible specific identity.\n"
        "6. The answer need not use a particular final-answer format.\n\n"
        "Examples:\n"
        "- Asked for artwork type; reference says 'charcoal sketch by Renoir'; report says 'charcoal sketch by Picasso': yes, because the requested type is correct.\n"
        "- Reference says 'village scene'; report says 'Mediterranean village scene': yes.\n"
        "- Asked whether A has more than B; reference says 'No'; report says 'B has more than A': yes.\n"
        "- Reference says 'GameCube controller'; report says 'GameCube Nunchuk': no, because the specific identity conflicts.\n\n"
        "Output exactly one line and nothing else:\ncorrect: yes|no"
    )
    try:
        result = _call_judge("accuracy", prompt, max_tokens=32)
    except Exception:
        return 0.0
    match = re.search(r"correct\s*:\s*(yes|no)\b", result, re.IGNORECASE)
    return 1.0 if match and match.group(1).casefold() == "yes" else 0.0


def _query_utility(question: str, answer: Any, prediction: str, completion: Any) -> float:
    summary = _trajectory_summary(completion)
    if not summary:
        return 0.0
    prompt = (
        "You are an impartial judge evaluating the quality and utility of an agent's search trajectory.\n\n"
        f"[Question]\n{question}\n\n[Ground Truth]\n{_reference(answer)}\n\n"
        f"[Final Answer]\n{prediction or '(no final answer)'}\n\n[Trajectory]\n{summary}\n\n"
        "Evaluate image-search utility, text-search utility, logical query progression, modality complementarity, "
        "and the ratio of useful evidence to noise. Use 0.0 for no useful evidence, 0.3 for mostly noise, "
        "0.5 for mixed utility, 0.7 for a good search strategy, and 1.0 for targeted efficient retrieval.\n\n"
        "Output exactly one line and nothing else:\nscore: <float from 0.0 to 1.0>"
    )
    try:
        result = _call_judge("query", prompt, max_tokens=32)
    except Exception:
        return 0.0
    match = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", result, re.IGNORECASE)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(1))))


def _score_one(
    completion: Any,
    expected: Any,
    source_prompt: Any,
    explicit_question: Any,
) -> tuple[float, float, float, float, float]:
    query = _question(source_prompt, explicit_question)
    scored_completion, fatal = _fatal_prefix(completion)
    prediction = "" if fatal else _final_answer(scored_completion)
    format_score = _format_score(scored_completion)
    accuracy = _accuracy(query, expected, prediction) if prediction else 0.0
    query_utility = _query_utility(query, expected, prediction, scored_completion)
    total = format_score * (ACCURACY_WEIGHT * accuracy + QUERY_WEIGHT * query_utility)
    return total, accuracy, query_utility, format_score, float(fatal)


async def trajectory_reward(
    completions: list[Any],
    answer: list[Any],
    prompts: list[Any] | None = None,
    question: list[Any] | None = None,
    log_metric: Any = None,
    environments: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    source_prompts = prompts or [None] * len(completions)
    questions = question or [None] * len(completions)
    tasks = [
        asyncio.to_thread(_score_one, completion, expected, source_prompt, explicit_question)
        for completion, expected, source_prompt, explicit_question in zip(
            completions, answer, source_prompts, questions, strict=True
        )
    ]
    results = list(await asyncio.gather(*tasks))
    if environments is not None:
        for environment, row in zip(environments, results, strict=True):
            setattr(environment, "_sightline_fatal", bool(row[4]))
    if callable(log_metric) and results:
        names = (
            "reward/total",
            "reward/accuracy",
            "reward/query_utility",
            "reward/format",
            "reward/fatal",
        )
        for metric_index, name in enumerate(names):
            log_metric(name, sum(row[metric_index] for row in results) / len(results))
    return [row[0] for row in results]
