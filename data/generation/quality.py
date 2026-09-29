from __future__ import annotations

import re
from typing import Any

from .model_api import RelayModel


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def deterministic_leak_check(question: str, hidden_entities: list[str]) -> bool:
    normalized = _normal(question)
    return not any(_normal(entity) and _normal(entity) in normalized for entity in hidden_entities)


QUALITY_KEYS = (
    "answer_invariant",
    "unique_entity_path",
    "no_alias_leakage",
    "source_anchor_only",
    "visually_grounded",
    "needs_external_tools",
    "not_solved_by_one_reverse_image_search",
    "nontrivial",
)


def _strict_accept(result: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """Treat malformed or missing model verdicts as rejection, never as approval."""
    return all(result.get(key) is True for key in keys)


def quality_check(model: RelayModel, *, question: str, answer: str, source_title: str, image_url: str, hidden_entities: list[str]) -> dict[str, Any]:
    prompt = f"""You are a conservative dataset auditor. A false positive is worse than rejecting a sample.
Return JSON only with these eight boolean keys:
answer_invariant, unique_entity_path, no_alias_leakage, source_anchor_only, visually_grounded,
needs_external_tools, not_solved_by_one_reverse_image_search, nontrivial.

Audit the candidate as a multi-hop visual-to-text research task:
- answer_invariant: the stated answer is entailed by the answer-node evidence and the wording is unambiguous;
- unique_entity_path: the descriptions identify exactly one intended entity at every bridge, not a plausible class;
- no_alias_leakage concerns a listed hidden entity name, its alias or URL slug, or the exact expected answer.
  Do not infer leakage merely because a legitimate bridge clue is
  a proper name or makes the bridge uniquely identifiable; that is the purpose of the bridge clue;
- source_anchor_only: the visual source is referenced only through the image, without a textual paraphrase,
  location, date, or identifying attribute attached to that source. Relational navigation after visual
  identification, such as "its city", "its architect", or "the park linked from its page", is allowed and
  must not be treated as a textual description of the source;
- visually_grounded: the attached image is a useful representative image of the source/anchor entity, not the answer entity;
- the source/anchor must be referred to through a visual expression (for example, "the entity shown in the image");
  a textual description, location, date, or other identifying prose attached to the visual source itself is a
  leakage and must fail source_anchor_only. Do not reject relational navigation or named evidence that
  describes a later bridge;
- needs_external_tools: a capable model cannot solve it from pixels plus parametric memory alone;
- not_solved_by_one_reverse_image_search: a single reverse-image lookup of the attached image would not reveal the answer directly;
- nontrivial: solving requires at least two evidence-collection/reasoning steps.
Reject generic captions, one-hop lookups, answer-coincident images, vague descriptors, and any uncertainty.

Source entity (auditor-only metadata): {source_title}
Expected answer: {answer}
Question: {question}
Hidden names and aliases that must not occur: {hidden_entities}
Attached source image URL: {image_url}
"""
    result = model.json(prompt, image_url=image_url, max_output_tokens=650)
    # Exact names and answers are a deterministic string property. The model
    # separately audits semantic source paraphrases through source_anchor_only.
    result["no_alias_leakage"] = deterministic_leak_check(question, [*hidden_entities, answer])
    result["accepted"] = _strict_accept(result, QUALITY_KEYS)
    return result


def tool_necessity_check(model: RelayModel, *, question: str, answer: str, image_url: str) -> dict[str, Any]:
    """Reproduce the two-stage frozen-VLM difficulty filter used before rollouts."""
    prompt = f"""You are a strict difficulty filter for a multimodal search dataset. Return JSON only:
{{"answerable_without_tools": true|false, "single_reverse_image_sufficient": true|false,
  "reason": "short explanation"}}

Given the image and question, decide whether a strong vision-language model could answer correctly
without calling any external retrieval or image-processing tool. Then decide whether exactly one
reverse-image-search call on the initial image would reveal the answer directly. Use true only when
you are confident; uncertainty must be false for both flags. Treat an answer as tool-free when it can
be reliably inferred or guessed from geographic, temporal, taxonomic, or category clues already stated
in the question, even if the exact sentence is not visible in the image. The intended item is rejected if either
flag is true. The question must require identifying the visual anchor and following multiple external
facts rather than reading the answer from the image.

Question: {question}
Reference answer (filter-only metadata): {answer}
"""
    result = model.json(prompt, image_url=image_url, max_output_tokens=300)
    result["accepted"] = result.get("answerable_without_tools") is False and result.get("single_reverse_image_sufficient") is False
    return result


def final_answer_check(model: RelayModel, *, question: str, answer: str, response: str) -> bool:
    result = model.json(
        f"""Return JSON only {{"correct": true|false}}.
You are an impartial answer judge. Accept semantic equivalence, but reject guesses, contradictions,
unsupported alternatives, or an answer that only repeats an entity without answering the requested fact.
Question: {question}
Expected answer: {answer}
Deep research response: {response[:12000]}""",
        max_output_tokens=80,
    )
    return result.get("correct") is True


def process_check(model: RelayModel, *, question: str, answer: str, trace: str) -> dict[str, Any]:
    result = model.json(
        f"""Return JSON only with boolean key accepted, boolean key no_redundant_search, and numeric process_score in [0,1].
Judge the complete tool trajectory, not just the final answer. Check: (1) image search, if used, is
followed by textual verification; (2) queries are targeted and progressively refine the search; (3)
tool observations are used consistently; (4) the agent does not repeat failed or uninformative calls;
(5) visual enhancement/OCR is used only when justified; and (6) the final answer is supported by the
collected evidence. Do not reward a correct answer obtained through an invalid, fabricated, or circular trace.
Reject if the trajectory has a fatal tool-error cascade or if the reasoning contradicts observations.
Question: {question}
Expected answer: {answer}
Trajectory trace:
{trace[:14000]}""",
        max_output_tokens=250,
    )
    result["accepted"] = result.get("accepted") is True and result.get("no_redundant_search") is True and float(result.get("process_score", 0) or 0) >= 0.6
    return result
