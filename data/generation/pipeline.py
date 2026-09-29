from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter
import requests

from .clip_score import ClipScorer
from .model_api import RelayModel
from .quality import quality_check, tool_necessity_check
from .trajectory import AgentTrajectoryClient
from .tool_schema import TOOLS
from .wikimedia import WikimediaClient


QUESTION_SYSTEM = "You construct difficult but answerable image-grounded multi-hop questions. Return only strict JSON and reject shortcuts."
GENERIC_ANSWERS = {
    "valid name",
    "one year",
    "unknown",
    "not known",
    "none",
    "various",
}


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _construct_question(
    model: RelayModel,
    source: Any,
    bridges: list[tuple[Any, str]],
    answer_page: Any,
    *,
    avoid_answers: list[str] | None = None,
) -> dict[str, Any]:
    path = [{"entity": source.title, "role": "source", "relation": "image anchor"}]
    path.extend({"entity": page.title, "role": "bridge", "relation": relation} for page, relation in bridges)
    path.append({"entity": answer_page.title, "role": "answer", "relation": "final node"})
    prompt = f"""Given this directed Wikipedia path, write one precise canonical question whose answer is a short fact from the final answer node.
The source node will later be replaced by an image reference. The question must require every edge in the path,
must not be answerable by naming the final entity, and must not ask for a visual caption. Select one unambiguous
attribute from the final node; the answer must be at most six whitespace-separated tokens.
The answer must not be predictable from source or bridge geography, dates, category words, or common knowledge.
Do not choose a broad native range such as a country or "X waters" when earlier clues locate the path in that
same region. Prefer a specific terminal fact whose type differs from the clues used to identify the path.
Before returning, simulate a tool-free solver. If the answer can be guessed from a category word such as the
animal group, place type, profession, or product class, choose a different terminal fact. Do not reuse any
previously rejected answer: {json.dumps(avoid_answers or [], ensure_ascii=False)}.
Return {{\"question\": string, \"answer\": string, \"answer_evidence\": string}} and no other text.
Path: {json.dumps(path, ensure_ascii=False)}
Source extract: {source.extract[:5000]}
Bridge extracts: {json.dumps([page.extract[:1800] for page, _ in bridges], ensure_ascii=False)}
Answer extract: {answer_page.extract[:7000]}"""
    return model.json(prompt, system=QUESTION_SYSTEM, max_output_tokens=800)


def _rewrite_question(model: RelayModel, *, canonical: dict[str, Any], source: Any, bridges: list[tuple[Any, str]], answer_page: Any) -> dict[str, Any]:
    hidden = [
        alias
        for page in [source, *(page for page, _ in bridges), answer_page]
        for alias in (page.title, *getattr(page, "aliases", ()))
    ]
    prompt = f"""Rewrite the canonical question into a fuzzy image-grounded multi-hop question.
Rewrite one entity at a time from the farthest bridge toward the source anchor. For every replacement,
use a relational or attribute-based descriptor grounded in that entity's extract and verify that the
partially rewritten question still identifies exactly one entity. Then replace the source mention with
the visual referring expression \"the entity shown in the image\". Do not reveal any entity name, alias,
URL slug, or answer. Keep the answer unchanged and retain every relation needed for the original path.
Do not place the answer, an obvious inflection of an answer word, or a near-synonym of the answer in
any source or bridge clue. Generic interrogative type words such as "year", "family", or "waters" are allowed.
Return {{\"question\": string, \"answer\": string, \"descriptors\": [string], \"rewrite_order\": [string]}} and no other text.
Canonical: {canonical}
Hidden entities: {hidden}
Source extract: {source.extract[:3500]}
Bridge extracts: {json.dumps([page.extract[:2500] for page, _ in bridges], ensure_ascii=False)}
Answer extract: {answer_page.extract[:3500]}"""
    result = model.json(prompt, system=QUESTION_SYSTEM, max_output_tokens=900)
    # Keep the intended rewrite order explicit in the record. The final
    # auditor still verifies uniqueness and leakage against all titles/aliases.
    result.setdefault("rewrite_order", [page.title for page, _ in reversed(bridges)] + [source.title])
    result["hidden_entities"] = hidden
    return result


def _rewrite_visual_anchor(model: RelayModel, *, question: str, source: Any) -> str:
    """Remove source-entity prose while preserving the path and every other clue."""
    result = model.json(
        f"""Return JSON only {{"question": string}}. Rewrite this question so its source node is referenced
only by a visual expression such as "the entity shown in the image" or "the place shown in the image".
Use the exact phrase "the entity shown in the image" once and return a complete grammatical question.
Do not describe, name, alias, locate, date, or otherwise identify the source node in text. Keep every
bridge relation, the requested attribute, and the answer unchanged. Do not rewrite any bridge or answer.
Candidate question: {question}
Source title (auditor-only, never output): {source.title}
Source extract (do not copy): {source.extract[:900]}""",
        system=QUESTION_SYSTEM,
        max_output_tokens=700,
    )
    rewritten = str(result.get("question") or question).strip()
    # Normalize only obvious token-boundary glitches. Semantic source leakage
    # is rejected by the deterministic and model audits below.
    rewritten = re.sub(
        r"\b(?:the\s+){2,}(?=(?:entity|place|building|person|site) shown in the image)",
        lambda match: "The " if match.group(0)[0].isupper() else "the ",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(r"(shown in the image)(?=[A-Za-z])", r"\1, ", rewritten, flags=re.IGNORECASE)
    return rewritten.strip()


def _anchor_description(model: RelayModel, source: Any) -> str:
    result = model.json(
        f"""Return JSON only {{\"description\": string}}. Write a short concrete visual description of the source entity
that would help rank representative Wikimedia images. Do not use aliases or answer facts; prefer a scene,
building, object, person, or landmark description grounded in this extract.
Source title: {source.title}
Extract: {source.extract[:1800]}""",
        max_output_tokens=120,
    )
    return str(result.get("description") or source.title).strip()


def _select_image(client: WikimediaClient, scorer: ClipScorer, source: Any, *, description: str, limit: int, threshold: float, skip_clip: bool) -> dict[str, Any] | None:
    candidates = client.commons_images(source.title, limit=limit)
    if not candidates:
        return None
    if skip_clip:
        candidates[0]["clip_score"] = None
        return candidates[0]
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            candidate = dict(candidate)
            candidate["clip_score"] = scorer.score(candidate["url"], description)
            if candidate["clip_score"] >= threshold:
                scored.append(candidate)
        except Exception as exc:
            candidate = dict(candidate)
            candidate["clip_error"] = str(exc)
    return max(scored, key=lambda item: item["clip_score"]) if scored else None


def _download_image(url: str, output: Path) -> Path:
    response = None
    last_error: requests.RequestException | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45, headers={"User-Agent": "SightlineDataBuilder/0.1 (research; contact repository maintainer)"})
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < 3:
                time.sleep(float(response.headers.get("Retry-After", min(2 ** attempt, 8))))
                continue
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))
    if response is None:
        raise RuntimeError(f"image download failed: {last_error}")
    if response.status_code == 429 and "thumb.wikimedia.org" not in url:
        fallback = url.replace("https://upload.wikimedia.org/wikipedia/commons/", "https://thumb.wikimedia.org/wikipedia/commons/thumb/")
        parts = fallback.split("/", 7)
        if len(parts) >= 8:
            filename = parts[7].split("?", 1)[0]
            fallback = "/".join(parts[:7] + [filename, f"960px-{filename}"])
            response = requests.get(fallback, timeout=45, headers={"User-Agent": "SightlineDataBuilder/0.1 (research; contact repository maintainer)"})
    response.raise_for_status()
    image = Image.open(__import__("io").BytesIO(response.content)).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=92)
    return output


def _make_enhancement(source: Path, output: Path, rng: random.Random) -> dict[str, Any]:
    image = Image.open(source).convert("RGB")
    operation = rng.choice(["blur", "downsample", "perspective_hint"])
    if operation == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=2.2))
    elif operation == "downsample":
        small = image.resize((max(64, image.width // 3), max(64, image.height // 3)))
        image = small.resize(image.size)
    else:
        image = image.rotate(rng.choice([-7, 7]), expand=True, fillcolor="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=85)
    return {"operation": operation, "image": str(output.resolve()), "restoration_tool": {"blur": "sharpen", "downsample": "super_resolution", "perspective_hint": "perspective_correct"}[operation]}


def build_one(args: argparse.Namespace, *, client: WikimediaClient, model: RelayModel, judge: RelayModel, scorer: ClipScorer, rng: random.Random) -> dict[str, Any]:
    if args.seed_title:
        seed = client.page(args.seed_title, require_image=True)
        sampled = client.sample_path_from_seed(seed, attempts=args.path_attempts) if seed else None
    else:
        sampled = client.sample_path(attempts=args.path_attempts, domain=args.source_domain)
    if sampled is None:
        return {"status": "rejected", "stage": "path", "reason": "no valid path"}
    source, bridges, answer_page = sampled
    path_record = [
        {"title": page.title, "relation": relation}
        for page, relation in [(source, "image anchor"), *bridges, (answer_page, "answer")]
    ]
    description = _anchor_description(model, source)
    image = _select_image(client, scorer, source, description=description, limit=args.image_candidates, threshold=args.clip_threshold, skip_clip=args.skip_clip)
    if image is None:
        return {"status": "rejected", "stage": "visual_grounding", "reason": "no image passed candidate filter", "source": source.title}
    local_image = _download_image(image["url"], args.output_dir / "images" / f"{source.title.replace(' ', '_')}.jpg")

    canonical_audits: list[dict[str, Any]] = []
    rejected_answers: list[str] = []
    selected: dict[str, Any] | None = None
    for canonical_attempt in range(args.canonical_attempts):
        canonical = _construct_question(
            model, source, bridges, answer_page, avoid_answers=rejected_answers
        )
        rewrite_audits: list[dict[str, Any]] = []
        question = ""
        answer = ""
        audit: dict[str, Any] = {}
        for rewrite_attempt in range(args.rewrite_attempts):
            rewritten = _rewrite_question(model, canonical=canonical, source=source, bridges=bridges, answer_page=answer_page)
            candidate_question = str(rewritten.get("question", "")).strip()
            candidate_answer = str(rewritten.get("answer") or canonical.get("answer", "")).strip()
            reason = None
            if not candidate_question or not candidate_answer:
                reason = "empty question or answer"
            elif len(candidate_answer.split()) > 6:
                reason = "answer exceeds six tokens"
            elif candidate_answer.casefold().strip(" .") in GENERIC_ANSWERS:
                reason = "answer is generic"
            else:
                candidate_question = _rewrite_visual_anchor(model, question=candidate_question, source=source)
                if "image" not in candidate_question.casefold():
                    reason = "source anchor is not visual"
            if reason:
                rewrite_audits.append({"attempt": rewrite_attempt + 1, "stage": "rewrite", "reason": reason})
                continue
            candidate_audit = quality_check(
                model=judge,
                question=candidate_question,
                answer=candidate_answer,
                source_title=source.title,
                image_url=image["url"],
                hidden_entities=list(rewritten.get("hidden_entities") or []),
            )
            rewrite_audits.append({"attempt": rewrite_attempt + 1, "stage": "quality", "question": candidate_question, "answer": candidate_answer, "audit": candidate_audit})
            if candidate_audit.get("accepted"):
                question, answer, audit = candidate_question, candidate_answer, candidate_audit
                break
        attempt_record: dict[str, Any] = {
            "attempt": canonical_attempt + 1,
            "canonical": canonical,
            "rewrite_audits": rewrite_audits,
        }
        if not audit.get("accepted"):
            attempt_record["stage"] = "quality"
            canonical_audits.append(attempt_record)
            canonical_answer = str(canonical.get("answer", "")).strip()
            if canonical_answer:
                rejected_answers.append(canonical_answer)
            continue
        tool_filter = tool_necessity_check(judge, question=question, answer=answer, image_url=image["url"])
        attempt_record.update({"stage": "tool_filter", "question": question, "answer": answer, "audit": audit, "tool_filter": tool_filter})
        canonical_audits.append(attempt_record)
        if tool_filter.get("accepted"):
            selected = {"canonical": canonical, "question": question, "answer": answer, "audit": audit, "rewrite_audits": rewrite_audits, "tool_filter": tool_filter}
            break
        rejected_answers.append(answer)

    if selected is None:
        last = canonical_audits[-1] if canonical_audits else {}
        return {
            "status": "rejected",
            "stage": last.get("stage", "canonical"),
            "reason": "no terminal fact passed rewrite, quality, and tool-necessity audits",
            "source": source.title,
            "path": path_record,
            "canonical": last.get("canonical", {}),
            "question": last.get("question", ""),
            "answer": last.get("answer", ""),
            "audit": last.get("audit", {}),
            "rewrite_audits": last.get("rewrite_audits", []),
            "canonical_audits": canonical_audits,
            "tool_filter": last.get("tool_filter"),
            "image": image,
        }
    canonical = selected["canonical"]
    question = selected["question"]
    answer = selected["answer"]
    audit = selected["audit"]
    rewrite_audits = selected["rewrite_audits"]
    tool_filter = selected["tool_filter"]
    result: dict[str, Any] = {
        "status": "accepted",
        "source": {"title": source.title, "url": source.url},
        "path": path_record,
        "question": question,
        "answer": answer,
        "image": image,
        "image_description": description,
        "local_image": str(local_image.resolve()),
        "source_image": str(local_image.resolve()),
        "audit": audit,
        "rewrite_audits": rewrite_audits,
        "canonical_audits": canonical_audits,
        "tool_filter": tool_filter,
        "source_domain": source.domain,
    }
    if rng.random() < args.enhancement_ratio:
        degraded = _make_enhancement(local_image, args.output_dir / "images" / f"{source.title.replace(' ', '_')}_degraded.jpg", rng)
        result["enhancement"] = degraded
        result["local_image"] = degraded["image"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image-grounded multi-hop questions and expert tool trajectories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--trajectories-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--path-attempts", type=int, default=10)
    parser.add_argument("--canonical-attempts", type=int, default=3)
    parser.add_argument("--rewrite-attempts", type=int, default=3)
    parser.add_argument("--seed-title", default=None, help="Use a specific page for a deterministic connectivity smoke test.")
    parser.add_argument("--source-domain", choices=["person", "building_place", "location", "organism", "artifact"], default=None)
    parser.add_argument("--image-candidates", type=int, default=8)
    parser.add_argument("--clip-threshold", type=float, default=0.28)
    parser.add_argument("--skip-clip", action="store_true", help="Use the first valid Commons image; intended only for connectivity smoke tests.")
    parser.add_argument("--generate-trajectories", action="store_true")
    parser.add_argument("--agent-url", default=None)
    parser.add_argument("--no-enhancement", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.no_enhancement:
        args.enhancement_ratio = 0.0
    else:
        args.enhancement_ratio = 0.1
    rng = random.Random(args.seed)
    client = WikimediaClient(seed=args.seed)
    model = RelayModel()
    judge = RelayModel(
        base_url=os.environ.get("JUDGE_MODEL_BASE_URL") or None,
        api_key=os.environ.get("JUDGE_API_KEY") or None,
        model=os.environ.get("JUDGE_MODEL_NAME", model.model),
    )
    scorer = ClipScorer()
    audit_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    trajectory_client = AgentTrajectoryClient(agent_url=args.agent_url) if args.generate_trajectories else None
    for index in range(args.samples):
        try:
            row = build_one(args, client=client, model=model, judge=judge, scorer=scorer, rng=rng)
            row["sample_index"] = index
            if row.get("status") == "accepted" and trajectory_client:
                row["trajectories"] = []
                for _ in range(args.trajectories_per_sample):
                    trajectory = trajectory_client.generate(image_url=Path(row["local_image"]).resolve().as_uri(), question=row["question"], answer=row["answer"], judge=judge)
                    if trajectory:
                        row["trajectories"].append({"messages": trajectory.messages, "process": trajectory.process})
                if not row["trajectories"]:
                    row["status"] = "rejected"
                    row["stage"] = "trajectory"
                    row["reason"] = "no trajectory passed answer/process rejection sampling"
            audit_rows.append(row)
            if row.get("status") == "accepted":
                accepted.append(row)
            source_value = row.get("source")
            source_title = source_value.get("title") if isinstance(source_value, dict) else source_value
            print(json.dumps({"sample": index, "status": row.get("status"), "stage": row.get("stage", "complete"), "source": source_title}, ensure_ascii=False), flush=True)
        except Exception as exc:
            row = {"sample_index": index, "status": "error", "stage": "exception", "reason": str(exc), "traceback": traceback.format_exc(limit=4)}
            audit_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)
    _save_jsonl(args.output_dir / "audit.jsonl", audit_rows)
    _save_jsonl(args.output_dir / "accepted.jsonl", accepted)
    if accepted:
        sft_rows = []
        rl_rows = []
        for row in accepted:
            for trajectory in row.get("trajectories", []):
                sft_rows.append({"messages": trajectory["messages"], "images": [row["local_image"]], "tools": TOOLS})
            rl_rows.append({"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["question"]}]}], "answer": [row["answer"]], "images": [row["local_image"]], "dataset": "generated"})
        _save_jsonl(args.output_dir / "sft.jsonl", sft_rows)
        _save_jsonl(args.output_dir / "rl.jsonl", rl_rows)


if __name__ == "__main__":
    main()
