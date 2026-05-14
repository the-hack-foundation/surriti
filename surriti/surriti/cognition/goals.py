"""Goal synthesis.

Heuristic pre-filter: scan the recent batch of episodes for sentences
matching intentional verbs ("want to", "trying to", "working on",
"goal is", "improve", "learning", "aim to", future-tense markers).
Those candidate sentences are batched into a single LLM call that
returns distinct, durable goals. Each goal is persisted as
``EntityNode(labels=['goal'])`` linked to the subject by a
``pursues_goal`` edge with ``memory_class='goal'``. Goals decay
slowly (``stability='persistent'``) and are superseded only when
later episodes contradict them through the existing temporal
machinery.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from surriti.cognition._jsonio import parse_json_loose, snake_case
from surriti.cognition._writes import (
    cache_on_subject,
    upsert_synthetic_edge,
    upsert_synthetic_entity,
)
from surriti.cognition.prompts import GOAL_RATIFY_SYSTEM
from surriti.search import _unwrap

logger = logging.getLogger(__name__)


# Intentional / goal-language patterns.
_GOAL_PATTERNS = [
    re.compile(r"\bI (?:want|wanna|need|plan|hope) to ([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bI(?:'m| am) (?:trying|working|learning) (?:to|on) ([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bI(?:'m| am) (?:going to|gonna) ([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bmy goal (?:is|was) (?:to )?([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bI aim to ([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bI want ([^.?!]+)", re.IGNORECASE),
    re.compile(r"\bI(?:'d| would) like to ([^.?!]+)", re.IGNORECASE),
]


def _scan_goal_sentences(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for pat in _GOAL_PATTERNS:
        for m in pat.finditer(text):
            phrase = (m.group(0) if m.lastindex is None else m.group(0)).strip()
            phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:")
            if 8 <= len(phrase) <= 200:
                out.append(phrase)
    return out


# Goal names extracted via snake_case from raw "I'm working on …" matches
# routinely come back as ``i_m_*`` / ``i_am_*`` / ``my_*`` — these are
# pronoun fragments, not durable goals, and pollute the active-goals list
# (e.g. "i_m_working_on_your_memory"). Reject names that begin with these
# tokens or are otherwise too short / pure punctuation.
_BAD_GOAL_PREFIXES = (
    "i_", "i_m_", "i_am_", "im_", "my_", "we_", "we_re_", "we_are_",
    "you_", "they_", "the_user_", "user_",
)


def _is_clean_goal_name(name: str) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    if len(n) < 4:
        return False
    if n in {"none", "null", "n_a", "na"}:
        return False
    for pref in _BAD_GOAL_PREFIXES:
        if n == pref.rstrip("_") or n.startswith(pref):
            return False
    return True


async def _resolve_speaker_uuid(
    driver: Any, *, group_id: str, episode_uuids: list[str]
) -> str | None:
    """Find the subject most recently associated with ``user`` mentions
    in this batch. Falls back to the canonical user entity for the
    group if one exists."""

    rows = _unwrap(
        await driver.query(
            """
            SELECT record::id(out) AS entity_uuid
            FROM mentions
            WHERE group_id = $g AND record::id(in) IN $eps;
            """,
            {"g": group_id, "eps": list(episode_uuids)},
        )
    )
    candidates = [str(r.get("entity_uuid")) for r in rows if r.get("entity_uuid")]
    if not candidates:
        return None
    # Heaviest-mentioned entity.
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return best


async def synthesize_goals(
    driver: Any,
    llm: Any,
    embedder: Any,
    *,
    group_id: str,
    episode_uuids: list[str],
) -> int:
    """Synthesize goals from the recent episode batch. Returns count of
    goal edges written / refreshed."""

    if not episode_uuids:
        return 0
    rows = _unwrap(
        await driver.query(
            "SELECT uuid, content FROM episode WHERE group_id = $g AND uuid IN $u;",
            {"g": group_id, "u": list(episode_uuids)},
        )
    )
    sentences: list[tuple[str, str]] = []  # (episode_uuid, phrase)
    for r in rows:
        for s in _scan_goal_sentences(str(r.get("content") or "")):
            sentences.append((str(r.get("uuid")), s))
    if not sentences:
        return 0

    # Existing goals supplied to the LLM so it dedupes against them.
    existing_rows = _unwrap(
        await driver.query(
            "SELECT name, summary FROM entity WHERE group_id = $g AND 'goal' IN labels;",
            {"g": group_id},
        )
    )
    existing_names = [str(r.get("name") or "") for r in existing_rows if r.get("name")]

    user = (
        "GOAL CANDIDATES (one per line, indexed):\n"
        + "\n".join(f"[{i}] {s}" for i, (_, s) in enumerate(sentences))
        + ("\n\nEXISTING_GOALS: " + ", ".join(existing_names) if existing_names else "")
    )
    raw = None
    try:
        raw = await llm.synthesize(GOAL_RATIFY_SYSTEM, user)
    except Exception:
        logger.exception("goal synthesis LLM call failed; falling back to heuristics")

    parsed = parse_json_loose(raw)
    accepted: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = snake_case(item.get("name") or "")
            if not _is_clean_goal_name(name):
                continue
            try:
                conf = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            accepted.append(
                {
                    "name": name,
                    "description": str(item.get("description") or "").strip(),
                    "domain": str(item.get("domain") or "").strip() or None,
                    "time_horizon": str(item.get("time_horizon") or "unknown"),
                    "confidence": max(0.0, min(1.0, conf)),
                }
            )
    if not accepted:
        # Fallback: collapse first sentence into a single anonymous goal.
        first = sentences[0][1]
        fb_name = snake_case(first[:48])
        if not _is_clean_goal_name(fb_name):
            logger.debug(
                "synthesize_goals: rejected fallback goal name=%r", fb_name
            )
            return 0
        accepted.append(
            {
                "name": fb_name,
                "description": first,
                "domain": None,
                "time_horizon": "unknown",
                "confidence": 0.55,
            }
        )

    subject_uuid = await _resolve_speaker_uuid(driver, group_id=group_id, episode_uuids=episode_uuids)
    if not subject_uuid:
        logger.debug("synthesize_goals: no speaker subject resolvable; skipping")
        return 0

    written = 0
    now = datetime.now(timezone.utc)
    for goal in accepted:
        goal_uuid = await upsert_synthetic_entity(
            driver,
            group_id=group_id,
            name=goal["name"],
            summary=goal["description"],
            label="goal",
            now=now,
        )
        await upsert_synthetic_edge(
            driver,
            embedder,
            group_id=group_id,
            subject_uuid=subject_uuid,
            object_uuid=goal_uuid,
            predicate="pursues_goal",
            fact_text=goal["description"] or f"pursues goal: {goal['name']}",
            memory_class="goal",
            confidence=float(goal["confidence"]),
            stability="persistent",
            extra_attrs={
                "domain": goal["domain"],
                "time_horizon": goal["time_horizon"],
            },
            now=now,
        )
        await cache_on_subject(
            driver, subject_uuid=subject_uuid, field="goals_active", value=goal_uuid
        )
        written += 1
    logger.debug("synthesize_goals: group=%s written=%d", group_id, written)
    return written
