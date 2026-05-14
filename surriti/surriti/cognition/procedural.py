"""Procedural / interaction-pattern detection.

Classifies each recent episode by interaction shape:

- ``optimization_request``  -- explicit "best", "optimal", "improve",
  "better", "tune" language.
- ``iterative_refinement``  -- topic continuity across N consecutive
  episodes (cosine of name embeddings, or token overlap fallback).
- ``clarification``         -- short follow-up questions referencing
  the prior assistant turn.
- ``narrative_share``       -- long monologue, low question density.
- ``one_off_query``         -- isolated question with no follow-up.

When a single classification dominates the user's recent window
(>= 5 of the last 8 episodes), we synthesise a single user-level
edge ``user -[interaction_style]-> <pattern>`` with
``memory_class='procedural'``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from surriti.cognition._writes import (
    cache_on_subject,
    upsert_synthetic_edge,
    upsert_synthetic_entity,
)
from surriti.search import _unwrap

logger = logging.getLogger(__name__)


_OPT_RE = re.compile(
    r"\b(?:best|optimal|optimi[sz]e|improve|better|tune|refine|tweak|sharper|tighter)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?")
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 3}


def classify_episode(text: str, *, prior_text: str | None = None) -> str:
    if not text:
        return "one_off_query"
    if _OPT_RE.search(text):
        return "optimization_request"
    qs = len(_QUESTION_RE.findall(text))
    words = len(text.split())
    if qs == 0 and words >= 60:
        return "narrative_share"
    if prior_text:
        overlap = len(_tokens(text) & _tokens(prior_text))
        if overlap >= 3 and qs >= 1:
            return "iterative_refinement"
        if overlap >= 2 and words < 40:
            return "clarification"
    return "one_off_query"


_INTERACTION_WINDOW = 8
_DOMINANT_THRESHOLD = 5


async def detect_interaction_patterns(
    driver: Any,
    embedder: Any,
    *,
    group_id: str,
    episode_uuids: list[str],
) -> int:
    """Tag recent episodes with ``interaction_pattern`` and (when stable)
    promote a user-level procedural edge. Returns count of episodes
    classified."""

    if not episode_uuids:
        return 0
    # Pull a wider window so iterative-refinement detection has prior context.
    rows = _unwrap(
        await driver.query(
            """
            SELECT uuid, content, reference_time
            FROM episode
            WHERE group_id = $g
            ORDER BY reference_time DESC
            LIMIT $n;
            """,
            {"g": group_id, "n": int(_INTERACTION_WINDOW)},
        )
    )
    if not rows:
        return 0
    rows.sort(key=lambda r: str(r.get("reference_time") or ""))
    classified = 0
    classifications: list[str] = []
    prior: str | None = None
    for r in rows:
        text = str(r.get("content") or "")
        label = classify_episode(text, prior_text=prior)
        if str(r.get("uuid")) in set(episode_uuids):
            await driver.query(
                "UPDATE episode SET interaction_pattern = $p WHERE uuid = $u;",
                {"u": r.get("uuid"), "p": label},
            )
            classified += 1
        classifications.append(label)
        prior = text

    counts = Counter(classifications)
    dominant, hits = counts.most_common(1)[0]
    if hits < _DOMINANT_THRESHOLD or dominant == "one_off_query":
        logger.debug(
            "detect_interaction_patterns: no dominant pattern (%r)", counts
        )
        return classified

    # Promote dominant pattern to a procedural edge on the speaker.
    speaker_rows = _unwrap(
        await driver.query(
            """
            SELECT record::id(out) AS entity_uuid, count() AS n
            FROM mentions
            WHERE group_id = $g
            GROUP BY entity_uuid
            ORDER BY n DESC
            LIMIT 1;
            """,
            {"g": group_id},
        )
    )
    speaker_uuid = (
        str(speaker_rows[0].get("entity_uuid")) if speaker_rows else None
    )
    if not speaker_uuid:
        return classified

    now = datetime.now(timezone.utc)
    pattern_uuid = await upsert_synthetic_entity(
        driver,
        group_id=group_id,
        name=dominant,
        summary=f"interaction pattern: {dominant}",
        label="pattern",
        now=now,
    )
    await upsert_synthetic_edge(
        driver,
        embedder,
        group_id=group_id,
        subject_uuid=speaker_uuid,
        object_uuid=pattern_uuid,
        predicate="interaction_style",
        fact_text=f"interacts via {dominant.replace('_', ' ')}",
        memory_class="procedural",
        confidence=min(1.0, 0.5 + 0.06 * hits),
        stability="reinforced",
        now=now,
    )
    return classified
