"""Predictive context bundle.

Computes a small per-group prediction object summarising what the user
is likely to ask / prefer / focus on next, derived from active goals,
dominant domains, and the recent interaction pattern. Persisted as a
single ``community`` row with ``kind='prediction'`` per group; refreshed
each cognition pass. ``recall(depth='deep')`` reads it.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from surriti.cognition._jsonio import parse_json_loose
from surriti.cognition.prompts import PREDICTION_SYSTEM
from surriti.search import _unwrap

logger = logging.getLogger(__name__)


async def synthesize_prediction(
    driver: Any, llm: Any, *, group_id: str
) -> dict[str, Any] | None:
    """Refresh and return the prediction bundle for ``group_id``."""

    # Active goals
    goal_rows = _unwrap(
        await driver.query(
            """
            SELECT name, summary FROM entity
            WHERE group_id = $g AND 'goal' IN labels
            LIMIT 10;
            """,
            {"g": group_id},
        )
    )
    goals = [
        {"name": str(r.get("name") or ""), "summary": str(r.get("summary") or "")}
        for r in goal_rows
    ]

    # Dominant domains
    domain_rows = _unwrap(
        await driver.query(
            "SELECT domain FROM entity WHERE group_id = $g AND domain IS NOT NONE;",
            {"g": group_id},
        )
    )
    domain_counts = Counter(str(r.get("domain")) for r in domain_rows if r.get("domain"))
    domains = [d for d, _ in domain_counts.most_common(5)]

    # Recent interaction patterns
    pattern_rows = _unwrap(
        await driver.query(
            """
            SELECT interaction_pattern, reference_time FROM episode
            WHERE group_id = $g AND interaction_pattern IS NOT NONE
            ORDER BY reference_time DESC
            LIMIT 8;
            """,
            {"g": group_id},
        )
    )
    pattern_counts = Counter(
        str(r.get("interaction_pattern"))
        for r in pattern_rows
        if r.get("interaction_pattern")
    )
    dominant_pattern = pattern_counts.most_common(1)[0][0] if pattern_counts else None

    if not goals and not domains and not dominant_pattern:
        return None

    user = (
        f"ACTIVE_GOALS: {goals}\n"
        f"DOMINANT_DOMAINS: {domains}\n"
        f"DOMINANT_INTERACTION_PATTERN: {dominant_pattern!r}"
    )
    bundle: dict[str, Any] | None = None
    try:
        raw = await llm.synthesize(PREDICTION_SYSTEM, user)
        parsed = parse_json_loose(raw)
        if isinstance(parsed, dict):
            bundle = {
                "likely_next_topics": list(parsed.get("likely_next_topics") or [])[:5],
                "likely_preferences": list(parsed.get("likely_preferences") or [])[:5],
                "likely_questions": list(parsed.get("likely_questions") or [])[:5],
            }
    except Exception:
        logger.exception("prediction synthesis LLM call failed; using heuristic fallback")

    if bundle is None:
        bundle = {
            "likely_next_topics": domains[:3],
            "likely_preferences": [g["name"] for g in goals[:3]],
            "likely_questions": [],
        }
    bundle["dominant_pattern"] = dominant_pattern
    bundle["refreshed_at"] = datetime.now(timezone.utc).isoformat()

    # Upsert prediction sidecar.
    existing = _unwrap(
        await driver.query(
            "SELECT uuid FROM community WHERE group_id = $g AND kind = 'prediction' LIMIT 1;",
            {"g": group_id},
        )
    )
    if existing:
        await driver.query(
            "UPDATE community SET payload = $p WHERE uuid = $u;",
            {"u": existing[0].get("uuid"), "p": bundle},
        )
    else:
        await driver.query(
            """
            CREATE community CONTENT {
                uuid: $u, group_id: $g, name: 'prediction',
                kind: 'prediction', summary: '', payload: $p,
                created_at: time::now()
            };
            """,
            {"u": str(uuid4()), "g": group_id, "p": bundle},
        )
    return bundle
