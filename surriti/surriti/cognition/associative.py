"""Associative edge weighting."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from surriti.cognition.decay import effective_confidence
from surriti.edges import EntityEdge
from surriti.search import _unwrap
from surriti.utils import parse_edge

logger = logging.getLogger(__name__)


async def refresh_weights(
    driver: Any,
    *,
    group_id: str,
    half_life_overrides: dict[str, float] | None = None,
) -> int:
    """Recompute composite associative weight for active edges.

    Linear vitality is computed lazily from the edge timestamps.  We update
    ``weight`` here but intentionally do not materialize ``decay_score`` on
    every pass; recall reinforcement owns score writes so daily decay is not
    counted twice between writes.
    """

    rows = _unwrap(
        await driver.query(
            """
            SELECT *,
                record::id(in)  AS source_node_uuid,
                record::id(out) AS target_node_uuid
            FROM relates_to
            WHERE group_id = $g AND status = "active";
            """,
            {"g": group_id},
        )
    )
    if not rows:
        return 0

    edges: list[EntityEdge] = [parse_edge(r) for r in rows]
    now = datetime.now(timezone.utc)

    entity_freq: Counter[str] = Counter()
    episode_freq: Counter[str] = Counter()
    for e in edges:
        entity_freq[e.source_node_uuid] += 1
        entity_freq[e.target_node_uuid] += 1
        for ep in e.episodes or []:
            episode_freq[ep] += 1

    updated = 0
    for e in edges:
        decay = effective_confidence(e, now=now, half_life_overrides=half_life_overrides)
        touched = (
            (entity_freq[e.source_node_uuid] - 1)
            + (entity_freq[e.target_node_uuid] - 1)
            + sum(max(0, episode_freq[ep] - 1) for ep in (e.episodes or []))
        )
        assoc_boost = min(1.0, touched / 12.0)
        weight = max(0.0, min(4.0, decay * (1.0 + assoc_boost)))
        await driver.query(
            "UPDATE relates_to SET weight = $w WHERE uuid = $u;",
            {"u": e.uuid, "w": float(weight)},
        )
        updated += 1
    logger.debug("refresh_weights: group=%s updated=%d", group_id, updated)
    return updated
