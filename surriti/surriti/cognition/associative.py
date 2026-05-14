"""Associative edge weighting.

After reinforcement / decay are refreshed, recompute a composite
``weight`` on each edge in the group::

    weight = clip(decay_score * (1 + assoc_boost), 0, 4)

where ``assoc_boost`` reflects how many *other* edges share an entity or
an episode with this one (a coarse proxy for hippocampal associative
strength). The full N x N co-occurrence matrix is overkill for this
purpose; we use cheap counts derivable from a single SurrealQL pass.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from surriti.cognition.decay import effective_confidence
from surriti.edges import EntityEdge
from surriti.search import _unwrap
from surriti.utils import _strip_record_id, parse_edge

logger = logging.getLogger(__name__)


async def refresh_weights(
    driver: Any,
    *,
    group_id: str,
    half_life_overrides: dict[str, float] | None = None,
) -> int:
    """Recompute ``decay_score`` and ``weight`` for every active edge in
    ``group_id``. Returns the number of edges updated."""

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

    # Co-occurrence counts: how many edges touch each entity / episode.
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
        # Boost: average co-occurrence frequency of touched entities + episodes
        # minus self contribution. Bounded for sanity.
        touched = (
            (entity_freq[e.source_node_uuid] - 1)
            + (entity_freq[e.target_node_uuid] - 1)
            + sum(max(0, episode_freq[ep] - 1) for ep in (e.episodes or []))
        )
        assoc_boost = min(1.0, touched / 12.0)
        weight = max(0.0, min(4.0, decay * (1.0 + assoc_boost)))
        await driver.query(
            "UPDATE relates_to SET decay_score = $d, weight = $w WHERE uuid = $u;",
            {"u": e.uuid, "d": float(decay), "w": float(weight)},
        )
        updated += 1
    logger.debug("refresh_weights: group=%s updated=%d", group_id, updated)
    return updated
