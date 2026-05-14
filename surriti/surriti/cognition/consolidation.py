"""Episodic -> semantic consolidation.

When a single ``fact_key`` accumulates many supporting episodes spanning
a meaningful time window, mint a single ``memory_class='consolidated'``
abstraction edge whose ``consolidates`` field lists the supporting edge
UUIDs. The originals stay queryable for provenance; ``recall()`` will
prefer the consolidated edge when both surface (Phase E reranking).
This is the "hippocampus -> cortex" pass.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from surriti.cognition._writes import upsert_synthetic_edge
from surriti.search import _unwrap
from surriti.utils import parse_edge

logger = logging.getLogger(__name__)


async def consolidate(
    driver: Any,
    embedder: Any,
    *,
    group_id: str,
    threshold: int = 8,
    min_span_days: float = 14.0,
) -> int:
    """Run the consolidation pass for ``group_id``. Returns number of
    consolidated edges written / refreshed."""

    rows = _unwrap(
        await driver.query(
            """
            SELECT *,
                record::id(in)  AS source_node_uuid,
                record::id(out) AS target_node_uuid
            FROM relates_to
            WHERE group_id = $g
              AND status = 'active'
              AND stability != 'consolidated'
              AND fact_key != '';
            """,
            {"g": group_id},
        )
    )
    if not rows:
        return 0
    edges = [parse_edge(r) for r in rows]

    # Group by fact_key.
    buckets: dict[str, list[Any]] = {}
    for e in edges:
        if not e.fact_key:
            continue
        buckets.setdefault(e.fact_key, []).append(e)

    # We also need per-edge supporting-episode timestamps.
    episode_uuids = sorted({u for e in edges for u in (e.episodes or [])})
    ep_times: dict[str, datetime] = {}
    if episode_uuids:
        ep_rows = _unwrap(
            await driver.query(
                "SELECT uuid, reference_time FROM episode WHERE uuid IN $u;",
                {"u": list(episode_uuids)},
            )
        )
        for er in ep_rows:
            t = er.get("reference_time")
            if isinstance(t, str):
                try:
                    t = datetime.fromisoformat(t.replace("Z", "+00:00"))
                except ValueError:
                    t = None
            if isinstance(t, datetime):
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                ep_times[str(er.get("uuid"))] = t

    written = 0
    now = datetime.now(timezone.utc)
    for fact_key, bucket in buckets.items():
        # Combined supporting episode set across all edges sharing the key.
        all_eps: set[str] = set()
        for e in bucket:
            all_eps.update(e.episodes or [])
        if len(all_eps) < threshold:
            continue
        ts = [ep_times[u] for u in all_eps if u in ep_times]
        if len(ts) < 2:
            continue
        span_days = (max(ts) - min(ts)).total_seconds() / 86_400.0
        if span_days < min_span_days:
            continue

        # Strongest edge (highest confidence) provides the canonical
        # subject/object/predicate/fact for the consolidated row.
        canon = max(bucket, key=lambda e: float(e.confidence or 0))
        avg_conf = sum(float(e.confidence or 0) for e in bucket) / len(bucket)

        # Skip if already consolidated (idempotency via the
        # ``::consolidated`` qualifier on the fact_key).
        await upsert_synthetic_edge(
            driver,
            embedder,
            group_id=group_id,
            subject_uuid=canon.source_node_uuid,
            object_uuid=canon.target_node_uuid,
            predicate=canon.canonical_name or canon.name,
            fact_text=canon.fact or f"{canon.name} (consolidated)",
            memory_class="consolidated",
            confidence=min(1.0, avg_conf + 0.1),
            supporting_edge_uuids=[e.uuid for e in bucket],
            consolidates=[e.uuid for e in bucket],
            stability="consolidated",
            extra_attrs={"consolidated_from_key": fact_key, "support_count": len(all_eps)},
            fact_key_qualifier="consolidated",
            now=now,
        )
        written += 1
    logger.debug("consolidate: group=%s written=%d", group_id, written)
    return written
