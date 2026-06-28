"""Episodic -> semantic consolidation."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from surriti.cognition._writes import upsert_synthetic_edge
from surriti.cognition.decay import effective_confidence, is_decay_protected
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
    buckets: dict[str, list[Any]] = {}
    for e in edges:
        if e.fact_key:
            buckets.setdefault(e.fact_key, []).append(e)

    episode_uuids = sorted({u for e in edges for u in (e.episodes or [])})
    ep_times: dict[str, datetime] = {}
    if episode_uuids:
        ep_rows = _unwrap(await driver.query("SELECT uuid, reference_time FROM episode WHERE uuid IN $u;", {"u": list(episode_uuids)}))
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
        all_eps: set[str] = set()
        for e in bucket:
            all_eps.update(e.episodes or [])
        if len(all_eps) < threshold:
            continue
        ts = [ep_times[u] for u in all_eps if u in ep_times]
        if len(ts) < 2:
            continue
        if (max(ts) - min(ts)).total_seconds() / 86_400.0 < min_span_days:
            continue
        canon = max(bucket, key=lambda e: float(e.confidence or 0))
        avg_conf = sum(float(e.confidence or 0) for e in bucket) / len(bucket)
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


def _summary_text(edges: list[Any]) -> str:
    facts = [str(e.fact or e.name or "").strip() for e in edges if (e.fact or e.name)]
    if not facts:
        return "Low-vitality memory cluster."
    text = "; ".join(facts[:5])
    if len(facts) > 5:
        text += f"; and {len(facts) - 5} related older facts"
    return f"Low-vitality memory cluster: {text}"


async def consolidate_stagnant_edges(
    driver: Any,
    embedder: Any,
    *,
    group_id: str,
    min_edges_per_summary: int = 5,
    max_edges_per_pass: int = 120,
) -> int:
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
            LIMIT $limit;
            """,
            {"g": group_id, "limit": int(max_edges_per_pass)},
        )
    )
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    buckets: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in rows:
        try:
            edge = parse_edge(row)
        except Exception:
            continue
        if is_decay_protected(edge) or effective_confidence(edge, now=now) > 0.0:
            continue
        key = (edge.source_node_uuid or "unknown", edge.memory_class or "objective", edge.domain or edge.canonical_name or edge.name or "misc")
        buckets[key].append(edge)

    written = 0
    for (_, memory_class, bucket_name), bucket in buckets.items():
        if len(bucket) < min_edges_per_summary:
            continue
        canon = bucket[0]
        await upsert_synthetic_edge(
            driver,
            embedder,
            group_id=group_id,
            subject_uuid=canon.source_node_uuid,
            object_uuid=canon.target_node_uuid,
            predicate=f"archived_summary_{bucket_name}",
            fact_text=_summary_text(bucket),
            memory_class="archived_summary",
            confidence=0.5,
            supporting_edge_uuids=[e.uuid for e in bucket],
            consolidates=[e.uuid for e in bucket],
            stability="consolidated",
            extra_attrs={"summary_type": "stagnant", "source_memory_class": memory_class, "source_count": len(bucket), "lossy": True},
            fact_key_qualifier=f"archived::{bucket_name}",
            now=now,
        )
        written += 1
    logger.debug("consolidate_stagnant_edges: group=%s written=%d", group_id, written)
    return written
