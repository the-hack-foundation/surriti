"""Reinforcement pass."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from surriti.cognition.decay import DEFAULT_RECALL_BOOST, effective_confidence
from surriti.search import _unwrap
from surriti.utils import parse_edge

logger = logging.getLogger(__name__)

_REINFORCED_THRESHOLD = 3
_PERSISTENT_THRESHOLD = 7
_PERSISTENT_MIN_SPAN_DAYS = 7.0


async def reinforce_recent_edges(
    driver: Any,
    *,
    group_id: str,
    episode_uuids: list[str],
) -> int:
    if not episode_uuids:
        return 0

    rows = _unwrap(
        await driver.query(
            """
            SELECT uuid, episodes, valid_at, last_reinforced_at, stability,
                   reinforcement_count
            FROM relates_to
            WHERE group_id = $g
              AND episodes ANYINSIDE $eps;
            """,
            {"g": group_id, "eps": list(episode_uuids)},
        )
    )
    if not rows:
        return 0

    all_ep_uuids: set[str] = set()
    for row in rows:
        for u in row.get("episodes") or []:
            if u:
                all_ep_uuids.add(str(u))
    ep_times: dict[str, datetime] = {}
    if all_ep_uuids:
        ep_rows = _unwrap(
            await driver.query(
                "SELECT uuid, reference_time FROM episode WHERE uuid IN $u;",
                {"u": list(all_ep_uuids)},
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

    updated = 0
    for row in rows:
        ep_list = [str(u) for u in (row.get("episodes") or []) if u]
        count = len(set(ep_list))
        if count <= 0:
            continue
        ts = [ep_times[u] for u in ep_list if u in ep_times]
        last_seen = max(ts) if ts else None
        span_days = ((max(ts) - min(ts)).total_seconds() / 86_400.0) if len(ts) >= 2 else 0.0

        stability = row.get("stability") or "episodic"
        if stability != "consolidated":
            if count >= _PERSISTENT_THRESHOLD and span_days >= _PERSISTENT_MIN_SPAN_DAYS:
                stability = "persistent"
            elif count >= _REINFORCED_THRESHOLD:
                stability = "reinforced" if stability == "episodic" else stability

        await driver.query(
            """
            UPDATE relates_to SET
                reinforcement_count = $c,
                last_reinforced_at  = $t,
                stability           = $s
            WHERE uuid = $u;
            """,
            {
                "u": row.get("uuid"),
                "c": int(count),
                "t": last_seen,
                "s": stability,
            },
        )
        updated += 1
    logger.debug("reinforce_recent_edges: group=%s updated=%d", group_id, updated)
    return updated


async def reinforce_edges_on_recall(
    driver: Any,
    *,
    group_id: str,
    edge_uuids: list[str],
    amount: int = 1,
) -> int:
    uuids = list(dict.fromkeys(str(u) for u in edge_uuids if u))
    if not uuids:
        return 0
    inc = max(1, int(amount))
    now = datetime.now(timezone.utc)
    rows = _unwrap(
        await driver.query(
            """
            SELECT *,
                record::id(in)  AS source_node_uuid,
                record::id(out) AS target_node_uuid
            FROM relates_to
            WHERE group_id = $group_id
              AND uuid IN $uuids;
            """,
            {"group_id": group_id, "uuids": uuids},
        )
    )
    updated = 0
    for row in rows:
        edge_uuid = row.get("uuid")
        if not edge_uuid:
            continue
        try:
            current_score = effective_confidence(parse_edge(row), now=now)
        except Exception:
            current_score = float(row.get("decay_score") or 1.0)
        next_score = max(0.0, min(1.0, current_score + (DEFAULT_RECALL_BOOST * inc)))
        await driver.query(
            """
            UPDATE relates_to SET
                recall_count = $recall_count,
                last_recalled_at = $last_recalled_at,
                decay_score = $decay_score
            WHERE group_id = $group_id
              AND uuid = $uuid;
            """,
            {
                "group_id": group_id,
                "uuid": edge_uuid,
                "recall_count": int(row.get("recall_count") or 0) + inc,
                "last_recalled_at": now,
                "decay_score": float(next_score),
            },
        )
        updated += 1
    return updated
