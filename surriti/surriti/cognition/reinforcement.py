"""Reinforcement pass.

For the batch of episodes new since the last cognition pass, scan the
edges that reference any of those episodes (via ``relates_to.episodes``)
and update:

- ``reinforcement_count``  -- distinct supporting episodes (len of the
  ``episodes`` array).
- ``last_reinforced_at``   -- max(``valid_at``, latest supporting
  episode's ``reference_time``).
- ``stability``            -- escalates ``episodic`` -> ``reinforced``
  (count >= 3) -> ``persistent`` (count >= 7 AND span >= 7d).

The pass is *purely heuristic*; no LLM. It is the cheapest cognition
component and feeds every later phase, so it always runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from surriti.search import _unwrap

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
    """Update ``reinforcement_count`` / ``last_reinforced_at`` /
    ``stability`` for every edge whose ``episodes`` array overlaps with
    the supplied ``episode_uuids``. Returns the number of edges updated.
    """

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

    # Pull the supporting episodes' reference_times so we can compute
    # ``last_reinforced_at`` and the time span for stability escalation.
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
                if stability == "reinforced" and count >= _PERSISTENT_THRESHOLD:
                    # caught in the branch above when span is met; stay
                    # reinforced otherwise.
                    pass

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
