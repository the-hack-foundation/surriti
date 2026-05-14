"""Decay function for the cognitive layer.

Effective confidence of an edge at read-time is::

    eff = base_confidence
        * exp(-ln(2) * Δt_days / half_life_days(stability))
        * (1 + log1p(reinforcement_count) / log(10))

clipped to ``[0, 1]``. The function is pure and cheap; ``recall()``
uses it as a multiplier on ranking scores, and the cognition pass
periodically snapshots the value to ``edge.decay_score`` so SurrealQL
can ``ORDER BY decay_score`` without recomputing.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from surriti.edges import EntityEdge

# Default half-lives in days, keyed by ``EntityEdge.stability``.
_DEFAULT_HALF_LIFE_DAYS: dict[str, float] = {
    "episodic": 30.0,
    "reinforced": 90.0,
    "persistent": 365.0,
    "consolidated": math.inf,
}


def half_life_for(stability: str, overrides: dict[str, float] | None = None) -> float:
    table = _DEFAULT_HALF_LIFE_DAYS
    if overrides:
        table = {**table, **overrides}
    return table.get(stability or "episodic", table["episodic"])


def effective_confidence(
    edge: EntityEdge,
    *,
    now: datetime | None = None,
    half_life_overrides: dict[str, float] | None = None,
) -> float:
    """Pure decay calculation; safe to call on any ``EntityEdge`` row."""

    now = now or datetime.now(timezone.utc)
    base = float(edge.confidence if edge.confidence is not None else 1.0)
    last = edge.last_reinforced_at or edge.valid_at or edge.created_at
    if last is None:
        last = now
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - last).total_seconds() / 86_400.0)
    hl = half_life_for(edge.stability, half_life_overrides)
    if math.isinf(hl):
        decay = 1.0
    else:
        decay = math.exp(-math.log(2.0) * delta_days / hl)
    count = max(1, int(edge.reinforcement_count or 1))
    reinforce = 1.0 + math.log1p(count - 1) / math.log(10.0)
    score = base * decay * reinforce
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
