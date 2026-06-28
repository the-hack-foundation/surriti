"""Linear memory-vitality decay for the cognitive layer.

Surriti stores long-term fact priority in ``EntityEdge.decay_score`` on a
simple 0..1 scale:

    1.00 == 100 vitality points
    0.50 == 50 vitality points
    0.00 == stagnant / archived candidate

The production policy is intentionally inspectable: a memory loses one point
per full day (``0.01``), and recall reinforcement adds four points (``0.04``;
see ``reinforcement.reinforce_edges_on_recall``). Zero-score facts are
handled by the associative pass.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from surriti.edges import EntityEdge

_DEFAULT_HALF_LIFE_DAYS: dict[str, float] = {
    "episodic": 30.0,
    "reinforced": 90.0,
    "persistent": 365.0,
    "consolidated": math.inf,
}

DEFAULT_DECAY_POINTS_PER_DAY = 0.01
DEFAULT_RECALL_BOOST = 0.04

PROTECTED_MEMORY_CLASSES: frozenset[str] = frozenset(
    {
        "constraint",
        "style",
        "preference",
        "goal",
        "trait",
        "self_model",
        "consolidated",
        "archived_summary",
    }
)


def half_life_for(stability: str, overrides: dict[str, float] | None = None) -> float:
    """Compatibility shim for older callers that import half-life helpers."""

    table = _DEFAULT_HALF_LIFE_DAYS
    if overrides:
        table = {**table, **overrides}
    return table.get(stability or "episodic", table["episodic"])


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def is_decay_protected(edge: EntityEdge) -> bool:
    """Return True when an edge should not be archived by passive decay."""

    memory_class = str(edge.memory_class or "objective").strip().lower() or "objective"
    return memory_class in PROTECTED_MEMORY_CLASSES or edge.stability == "consolidated"


def linear_vitality(
    edge: EntityEdge,
    *,
    now: datetime | None = None,
    points_per_day: float = DEFAULT_DECAY_POINTS_PER_DAY,
) -> float:
    """Return the current 0..1 vitality score after lazy daily decay."""

    now = now or datetime.now(timezone.utc)
    score = float(edge.decay_score if edge.decay_score is not None else 1.0)

    if is_decay_protected(edge):
        return max(0.0, min(1.0, score))

    last = (
        edge.last_decayed_at
        or edge.last_recalled_at
        or edge.last_reinforced_at
        or edge.valid_at
        or edge.created_at
    )
    last = _as_aware(last)
    if last is None:
        return max(0.0, min(1.0, score))

    days = max(0, int((now - last).total_seconds() // 86_400))
    decayed = score - (days * max(0.0, float(points_per_day)))
    return max(0.0, min(1.0, decayed))


def effective_confidence(
    edge: EntityEdge,
    *,
    now: datetime | None = None,
    half_life_overrides: dict[str, float] | None = None,
) -> float:
    """Current ranking vitality for an edge."""

    _ = half_life_overrides
    return linear_vitality(edge, now=now)
