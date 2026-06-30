"""Linear memory-vitality decay for the cognitive layer."""

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
DEFAULT_REINFORCEMENT_BOOST = 0.03
MAX_RECALL_BOOST = 0.2
MAX_REINFORCEMENT_BOOST = 0.25

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
    memory_class = str(edge.memory_class or "objective").strip().lower() or "objective"
    return memory_class in PROTECTED_MEMORY_CLASSES or edge.stability == "consolidated"


def linear_vitality(
    edge: EntityEdge,
    *,
    now: datetime | None = None,
    points_per_day: float = DEFAULT_DECAY_POINTS_PER_DAY,
) -> float:
    now = now or datetime.now(timezone.utc)
    score = float(edge.decay_score if edge.decay_score is not None else 1.0)
    if is_decay_protected(edge):
        return max(0.0, min(1.0, score))
    last = edge.last_recalled_at or edge.last_reinforced_at or edge.valid_at or edge.created_at
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
    _ = half_life_overrides
    base = max(0.0, min(1.0, float(edge.confidence)))
    vitality = linear_vitality(edge, now=now)
    if is_decay_protected(edge):
        return base

    reinforcement_count = max(1, int(edge.reinforcement_count or 1))
    reinforcement_boost = min(
        MAX_REINFORCEMENT_BOOST,
        math.log1p(reinforcement_count - 1) * DEFAULT_REINFORCEMENT_BOOST,
    )
    recall_count = max(0, int(edge.recall_count or 0))
    recall_boost = min(MAX_RECALL_BOOST, recall_count * DEFAULT_RECALL_BOOST)

    score = (base * vitality) + reinforcement_boost + recall_boost
    return max(0.0, min(1.0, score))
