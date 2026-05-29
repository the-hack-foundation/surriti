"""Configuration for the cognitive abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CognitionConfig:
    """Tunables for the second-pass cognitive synthesizer.

    Defaults are chosen for "good for production": background
    debounced synthesis with hybrid heuristic + LLM ratification,
    decay-aware recall, periodic domain labelling, and a low-cost
    consolidation threshold. Pass ``cognition=False`` to
    :class:`~surriti.Surriti` to disable the whole layer.
    """

    enabled: bool = True
    """Master switch. When ``False``, the scheduler is never started
    and ``add_episode()`` does not call ``notify()``; ``recall()``
    behaves exactly as it did before this layer existed."""

    idle_seconds: float = 8.0
    """Debounce window after the last ``notify(group_id)`` before a
    cognition pass fires."""

    batch_threshold: int = 5
    """Force-fire the cognition pass when this many episodes have
    accumulated for a group, even if the idle window has not elapsed."""

    decay_aware_recall: bool = True
    """When ``True``, ``Surriti.recall()`` multiplies edge ranking
    scores by the decay function (more recent / more reinforced
    facts surface first)."""

    consolidation_threshold: int = 8
    """A ``fact_key`` with at least this many supporting episodes
    spanning ``consolidation_min_span_days`` is collapsed into a
    single ``memory_class='consolidated'`` abstraction edge."""

    consolidation_min_span_days: float = 14.0
    """Minimum time span (in days) between earliest and latest
    supporting episode before a fact is eligible for consolidation."""

    domain_labeling_every_n_passes: int = 3
    """Run domain-aware community labelling once every N cognition
    passes per group. ``1`` runs every pass; ``0`` disables."""

    affect_extraction: bool = True
    """When ``True``, the extraction prompt asks for an ``affect``
    block per episode and edges inherit valence/intensity from their
    supporting episode."""

    belief_extraction: bool = True
    """When ``True``, the extractor flags epistemic statements as
    beliefs (``is_belief=True``) and contradiction detection skips
    belief-vs-objective collisions."""

    trait_synthesis: bool = True
    """Enable Phase C trait synthesis (heuristic candidates + LLM
    ratify, persisted as ``EntityNode(labels=['trait'])`` plus
    ``has_trait`` edges)."""

    goal_synthesis: bool = True
    """Enable Phase C goal synthesis (intentional-verb pre-filter +
    LLM batch, persisted as ``EntityNode(labels=['goal'])`` plus
    ``pursues_goal`` edges)."""

    procedural_synthesis: bool = True
    """Enable Phase C procedural / interaction-pattern detection."""

    consolidation: bool = True
    """Enable Phase D episodic-to-semantic consolidation."""

    prediction: bool = True
    """Enable Phase D predictive-bundle synthesis (surfaced via
    ``recall(depth='deep')``)."""

    self_awareness: bool = True
    """Enable self-awareness pass: extract structured self-model
    (traits, beliefs, patterns) from self-episodes via LLM."""

    max_episodes_per_pass: int = 32
    """Upper bound on how many recent episodes to consider per
    cognition pass per group. Caps both LLM cost and memory."""

    decay_half_life_days: dict[str, float] | None = None
    """Override the per-stability decay half-lives in days. Defaults
    (when ``None``): ``episodic=30``, ``reinforced=90``,
    ``persistent=365``, ``consolidated=infinity``."""
