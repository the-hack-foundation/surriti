"""Configuration for the cognitive abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CognitionConfig:
    """Tunables for the second-pass cognitive synthesizer."""

    enabled: bool = True
    idle_seconds: float = 8.0
    batch_threshold: int = 5

    decay_aware_recall: bool = True

    consolidation_threshold: int = 8
    consolidation_min_span_days: float = 14.0
    domain_labeling_every_n_passes: int = 3

    affect_extraction: bool = True
    belief_extraction: bool = True
    trait_synthesis: bool = True
    goal_synthesis: bool = True
    procedural_synthesis: bool = True
    consolidation: bool = True
    stagnant_consolidation: bool = True
    stagnant_min_edges_per_summary: int = 5
    stagnant_max_edges_per_pass: int = 120
    prediction: bool = True
    self_awareness: bool = True

    max_episodes_per_pass: int = 32
    decay_half_life_days: dict[str, float] | None = None
