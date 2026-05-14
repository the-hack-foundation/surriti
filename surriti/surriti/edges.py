"""Edge models. These map to SurrealDB edge tables created via RELATE."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from surriti.nodes import _Base, utc_now


def make_fact_key(
    group_id: str,
    subject_uuid: str,
    predicate: str,
    object_uuid: str,
    *,
    qualifier_hash: str = "",
) -> str:
    """Deterministic dedupe key for a (subject, predicate, object) triple.

    Used to populate ``relates_to.fact_key`` on insert and to look up
    equivalent edges across episodes without relying on natural-language
    fact-text comparison. The format is:

        ``"<group_id>::<subject_uuid>::<predicate_lower>::<object_uuid>[::<qualifier_hash>]"``

    ``predicate`` should be the *canonical* relation name (frame
    ``canonical_name``) when a frame is available; the raw extractor
    string is acceptable as a fallback for backward compatibility.
    ``qualifier_hash`` (when supplied) keeps qualified variants
    (e.g. ``lives_in(Florida, season=winter)``) in distinct slots.
    """

    parts = [
        (group_id or "").strip(),
        (subject_uuid or "").strip(),
        (predicate or "").strip().lower(),
        (object_uuid or "").strip(),
    ]
    if qualifier_hash:
        parts.append(qualifier_hash)
    return "::".join(parts)


class _Edge(_Base):
    source_node_uuid: str
    target_node_uuid: str


class EpisodicEdge(_Edge):
    """episode -> MENTIONS -> entity"""


class EntityEdge(_Edge):
    """entity -> RELATES_TO -> entity (semantic fact edge)."""

    name: str = Field(description="Relation label / predicate")
    fact: str = ""
    fact_embedding: list[float] | None = None
    episodes: list[str] = Field(default_factory=list)
    """UUIDs of EpisodicNodes that support this fact."""

    valid_at: datetime | None = None
    """When the fact became true in the real world."""

    invalid_at: datetime | None = None
    """When the fact stopped being true (set when contradicted)."""

    expired_at: datetime | None = None
    """Transaction time when this edge was marked as superseded."""

    # Generic temporal-state metadata mirrored from ExtractedFact. None of
    # these encode any hardcoded predicate vocabulary; they're set per edge
    # by the extractor (or by the deterministic singleton-slot closer).
    status: str = "active"
    """"active" while the edge represents current truth; "superseded" when
    closed by a newer assertion; reserved values for future use."""
    polarity: str = "positive"
    source_type: str = "user"
    """Provenance: "user" facts are authoritative and trigger the
    deterministic singleton closer; "assistant"/"tool"/"system" facts
    skip it."""
    confidence: float = 1.0
    temporal: bool = False
    singleton: bool = False
    domain: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    """UUIDs of edges this one closed (set on insert when the singleton
    closer fires)."""
    superseded_by: str | None = None
    """UUID of the edge that closed this one (set on the closed edge)."""

    fact_key: str = ""
    """Deterministic ``(group_id, subject, predicate, object)`` key used
    to dedupe equivalent triples across episodes. Computed by
    :func:`make_fact_key` on insert; backfilled for legacy rows by the
    schema migration helper."""

    # Relation-frame layer (generalized predicate metadata). All optional
    # so legacy rows continue to load; populated on insert when a frame
    # resolves for the predicate.
    relation_frame_id: str | None = None
    """UUID of the :class:`~surriti.relation_frames.RelationFrame` this
    edge instantiates, when one was resolved at write time."""
    canonical_name: str = ""
    """Frame canonical predicate name (e.g. ``spouse_of``). Denormalized
    from the frame so slot/alias queries stay a single-table lookup."""
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    """Free-form qualifier scope (e.g. ``{"season": "winter"}``). Hashed
    into ``fact_key`` so qualified variants occupy distinct slots."""
    roles: dict[str, str] = Field(default_factory=dict)
    """Semantic argument roles surfaced by the extractor (e.g.
    ``{"subject": "self", "object": "spouse"}``)."""
    conflict_group_id: str | None = None
    """Set when an unresolved contradiction grouped this edge with one
    or more peers; surface via ``Surriti.get_conflicts()``."""
    derived: bool = False
    """True when this edge was synthesized by the engine (e.g. the
    inverse of an ``inverse_pair`` frame) rather than extracted directly."""
    derived_from: str | None = None
    """UUID of the source edge this one was derived from."""

    memory_class: str = "objective"
    """Closed-vocabulary kind tag carried from :class:`ExtractedFact`.
    See ``ExtractedFact.memory_class`` for values. Persisted in
    ``attributes['memory_class']`` on the row so legacy rows (without
    the field set) default to ``"objective"`` on read."""

    # Cognitive-layer fields (additive). ``surriti.cognition`` populates
    # these in background passes; legacy rows simply read defaults. The
    # public ``Surriti`` API surface is unchanged -- these are extra
    # signals consumed by rerankers and ``recall()`` enrichment.
    weight: float = 1.0
    """Composite associative weight = ``confidence * decay * (1 + assoc_boost)``
    snapshotted by the cognition pass. Used as an extra reranking
    signal when ``decay_aware`` is enabled."""
    reinforcement_count: int = 1
    """Number of distinct episodes that have re-asserted this fact."""
    last_reinforced_at: datetime | None = None
    """Timestamp of the most recent reinforcement (newest supporting
    episode)."""
    decay_score: float = 1.0
    """Snapshotted ``effective_confidence`` produced by the decay
    function; refreshed on each cognition pass."""
    stability: str = "episodic"
    """Lifecycle bucket -- one of ``episodic``, ``reinforced``,
    ``persistent``, ``consolidated``. Drives the per-bucket decay
    half-life."""
    valence: float | None = None
    """Emotional polarity in [-1, 1] inherited from the supporting
    episode's ``affect`` tag (Phase C affect pass)."""
    intensity: float | None = None
    """Emotional intensity in [0, 1] inherited from the supporting
    episode's ``affect`` tag."""
    consolidates: list[str] = Field(default_factory=list)
    """When non-empty, this is a synthesized abstraction edge that
    summarises the listed supporting edge UUIDs (Phase D consolidation)."""
    is_belief: bool = False
    """``True`` when the fact is a subjective belief rather than an
    objective claim (Phase C perspective layer). Beliefs are excluded
    from contradiction detection against ``objective`` facts."""
    belief_holder: str | None = None
    """UUID of the entity that holds the belief, when ``is_belief`` is
    ``True``."""

    attributes: dict[str, Any] = Field(default_factory=dict)


class CommunityEdge(_Edge):
    """community -> HAS_MEMBER -> entity"""
