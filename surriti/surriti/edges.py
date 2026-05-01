"""Edge models. These map to SurrealDB edge tables created via RELATE."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from surriti.nodes import _Base, utc_now


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

    attributes: dict[str, Any] = Field(default_factory=dict)


class CommunityEdge(_Edge):
    """community -> HAS_MEMBER -> entity"""
