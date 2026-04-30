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

    attributes: dict[str, Any] = Field(default_factory=dict)


class CommunityEdge(_Edge):
    """community -> HAS_MEMBER -> entity"""
