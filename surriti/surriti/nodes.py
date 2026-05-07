"""Node models mirroring Graphiti's Episodic / Entity / Community concepts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EpisodeType(str, Enum):
    message = "message"
    json = "json"
    text = "text"
    fact_triple = "fact_triple"


class _Base(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class EpisodicNode(_Base):
    """A raw input episode (chat message, document, structured event)."""

    name: str
    source: EpisodeType = EpisodeType.message
    source_description: str = ""
    content: str = ""
    reference_time: datetime = Field(default_factory=utc_now)
    entity_edges: list[str] = Field(default_factory=list)
    """UUIDs of EntityEdges derived from this episode."""


class EntityNode(_Base):
    """A discrete entity (person, place, thing) extracted from episodes."""

    name: str
    name_embedding: list[float] | None = None
    summary: str = ""
    labels: list[str] = Field(default_factory=lambda: ["Entity"])
    attributes: dict[str, Any] = Field(default_factory=dict)
    # Dossier / profile fields. ``profiles.refresh_entity_profiles``
    # materialises these after each ingest. Kept optional so existing
    # callers and stored rows are unaffected.
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    profile_summary: str = ""
    profile_embedding: list[float] | None = None
    salience: float = 0.0
    mention_count: int = 0
    last_seen_at: datetime | None = None
    merged_into: str | None = None


class EntityAlias(_Base):
    """A known surface form of an entity within a tenant.

    Stored in the ``entity_alias`` table. ``normalized_alias`` is the
    casefolded / whitespace-collapsed form used for O(1) lookup before
    semantic / LLM resolution runs.
    """

    alias: str
    normalized_alias: str
    entity_uuid: str
    confidence: float = 1.0
    source_episode_uuid: str | None = None


class CommunityNode(_Base):
    """A cluster of related entities."""

    name: str
    name_embedding: list[float] | None = None
    summary: str = ""
