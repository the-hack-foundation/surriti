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


class CommunityNode(_Base):
    """A cluster of related entities."""

    name: str
    name_embedding: list[float] | None = None
    summary: str = ""
