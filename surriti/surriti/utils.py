"""Internal helpers: record parsing and SurrealDB row normalisation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from surriti.edges import CommunityEdge, EntityEdge, EpisodicEdge
from surriti.nodes import (
    CommunityNode,
    EntityAlias,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)


def _coerce_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _strip_record_id(value: Any) -> str:
    """SurrealDB record IDs come back as ``RecordID('table', 'key')`` objects
    or ``table:key`` strings. We only ever need the key portion when joining
    against our application-managed ``uuid`` field."""

    if value is None:
        return ""
    text = str(value)
    if ":" in text:
        return text.split(":", 1)[1].strip("⟨⟩")
    return text


def parse_episode(row: dict[str, Any]) -> EpisodicNode:
    return EpisodicNode(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        name=row.get("name", ""),
        source=EpisodeType(row.get("source", "message")),
        source_description=row.get("source_description", ""),
        content=row.get("content", ""),
        reference_time=_coerce_dt(row.get("reference_time")) or _utcnow(),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
        entity_edges=list(row.get("entity_edges") or []),
        affect=dict(row.get("affect") or {}),
        interaction_pattern=row.get("interaction_pattern"),
    )


def parse_entity(row: dict[str, Any]) -> EntityNode:
    return EntityNode(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        name=row.get("name", ""),
        summary=row.get("summary", "") or "",
        labels=list(row.get("labels") or ["Entity"]),
        attributes=dict(row.get("attributes") or {}),
        name_embedding=row.get("name_embedding"),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
        canonical_name=row.get("canonical_name"),
        aliases=list(row.get("aliases") or []),
        profile_summary=row.get("profile_summary", "") or "",
        profile_embedding=row.get("profile_embedding"),
        salience=float(row.get("salience") or 0.0),
        mention_count=int(row.get("mention_count") or 0),
        last_seen_at=_coerce_dt(row.get("last_seen_at")),
        merged_into=row.get("merged_into"),
        traits=list(row.get("traits") or []),
        goals_active=list(row.get("goals_active") or []),
        domain=row.get("domain"),
    )


def parse_entity_alias(row: dict[str, Any]) -> EntityAlias:
    return EntityAlias(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        alias=row.get("alias", ""),
        normalized_alias=row.get("normalized_alias", ""),
        entity_uuid=row.get("entity_uuid", ""),
        confidence=float(row.get("confidence") or 1.0),
        source_episode_uuid=row.get("source_episode_uuid"),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
    )


def parse_edge(row: dict[str, Any]) -> EntityEdge:
    return EntityEdge(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        source_node_uuid=row.get("source_node_uuid")
        or _strip_record_id(row.get("in")),
        target_node_uuid=row.get("target_node_uuid")
        or _strip_record_id(row.get("out")),
        name=row.get("name", ""),
        fact=row.get("fact", ""),
        fact_embedding=row.get("fact_embedding"),
        episodes=list(row.get("episodes") or []),
        valid_at=_coerce_dt(row.get("valid_at")),
        invalid_at=_coerce_dt(row.get("invalid_at")),
        expired_at=_coerce_dt(row.get("expired_at")),
        status=row.get("status") or "active",
        polarity=row.get("polarity") or "positive",
        source_type=row.get("source_type") or "user",
        confidence=float(row.get("confidence")) if row.get("confidence") is not None else 1.0,
        temporal=bool(row.get("temporal") or False),
        singleton=bool(row.get("singleton") or False),
        domain=row.get("domain"),
        supersedes=list(row.get("supersedes") or []),
        superseded_by=row.get("superseded_by"),
        fact_key=row.get("fact_key", "") or "",
        relation_frame_id=row.get("relation_frame_id"),
        canonical_name=row.get("canonical_name", "") or "",
        qualifiers=dict(row.get("qualifiers") or {}),
        roles=dict(row.get("roles") or {}),
        conflict_group_id=row.get("conflict_group_id"),
        derived=bool(row.get("derived") or False),
        derived_from=row.get("derived_from"),
        attributes=dict(row.get("attributes") or {}),
        weight=float(row.get("weight")) if row.get("weight") is not None else 1.0,
        reinforcement_count=int(row.get("reinforcement_count") or 1),
        last_reinforced_at=_coerce_dt(row.get("last_reinforced_at")),
        decay_score=float(row.get("decay_score")) if row.get("decay_score") is not None else 1.0,
        stability=str(row.get("stability") or "episodic"),
        valence=float(row.get("valence")) if row.get("valence") is not None else None,
        intensity=float(row.get("intensity")) if row.get("intensity") is not None else None,
        consolidates=list(row.get("consolidates") or []),
        is_belief=bool(row.get("is_belief") or False),
        belief_holder=row.get("belief_holder"),
        memory_class=str(
            (row.get("attributes") or {}).get("memory_class") or "objective"
        ).strip().lower() or "objective",
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
    )


def parse_episodic_edge(row: dict[str, Any]) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        source_node_uuid=_strip_record_id(row.get("in")),
        target_node_uuid=_strip_record_id(row.get("out")),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
    )


def parse_community_edge(row: dict[str, Any]) -> CommunityEdge:
    return CommunityEdge(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        source_node_uuid=_strip_record_id(row.get("in")),
        target_node_uuid=_strip_record_id(row.get("out")),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
    )


def parse_community(row: dict[str, Any]) -> CommunityNode:
    return CommunityNode(
        uuid=row.get("uuid") or _strip_record_id(row.get("id")),
        group_id=row.get("group_id", ""),
        name=row.get("name", ""),
        summary=row.get("summary", "") or "",
        name_embedding=row.get("name_embedding"),
        created_at=_coerce_dt(row.get("created_at")) or _utcnow(),
        kind=str(row.get("kind") or "cluster"),
        domain=row.get("domain"),
        payload=dict(row.get("payload") or {}),
    )
