"""Hybrid search over the SurrealDB-backed knowledge graph.

The strategy mirrors Graphiti's edge search: run a vector KNN query and a
BM25 full-text query in parallel, then fuse the rankings with a configurable
reranker (RRF, MMR, cross-encoder, node-distance, episode-mentions).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from surriti.driver import SurrealDriver
from surriti.edges import EntityEdge
from surriti.nodes import CommunityNode, EntityNode, EpisodicNode
from surriti.rerankers import (
    CrossEncoderClient,
    cross_encoder_rerank,
    episode_mentions_rerank,
    mmr_rerank,
    rrf,
)
from surriti.search_filters import SearchFilters, edge_passes_filters, node_passes_filters
from surriti.utils import (
    _strip_record_id,
    parse_community,
    parse_edge,
    parse_entity,
    parse_episode,
)

DEFAULT_LIMIT = 10
RRF_K = 60  # Standard Reciprocal Rank Fusion smoothing constant.


class Reranker(str, Enum):
    rrf = "rrf"
    mmr = "mmr"
    cross_encoder = "cross_encoder"
    node_distance = "node_distance"
    episode_mentions = "episode_mentions"


@dataclass
class SearchConfig:
    limit: int = DEFAULT_LIMIT
    candidate_limit: int = 50
    use_vector: bool = True
    use_fulltext: bool = True
    only_valid: bool = True
    """If True, exclude edges whose ``invalid_at`` / ``expired_at`` are in the past."""
    focal_uuid: str | None = None
    """Optional entity UUID; results are re-ranked by hop distance to it."""
    reranker: Reranker = Reranker.rrf
    mmr_lambda: float = 0.5
    cross_encoder: CrossEncoderClient | None = None
    include_nodes: bool = False
    include_episodes: bool = False
    include_communities: bool = False
    filters: SearchFilters | None = None
    decay_aware: bool = False
    """Multiply post-fusion scores by ``effective_confidence(edge, now)``
    so reinforced/persistent edges outrank stale ones. Off by default
    to preserve current ``Surriti.search`` semantics; ``Surriti.recall``
    opts in."""
    decay_half_life_overrides: dict[str, float] | None = None


@dataclass
class SearchResults:
    edges: list[EntityEdge] = field(default_factory=list)
    nodes: list[EntityNode] = field(default_factory=list)
    episodes: list[EpisodicNode] = field(default_factory=list)
    communities: list[CommunityNode] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    """Final fused score per edge UUID."""


def _rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    return rrf(rankings, k=k)


async def _vector_search_edges(
    driver: SurrealDriver,
    query_embedding: list[float],
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = "WHERE fact_embedding IS NOT NONE"
    if group_id is not None:
        where += " AND group_id = $group_id"
    surql = f"""
        SELECT *
        FROM relates_to
        {where}
            AND fact_embedding <|{limit},40|> $vec
        LIMIT {limit};
    """
    rows = await driver.query(surql, {"vec": query_embedding, "group_id": group_id})
    return _unwrap(rows)


async def _fulltext_search_edges(
    driver: SurrealDriver,
    query: str,
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = "WHERE fact @1@ $q"
    if group_id is not None:
        where += " AND group_id = $group_id"
    surql = f"""
        SELECT *, search::score(1) AS score
        FROM relates_to
        {where}
        ORDER BY score DESC
        LIMIT {limit};
    """
    rows = await driver.query(surql, {"q": query, "group_id": group_id})
    return _unwrap(rows)


def _unwrap(rows: Any) -> list[dict[str, Any]]:
    """Normalise SurrealDB query results to a flat list of dicts.

    Handles the three shapes returned across SDK versions:
    - ``[{"result": [...]}, ...]``       (legacy SDK 1.x multi-statement)
    - ``[[...], ...]``                   (SDK 1.x list-of-lists)
    - ``[{...}, {...}, ...]`` or ``{...}`` (SDK 2.0 flat result of last stmt)
    """

    if rows is None:
        return []
    if isinstance(rows, dict):
        if "result" in rows:
            return list(rows["result"] or [])
        return [rows]
    if not isinstance(rows, list):
        return list(rows)
    if not rows:
        return []
    # SDK 2.0: flat list of row dicts (no "result" wrapper).
    if all(isinstance(r, dict) and "result" not in r for r in rows):
        return list(rows)
    last = rows[-1]
    if isinstance(last, dict) and "result" in last:
        return list(last["result"] or [])
    if isinstance(last, list):
        return list(last)
    return list(rows)


def _filter_valid(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    out = []
    for e in edges:
        invalid_at = _as_aware(e.get("invalid_at"))
        expired_at = _as_aware(e.get("expired_at"))
        if invalid_at and invalid_at <= now:
            continue
        if expired_at and expired_at <= now:
            continue
        # Generic status guard: only "active" edges count as current truth.
        # Older rows that pre-date the field default to "active" by virtue of
        # the schema DEFAULT, so this is backwards-compatible.
        status = e.get("status")
        if status is not None and status != "active":
            continue
        out.append(e)
    return out


def _as_aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def hybrid_search(
    driver: SurrealDriver,
    *,
    query: str,
    query_embedding: list[float] | None,
    group_id: str | None = None,
    config: SearchConfig | None = None,
    ego_filter: list[str] | None = None,
) -> SearchResults:
    cfg = config or SearchConfig()
    rankings: list[list[str]] = []
    raw_by_uuid: dict[str, dict[str, Any]] = {}

    if cfg.use_vector and query_embedding is not None:
        vector_hits = await _vector_search_edges(
            driver, query_embedding, group_id, cfg.candidate_limit
        )
        for hit in vector_hits:
            raw_by_uuid.setdefault(hit["uuid"], hit)
        rankings.append([h["uuid"] for h in vector_hits])

    if cfg.use_fulltext and query.strip():
        ft_hits = await _fulltext_search_edges(
            driver, query, group_id, cfg.candidate_limit
        )
        for hit in ft_hits:
            raw_by_uuid.setdefault(hit["uuid"], hit)
        rankings.append([h["uuid"] for h in ft_hits])

    fused = _rrf_merge(rankings)

    candidates = [raw_by_uuid[u] for u in fused if u in raw_by_uuid]
    if cfg.only_valid:
        candidates = _filter_valid(candidates)
    candidates = [c for c in candidates if edge_passes_filters(c, cfg.filters)]

    # Ego filter: keep only edges whose source or target is in the
    # provided uuid set. Used by ``Surriti.recall`` to limit fact
    # retrieval to the entities mentioned in the user's query.
    if ego_filter:
        ego = set(ego_filter)
        candidates = [
            c for c in candidates
            if _strip_record_id(c.get("in")) in ego
            or _strip_record_id(c.get("out")) in ego
        ]

    candidates = await _apply_reranker(
        driver=driver,
        candidates=candidates,
        fused=fused,
        cfg=cfg,
        query=query,
        query_embedding=query_embedding,
    )

    # Decay-aware multiplicative rescoring. We apply this *after* the
    # configured reranker so it never displaces a focal/MMR/CE-driven
    # ordering -- it only down-weights stale edges within the chosen
    # candidate set. The fused score map is updated so callers reading
    # ``SearchResults.scores`` see the same number we ranked by.
    if cfg.decay_aware and candidates:
        from surriti.cognition.decay import effective_confidence
        from surriti.utils import parse_edge as _pe

        now = datetime.now(timezone.utc)
        for c in candidates:
            try:
                eff = effective_confidence(
                    _pe(c),
                    now=now,
                    half_life_overrides=cfg.decay_half_life_overrides,
                )
            except Exception:
                eff = 1.0
            uid = c.get("uuid")
            if uid is None:
                continue
            base = fused.get(uid, 0.0) or 0.0
            fused[uid] = float(base) * float(eff)
        candidates.sort(key=lambda e: fused.get(e["uuid"], 0.0), reverse=True)

    edges = [parse_edge(c) for c in candidates[: cfg.limit]]
    return SearchResults(edges=edges, scores=fused)


async def _apply_reranker(
    *,
    driver: SurrealDriver,
    candidates: list[dict[str, Any]],
    fused: dict[str, float],
    cfg: SearchConfig,
    query: str,
    query_embedding: list[float] | None,
) -> list[dict[str, Any]]:
    if cfg.focal_uuid or cfg.reranker is Reranker.node_distance:
        if not cfg.focal_uuid:
            raise ValueError("Reranker.node_distance requires SearchConfig.focal_uuid")
        return await _rerank_by_focal_distance(driver, candidates, cfg.focal_uuid, fused)
    if cfg.reranker is Reranker.mmr:
        return mmr_rerank(
            candidates=candidates,
            query_embedding=query_embedding,
            embedding_field="fact_embedding",
            lambda_mult=cfg.mmr_lambda,
            limit=cfg.limit,
        )
    if cfg.reranker is Reranker.cross_encoder:
        if cfg.cross_encoder is None:
            raise ValueError("Reranker.cross_encoder requires SearchConfig.cross_encoder")
        return await cross_encoder_rerank(
            candidates=candidates,
            query=query,
            text_field="fact",
            cross_encoder=cfg.cross_encoder,
            limit=cfg.limit,
        )
    if cfg.reranker is Reranker.episode_mentions:
        return episode_mentions_rerank(candidates, cfg.limit)
    # Default: RRF fused order
    candidates.sort(key=lambda e: fused.get(e["uuid"], 0.0), reverse=True)
    return candidates


async def _rerank_by_focal_distance(
    driver: SurrealDriver,
    candidates: list[dict[str, Any]],
    focal_uuid: str,
    fused: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Order candidates by hop distance from the focal entity (closer first).

    Falls back to vector/text fused score when distances tie.
    """

    # One BFS up to depth 3 from the focal entity, recording each edge's UUID
    # alongside the depth at which it was discovered.
    surql = """
        LET $focal = (SELECT * FROM entity WHERE uuid = $focal_uuid LIMIT 1)[0];
        RETURN IF $focal == NONE THEN [] ELSE
            array::concat(
                (SELECT uuid, 1 AS depth FROM $focal->relates_to),
                (SELECT uuid, 1 AS depth FROM $focal<-relates_to),
                (SELECT uuid, 2 AS depth FROM $focal->relates_to->entity->relates_to),
                (SELECT uuid, 2 AS depth FROM $focal->relates_to->entity<-relates_to)
            )
        END;
    """
    rows = await driver.query(surql, {"focal_uuid": focal_uuid})
    distance_records = _unwrap(rows)
    distances: dict[str, int] = {}
    for record in distance_records:
        uuid = record.get("uuid")
        depth = record.get("depth", 99)
        if uuid and (uuid not in distances or depth < distances[uuid]):
            distances[uuid] = depth

    def sort_key(edge: dict[str, Any]) -> tuple[int, float]:
        score = (fused or {}).get(edge["uuid"], edge.get("_score", 0.0))
        return (distances.get(edge["uuid"], 99), -float(score))

    return sorted(candidates, key=sort_key)


def parse_entities(rows: list[dict[str, Any]]) -> list[EntityNode]:
    return [parse_entity(r) for r in rows]


def parse_episodes(rows: list[dict[str, Any]]) -> list[EpisodicNode]:
    return [parse_episode(r) for r in rows]


# ---------------------------------------------------------------- node / episode search


async def _vector_search_nodes(
    driver: SurrealDriver,
    query_embedding: list[float],
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = "WHERE name_embedding IS NOT NONE"
    if group_id is not None:
        where += " AND group_id = $group_id"
    rows = await driver.query(
        f"""
        SELECT * FROM entity
        {where}
            AND name_embedding <|{limit},40|> $vec
        LIMIT {limit};
        """,
        {"vec": query_embedding, "group_id": group_id},
    )
    return _unwrap(rows)


async def _fulltext_search_nodes(
    driver: SurrealDriver,
    query: str,
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    # The entity FT index covers (name, summary); a single match on `name`
    # is sufficient for SurrealDB to use it.
    where = "WHERE name @1@ $q"
    if group_id is not None:
        where += " AND group_id = $group_id"
    rows = await driver.query(
        f"SELECT * FROM entity {where} LIMIT {limit};",
        {"q": query, "group_id": group_id},
    )
    return _unwrap(rows)


async def _fulltext_search_episodes(
    driver: SurrealDriver,
    query: str,
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = "WHERE content @1@ $q"
    if group_id is not None:
        where += " AND group_id = $group_id"
    rows = await driver.query(
        f"SELECT * FROM episode {where} LIMIT {limit};",
        {"q": query, "group_id": group_id},
    )
    return _unwrap(rows)


async def _vector_search_communities(
    driver: SurrealDriver,
    query_embedding: list[float],
    group_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = "WHERE name_embedding IS NOT NONE"
    if group_id is not None:
        where += " AND group_id = $group_id"
    rows = await driver.query(
        f"""
        SELECT * FROM community
        {where}
            AND name_embedding <|{limit},40|> $vec
        LIMIT {limit};
        """,
        {"vec": query_embedding, "group_id": group_id},
    )
    return _unwrap(rows)


async def search_nodes(
    driver: SurrealDriver,
    *,
    query: str,
    query_embedding: list[float] | None,
    group_id: str | None,
    limit: int,
    filters: SearchFilters | None = None,
) -> list[EntityNode]:
    rankings: list[list[str]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    if query_embedding is not None:
        for hit in await _vector_search_nodes(driver, query_embedding, group_id, limit):
            by_uuid.setdefault(hit["uuid"], hit)
        rankings.append([h["uuid"] for h in by_uuid.values()])
    if query.strip():
        for hit in await _fulltext_search_nodes(driver, query, group_id, limit):
            by_uuid.setdefault(hit["uuid"], hit)
    fused = _rrf_merge(rankings)
    rows = [by_uuid[u] for u in by_uuid if node_passes_filters(by_uuid[u], filters)]
    rows.sort(key=lambda r: fused.get(r["uuid"], 0.0), reverse=True)
    return [parse_entity(r) for r in rows[:limit]]


async def search_episodes(
    driver: SurrealDriver,
    *,
    query: str,
    group_id: str | None,
    limit: int,
) -> list[EpisodicNode]:
    if not query.strip():
        return []
    rows = await _fulltext_search_episodes(driver, query, group_id, limit)
    return [parse_episode(r) for r in rows]


async def search_communities(
    driver: SurrealDriver,
    *,
    query: str,
    query_embedding: list[float] | None,
    group_id: str | None,
    limit: int,
) -> list[CommunityNode]:
    rows: list[dict[str, Any]] = []
    if query_embedding is not None:
        rows.extend(await _vector_search_communities(driver, query_embedding, group_id, limit))
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        seen.setdefault(r["uuid"], r)
    return [parse_community(r) for r in list(seen.values())[:limit]]
