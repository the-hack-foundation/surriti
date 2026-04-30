"""Pre-built :class:`~surriti.search.SearchConfig` recipes mirroring
``graphiti_core.search.search_config_recipes``.

Each recipe is a fresh instance, so callers may freely mutate it.
"""

from __future__ import annotations

from surriti.search import Reranker, SearchConfig

DEFAULT_LIMIT = 10


def _edge_rrf(limit: int = DEFAULT_LIMIT) -> SearchConfig:
    return SearchConfig(limit=limit, reranker=Reranker.rrf)


def _edge_mmr(limit: int = DEFAULT_LIMIT) -> SearchConfig:
    return SearchConfig(limit=limit, reranker=Reranker.mmr, mmr_lambda=0.5)


def _edge_node_distance(focal_uuid: str, limit: int = DEFAULT_LIMIT) -> SearchConfig:
    return SearchConfig(limit=limit, reranker=Reranker.node_distance, focal_uuid=focal_uuid)


def _edge_episode_mentions(limit: int = DEFAULT_LIMIT) -> SearchConfig:
    return SearchConfig(limit=limit, reranker=Reranker.episode_mentions)


def _edge_cross_encoder(limit: int = DEFAULT_LIMIT) -> SearchConfig:
    return SearchConfig(limit=limit, reranker=Reranker.cross_encoder)


# --- Edge-only recipes ------------------------------------------------------
EDGE_HYBRID_SEARCH_RRF = _edge_rrf()
EDGE_HYBRID_SEARCH_MMR = _edge_mmr()
EDGE_HYBRID_SEARCH_EPISODE_MENTIONS = _edge_episode_mentions()
EDGE_HYBRID_SEARCH_CROSS_ENCODER = _edge_cross_encoder()


def edge_hybrid_search_node_distance(focal_uuid: str) -> SearchConfig:
    """Factory: edges reranked by hop distance to ``focal_uuid``."""
    return _edge_node_distance(focal_uuid)


# --- Combined recipes (edges + nodes + episodes) ----------------------------
def _combined(reranker: Reranker, limit: int = DEFAULT_LIMIT, **kw) -> SearchConfig:
    return SearchConfig(
        limit=limit,
        reranker=reranker,
        include_nodes=True,
        include_episodes=True,
        include_communities=True,
        **kw,
    )


COMBINED_HYBRID_SEARCH_RRF = _combined(Reranker.rrf)
COMBINED_HYBRID_SEARCH_MMR = _combined(Reranker.mmr, mmr_lambda=0.5)
COMBINED_HYBRID_SEARCH_CROSS_ENCODER = _combined(Reranker.cross_encoder)


# --- Node-only recipes ------------------------------------------------------
NODE_HYBRID_SEARCH_RRF = SearchConfig(
    limit=DEFAULT_LIMIT, include_nodes=True, use_vector=True, use_fulltext=True,
    reranker=Reranker.rrf,
)
NODE_HYBRID_SEARCH_MMR = SearchConfig(
    limit=DEFAULT_LIMIT, include_nodes=True, reranker=Reranker.mmr, mmr_lambda=0.5,
)
NODE_HYBRID_SEARCH_EPISODE_MENTIONS = SearchConfig(
    limit=DEFAULT_LIMIT, include_nodes=True, reranker=Reranker.episode_mentions,
)
NODE_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    limit=DEFAULT_LIMIT, include_nodes=True, reranker=Reranker.cross_encoder,
)


# --- Community recipes ------------------------------------------------------
COMMUNITY_HYBRID_SEARCH_RRF = SearchConfig(
    limit=DEFAULT_LIMIT, include_communities=True, reranker=Reranker.rrf,
)
COMMUNITY_HYBRID_SEARCH_MMR = SearchConfig(
    limit=DEFAULT_LIMIT, include_communities=True, reranker=Reranker.mmr, mmr_lambda=0.5,
)
COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    limit=DEFAULT_LIMIT, include_communities=True, reranker=Reranker.cross_encoder,
)


__all__ = [
    "COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
    "COMBINED_HYBRID_SEARCH_MMR",
    "COMBINED_HYBRID_SEARCH_RRF",
    "COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER",
    "COMMUNITY_HYBRID_SEARCH_MMR",
    "COMMUNITY_HYBRID_SEARCH_RRF",
    "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
    "EDGE_HYBRID_SEARCH_EPISODE_MENTIONS",
    "EDGE_HYBRID_SEARCH_MMR",
    "EDGE_HYBRID_SEARCH_RRF",
    "NODE_HYBRID_SEARCH_CROSS_ENCODER",
    "NODE_HYBRID_SEARCH_EPISODE_MENTIONS",
    "NODE_HYBRID_SEARCH_MMR",
    "NODE_HYBRID_SEARCH_RRF",
    "edge_hybrid_search_node_distance",
]
