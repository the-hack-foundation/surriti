"""Reranker strategies (RRF, MMR, cross-encoder, node-distance, episode-mentions)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from surriti.embedder import cosine_similarity


class CrossEncoderClient(ABC):
    """Same shape as Graphiti's CrossEncoderClient."""

    @abstractmethod
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        ...


class DummyCrossEncoder(CrossEncoderClient):
    """Token-overlap scorer. Useful for tests without an LLM."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        q_tokens = {t.lower() for t in query.split() if t}
        scored: list[tuple[str, float]] = []
        for p in passages:
            p_tokens = {t.lower() for t in p.split() if t}
            denom = max(len(q_tokens), 1)
            scored.append((p, len(q_tokens & p_tokens) / denom))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rankings:
        for rank, uuid in enumerate(ranked):
            scores[uuid] += 1.0 / (k + rank + 1)
    return dict(scores)


def mmr_rerank(
    *,
    candidates: list[dict[str, Any]],
    query_embedding: list[float] | None,
    embedding_field: str,
    lambda_mult: float = 0.5,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance.

    ``score = λ * sim(query, c) - (1-λ) * max_{s in selected} sim(c, s)``
    """

    if query_embedding is None or not candidates:
        return candidates[:limit]

    pool = list(candidates)
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < limit:
        best_idx = 0
        best_score = -1e9
        for i, cand in enumerate(pool):
            emb = cand.get(embedding_field)
            if emb is None:
                continue
            relevance = cosine_similarity(query_embedding, emb)
            redundancy = 0.0
            for s in selected:
                s_emb = s.get(embedding_field)
                if s_emb is not None:
                    redundancy = max(redundancy, cosine_similarity(emb, s_emb))
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(pool.pop(best_idx))
    return selected


async def cross_encoder_rerank(
    *,
    candidates: list[dict[str, Any]],
    query: str,
    text_field: str,
    cross_encoder: CrossEncoderClient,
    limit: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    passages = [str(c.get(text_field, "")) for c in candidates]
    ranked = await cross_encoder.rank(query, passages)
    order = {p: i for i, (p, _) in enumerate(ranked)}
    candidates.sort(key=lambda c: order.get(str(c.get(text_field, "")), len(candidates)))
    return candidates[:limit]


def episode_mentions_rerank(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank edges by how many episodes mention them (descending)."""

    return sorted(
        candidates,
        key=lambda c: len(c.get("episodes") or []),
        reverse=True,
    )[:limit]
