"""Temporal/contradiction handling.

Graphiti expires older facts when newer episodes contradict them. Surriti
implements the same idea by:

1. Finding existing edges that are semantically similar to the new fact (via
   the same hybrid search the retrieval API uses).
2. Asking the LLM client which of those existing facts the new one
   invalidates.
3. Setting ``invalid_at`` (real-world time) and ``expired_at`` (transaction
   time) on the contradicted edges so future ``only_valid=True`` searches
   skip them while history remains queryable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from surriti.driver import SurrealDriver
from surriti.edges import EntityEdge
from surriti.llm import LLMClient
from surriti.search import _fulltext_search_edges, _vector_search_edges, _unwrap
from surriti.utils import parse_edge


async def find_similar_edges(
    driver: SurrealDriver,
    *,
    fact: str,
    fact_embedding: list[float] | None,
    group_id: str,
    limit: int = 10,
) -> list[EntityEdge]:
    rows: list[dict] = []
    if fact_embedding is not None:
        rows.extend(await _vector_search_edges(driver, fact_embedding, group_id, limit))
    if fact:
        rows.extend(await _fulltext_search_edges(driver, fact, group_id, limit))

    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["uuid"], row)
    return [parse_edge(r) for r in seen.values()]


async def invalidate_edges(
    driver: SurrealDriver,
    edge_uuids: Iterable[str],
    *,
    invalid_at: datetime,
) -> None:
    uuids = list(edge_uuids)
    if not uuids:
        return
    expired_at = datetime.now(timezone.utc)
    await driver.query(
        """
        UPDATE relates_to
        SET invalid_at = $invalid_at, expired_at = $expired_at
        WHERE uuid IN $uuids AND (invalid_at IS NONE OR invalid_at > $invalid_at);
        """,
        {
            "uuids": uuids,
            "invalid_at": invalid_at,
            "expired_at": expired_at,
        },
    )


async def resolve_contradictions(
    driver: SurrealDriver,
    *,
    llm: LLMClient,
    new_fact: str,
    new_fact_embedding: list[float] | None,
    new_valid_at: datetime,
    group_id: str,
    similarity_limit: int = 10,
) -> list[EntityEdge]:
    """Run the full contradiction pipeline. Returns the invalidated edges."""

    candidates = await find_similar_edges(
        driver,
        fact=new_fact,
        fact_embedding=new_fact_embedding,
        group_id=group_id,
        limit=similarity_limit,
    )
    if not candidates:
        return []

    fact_strings = [c.fact for c in candidates]
    contradicted_idx = await llm.find_contradictions(new_fact, fact_strings)
    if not contradicted_idx:
        return []

    invalidated = [candidates[i] for i in contradicted_idx if 0 <= i < len(candidates)]
    await invalidate_edges(
        driver, [e.uuid for e in invalidated], invalid_at=new_valid_at
    )
    return invalidated
