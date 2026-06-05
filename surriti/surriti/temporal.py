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
from surriti.llm import ContradictionCandidate, ExtractedFact, LLMClient
from surriti.search import _fulltext_search_edges, _vector_search_edges, _unwrap
from surriti.utils import parse_edge


async def find_similar_edges(
    driver: SurrealDriver,
    *,
    fact: str,
    fact_embedding: list[float] | None,
    group_id: str,
    limit: int = 10,
    co_subject_uuid: str | None = None,
    co_object_uuid: str | None = None,
    only_active: bool = True,
) -> list[EntityEdge]:
    """Return edges similar to the new fact for contradiction reasoning.

    The candidate pool is the union of:

    * vector-search hits on ``fact_embedding`` (when supplied),
    * fulltext-search hits on ``fact``,
    * **co-object active peers** -- every active edge in ``group_id``
      whose target is ``co_object_uuid``. This guarantees that
      transfer-of-state events ("X sold Y", "X moved Y to ...") see
      every prior fact attached to ``Y`` regardless of how
      semantically distant their predicates are from the new event,
      so the contradiction LLM has the full set of dependencies on
      the affected object to reason over,
    * **co-subject active peers** when ``co_subject_uuid`` is given
      and no ``co_object_uuid`` is supplied (rare; reserved for
      future per-subject sweeps).

    When ``only_active`` is true (default), superseded / invalidated
    rows are filtered out so the LLM only sees current truth.
    """
    rows: list[dict] = []
    if fact_embedding is not None:
        rows.extend(await _vector_search_edges(driver, fact_embedding, group_id, limit))
    if fact:
        rows.extend(await _fulltext_search_edges(driver, fact, group_id, limit))

    # Fallback: when vector and fulltext searches return nothing (common with
    # DummyEmbedder or very short facts), sweep active edges scoped to the
    # same subject/object so the contradiction LLM gets a relevant pool rather
    # than every edge in the group.
    if not rows:
        conditions = [
            'group_id = $group_id',
            'status = "active"',
            "invalid_at IS NONE",
        ]
        params: dict = {"group_id": group_id, "limit": limit * 2}
        if co_subject_uuid:
            conditions.append('in = type::record("entity", $subj)')
            params["subj"] = co_subject_uuid
        if co_object_uuid:
            conditions.append('out = type::record("entity", $obj)')
            params["obj"] = co_object_uuid
        rows = _unwrap(
            await driver.query(
                f"""
                SELECT * FROM relates_to
                WHERE {" AND ".join(conditions)}
                LIMIT $limit;
                """,
                params,
            )
        )

    if co_object_uuid:
        co_rows = _unwrap(
            await driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND out = type::record("entity", $obj)
                    AND status = "active"
                    AND invalid_at IS NONE
                LIMIT $limit;
                """,
                {"group_id": group_id, "obj": co_object_uuid, "limit": limit * 2},
            )
        )
        rows.extend(co_rows)
    elif co_subject_uuid:
        co_rows = _unwrap(
            await driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND in = type::record("entity", $sub)
                    AND status = "active"
                    AND invalid_at IS NONE
                LIMIT $limit;
                """,
                {"group_id": group_id, "sub": co_subject_uuid, "limit": limit * 2},
            )
        )
        rows.extend(co_rows)

    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["uuid"], row)

    edges = [parse_edge(r) for r in seen.values()]
    if only_active:
        edges = [
            e for e in edges
            if e.status == "active" and e.invalid_at is None
        ]
    return edges


async def invalidate_edges(
    driver: SurrealDriver,
    edge_uuids: Iterable[str],
    *,
    invalid_at: datetime,
    superseded_by: str | None = None,
) -> None:
    uuids = list(edge_uuids)
    if not uuids:
        return
    expired_at = datetime.now(timezone.utc)
    await driver.query(
        """
        UPDATE relates_to
        SET invalid_at = $invalid_at, expired_at = $expired_at,
            status = "superseded", superseded_by = $superseded_by
        WHERE uuid IN $uuids AND (invalid_at IS NONE OR invalid_at > $invalid_at);
        """,
        {
            "uuids": uuids,
            "invalid_at": invalid_at,
            "expired_at": expired_at,
            "superseded_by": superseded_by,
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
    new_fact_struct: ExtractedFact | None = None,
    new_edge_uuid: str | None = None,
    new_subject_uuid: str | None = None,
    new_object_uuid: str | None = None,
) -> list[EntityEdge]:
    """Run the full contradiction pipeline. Returns the invalidated edges.

    When ``new_fact_struct`` is supplied, structured
    :class:`~surriti.llm.ContradictionCandidate` records are built from
    the similar edges and forwarded to the LLM client so the prompt can
    reason about subject/predicate/object/domain explicitly. When
    ``new_edge_uuid`` is supplied, invalidated edges are stamped with
    ``superseded_by`` pointing at it. When ``new_object_uuid`` is
    supplied, the candidate set is enriched with every active edge
    sharing that object so transfer-of-state events surface their
    dependencies.
    """

    candidates_edges = await find_similar_edges(
        driver,
        fact=new_fact,
        fact_embedding=new_fact_embedding,
        group_id=group_id,
        limit=similarity_limit,
        co_subject_uuid=new_subject_uuid,
        co_object_uuid=new_object_uuid,
    )
    # Never let a candidate include the new edge itself, or any edge
    # already invalidated.
    if new_edge_uuid:
        candidates_edges = [c for c in candidates_edges if c.uuid != new_edge_uuid]
    if not candidates_edges:
        return []

    # Belief vs objective filter. A subjective belief ("I think X") and
    # a literal objective fact ("X is Y") are not in the same epistemic
    # class -- one cannot invalidate the other. Belief-vs-belief and
    # objective-vs-objective contradictions are still candidates.
    new_is_belief = bool(getattr(new_fact_struct, "is_belief", False))
    candidates_edges = [
        c for c in candidates_edges
        if bool(getattr(c, "is_belief", False)) == new_is_belief
        or (getattr(c, "memory_class", None) == "belief") == new_is_belief
    ]
    if not candidates_edges:
        return []

    fact_strings = [c.fact for c in candidates_edges]

    # Build structured candidates whenever we have a new ExtractedFact in
    # scope. Stub clients ignore them; production clients use them.
    structured: list[ContradictionCandidate] | None = None
    if new_fact_struct is not None:
        structured = []
        for edge in candidates_edges:
            structured.append(
                ContradictionCandidate(
                    uuid=edge.uuid,
                    subject=edge.source_node_uuid,
                    predicate=edge.name,
                    object=edge.target_node_uuid,
                    fact=edge.fact,
                    domain=edge.domain,
                    valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
                    invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None,
                )
            )

    contradicted_idx = await llm.find_contradictions(
        new_fact,
        fact_strings,
        candidates=structured,
        new_fact_struct=new_fact_struct,
    )
    if not contradicted_idx:
        return []

    invalidated = [
        candidates_edges[i] for i in contradicted_idx if 0 <= i < len(candidates_edges)
    ]
    await invalidate_edges(
        driver,
        [e.uuid for e in invalidated],
        invalid_at=new_valid_at,
        superseded_by=new_edge_uuid,
    )
    return invalidated
