"""Shared write helpers for the cognition layer.

Centralises the SurrealDB INSERT / RELATE patterns used by trait /
goal / consolidation modules so they all share the same field set and
keep parity with ``Surriti._insert_edge``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from surriti.edges import make_fact_key
from surriti.search import _unwrap

logger = logging.getLogger(__name__)


async def upsert_synthetic_entity(
    driver: Any,
    *,
    group_id: str,
    name: str,
    summary: str,
    label: str,
    now: datetime | None = None,
) -> str:
    """Upsert a synthesized ``EntityNode`` (trait / goal / domain / pattern).

    Returns the entity UUID (existing or newly created). The unique
    ``(group_id, name)`` index drives idempotency.
    """

    now = now or datetime.now(timezone.utc)
    existing = _unwrap(
        await driver.query(
            "SELECT uuid FROM entity WHERE group_id = $g AND name = $n LIMIT 1;",
            {"g": group_id, "n": name},
        )
    )
    if existing:
        uuid_ = str(existing[0].get("uuid"))
        await driver.query(
            "UPDATE entity SET summary = $s, last_seen_at = $t WHERE uuid = $u;",
            {"u": uuid_, "s": summary, "t": now},
        )
        return uuid_
    uuid_ = str(uuid4())
    await driver.query(
        """
        CREATE entity CONTENT {
            uuid: $u, group_id: $g, name: $n, summary: $s,
            labels: ['Entity', $l], attributes: {},
            created_at: $t, last_seen_at: $t,
            aliases: [], traits: [], goals_active: []
        };
        """,
        {"u": uuid_, "g": group_id, "n": name, "s": summary, "l": label, "t": now},
    )
    return uuid_


async def upsert_synthetic_edge(
    driver: Any,
    embedder: Any,
    *,
    group_id: str,
    subject_uuid: str,
    object_uuid: str,
    predicate: str,
    fact_text: str,
    memory_class: str,
    confidence: float,
    supporting_edge_uuids: list[str] | None = None,
    consolidates: list[str] | None = None,
    stability: str = "reinforced",
    is_belief: bool = False,
    belief_holder: str | None = None,
    extra_attrs: dict[str, Any] | None = None,
    fact_key_qualifier: str = "",
    now: datetime | None = None,
) -> str:
    """Upsert a synthesized RELATES_TO edge by ``fact_key``. Returns the
    edge UUID."""

    now = now or datetime.now(timezone.utc)
    fact_key = make_fact_key(
        group_id, subject_uuid, predicate, object_uuid,
        qualifier_hash=fact_key_qualifier,
    )

    fact_embedding = None
    try:
        if embedder is not None:
            embeddings = await embedder.create([fact_text])
            if embeddings:
                fact_embedding = list(embeddings[0])
    except Exception:
        logger.debug("synthetic edge embedding failed; storing without vector", exc_info=True)

    supporting = sorted({u for u in (supporting_edge_uuids or []) if u})
    consolidates = sorted({u for u in (consolidates or []) if u})
    attrs = {"memory_class": memory_class, "supporting_edges": supporting}
    if extra_attrs:
        attrs.update(extra_attrs)

    existing = _unwrap(
        await driver.query(
            "SELECT uuid FROM relates_to WHERE group_id = $g AND fact_key = $k LIMIT 1;",
            {"g": group_id, "k": fact_key},
        )
    )
    if existing:
        edge_uuid = str(existing[0].get("uuid"))
        await driver.query(
            """
            UPDATE relates_to SET
                fact = $f, confidence = $c, last_reinforced_at = $t,
                consolidates = $cons,
                attributes = object::extend(attributes, $attrs)
            WHERE uuid = $u;
            """,
            {"u": edge_uuid, "f": fact_text, "c": float(confidence), "t": now,
             "cons": consolidates, "attrs": attrs},
        )
        return edge_uuid

    edge_uuid = str(uuid4())
    await driver.query(
        """
        RELATE (type::record("entity", $su))->relates_to->(type::record("entity", $tu))
        CONTENT {
            uuid: $u, group_id: $g, name: $p,
            canonical_name: $p,
            fact: $f, fact_embedding: $emb,
            episodes: [], valid_at: $t, created_at: $t,
            last_reinforced_at: $t,
            attributes: $attrs,
            status: 'active', polarity: 'positive', source_type: 'system',
            confidence: $c, temporal: false, singleton: false,
            derived: true,
            derived_from: $df,
            fact_key: $k,
            weight: 1.0, reinforcement_count: 1,
            decay_score: 1.0, stability: $stab,
            consolidates: $cons,
            is_belief: $belief, belief_holder: $bh
        };
        """,
        {
            "su": subject_uuid, "tu": object_uuid, "u": edge_uuid,
            "g": group_id, "p": predicate, "f": fact_text, "emb": fact_embedding,
            "t": now, "attrs": attrs, "c": float(confidence), "k": fact_key,
            "df": (supporting[0] if supporting else None),
            "stab": stability, "cons": consolidates,
            "belief": bool(is_belief), "bh": belief_holder,
        },
    )
    return edge_uuid


async def cache_on_subject(
    driver: Any, *, subject_uuid: str, field: str, value: str
) -> None:
    """Append ``value`` to the entity's ``traits`` / ``goals_active``
    cache list (deduplicated)."""

    if field not in ("traits", "goals_active"):
        raise ValueError(f"unsupported subject cache field: {field!r}")
    await driver.query(
        f"""
        UPDATE entity SET {field} = array::distinct(array::concat({field}, [$v]))
        WHERE uuid = $u;
        """,
        {"u": subject_uuid, "v": value},
    )
