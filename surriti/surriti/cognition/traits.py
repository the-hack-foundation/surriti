"""Trait synthesis -- the keystone of the cognitive layer.

For each subject entity in a group, we mine candidate traits from the
edges incident to that entity:

- **Reinforcement-driven candidates** -- ``(predicate, object_uuid)``
  pairs that recur across multiple supporting episodes.
- **Predicate-frequency candidates** -- predicates that fire on the
  subject many times (regardless of object) suggest a behavioural
  tendency.

Candidates are then handed to the LLM (one batched call per subject)
which names, describes, scores, and discards them. Surviving traits
become ``EntityNode(labels=['trait'])`` rows attached to the subject by
``has_trait`` ``RELATES_TO`` edges with ``memory_class='trait'``. All
writes are deduped by ``fact_key`` so re-runs reinforce rather than
duplicate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from surriti.cognition._jsonio import parse_json_loose, snake_case
from surriti.cognition._writes import (
    cache_on_subject,
    upsert_synthetic_edge,
    upsert_synthetic_entity,
)
from surriti.cognition.prompts import TRAIT_RATIFY_SYSTEM
from surriti.edges import EntityEdge
from surriti.search import _unwrap
from surriti.utils import parse_edge
from surriti.validators import IDENTITY_PREDICATES

logger = logging.getLogger(__name__)


_MIN_REINFORCEMENT = 2  # candidate must hit at least this often
_MAX_CANDIDATES_PER_SUBJECT = 12
_MAX_SUPPORTING_FACTS_SHOWN = 4

# Predicates that name a single slot-value (employer, residence, age,
# birthday, locker number, current vehicle) rather than a behavioural
# tendency. Together with ``IDENTITY_PREDICATES`` and the per-edge
# ``singleton`` flag, these are excluded from trait candidate buckets:
# a trait describes recurring behaviour, not one-of-a-kind facts.
_NON_TRAIT_PREDICATES: frozenset[str] = frozenset(
    {
        "has_trait",
        "pursues_goal",
        "is_age",
        "has_birthday",
        "has_age",
        "is_a",
    }
) | IDENTITY_PREDICATES


def _select_candidate_subjects(
    edges: list[EntityEdge], episode_uuids: set[str]
) -> list[str]:
    """Subjects whose edges overlap with the recent episode batch."""

    seen: set[str] = set()
    for e in edges:
        if any(ep in episode_uuids for ep in (e.episodes or [])):
            seen.add(e.source_node_uuid)
    return list(seen)


def _candidates_for_subject(
    subject_uuid: str, edges: list[EntityEdge]
) -> list[dict[str, Any]]:
    incident = [
        e for e in edges
        if e.source_node_uuid == subject_uuid
        and e.status == "active"
        and e.memory_class in ("objective", "preference", "style", "trait", "sentiment")
        and not e.is_belief
        # Singletons describe slot-values (one current employer, one
        # residence, one age) — not behavioural tendencies. They get
        # closed by the singleton-slot closer, not synthesized as
        # traits. Identity / value-equation predicates are likewise
        # excluded by name as a defence-in-depth.
        and not e.singleton
        and (e.canonical_name or e.name or "").lower() not in _NON_TRAIT_PREDICATES
    ]
    if not incident:
        return []

    # Bucket by (predicate, object) and by predicate alone.
    pair_buckets: dict[tuple[str, str], list[EntityEdge]] = defaultdict(list)
    pred_buckets: dict[str, list[EntityEdge]] = defaultdict(list)
    for e in incident:
        pred = (e.canonical_name or e.name or "").strip().lower()
        if not pred:
            continue
        pair_buckets[(pred, e.target_node_uuid)].append(e)
        pred_buckets[pred].append(e)

    cand: list[dict[str, Any]] = []
    for (pred, obj), bucket in pair_buckets.items():
        reinforce = sum(max(1, e.reinforcement_count) for e in bucket)
        if reinforce < _MIN_REINFORCEMENT and len(bucket) < _MIN_REINFORCEMENT:
            continue
        cand.append(
            {
                "kind": "pair",
                "predicate": pred,
                "object_uuid": obj,
                "reinforcement": reinforce,
                "supporting_edges": bucket,
            }
        )
    for pred, bucket in pred_buckets.items():
        if len(bucket) < _MIN_REINFORCEMENT + 1:
            continue
        cand.append(
            {
                "kind": "predicate",
                "predicate": pred,
                "object_uuid": None,
                "reinforcement": sum(max(1, e.reinforcement_count) for e in bucket),
                "supporting_edges": bucket,
            }
        )

    cand.sort(key=lambda c: c["reinforcement"], reverse=True)
    return cand[:_MAX_CANDIDATES_PER_SUBJECT]


def _render_candidate(c: dict[str, Any]) -> str:
    examples = []
    for e in c["supporting_edges"][:_MAX_SUPPORTING_FACTS_SHOWN]:
        if e.fact:
            examples.append(f"  - {e.fact}")
        else:
            examples.append(f"  - {e.name}")
    head = (
        f"({c['kind']}) predicate={c['predicate']!r} "
        f"reinforcement={c['reinforcement']}"
    )
    return head + "\n" + "\n".join(examples)


def _heuristic_trait_name(c: dict[str, Any]) -> str:
    """Fallback when no LLM is available: snake_case the predicate +
    optionally the object's name (if we can pull it from a supporting
    edge's ``fact``). Never produces empty."""

    pred = c["predicate"] or "tendency"
    if c["kind"] == "predicate":
        return snake_case(f"{pred}_pattern")
    return snake_case(f"{pred}")


async def _load_subject_edges(
    driver: Any, group_id: str, subject_uuid: str | None = None
) -> list[EntityEdge]:
    if subject_uuid:
        rows = _unwrap(
            await driver.query(
                """
                SELECT *,
                    record::id(in)  AS source_node_uuid,
                    record::id(out) AS target_node_uuid
                FROM relates_to
                WHERE group_id = $g AND record::id(in) = $s;
                """,
                {"g": group_id, "s": subject_uuid},
            )
        )
    else:
        rows = _unwrap(
            await driver.query(
                """
                SELECT *,
                    record::id(in)  AS source_node_uuid,
                    record::id(out) AS target_node_uuid
                FROM relates_to WHERE group_id = $g;
                """,
                {"g": group_id},
            )
        )
    return [parse_edge(r) for r in rows]


async def synthesize_traits(
    driver: Any,
    llm: Any,
    embedder: Any,
    *,
    group_id: str,
    episode_uuids: list[str],
) -> int:
    """Run the trait synthesis pass for ``group_id``. Returns the number
    of trait edges written or refreshed."""

    if not episode_uuids:
        return 0
    all_edges = await _load_subject_edges(driver, group_id)
    if not all_edges:
        return 0
    subjects = _select_candidate_subjects(all_edges, set(episode_uuids))
    if not subjects:
        return 0

    written = 0
    now = datetime.now(timezone.utc)
    for subj_uuid in subjects:
        candidates = _candidates_for_subject(subj_uuid, all_edges)
        if not candidates:
            continue
        accepted = await _ratify_candidates(llm, candidates)
        if not accepted:
            continue
        for trait in accepted:
            written += await _persist_trait(
                driver,
                embedder,
                group_id=group_id,
                subject_uuid=subj_uuid,
                trait=trait,
                candidates=candidates,
                now=now,
            )
    logger.debug("synthesize_traits: group=%s written=%d", group_id, written)
    return written


async def _ratify_candidates(
    llm: Any, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Hand candidates to the LLM for naming/scoring; fall back to
    heuristics when the adapter has no ``synthesize`` method."""

    rendered = "CANDIDATE TRAITS:\n" + "\n\n".join(
        f"[{i}] {_render_candidate(c)}" for i, c in enumerate(candidates)
    )
    raw = None
    try:
        raw = await llm.synthesize(TRAIT_RATIFY_SYSTEM, rendered)
    except Exception:
        logger.exception("trait synthesis LLM call failed; falling back to heuristics")
        raw = None

    parsed = parse_json_loose(raw)
    if isinstance(parsed, list) and parsed:
        out: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = snake_case(item.get("name") or "")
            if not name:
                continue
            try:
                conf = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            indices = item.get("supporting_indices") or []
            indices = [int(i) for i in indices if isinstance(i, (int, str)) and str(i).isdigit()]
            supporting_edges = []
            for i in indices:
                if 0 <= i < len(candidates):
                    supporting_edges.extend(candidates[i]["supporting_edges"])
            if not supporting_edges and indices == [] and candidates:
                # accept LLM-named trait without binding indices; attach to
                # the strongest candidate's supporters.
                supporting_edges = candidates[0]["supporting_edges"]
            out.append(
                {
                    "name": name,
                    "description": str(item.get("description") or "").strip(),
                    "confidence": max(0.0, min(1.0, conf)),
                    "supporting_edges": supporting_edges,
                }
            )
        if out:
            return out

    # Heuristic fallback: accept the top 3 candidates by reinforcement.
    return [
        {
            "name": _heuristic_trait_name(c),
            "description": "",
            "confidence": min(0.95, 0.4 + 0.05 * c["reinforcement"]),
            "supporting_edges": c["supporting_edges"],
        }
        for c in candidates[:3]
    ]


async def _persist_trait(
    driver: Any,
    embedder: Any,
    *,
    group_id: str,
    subject_uuid: str,
    trait: dict[str, Any],
    candidates: list[dict[str, Any]],
    now: datetime,
) -> int:
    """Upsert the trait entity + ``has_trait`` edge. Returns 1 on new
    write, 1 on refresh (always 1 for accounting simplicity)."""

    trait_name = trait["name"]
    description = trait["description"]
    fact_text = description or f"has trait: {trait_name}"

    trait_uuid = await upsert_synthetic_entity(
        driver,
        group_id=group_id,
        name=trait_name,
        summary=description,
        label="trait",
        now=now,
    )
    await upsert_synthetic_edge(
        driver,
        embedder,
        group_id=group_id,
        subject_uuid=subject_uuid,
        object_uuid=trait_uuid,
        predicate="has_trait",
        fact_text=fact_text,
        memory_class="trait",
        confidence=float(trait["confidence"]),
        supporting_edge_uuids=[e.uuid for e in trait["supporting_edges"]],
        stability="reinforced",
        now=now,
    )
    await cache_on_subject(driver, subject_uuid=subject_uuid, field="traits", value=trait_uuid)
    return 1
