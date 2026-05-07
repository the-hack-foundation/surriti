"""Canonical entity resolution layer.

Replaces ``Surriti._upsert_entities``'s exact-name keying with a layered
pipeline that catches aliases, casing/typo variants, and context-dependent
references before they ever create a duplicate node.

Pipeline (per mention):

    A. **Alias hit** -- O(1) lookup against ``entity_alias`` by normalized
       form. Fastest and free.
    B. **Exact name** -- the legacy casefold key. Catches the trivial
       case without a vector roundtrip.
    C. **Semantic candidates** -- batched name embeddings, KNN against
       ``entity.name_embedding``. A single confident hit (cosine >=
       ``threshold``) resolves; multiple candidates fall through to D.
    D. **LLM tiebreak** -- one structured prompt per ambiguous mention
       carrying the episode context + top-3 candidates. Skipped when
       the resolver is constructed with ``use_llm=False`` or no LLM
       implements the hook.

Anything still unresolved is treated as ``"new"`` -- a fresh entity is
created downstream. ``EntityAlias`` rows are written every time a
mention resolves to a non-canonical surface form so future ingests
short-circuit on the alias hit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

from surriti.embedder import EmbedderClient, cosine_similarity
from surriti.llm import ExtractedEntity, LLMClient
from surriti.nodes import EntityNode, utc_now
from surriti.utils import parse_entity

logger = logging.getLogger(__name__)


_PUNCT_RE = re.compile(r"[^\w\s'-]")


def normalize_alias(name: str) -> str:
    """Casefold + collapse whitespace + strip non-word punctuation.

    Idempotent. Used for the alias-lookup hash and for comparing case /
    whitespace / minor-punctuation variants across mentions.
    """
    if not name:
        return ""
    cleaned = _PUNCT_RE.sub(" ", str(name)).casefold()
    return " ".join(cleaned.split())


@dataclass
class ResolvedEntity:
    """Result of resolving one extracted mention to a canonical entity.

    ``resolution`` reports which pipeline stage decided the match. When
    ``canonical_uuid`` is ``None`` the mention is "new" and the caller
    must create a fresh entity for it.
    """

    mention: ExtractedEntity
    canonical_uuid: str | None
    canonical_name: str
    resolution: Literal["alias_hit", "exact_name", "semantic_match", "llm_match", "new"]
    confidence: float
    # Existing entity row (if any) so callers can skip re-fetching.
    existing: EntityNode | None = None


# Type for an optional LLM tiebreak callable. Implementations receive the
# mention plus a list of (candidate_uuid, candidate_name, summary) and
# return the winning candidate_uuid or ``None`` for "no match".
LLMResolver = Callable[
    [ExtractedEntity, list[tuple[str, str, str]], str],
    Awaitable[str | None],
]


async def _default_llm_resolver(
    llm: LLMClient | None,
    mention: ExtractedEntity,
    candidates: list[tuple[str, str, str]],
    episode_context: str,
) -> str | None:
    """Bridge to ``LLMClient.resolve_entity_alias`` if implemented."""
    if llm is None or not candidates:
        return None
    hook = getattr(llm, "resolve_entity_alias", None)
    if hook is None:
        return None
    try:
        return await hook(
            mention=mention,
            candidates=candidates,
            episode_context=episode_context,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LLM alias resolver raised %s; treating as no-match", exc)
        return None


async def resolve_entity_mentions(
    *,
    driver,
    embedder: EmbedderClient,
    llm: LLMClient | None,
    mentions: list[ExtractedEntity],
    group_id: str,
    episode_context: str = "",
    threshold: float = 0.86,
    use_llm: bool = True,
    create_missing: bool = True,
    episode_uuid: str | None = None,
    llm_resolver: LLMResolver | None = None,
) -> list[ResolvedEntity]:
    """Resolve a batch of entity mentions to canonical entities.

    See module docstring for the four-stage pipeline.

    Parameters
    ----------
    create_missing:
        When ``False`` (e.g. used inside ``recall``), unresolved mentions
        return ``ResolvedEntity(canonical_uuid=None, resolution="new")``
        but no entity row is written by this function. The caller is
        expected to handle creation. When ``True``, the contract is the
        same -- this layer never writes the entity itself, only writes
        alias rows for resolved matches; entity creation lives in the
        ingest path.
    """

    if not mentions:
        return []

    from surriti.search import _unwrap

    # ------------------------------------------------------------------
    # Pre-compute normalized keys; preserve order; dedupe by normalized
    # form within this batch so we don't issue duplicate queries.
    # ------------------------------------------------------------------
    norm_keys = [normalize_alias(m.name) for m in mentions]
    unique_norm = list({k for k in norm_keys if k})

    # ------------------------------------------------------------------
    # Stage A -- alias hit (single bulk query).
    # ------------------------------------------------------------------
    alias_to_entity: dict[str, dict[str, Any]] = {}
    if unique_norm:
        rows = _unwrap(
            await driver.query(
                "SELECT * FROM entity_alias WHERE group_id = $g "
                "AND normalized_alias IN $aliases;",
                {"g": group_id, "aliases": unique_norm},
            )
        )
        for r in rows:
            alias_to_entity.setdefault(r.get("normalized_alias", ""), r)

    # Fetch entity rows referenced by alias hits in one query.
    entity_uuids_needed = {
        r.get("entity_uuid") for r in alias_to_entity.values() if r.get("entity_uuid")
    }
    # Plus any rows we'll need for the exact-name pass.
    entity_rows_by_uuid: dict[str, EntityNode] = {}
    entity_by_norm_name: dict[str, EntityNode] = {}
    if unique_norm or entity_uuids_needed:
        rows = _unwrap(
            await driver.query(
                "SELECT * FROM entity WHERE group_id = $g;",
                {"g": group_id},
            )
        )
        for row in rows:
            node = parse_entity(row)
            if node.uuid:
                entity_rows_by_uuid[node.uuid] = node
            key = normalize_alias(node.name)
            if key and key not in entity_by_norm_name:
                entity_by_norm_name[key] = node

    results: list[ResolvedEntity | None] = [None] * len(mentions)
    unresolved_idx: list[int] = []
    for i, (mention, key) in enumerate(zip(mentions, norm_keys)):
        if not key:
            results[i] = ResolvedEntity(
                mention=mention,
                canonical_uuid=None,
                canonical_name=mention.name,
                resolution="new",
                confidence=0.0,
            )
            continue
        # Stage A
        alias_row = alias_to_entity.get(key)
        if alias_row:
            uuid = alias_row.get("entity_uuid")
            existing = entity_rows_by_uuid.get(uuid)
            if existing is not None:
                results[i] = ResolvedEntity(
                    mention=mention,
                    canonical_uuid=existing.uuid,
                    canonical_name=existing.canonical_name or existing.name,
                    resolution="alias_hit",
                    confidence=float(alias_row.get("confidence") or 1.0),
                    existing=existing,
                )
                continue
        # Stage B -- exact name
        existing = entity_by_norm_name.get(key)
        if existing is not None:
            results[i] = ResolvedEntity(
                mention=mention,
                canonical_uuid=existing.uuid,
                canonical_name=existing.canonical_name or existing.name,
                resolution="exact_name",
                confidence=1.0,
                existing=existing,
            )
            continue
        unresolved_idx.append(i)

    # ------------------------------------------------------------------
    # Stage C -- semantic KNN. One batched embedding call, then in-memory
    # cosine against the same entity rows we already loaded. Keeping the
    # KNN client-side avoids hot-path SurrealDB HNSW queries when we
    # already have every candidate row.
    # ------------------------------------------------------------------
    if unresolved_idx and entity_rows_by_uuid:
        unresolved_mentions = [mentions[i] for i in unresolved_idx]
        try:
            vectors = await embedder.create_batch(
                [m.name for m in unresolved_mentions]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Embedder.create_batch failed (%s); skipping semantic", exc)
            vectors = []
        candidate_pool = [
            (n, n.name_embedding)
            for n in entity_rows_by_uuid.values()
            if n.name_embedding
        ]
        still_unresolved: list[int] = []
        ambiguous: list[tuple[int, list[tuple[EntityNode, float]]]] = []
        for local_i, vec in enumerate(vectors):
            i = unresolved_idx[local_i]
            scored: list[tuple[EntityNode, float]] = []
            for node, emb in candidate_pool:
                score = cosine_similarity(vec, emb)
                if score >= threshold:
                    scored.append((node, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            if not scored:
                still_unresolved.append(i)
                continue
            # Single confident hit, or a clear winner with a wide margin.
            top = scored[0]
            if len(scored) == 1 or (top[1] - scored[1][1]) >= 0.05:
                results[i] = ResolvedEntity(
                    mention=mentions[i],
                    canonical_uuid=top[0].uuid,
                    canonical_name=top[0].canonical_name or top[0].name,
                    resolution="semantic_match",
                    confidence=float(top[1]),
                    existing=top[0],
                )
            else:
                ambiguous.append((i, scored[:3]))
        unresolved_idx = still_unresolved + [i for i, _ in ambiguous]

        # --------------------------------------------------------------
        # Stage D -- LLM tiebreak (only on ambiguous matches).
        # --------------------------------------------------------------
        if use_llm and ambiguous:
            resolver = llm_resolver or (
                lambda mention, cands, ctx: _default_llm_resolver(
                    llm, mention, cands, ctx
                )
            )
            for i, scored in ambiguous:
                cand_payload = [
                    (n.uuid, n.canonical_name or n.name, n.profile_summary or n.summary)
                    for n, _ in scored
                ]
                winner = await resolver(mentions[i], cand_payload, episode_context)
                if winner:
                    chosen = next((n for n, _ in scored if n.uuid == winner), None)
                    if chosen is not None:
                        results[i] = ResolvedEntity(
                            mention=mentions[i],
                            canonical_uuid=chosen.uuid,
                            canonical_name=chosen.canonical_name or chosen.name,
                            resolution="llm_match",
                            confidence=0.9,
                            existing=chosen,
                        )
                        # An ambiguous match resolved by LLM means this
                        # surface form is now an alias of the chosen
                        # entity -- record it.
                        if i in unresolved_idx:
                            unresolved_idx.remove(i)

    # Anything still unresolved becomes "new".
    for i in unresolved_idx:
        if results[i] is None:
            results[i] = ResolvedEntity(
                mention=mentions[i],
                canonical_uuid=None,
                canonical_name=mentions[i].name,
                resolution="new",
                confidence=0.0,
            )

    final: list[ResolvedEntity] = [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Persist alias rows for non-canonical surface forms. Best-effort:
    # the unique index `entity_alias_unique` makes concurrent writes
    # safe to ignore.
    # ------------------------------------------------------------------
    if create_missing:
        await _record_aliases(
            driver=driver,
            resolved=final,
            group_id=group_id,
            episode_uuid=episode_uuid,
        )

    return final


async def _record_aliases(
    *,
    driver,
    resolved: list[ResolvedEntity],
    group_id: str,
    episode_uuid: str | None,
) -> None:
    for r in resolved:
        if r.canonical_uuid is None or r.existing is None:
            continue
        if r.resolution in ("alias_hit", "exact_name"):
            # Already represented (alias_hit was loaded from DB; exact_name
            # matches the canonical name and adds no signal).
            continue
        norm = normalize_alias(r.mention.name)
        if not norm:
            continue
        try:
            alias_uuid = str(uuid4())
            await driver.query(
                """
                CREATE type::record("entity_alias", $uuid) CONTENT {
                    uuid: $uuid,
                    group_id: $group_id,
                    alias: $alias,
                    normalized_alias: $normalized_alias,
                    entity_uuid: $entity_uuid,
                    confidence: $confidence,
                    source_episode_uuid: $source_episode_uuid,
                    created_at: $created_at
                };
                """,
                {
                    "uuid": alias_uuid,
                    "group_id": group_id,
                    "alias": r.mention.name,
                    "normalized_alias": norm,
                    "entity_uuid": r.canonical_uuid,
                    "confidence": float(r.confidence),
                    "source_episode_uuid": episode_uuid,
                    "created_at": utc_now(),
                },
            )
        except Exception as exc:
            # Unique-index race or lookup churn; safe to ignore.
            if "entity_alias_unique" not in str(exc):
                logger.debug("alias write failed for %r: %s", r.mention.name, exc)


__all__ = [
    "ResolvedEntity",
    "normalize_alias",
    "resolve_entity_mentions",
]
