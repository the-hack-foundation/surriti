"""Per-entity profile (dossier) refresh.

A *profile* is a short natural-language summary plus an embedding that
collapses what the graph knows about one entity. Profiles are what
:meth:`Surriti.recall` and the visualizer dossier UI render -- they
are the cheap, prebuilt "page about X" that backs every read.

Refresh strategy is **touched-only**: after each ``add_episode`` we
rebuild profiles for *only* the entities that participated in that
episode. There is no global crawl, no scheduler, and no background
worker -- just an ``asyncio.create_task`` (when ``profile_refresh =
"async"``) per ingest. ``backfill_profiles`` exists for the one-time
migration case.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from surriti.embedder import EmbedderClient
from surriti.llm import LLMClient
from surriti.nodes import EntityNode, utc_now
from surriti.utils import parse_edge, parse_entity

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _compose_summary(
    entity_name: str,
    facts: list[str],
    *,
    max_chars: int = 800,
) -> str:
    """Stitch the top facts into a compact summary string.

    Used as the default summarizer when the LLM client doesn't
    implement ``summarize_entity_profile``. Deterministic, free, and
    always good enough as a fallback.
    """
    if not facts:
        return ""
    bullets = []
    used = 0
    head = f"{entity_name}: "
    for f in facts:
        line = f.strip().rstrip(".")
        if not line:
            continue
        candidate = ("; " if bullets else "") + line
        if used + len(candidate) > max_chars:
            break
        bullets.append(line)
        used += len(candidate)
    return head + "; ".join(bullets) + "."


async def _summarize_with_llm(
    llm: LLMClient | None,
    entity_name: str,
    facts: list[str],
    max_chars: int,
) -> str:
    """Try the optional LLM summarizer; fall back to deterministic compose."""
    if llm is not None:
        hook = getattr(llm, "summarize_entity_profile", None)
        if hook is not None:
            try:
                summary = await hook(
                    entity_name=entity_name,
                    facts=facts,
                    max_chars=max_chars,
                )
                if summary:
                    return summary
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "summarize_entity_profile raised %s; falling back", exc
                )
    return _compose_summary(entity_name, facts, max_chars=max_chars)


async def refresh_entity_profiles(
    *,
    driver,
    embedder: EmbedderClient,
    llm: LLMClient | None,
    group_id: str,
    entity_uuids: list[str],
    max_facts: int = 30,
    max_chars: int = 800,
) -> int:
    """Rebuild profiles for the given entities. Returns count refreshed.

    Idempotent. Each entity:

    * latest ``max_facts`` valid facts are pulled (one query per entity
      -- entities-per-episode is small in practice);
    * a summary is composed (LLM if available, else deterministic);
    * the summary is embedded;
    * the entity row's ``profile_summary``, ``profile_embedding``,
      ``mention_count``, ``last_seen_at``, ``salience`` are updated in
      one statement.

    Failure on a single entity is logged and skipped -- one bad row
    must not poison a batch.
    """

    if not entity_uuids:
        return 0

    from surriti.search import _unwrap

    refreshed = 0
    for uuid in entity_uuids:
        try:
            erows = _unwrap(
                await driver.query(
                    "SELECT * FROM entity WHERE group_id = $g AND uuid = $u "
                    "LIMIT 1;",
                    {"g": group_id, "u": uuid},
                )
            )
            if not erows:
                continue
            entity = parse_entity(erows[0])

            # Pull facts where this entity is on either side.
            frows = _unwrap(
                await driver.query(
                    "SELECT * FROM relates_to WHERE group_id = $g "
                    "AND (in = $rec OR out = $rec) "
                    "AND invalid_at IS NONE "
                    "ORDER BY valid_at DESC LIMIT $lim;",
                    {
                        "g": group_id,
                        "rec": f"entity:{uuid}",
                        "lim": int(max_facts),
                    },
                )
            )
            facts: list[str] = []
            for fr in frows:
                edge = parse_edge(fr)
                if edge.fact:
                    facts.append(edge.fact)

            display_name = entity.canonical_name or entity.name

            # Append cached trait + goal labels (synthesised by the
            # cognitive layer) so the profile_summary is self-describing
            # and can be rendered straight into a system prompt without
            # an extra trait/goal lookup. We resolve the cached UUIDs
            # to their entity ``name`` in one bulk SELECT.
            trait_goal_uuids = list(
                dict.fromkeys(
                    list(getattr(entity, "traits", []) or [])
                    + list(getattr(entity, "goals_active", []) or [])
                )
            )
            trait_names: list[str] = []
            goal_names: list[str] = []
            if trait_goal_uuids:
                try:
                    sidecar_rows = _unwrap(
                        await driver.query(
                            "SELECT uuid, name, labels FROM entity "
                            "WHERE group_id = $g AND uuid IN $u;",
                            {"g": group_id, "u": trait_goal_uuids},
                        )
                    )
                    for sr in sidecar_rows:
                        labels = sr.get("labels") or []
                        nm = str(sr.get("name") or "").strip()
                        if not nm:
                            continue
                        if "trait" in labels:
                            trait_names.append(nm)
                        elif "goal" in labels:
                            goal_names.append(nm)
                except Exception:
                    logger.debug("profile sidecar fetch failed for %s", uuid)

            summary = await _summarize_with_llm(
                llm, display_name, facts, max_chars
            )
            tail_parts: list[str] = []
            if trait_names:
                tail_parts.append("Traits: " + ", ".join(trait_names[:6]))
            if goal_names:
                tail_parts.append("Active goals: " + ", ".join(goal_names[:6]))
            if tail_parts:
                tail = " " + " ".join(p + "." for p in tail_parts)
                if len(summary) + len(tail) <= max_chars + 240:
                    summary = (summary or display_name + ".") + tail
            embedding: list[float] = []
            if summary:
                try:
                    embedding = await embedder.create(summary)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("profile embed failed for %s: %s", uuid, exc)

            mention_count = int(entity.mention_count or 0) + 1
            salience = float(entity.salience or 0.0) + 1.0

            # Only include profile_embedding in the UPDATE when we have a
            # valid non-empty vector.  An empty embedding (e.g. when the
            # entity has no facts yet) would violate SurrealDB's dimension
            # constraint and cause a spurious error.
            if embedding:
                await driver.query(
                    """
                    UPDATE type::record("entity", $uuid) SET
                        profile_summary = $profile_summary,
                        profile_embedding = $profile_embedding,
                        mention_count = $mention_count,
                        salience = $salience,
                        last_seen_at = $last_seen_at;
                    """,
                    {
                        "uuid": uuid,
                        "profile_summary": summary,
                        "profile_embedding": embedding,
                        "mention_count": mention_count,
                        "salience": salience,
                        "last_seen_at": utc_now(),
                    },
                )
            else:
                await driver.query(
                    """
                    UPDATE type::record("entity", $uuid) SET
                        profile_summary = $profile_summary,
                        mention_count = $mention_count,
                        salience = $salience,
                        last_seen_at = $last_seen_at;
                    """,
                    {
                        "uuid": uuid,
                        "profile_summary": summary,
                        "mention_count": mention_count,
                        "salience": salience,
                        "last_seen_at": utc_now(),
                    },
                )
            refreshed += 1
        except Exception as exc:
            logger.warning("profile refresh failed for entity %s: %s", uuid, exc)
    return refreshed


async def backfill_profiles(
    *,
    driver,
    embedder: EmbedderClient,
    llm: LLMClient | None,
    group_id: str,
    batch_size: int = 50,
) -> int:
    """One-time backfill: refresh every entity in ``group_id`` once.

    Streams in batches so a multi-thousand-entity tenant doesn't load
    the world. Returns the total number of entities refreshed.
    """

    from surriti.search import _unwrap

    rows = _unwrap(
        await driver.query(
            "SELECT uuid FROM entity WHERE group_id = $g;",
            {"g": group_id},
        )
    )
    uuids = [r.get("uuid") for r in rows if r.get("uuid")]
    total = 0
    for i in range(0, len(uuids), batch_size):
        chunk = uuids[i : i + batch_size]
        total += await refresh_entity_profiles(
            driver=driver,
            embedder=embedder,
            llm=llm,
            group_id=group_id,
            entity_uuids=chunk,
        )
    return total


__all__ = ["refresh_entity_profiles", "backfill_profiles"]
