"""Self-awareness cognition pass.

Reads self-episodes (self_observation, self_correction, self_success,
self_pattern) from the graph, extracts structured self-model data
(traits, beliefs, patterns), and writes them back as edges on the
assistant's self-entity.

The self-model is then available for:
- Injection into recall/prompts so the assistant is aware of its own patterns
- Feedback loop: the model influences future behavior
"""

from __future__ import annotations

import logging
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from surriti.cognition.config import CognitionConfig
from surriti.nodes import EpisodeType

logger = logging.getLogger("surriti.cognition.self_awareness")


async def _complete_json(llm: Any, *, system: str, prompt: str) -> str | None:
    """Use the public LLM hook; tolerate legacy adapters with ``generate``."""

    if hasattr(llm, "synthesize"):
        response = await llm.synthesize(system, prompt)
        if response:
            return response
    if hasattr(llm, "generate"):
        return await llm.generate(prompt, system_prompt=system, temperature=0.3)
    return None


async def run_self_awareness_pass(
    *,
    driver: Any,
    llm: Any,
    group_id: str,
    episode_uuids: list[str],
    config: CognitionConfig,
) -> dict[str, Any]:
    """Run self-awareness cognition pass for a group.

    Steps:
    1. Query all self-episodes for this group
    2. Batch-process them through LLM to extract structured self-model
    3. Write traits, beliefs, and patterns back to the graph
    4. Return metrics

    Always succeeds (logs + swallows individual failures).
    """
    metrics: dict[str, Any] = {
        "self_episodes_read": 0,
        "self_traits_extracted": 0,
        "self_beliefs_extracted": 0,
        "self_patterns_detected": 0,
    }

    try:
        # 1. Query self-episodes
        self_episodes = await _query_self_episodes(driver, group_id)
        metrics["self_episodes_read"] = len(self_episodes)

        if not self_episodes:
            logger.debug("no self-episodes for group %s", group_id)
            return metrics

        # 2. Group by type for targeted processing
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ep in self_episodes:
            src = ep.get("source", "")
            if isinstance(src, str) and src.startswith("self_"):
                by_type[src].append(ep)

        # 3. Process each type with targeted LLM prompts
        for ep_type, episodes in by_type.items():
            if ep_type == EpisodeType.self_observation.value:
                traits, beliefs = await _extract_self_traits(
                    driver, llm, group_id, episodes, config
                )
                metrics["self_traits_extracted"] = traits
                metrics["self_beliefs_extracted"] = beliefs

            elif ep_type == EpisodeType.self_pattern.value:
                patterns = await _extract_self_patterns(
                    driver, llm, group_id, episodes, config
                )
                metrics["self_patterns_detected"] = patterns

            elif ep_type in (
                EpisodeType.self_correction.value,
                EpisodeType.self_success.value,
            ):
                # Corrections and successes feed into traits
                traits, beliefs = await _extract_traits_from_events(
                    driver, llm, group_id, episodes, config
                )
                metrics["self_traits_extracted"] += traits
                metrics["self_beliefs_extracted"] += beliefs

        logger.info(
            "self-awareness pass complete for group %s: %s",
            group_id,
            metrics,
        )

    except Exception:
        logger.exception("self-awareness pass failed for group %s", group_id)

    return metrics


async def _query_self_episodes(
    driver: Any,
    group_id: str,
) -> list[dict[str, Any]]:
    """Query all self-referential episodes for a group."""
    from surriti.search import _unwrap

    rows = _unwrap(
        await driver.query(
            """
            SELECT name, content, source, source_description,
                   reference_time, created_at, group_id
            FROM episode
            WHERE group_id = $group_id
                AND source CONTAINS 'self_'
            ORDER BY created_at DESC
            LIMIT 100;
            """,
            {"group_id": group_id},
        )
    )
    return rows


async def _extract_self_traits(
    driver: Any,
    llm: Any,
    group_id: str,
    episodes: list[dict[str, Any]],
    config: CognitionConfig,
) -> tuple[int, int]:
    """Extract self-traits from self_observation episodes.

    Returns (traits_extracted, beliefs_extracted).
    """
    if not episodes:
        return 0, 0

    # Build a prompt with all self-observations
    ep_texts = "\n\n".join(
        f"[{ep.get('source_description', '')}] {ep.get('content', '')}"
        for ep in episodes
    )

    prompt = textwrap.dedent(f"""\
        You are an AI assistant analyzing your own self-observations.
        Extract structured self-model data from these observations.

        Self-observations:
        {ep_texts}

        Return JSON with this structure:
        {{
          "traits": [
            {{"trait": "concise", "evidence": "multiple observations mention brevity", "confidence": 0.8}}
          ],
          "beliefs": [
            {{"belief": "I tend to be verbose in technical contexts", "confidence": 0.7}}
          ]
        }}

        Only include traits/beliefs directly supported by the observations.
        Be honest and specific.
    """)

    try:
        response = await _complete_json(
            llm,
            prompt=prompt,
            system=(
                "You are analyzing self-observations of an AI assistant. "
                "Return structured JSON. Be concise and accurate."
            ),
        )
        if not response:
            return 0, 0
        # Parse JSON response and write to graph
        import json
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("self-awareness: failed to parse LLM response as JSON")
            return 0, 0

        traits_count = 0
        beliefs_count = 0

        for trait in data.get("traits", []):
            await _write_self_trait(driver, group_id, trait)
            traits_count += 1

        for belief in data.get("beliefs", []):
            await _write_self_belief(driver, group_id, belief)
            beliefs_count += 1

        return traits_count, beliefs_count

    except Exception:
        logger.exception("self-awareness: trait extraction failed")
        return 0, 0


async def _extract_self_patterns(
    driver: Any,
    llm: Any,
    group_id: str,
    episodes: list[dict[str, Any]],
    config: CognitionConfig,
) -> int:
    """Extract behavioral patterns from self_pattern episodes."""
    if not episodes:
        return 0

    ep_texts = "\n\n".join(
        f"[{ep.get('source_description', '')}] {ep.get('content', '')}"
        for ep in episodes
    )

    prompt = textwrap.dedent(f"""\
        You are an AI assistant analyzing your own behavioral patterns.
        Extract structured pattern data.

        Self-pattern observations:
        {ep_texts}

        Return JSON:
        {{
          "patterns": [
            {{"pattern": "prefers structured responses", "frequency": "high", "context": "technical queries"}}
          ]
        }}
    """)

    try:
        response = await _complete_json(
            llm,
            system="Return concise JSON describing assistant behavioral patterns.",
            prompt=prompt,
        )
        if not response:
            return 0
        import json
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return 0

        patterns_count = 0
        for pattern in data.get("patterns", []):
            await _write_self_pattern(driver, group_id, pattern)
            patterns_count += 1

        return patterns_count

    except Exception:
        logger.exception("self-awareness: pattern extraction failed")
        return 0


async def _extract_traits_from_events(
    driver: Any,
    llm: Any,
    group_id: str,
    episodes: list[dict[str, Any]],
    config: CognitionConfig,
) -> tuple[int, int]:
    """Extract traits from self_correction and self_success episodes."""
    if not episodes:
        return 0, 0

    ep_texts = "\n\n".join(
        f"[{ep.get('source_description', '')}] {ep.get('content', '')}"
        for ep in episodes
    )

    prompt = textwrap.dedent(f"""\
        You are an AI assistant analyzing your own corrections and successes.
        Extract traits and beliefs from these events.

        Self-corrections/successes:
        {ep_texts}

        Return JSON:
        {{
          "traits": [
            {{"trait": "adaptable", "evidence": "acknowledges and corrects mistakes", "confidence": 0.9}}
          ],
          "beliefs": [
            {{"belief": "I value accuracy over speed", "confidence": 0.7}}
          ]
        }}
    """)

    try:
        response = await _complete_json(
            llm,
            system="Return concise JSON describing assistant traits and beliefs.",
            prompt=prompt,
        )
        if not response:
            return 0, 0
        import json
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return 0, 0

        traits_count = 0
        beliefs_count = 0

        for trait in data.get("traits", []):
            await _write_self_trait(driver, group_id, trait)
            traits_count += 1

        for belief in data.get("beliefs", []):
            await _write_self_belief(driver, group_id, belief)
            beliefs_count += 1

        return traits_count, beliefs_count

    except Exception:
        logger.exception("self-awareness: event trait extraction failed")
        return 0, 0


async def _write_self_trait(
    driver: Any,
    group_id: str,
    trait_data: dict[str, Any],
) -> None:
    """Write a self-trait edge to the graph."""
    trait_name = trait_data.get("trait", "")
    if not trait_name:
        return

    self_entity = await _get_self_entity(driver, group_id)
    if not self_entity:
        return

    trait_uuid = f"trait_{trait_name.replace(' ', '_')}_{group_id}"
    confidence = trait_data.get("confidence", 0.5)
    evidence = trait_data.get("evidence", "")

    await driver.query(
        """
        CREATE type::record("entity", $uuid) CONTENT {
            uuid: $uuid,
            group_id: $group_id,
            name: $name,
            summary: $summary,
            labels: $labels,
            created_at: $created_at
        };
        """,
        {
            "uuid": trait_uuid,
            "group_id": group_id,
            "name": trait_name,
            "summary": evidence or f"Self-trait: {trait_name}",
            "labels": ["SelfTrait", "Trait"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    await driver.query(
        """
        RELATE (type::record("entity", $src))->has_trait->(type::record("entity", $dst))
        CONTENT {
            uuid: $edge_uuid,
            group_id: $group_id,
            created_at: $created_at,
            fact: $fact,
            confidence: $confidence,
            is_belief: false
        };
        """,
        {
            "src": self_entity["uuid"],
            "dst": trait_uuid,
            "edge_uuid": f"edge_trait_{trait_uuid}_{group_id}",
            "group_id": group_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fact": f"has_trait: {trait_name}",
            "confidence": confidence,
        },
    )


async def _write_self_belief(
    driver: Any,
    group_id: str,
    belief_data: dict[str, Any],
) -> None:
    """Write a self-belief edge to the graph."""
    belief_text = belief_data.get("belief", "")
    if not belief_text:
        return

    self_entity = await _get_self_entity(driver, group_id)
    if not self_entity:
        return

    belief_uuid = f"belief_{len(belief_text)}_{group_id}"
    confidence = belief_data.get("confidence", 0.5)

    await driver.query(
        """
        CREATE type::record("entity", $uuid) CONTENT {
            uuid: $uuid,
            group_id: $group_id,
            name: $name,
            summary: $summary,
            labels: $labels,
            created_at: $created_at
        };
        """,
        {
            "uuid": belief_uuid,
            "group_id": group_id,
            "name": "self_belief",
            "summary": belief_text,
            "labels": ["SelfBelief"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    await driver.query(
        """
        RELATE (type::record("entity", $src))->has_belief->(type::record("entity", $dst))
        CONTENT {
            uuid: $edge_uuid,
            group_id: $group_id,
            created_at: $created_at,
            fact: $fact,
            confidence: $confidence,
            is_belief: true
        };
        """,
        {
            "src": self_entity["uuid"],
            "dst": belief_uuid,
            "edge_uuid": f"edge_belief_{belief_uuid}_{group_id}",
            "group_id": group_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fact": belief_text,
            "confidence": confidence,
        },
    )


async def _write_self_pattern(
    driver: Any,
    group_id: str,
    pattern_data: dict[str, Any],
) -> None:
    """Write a self-pattern edge to the graph."""
    pattern_name = pattern_data.get("pattern", "")
    if not pattern_name:
        return

    self_entity = await _get_self_entity(driver, group_id)
    if not self_entity:
        return

    pattern_uuid = f"pattern_{pattern_name.replace(' ', '_')}_{group_id}"

    await driver.query(
        """
        CREATE type::record("entity", $uuid) CONTENT {
            uuid: $uuid,
            group_id: $group_id,
            name: $name,
            summary: $summary,
            labels: $labels,
            created_at: $created_at
        };
        """,
        {
            "uuid": pattern_uuid,
            "group_id": group_id,
            "name": pattern_name,
            "summary": str(pattern_data),
            "labels": ["SelfPattern", "Pattern"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    await driver.query(
        """
        RELATE (type::record("entity", $src))->has_pattern->(type::record("entity", $dst))
        CONTENT {
            uuid: $edge_uuid,
            group_id: $group_id,
            created_at: $created_at,
            fact: $fact,
            confidence: $confidence
        };
        """,
        {
            "src": self_entity["uuid"],
            "dst": pattern_uuid,
            "edge_uuid": f"edge_pattern_{pattern_uuid}_{group_id}",
            "group_id": group_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fact": f"has_pattern: {pattern_name}",
            "confidence": 0.7,
        },
    )


async def _get_self_entity(
    driver: Any,
    group_id: str,
) -> dict[str, Any] | None:
    """Get the self-entity for a group, or None if not found."""
    from surriti.search import _unwrap

    self_entity_name = f"assistant_{group_id}" if group_id else "assistant"

    rows = _unwrap(
        await driver.query(
            """
            SELECT * FROM entity
            WHERE group_id = $group_id
                AND name = $name
            LIMIT 1;
            """,
            {"group_id": group_id, "name": self_entity_name},
        )
    )

    return rows[0] if rows else None
