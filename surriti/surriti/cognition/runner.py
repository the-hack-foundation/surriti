"""Single cognition pass orchestrator.

Sequencing rationale (each step depends only on prior steps):

1. **affect**       -- tag episodes; cheap, no LLM.
2. **perspective**  -- promote belief edges; cheap, no LLM.
3. **reinforcement**-- update ``reinforcement_count`` / ``stability``.
4. **associative**  -- recompute ``decay_score`` / ``weight``.
5. **traits**       -- LLM-ratified hybrid synthesis.
6. **self-awareness** -- extract structured self-model from self-episodes.
7. **goals**        -- LLM-ratified hybrid synthesis.
8. **procedural**   -- pattern detection + (occasional) edge promotion.
9. **consolidation**-- mint consolidated abstractions when threshold met.
10. **clustering**   -- domain labelling, every Nth pass only.
11. **prediction**  -- refresh per-group prediction sidecar.

Every step is wrapped in a ``try/except`` -- any failure is logged
and skipped; cognition must never break ingest. Returns a small
metrics dict the scheduler logs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from surriti.cognition import affect as _affect
from surriti.cognition import associative as _assoc
from surriti.cognition import clustering as _clust
from surriti.cognition import consolidation as _consol
from surriti.cognition import goals as _goals
from surriti.cognition import perspective as _persp
from surriti.cognition import prediction as _pred
from surriti.cognition import procedural as _proc
from surriti.cognition import reinforcement as _reinf
from surriti.cognition import self_awareness as _selfaware
from surriti.cognition import traits as _traits
from surriti.cognition.config import CognitionConfig

logger = logging.getLogger("surriti.cognition")

_COGNITION_VERSION = "2026-06-linear-vitality"


async def run_cognition_pass(
    *,
    driver: Any,
    llm: Any,
    embedder: Any,
    group_id: str,
    episode_uuids: list[str],
    config: CognitionConfig,
    pass_count: int = 0,
) -> dict[str, Any]:
    """Run one cognition pass for a group. Always succeeds (logs +
    swallows individual step failures)."""

    started = time.monotonic()
    metrics: dict[str, Any] = {"group_id": group_id, "episodes": len(episode_uuids)}

    async def _safe(step: str, coro):  # type: ignore[no-untyped-def]
        try:
            return await coro
        except Exception:
            logger.exception("cognition step %r failed for group=%s", step, group_id)
            return None

    if config.affect_extraction:
        metrics["affect_tagged"] = await _safe(
            "affect", _affect.tag_episode_affect(driver, group_id=group_id, episode_uuids=episode_uuids)
        )
    if config.belief_extraction:
        metrics["beliefs_promoted"] = await _safe(
            "perspective", _persp.tag_beliefs(driver, group_id=group_id, episode_uuids=episode_uuids)
        )

    metrics["edges_reinforced"] = await _safe(
        "reinforcement",
        _reinf.reinforce_recent_edges(driver, group_id=group_id, episode_uuids=episode_uuids),
    )
    metrics["weights_refreshed"] = await _safe(
        "associative",
        _assoc.refresh_weights(
            driver,
            group_id=group_id,
            half_life_overrides=config.decay_half_life_days,
        ),
    )

    if config.trait_synthesis:
        metrics["traits_synthesized"] = await _safe(
            "traits",
            _traits.synthesize_traits(
                driver, llm, embedder, group_id=group_id, episode_uuids=episode_uuids
            ),
        )
    if config.self_awareness:
        metrics["self_model_updated"] = await _safe(
            "self_awareness",
            _selfaware.run_self_awareness_pass(
                driver=driver,
                llm=llm,
                group_id=group_id,
                episode_uuids=episode_uuids,
                config=config,
            ),
        )
    if config.goal_synthesis:
        metrics["goals_synthesized"] = await _safe(
            "goals",
            _goals.synthesize_goals(
                driver, llm, embedder, group_id=group_id, episode_uuids=episode_uuids
            ),
        )
    if config.procedural_synthesis:
        metrics["episodes_classified"] = await _safe(
            "procedural",
            _proc.detect_interaction_patterns(
                driver, embedder, group_id=group_id, episode_uuids=episode_uuids
            ),
        )
    if config.consolidation:
        metrics["edges_consolidated"] = await _safe(
            "consolidation",
            _consol.consolidate(
                driver,
                embedder,
                group_id=group_id,
                threshold=config.consolidation_threshold,
                min_span_days=config.consolidation_min_span_days,
            ),
        )
    if config.consolidation and getattr(config, "stagnant_consolidation", False):
        metrics["low_vitality_consolidated"] = await _safe(
            "low_vitality_consolidation",
            _consol.consolidate_stagnant_edges(
                driver,
                embedder,
                group_id=group_id,
                min_edges_per_summary=getattr(config, "stagnant_min_edges_per_summary", 5),
                max_edges_per_pass=getattr(config, "stagnant_max_edges_per_pass", 120),
            ),
        )
    if (
        config.domain_labeling_every_n_passes > 0
        and pass_count % max(1, config.domain_labeling_every_n_passes) == 0
    ):
        metrics["domains_labelled"] = await _safe(
            "clustering",
            _clust.label_community_domains(driver, llm, group_id=group_id),
        )
    if config.prediction:
        bundle = await _safe(
            "prediction", _pred.synthesize_prediction(driver, llm, group_id=group_id)
        )
        metrics["prediction"] = "ok" if bundle else "skipped"

    if episode_uuids:
        await _safe(
            "mark_processed",
            driver.query(
                """
                UPDATE episode SET
                    cognition_processed_at = $processed_at,
                    cognition_version = $version
                WHERE group_id = $group_id
                  AND uuid IN $episode_uuids;
                """,
                {
                    "group_id": group_id,
                    "episode_uuids": list(episode_uuids),
                    "processed_at": datetime.now(timezone.utc),
                    "version": _COGNITION_VERSION,
                },
            ),
        )

    metrics["duration_ms"] = int((time.monotonic() - started) * 1000)
    logger.info("cognition pass complete: %s", metrics)
    return metrics
