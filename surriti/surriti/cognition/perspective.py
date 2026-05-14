"""Belief / perspective layer.

Marks edges as beliefs when the supporting episode contains epistemic
markers ("I think", "I believe", "feels like", "seems", "might",
"suspect"). Beliefs:

- get ``is_belief = True`` and ``belief_holder = <speaker_uuid>``;
- get ``memory_class = 'belief'`` (overriding ``objective``) so
  ``recall(memory_classes=['belief'])`` can fetch them;
- are skipped by ``temporal.resolve_contradictions`` when paired
  against an objective fact (the belief filter is enforced by a
  helper consumed in ``temporal.py``).

The marking is a *post-hoc* pass over edges incident to the recent
episode batch -- this keeps the existing extraction prompt unchanged
when belief extraction is opted out, but still upgrades belief-shaped
facts to first-class beliefs at cognition time.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from surriti.search import _unwrap

logger = logging.getLogger(__name__)


_BELIEF_RE = re.compile(
    r"\b(?:I think|I believe|I feel(?: like)?|feels like|seems(?: like)?|might be|"
    r"I suspect|in my opinion|I guess|probably)\b",
    re.IGNORECASE,
)


def looks_like_belief(text: str) -> bool:
    return bool(text) and bool(_BELIEF_RE.search(text))


async def tag_beliefs(
    driver: Any, *, group_id: str, episode_uuids: list[str]
) -> int:
    """For each episode in the batch flagged as belief-bearing, mark
    every edge that cites it as a belief held by the speaker. Returns
    count of edges promoted."""

    if not episode_uuids:
        return 0
    ep_rows = _unwrap(
        await driver.query(
            "SELECT uuid, content FROM episode WHERE group_id = $g AND uuid IN $u;",
            {"g": group_id, "u": list(episode_uuids)},
        )
    )
    belief_eps = [str(r.get("uuid")) for r in ep_rows if looks_like_belief(str(r.get("content") or ""))]
    if not belief_eps:
        return 0

    # Determine candidate speakers: entities mentioned by the belief
    # episodes. We tag any edge whose subject is one of these
    # candidates and which cites a belief episode -- the subject of
    # such an edge IS the speaker holding the belief. This is more
    # robust than picking a single "most-mentioned" speaker, which
    # ties non-deterministically when multiple entities share the
    # same mention count.
    mention_rows = _unwrap(
        await driver.query(
            """
            SELECT record::id(out) AS entity_uuid
            FROM mentions
            WHERE group_id = $g AND record::id(in) IN $eps;
            """,
            {"g": group_id, "eps": belief_eps},
        )
    )
    speaker_candidates = {
        str(row.get("entity_uuid"))
        for row in mention_rows
        if row.get("entity_uuid")
    }
    if not speaker_candidates:
        return 0

    # Promote edges that cite belief-bearing episodes and whose
    # subject is one of the mentioned entities.
    edge_rows = _unwrap(
        await driver.query(
            """
            SELECT uuid, attributes, record::id(in) AS speaker
            FROM relates_to
            WHERE group_id = $g
              AND record::id(in) IN $candidates
              AND episodes ANYINSIDE $eps
              AND is_belief = false
              AND status = 'active';
            """,
            {
                "g": group_id,
                "candidates": list(speaker_candidates),
                "eps": belief_eps,
            },
        )
    )
    promoted = 0
    for er in edge_rows:
        speaker_uuid = str(er.get("speaker"))
        await driver.query(
            """
            UPDATE relates_to SET
                is_belief = true,
                belief_holder = $h,
                attributes = object::extend(attributes, { memory_class: 'belief' })
            WHERE uuid = $u;
            """,
            {"u": er.get("uuid"), "h": speaker_uuid},
        )
        promoted += 1
    logger.debug("tag_beliefs: group=%s promoted=%d", group_id, promoted)
    return promoted
