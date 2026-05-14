"""Domain-aware labelling for entity communities.

The existing ``Surriti.build_communities()`` builds connected
components over active ``relates_to`` edges and gives each cluster a
``name`` taken from its most-connected entity. The cognition layer
extends that primitive with a *domain label*: a single short
snake_case term (``fortnite``, ``distance_running``) derived from the
member entities' names + edge facts. We persist the label to
``community.domain`` and denormalise onto each member's ``entity.domain``
so ``recall()`` can filter cheaply by domain without an extra hop.

Runs every ``CognitionConfig.domain_labeling_every_n_passes`` passes
to amortise the LLM cost (one ``synthesize`` call per cluster).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from surriti.cognition._jsonio import snake_case
from surriti.cognition.prompts import DOMAIN_LABEL_SYSTEM
from surriti.search import _unwrap

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}")
_STOPWORDS = {
    "have", "with", "this", "that", "from", "user", "also", "into",
    "their", "there", "about", "they", "your", "more", "much", "some",
    "what", "when", "would", "could", "should", "really", "actually",
}


def _top_terms(texts: list[str], k: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for t in texts:
        for tok in _TOKEN_RE.findall(t.lower()):
            if tok in _STOPWORDS:
                continue
            counter[tok] += 1
    return [w for w, _ in counter.most_common(k)]


async def label_community_domains(
    driver: Any, llm: Any, *, group_id: str
) -> int:
    """Assign a ``domain`` label to each entity cluster in ``group_id``.
    Returns count of communities labelled."""

    communities = _unwrap(
        await driver.query(
            "SELECT uuid, name FROM community WHERE group_id = $g AND kind = 'cluster';",
            {"g": group_id},
        )
    )
    labelled = 0
    for c in communities:
        c_uuid = str(c.get("uuid"))
        members = _unwrap(
            await driver.query(
                """
                SELECT record::id(out) AS entity_uuid
                FROM has_member WHERE group_id = $g AND record::id(in) = $c;
                """,
                {"g": group_id, "c": c_uuid},
            )
        )
        member_uuids = [str(m.get("entity_uuid")) for m in members if m.get("entity_uuid")]
        if not member_uuids:
            continue
        ent_rows = _unwrap(
            await driver.query(
                "SELECT name, summary FROM entity WHERE uuid IN $u;",
                {"u": member_uuids},
            )
        )
        edge_rows = _unwrap(
            await driver.query(
                """
                SELECT fact FROM relates_to
                WHERE group_id = $g
                  AND (record::id(in) IN $u OR record::id(out) IN $u)
                  AND status = 'active'
                LIMIT 24;
                """,
                {"g": group_id, "u": member_uuids},
            )
        )
        names = [str(r.get("name") or "") for r in ent_rows]
        facts = [str(r.get("fact") or "") for r in edge_rows]
        terms = _top_terms(names + facts, k=8)
        if not terms:
            continue
        # LLM ratify; fall back to top term.
        label: str | None = None
        try:
            user = (
                "CLUSTER ENTITIES: " + ", ".join(names[:12])
                + "\nCLUSTER FACTS:\n" + "\n".join(f"- {f}" for f in facts[:8])
                + "\nTOP TOKENS: " + ", ".join(terms)
            )
            raw = await llm.synthesize(DOMAIN_LABEL_SYSTEM, user)
            if raw:
                label = snake_case(raw.splitlines()[0].strip().strip("\"'`"))
        except Exception:
            logger.exception("domain labelling LLM call failed; using top-term fallback")
        if not label:
            label = snake_case(terms[0])
        if not label:
            continue
        await driver.query(
            "UPDATE community SET domain = $d WHERE uuid = $u;",
            {"u": c_uuid, "d": label},
        )
        await driver.query(
            "UPDATE entity SET domain = $d WHERE uuid IN $u;",
            {"u": member_uuids, "d": label},
        )
        labelled += 1
    logger.debug("label_community_domains: group=%s labelled=%d", group_id, labelled)
    return labelled
