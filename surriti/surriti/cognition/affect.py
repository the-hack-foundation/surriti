"""Episode affect tagging.

A lightweight lexicon-based scorer that assigns each recent episode an
``affect`` dict ``{emotion, intensity, polarity}``. We deliberately
*do not* spend an extra LLM call here: real adapters can later override
``LLMClient.synthesize`` with a per-episode prompt if higher fidelity is
needed; the heuristic keeps the cognition pass cheap and offline-safe.
After tagging, edges supported by an affect-tagged episode inherit
``valence`` and ``intensity`` (max across supporting episodes).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from surriti.search import _unwrap

logger = logging.getLogger(__name__)


# Tiny, intentionally conservative lexicon. Each entry: keyword -> (emotion, polarity, weight).
_LEX: dict[str, tuple[str, float, float]] = {
    # frustration / negative
    "frustrated": ("frustration", -0.7, 1.0),
    "annoyed": ("frustration", -0.5, 0.8),
    "angry": ("frustration", -0.8, 1.0),
    "stuck": ("frustration", -0.5, 0.7),
    "hate": ("frustration", -0.7, 0.8),
    "tired": ("fatigue", -0.4, 0.6),
    "exhausted": ("fatigue", -0.7, 0.9),
    "sad": ("sadness", -0.6, 0.8),
    "worried": ("anxiety", -0.5, 0.7),
    "anxious": ("anxiety", -0.6, 0.8),
    "nervous": ("anxiety", -0.4, 0.6),
    "embarrassed": ("embarrassment", -0.5, 0.7),
    # excitement / positive
    "excited": ("excitement", 0.7, 0.9),
    "love": ("excitement", 0.7, 0.9),
    "happy": ("joy", 0.7, 0.9),
    "great": ("joy", 0.4, 0.5),
    "awesome": ("excitement", 0.7, 0.8),
    "amazing": ("excitement", 0.7, 0.8),
    "proud": ("pride", 0.7, 0.9),
    "confident": ("confidence", 0.6, 0.8),
    "winning": ("excitement", 0.6, 0.7),
    # urgency
    "urgent": ("urgency", 0.0, 0.9),
    "asap": ("urgency", 0.0, 0.9),
    "deadline": ("urgency", -0.2, 0.8),
    # uncertainty
    "confused": ("uncertainty", -0.3, 0.7),
    "unsure": ("uncertainty", -0.2, 0.6),
    "maybe": ("uncertainty", -0.1, 0.4),
}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def score_affect(text: str) -> dict[str, float | str]:
    """Return ``{}`` when no signal; otherwise an affect dict."""

    if not text:
        return {}
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return {}
    # Aggregate per emotion.
    bag: dict[str, list[tuple[float, float]]] = {}
    for t in tokens:
        hit = _LEX.get(t)
        if not hit:
            continue
        emotion, polarity, weight = hit
        bag.setdefault(emotion, []).append((polarity, weight))
    if not bag:
        return {}
    # Pick the dominant emotion by total weight.
    dominant = max(bag.items(), key=lambda kv: sum(w for _, w in kv[1]))[0]
    pol_w = bag[dominant]
    intensity = min(1.0, sum(w for _, w in pol_w) / 2.0)
    polarity = sum(p * w for p, w in pol_w) / max(1e-6, sum(w for _, w in pol_w))
    return {
        "emotion": dominant,
        "polarity": float(round(polarity, 3)),
        "intensity": float(round(intensity, 3)),
    }


async def tag_episode_affect(
    driver: Any, *, group_id: str, episode_uuids: list[str]
) -> int:
    """Tag each episode in the batch and propagate affect onto incident
    edges. Returns count of episodes tagged."""

    if not episode_uuids:
        return 0
    rows = _unwrap(
        await driver.query(
            "SELECT uuid, content FROM episode WHERE group_id = $g AND uuid IN $u;",
            {"g": group_id, "u": list(episode_uuids)},
        )
    )
    tagged = 0
    for r in rows:
        affect = score_affect(str(r.get("content") or ""))
        if not affect:
            continue
        await driver.query(
            "UPDATE episode SET affect = $a WHERE uuid = $u;",
            {"u": r.get("uuid"), "a": affect},
        )
        # Propagate to edges that cite this episode.
        await driver.query(
            """
            UPDATE relates_to SET
                valence = $v,
                intensity = math::max([intensity OR 0, $i])
            WHERE group_id = $g AND $u IN episodes;
            """,
            {"g": group_id, "u": r.get("uuid"),
             "v": float(affect.get("polarity", 0.0)),
             "i": float(affect.get("intensity", 0.0))},
        )
        tagged += 1
    logger.debug("tag_episode_affect: group=%s tagged=%d", group_id, tagged)
    return tagged
