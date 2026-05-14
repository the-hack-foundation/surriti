"""LLM prompt templates used by the cognition pass.

Kept in one module so prompt drift is auditable. Every template is a
plain ``str.format``-friendly string with named placeholders.
"""

from __future__ import annotations

TRAIT_RATIFY_SYSTEM = (
    "You are a memory-synthesis assistant. Given a list of CANDIDATE "
    "TRAITS for a person -- each candidate paired with a few example "
    "facts that suggested it -- return a JSON array of accepted traits. "
    "Each accepted trait is an object with fields: "
    "{\"name\": <short snake_case label>, \"description\": <one sentence>, "
    "\"confidence\": <float 0..1>, \"supporting_indices\": [<int>...] }. "
    "Discard candidates that are too vague, too specific to a single "
    "episode, or that simply restate a single fact. Prefer durable "
    "personality / preference / competency / habit signals over momentary "
    "states. Return ONLY the JSON array."
)

GOAL_RATIFY_SYSTEM = (
    "You are a memory-synthesis assistant. Given GOAL CANDIDATES "
    "extracted from a person's recent episodes -- each candidate is a "
    "short sentence that contained intentional language ('want', "
    "'trying to', 'goal', 'improve') -- return a JSON array of distinct, "
    "durable goals. Each accepted goal is an object: "
    "{\"name\": <short snake_case label>, \"description\": <one sentence>, "
    "\"domain\": <short label or empty>, \"time_horizon\": <\"short\"|\"medium\"|\"long\"|\"unknown\">, "
    "\"confidence\": <float 0..1>, \"supporting_indices\": [<int>...] }. "
    "Drop one-off requests, hypotheticals, or duplicates of existing "
    "goals supplied in EXISTING_GOALS. Return ONLY the JSON array."
)

DOMAIN_LABEL_SYSTEM = (
    "You label thematic clusters. Given a CLUSTER consisting of entity "
    "names and short fact strings, return a SINGLE short snake_case "
    "domain label (e.g. 'fortnite', 'distance_running', 'cooking', "
    "'machine_learning'). Return ONLY the label, no quotes, no "
    "explanation."
)

PREDICTION_SYSTEM = (
    "You are a predictive-context assistant. Given a person's recent "
    "interaction patterns, active goals, and dominant domains, return a "
    "JSON object with fields: "
    "{\"likely_next_topics\": [<short str>...], "
    " \"likely_preferences\": [<short str>...], "
    " \"likely_questions\":   [<short str>...] }. "
    "Keep each list <= 5 items. Return ONLY the JSON object."
)
