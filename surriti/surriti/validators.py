"""Post-extraction fact validation and repair.

The LLM is encouraged in :mod:`surriti.llm_clients` to follow strict
extraction rules, but small / fast models routinely violate them. This
module is the deterministic safety net that runs on every
:class:`~surriti.llm.ExtractedFact` between extraction and persistence.

It does two things:

1. **Repair** — fix predictable mistakes in place (lowercase predicates,
   rewrite self-loop identity facts to the speaker's stable id, ...).
2. **Reject** — return ``None`` for facts that cannot be salvaged
   (empty fields, banned placeholder objects like "world", unrepaired
   self-loops, ...). Callers drop those silently.

The validator is *intentionally* tiny and rule-based. Anything that
needs to "understand" the fact belongs in the contradiction layer.
"""

from __future__ import annotations

from surriti.llm import ExtractedFact

# Predicates that legitimately connect an entity to itself (identity /
# aliasing). Everything else with subject == object is LLM garbage we
# drop as defence-in-depth on top of the extraction prompt.
IDENTITY_PREDICATES: frozenset[str] = frozenset(
    {"is_named", "is_called", "is_self", "is_aka"}
)

# Filler "places" that small models love to emit for ``lives_in`` /
# ``located_in`` / ``from`` when no real location appears in the input.
# These are not entities and the corresponding facts are pure noise.
_LOCATION_FILLERS: frozenset[str] = frozenset(
    {"world", "earth", "universe", "everywhere", "anywhere", "somewhere"}
)
_LOCATION_PREDICATES: frozenset[str] = frozenset(
    {"lives_in", "located_in", "from", "resides_in", "based_in"}
)


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def repair_fact(
    fact: ExtractedFact,
    *,
    speaker_id: str | None = None,
    speaker_name: str | None = None,
) -> ExtractedFact | None:
    """Return a cleaned-up :class:`ExtractedFact` or ``None`` to drop it.

    Parameters
    ----------
    fact:
        The raw extracted fact. Mutated in place AND returned for
        convenience; callers should use the return value.
    speaker_id:
        Stable speaker identifier (e.g. ``"default"``) used to repair
        identity self-loops by rewriting the subject to the speaker.
    speaker_name:
        Display name of the speaker, currently unused; reserved for
        future heuristics.
    """

    fact.subject = _normalize_text(fact.subject)
    fact.object = _normalize_text(fact.object)
    fact.predicate = _normalize_text(fact.predicate).lower()

    if not fact.subject or not fact.object or not fact.predicate:
        return None

    # Banned placeholder objects -- common LLM filler that produces
    # "User lives_in world" style noise.
    if (
        fact.predicate in _LOCATION_PREDICATES
        and fact.object.lower() in _LOCATION_FILLERS
    ):
        return None

    # Self-loop handling. Identity predicates with a speaker_id can be
    # repaired by rewriting the subject to the speaker's stable id;
    # identity predicates without speaker context are kept as a last
    # resort (better to record the name than lose it). All other
    # self-loops are LLM garbage.
    if fact.subject == fact.object:
        if fact.predicate in IDENTITY_PREDICATES:
            if speaker_id and fact.subject != speaker_id and speaker_id != fact.object:
                fact.subject = speaker_id
                return fact
            # No repair available -- fall through and keep as-is.
            return fact
        return None

    # ``terminate`` requires a target object to close; we already
    # ensured object is non-empty above so this is just a guard for
    # readability and future-proofing.
    if fact.operation == "terminate" and not fact.object:
        return None

    return fact


__all__ = [
    "IDENTITY_PREDICATES",
    "repair_fact",
]
