"""Relation frames - the per-predicate metadata that drives generalized
temporal reasoning.

A :class:`RelationFrame` describes the *shape* of a relation type
(directionality, cardinality, contradiction policy, ...) instead of the
relation's domain semantics. The engine uses this metadata to decide,
generically:

* whether ``wife_of`` and ``husband_of`` are aliases of one symmetric
  ``spouse_of`` frame (and therefore should produce a single canonical
  edge regardless of phrasing),
* whether a new ``lives_in`` assertion should expire the prior one (the
  frame's ``cardinality`` is ``one_current``),
* whether two simultaneous claims should coexist (``many_current``) or
  trigger LLM-mediated contradiction detection (``uncertain``).

Frames are *data*, not code. Users register them at runtime, the engine
seeds a small default catalog of common shapes, and unknown predicates
either fall through to the existing per-fact heuristics or get
classified by the LLM on demand.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Literal

from pydantic import Field

from surriti.nodes import _Base

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Universal claim operations. Domain-agnostic; any user statement maps
# into one of these regardless of subject matter.
ClaimOperation = Literal["assert", "terminate", "correct", "qualify", "noop"]

Directionality = Literal["directed", "symmetric", "inverse_pair", "unknown"]
TemporalKind = Literal["state", "event", "timeless", "recurring", "unknown"]
Cardinality = Literal[
    "one_current", "many_current", "many_historical", "timeless", "unknown"
]
ContradictionPolicy = Literal["replace", "coexist", "negate", "uncertain"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RelationFrame(_Base):
    """Metadata describing one relation type.

    Frames are scoped per ``group_id`` so multi-tenant deployments stay
    isolated. ``aliases`` is the list of alternate predicate phrases that
    resolve to this frame (``["wife_of", "husband_of", "married_to"]``).
    """

    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""

    directionality: Directionality = "unknown"
    temporal_kind: TemporalKind = "unknown"
    cardinality: Cardinality = "unknown"
    contradiction_policy: ContradictionPolicy = "uncertain"

    inverse_name: str | None = None
    subject_role: str | None = None
    object_role: str | None = None

    confidence: float = 0.5

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def matches(self, predicate: str) -> bool:
        """Return True when ``predicate`` is this frame's canonical name
        or one of its aliases (case-insensitive)."""
        p = (predicate or "").strip().lower()
        if not p:
            return False
        if p == self.canonical_name.lower():
            return True
        return any(p == a.lower() for a in self.aliases)


# ---------------------------------------------------------------------------
# Slot keys
# ---------------------------------------------------------------------------


def qualifier_hash(qualifiers: dict[str, Any] | None) -> str:
    """Stable short hash of a qualifier dict.

    Two qualifier dicts hash the same iff they are structurally equal
    after JSON canonicalization (sorted keys). Empty/None hashes to ``""``
    so unqualified facts share a slot with each other.
    """
    if not qualifiers:
        return ""
    payload = json.dumps(qualifiers, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def make_slot_key(
    group_id: str,
    subject_uuid: str,
    canonical_name: str,
    qualifiers: dict[str, Any] | None = None,
) -> str:
    """Deterministic slot key for ``(subject, relation_frame, qualifiers)``.

    Used by the cardinality-driven slot closer to find the set of edges
    that occupy the same logical slot as a new claim, regardless of the
    raw alias the extractor used.
    """
    return "::".join(
        (
            (group_id or "").strip(),
            (subject_uuid or "").strip(),
            (canonical_name or "").strip().lower(),
            qualifier_hash(qualifiers),
        )
    )


def normalize_symmetric(
    subject_uuid: str, object_uuid: str
) -> tuple[str, str]:
    """Return ``(subject, object)`` ordered lexicographically.

    For symmetric frames this collapses ``A->B`` and ``B->A`` onto the
    same canonical edge so alias-equivalent statements written from
    either side dedupe deterministically.
    """
    if not subject_uuid or not object_uuid:
        return subject_uuid, object_uuid
    if subject_uuid <= object_uuid:
        return subject_uuid, object_uuid
    return object_uuid, subject_uuid


# ---------------------------------------------------------------------------
# Default frame catalog
# ---------------------------------------------------------------------------

# A tiny seed catalog that covers the relations every general assistant
# trips over within the first few episodes. Shipping these by default
# means brand-new deployments get spouse-direction collapse,
# current-location supersession, and identity handling out of the box -
# without paying an LLM round-trip for the first encounter.
DEFAULT_FRAMES: tuple[RelationFrame, ...] = (
    RelationFrame(
        canonical_name="spouse_of",
        aliases=["wife_of", "husband_of", "married_to", "partner_of"],
        directionality="symmetric",
        temporal_kind="state",
        cardinality="one_current",
        contradiction_policy="replace",
        confidence=0.9,
    ),
    RelationFrame(
        canonical_name="parent_of",
        aliases=["father_of", "mother_of", "mom_of", "dad_of"],
        directionality="inverse_pair",
        inverse_name="child_of",
        temporal_kind="timeless",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.9,
    ),
    RelationFrame(
        canonical_name="child_of",
        aliases=["son_of", "daughter_of"],
        directionality="inverse_pair",
        inverse_name="parent_of",
        temporal_kind="timeless",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.9,
    ),
    RelationFrame(
        canonical_name="sibling_of",
        aliases=["brother_of", "sister_of"],
        directionality="symmetric",
        temporal_kind="timeless",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.9,
    ),
    RelationFrame(
        canonical_name="friend_of",
        aliases=["friends_with"],
        directionality="symmetric",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.7,
    ),
    RelationFrame(
        canonical_name="lives_in",
        aliases=["resides_in", "based_in", "located_in", "from"],
        directionality="directed",
        temporal_kind="state",
        cardinality="one_current",
        contradiction_policy="replace",
        confidence=0.85,
    ),
    RelationFrame(
        canonical_name="works_at",
        aliases=["employed_by", "employee_of", "works_for"],
        directionality="directed",
        temporal_kind="state",
        cardinality="one_current",
        contradiction_policy="replace",
        confidence=0.85,
    ),
    RelationFrame(
        canonical_name="member_of",
        aliases=["belongs_to", "part_of"],
        directionality="directed",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.7,
    ),
    RelationFrame(
        canonical_name="owns",
        aliases=["has", "possesses"],
        directionality="directed",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.7,
    ),
    RelationFrame(
        canonical_name="owns_pet",
        aliases=["has_pet", "owns_dog", "owns_cat", "pet_is"],
        directionality="directed",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.8,
    ),
    RelationFrame(
        canonical_name="is_named",
        aliases=["is_called", "is_aka", "name_is", "goes_by"],
        directionality="directed",
        subject_role="self",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.95,
    ),
    RelationFrame(
        canonical_name="born_in",
        aliases=["was_born_in", "birthplace_is"],
        directionality="directed",
        temporal_kind="timeless",
        cardinality="one_current",
        contradiction_policy="replace",
        confidence=0.95,
    ),
    RelationFrame(
        canonical_name="located_in",
        aliases=["situated_in", "found_in"],
        directionality="directed",
        temporal_kind="state",
        cardinality="one_current",
        contradiction_policy="replace",
        confidence=0.7,
    ),
    RelationFrame(
        canonical_name="contains",
        aliases=["includes", "holds"],
        directionality="inverse_pair",
        inverse_name="part_of",
        temporal_kind="state",
        cardinality="many_current",
        contradiction_policy="coexist",
        confidence=0.7,
    ),
    RelationFrame(
        canonical_name="precedes",
        aliases=["before", "happens_before"],
        directionality="inverse_pair",
        inverse_name="follows",
        temporal_kind="event",
        cardinality="many_historical",
        contradiction_policy="coexist",
        confidence=0.7,
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RelationFrameRegistry:
    """In-process registry that resolves predicate phrases to frames.

    Resolution order:

    1. Exact match against a frame's ``canonical_name`` or ``aliases``
       in the requested ``group_id`` (or in the global fallback).
    2. Optional LLM classifier (when an ``llm_classifier`` callable is
       provided), which mints a fresh frame and persists it back into
       the registry for future hits.
    3. Returns ``None`` so callers can fall back to legacy per-fact
       metadata - this keeps the engine fully backward compatible while
       frames are being rolled out.
    """

    def __init__(
        self,
        *,
        seed_defaults: bool = True,
        llm_classifier: Any | None = None,
    ) -> None:
        # group_id -> {predicate.lower(): RelationFrame}
        self._by_group: dict[str, dict[str, RelationFrame]] = {}
        self._global: dict[str, RelationFrame] = {}
        self._llm_classifier = llm_classifier
        if seed_defaults:
            for frame in DEFAULT_FRAMES:
                self._register_global(frame)

    # ---- registration -------------------------------------------------

    def register(
        self, frame: RelationFrame, *, group_id: str | None = None
    ) -> RelationFrame:
        """Register ``frame`` either globally (``group_id is None``) or
        scoped to a single tenant. Returns the registered frame."""
        if group_id is None or group_id == "":
            self._register_global(frame)
        else:
            bucket = self._by_group.setdefault(group_id, {})
            for alias in self._alias_keys(frame):
                bucket[alias] = frame
        return frame

    def register_many(
        self,
        frames: Iterable[RelationFrame],
        *,
        group_id: str | None = None,
    ) -> None:
        for f in frames:
            self.register(f, group_id=group_id)

    def _register_global(self, frame: RelationFrame) -> None:
        for alias in self._alias_keys(frame):
            self._global[alias] = frame

    @staticmethod
    def _alias_keys(frame: RelationFrame) -> list[str]:
        keys = [frame.canonical_name.strip().lower()]
        keys.extend(a.strip().lower() for a in frame.aliases if a)
        return [k for k in keys if k]

    # ---- lookup -------------------------------------------------------

    def get(
        self, predicate: str, *, group_id: str = ""
    ) -> RelationFrame | None:
        """Return the frame for ``predicate`` (group-scoped first, then
        global), or ``None`` if no frame is registered."""
        key = (predicate or "").strip().lower()
        if not key:
            return None
        if group_id:
            bucket = self._by_group.get(group_id)
            if bucket is not None and key in bucket:
                return bucket[key]
        return self._global.get(key)

    async def resolve(
        self,
        predicate: str,
        *,
        group_id: str = "",
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ) -> RelationFrame | None:
        """Resolve ``predicate`` to a frame, classifying via LLM if
        necessary. Returns ``None`` when no classifier is configured and
        the predicate is unknown."""
        existing = self.get(predicate, group_id=group_id)
        if existing is not None:
            return existing
        if self._llm_classifier is None:
            return None
        try:
            frame = await self._llm_classifier(
                predicate=predicate,
                source_span=source_span,
                sample_subject=sample_subject,
                sample_object=sample_object,
            )
        except Exception:  # pragma: no cover - classifier is opt-in
            return None
        if frame is None:
            return None
        # Always include the original phrase as an alias so the next
        # resolve() hits the cache without re-classifying.
        alias_key = (predicate or "").strip().lower()
        if alias_key and not frame.matches(alias_key):
            frame.aliases = [*frame.aliases, alias_key]
        return self.register(frame, group_id=group_id or None)

    # ---- introspection ------------------------------------------------

    def all_frames(self, *, group_id: str = "") -> list[RelationFrame]:
        seen: dict[str, RelationFrame] = {}
        for frame in self._global.values():
            seen[frame.canonical_name] = frame
        if group_id:
            for frame in self._by_group.get(group_id, {}).values():
                seen[frame.canonical_name] = frame
        return list(seen.values())


__all__ = [
    "ClaimOperation",
    "Directionality",
    "TemporalKind",
    "Cardinality",
    "ContradictionPolicy",
    "RelationFrame",
    "RelationFrameRegistry",
    "DEFAULT_FRAMES",
    "qualifier_hash",
    "make_slot_key",
    "normalize_symmetric",
]
