"""LLM client interfaces and a deterministic stub.

Surriti calls an LLM for two things:

1. **Extraction** - given an episode, return entities and (subject, predicate,
   object, fact) triples.
2. **Contradiction detection** - given a candidate fact and a list of
   existing facts, decide which existing facts are invalidated.

Real adapters can wrap OpenAI, Anthropic, Gemini, etc. The bundled
:class:`DummyLLMClient` performs naive heuristics so tests and quick demos
work offline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# Words that are too generic to be useful entity names in the dummy extractor.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "from", "as", "by",
}


@dataclass
class ExtractedEntity:
    name: str
    summary: str = ""
    labels: list[str] = field(default_factory=lambda: ["Entity"])


FactOperation = Literal["assert", "terminate", "correct", "qualify", "noop"]


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    fact: str = ""
    """Natural-language sentence form. Defaults to empty; the engine
    falls back to ``"<subject> <predicate> <object>."`` when blank."""
    valid_at: str | None = None  # ISO 8601 timestamp; None = unknown
    invalid_at: str | None = None
    # Generic temporal-state metadata. The LLM (or caller) sets these per
    # fact; the engine reasons over them without any hardcoded predicate
    # vocabulary. See EXTRACTION_SYSTEM for the rubric.
    operation: FactOperation = "assert"
    """What to do with the fact: assert (default), terminate a prior
    matching fact, correct (terminate prior + assert new), qualify (add
    a scoped variant without closing peers), or noop."""
    temporal: bool = False
    """True when the fact describes a time-varying state of the subject."""
    singleton: bool = False
    """True when the subject can hold only ONE active value of this
    predicate at a time. Triggers the deterministic singleton-slot closer
    on assert."""
    domain: str | None = None
    """Short free-form bucket label (e.g. "employment", "residence") used
    to scope contradiction comparisons. Free text, never an enum."""
    memory_class: str = "objective"
    """Closed-vocabulary tag for the *kind* of fact this is. One of:
    ``"objective"`` (verifiable claim about the world — default),
    ``"preference"`` (soft user wish about the assistant or world),
    ``"style"`` (communication-style directive),
    ``"constraint"`` (hard rule / forbidden action),
    ``"trait"`` (persistent personal trait/value),
    ``"sentiment"`` (emotional pattern / opinion).
    Drives kind-aware singleton closure (cross-class facts coexist) and
    always-pinned recall for subjective classes (preference/style/constraint).
    Persisted on the edge in ``attributes['memory_class']``."""
    replaces: list[str] = field(default_factory=list)
    """Optional name hints ("<subject> <predicate> <object>") of prior
    facts this one replaces. Used by terminate/correct operations."""
    confidence: float = 1.0
    # Generalized claim metadata (optional). Populated by extractors that
    # emit structured claims; ignored by older extractors. The engine
    # uses these to resolve a :class:`~surriti.relation_frames.RelationFrame`
    # and build qualifier-aware slot keys without needing a hardcoded
    # predicate vocabulary.
    relation_phrase: str | None = None
    """Original natural-language relation phrase ("is the wife of") when
    the extractor preserves it separately from the normalized predicate."""
    qualifiers: dict[str, Any] = field(default_factory=dict)
    """Free-form scope qualifying the claim (e.g.
    ``{"season": "winter"}``); hashed into the slot key so qualified
    variants coexist with the unqualified slot."""
    argument_roles: dict[str, str] = field(default_factory=dict)
    """Semantic argument roles supplied by the extractor (e.g.
    ``{"object_role_relative_to_subject": "wife"}``); used by the
    direction-repair pass for symmetric/inverse-pair frames."""
    source_span: str | None = None
    """Verbatim span of the source text that produced this claim, when
    available. Forwarded to the LLM frame classifier on cold-start."""


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)


@dataclass
class ContradictionCandidate:
    """Structured view of an existing edge passed to contradiction detection.

    The contradiction layer historically saw only the natural-language
    ``fact`` string. That hides the structured signal — subject,
    predicate, object, domain, temporal validity — that the LLM needs
    to reason about same-domain conflicts. Adapters that receive a
    ``candidates`` list should render these fields in their prompt.
    """

    uuid: str
    subject: str
    predicate: str
    object: str
    fact: str
    domain: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None


class LLMClient(ABC):
    @abstractmethod
    async def extract(
        self,
        content: str,
        *,
        group_id: str = "",
        entity_types: dict[str, type] | None = None,
        custom_instructions: str | None = None,
        context: str | None = None,
    ) -> ExtractionResult:
        """Extract entities + facts from ``content``.

        ``content`` is the **current episode** to extract from. ``context``
        is optional read-only prior text (e.g. recent episodes) supplied
        purely for pronoun/entity resolution; facts whose source is in
        ``context`` should NOT be re-emitted. Implementations are free to
        ignore ``context`` (e.g. heuristic stubs) but real adapters MUST
        present it to the model in a clearly-fenced block."""

    @abstractmethod
    async def find_contradictions(
        self,
        new_fact: str,
        existing_facts: list[str],
        *,
        candidates: list[ContradictionCandidate] | None = None,
        new_fact_struct: ExtractedFact | None = None,
    ) -> list[int]:
        """Return the indexes in ``existing_facts`` invalidated by ``new_fact``.

        ``new_fact`` and ``existing_facts`` are the natural-language
        fact strings (kept for back-compat). Real adapters should also
        consume the structured ``candidates`` list and ``new_fact_struct``
        when provided to render richer prompts; stub clients are free
        to ignore them."""

    async def classify_relation_frame(
        self,
        *,
        predicate: str,
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ) -> "Any | None":
        """Optionally classify an unknown predicate into a RelationFrame.

        Default returns ``None`` so adapters that don't need dynamic
        classification keep working unchanged. Real adapters override
        this to ask the model for ``{canonical_name, aliases,
        directionality, temporal_kind, cardinality, contradiction_policy,
        inverse_name, subject_role, object_role, confidence}`` and
        return a :class:`~surriti.relation_frames.RelationFrame`.

        Returning ``None`` (or raising) makes the engine fall through to
        per-fact heuristics for that predicate.
        """
        del predicate, source_span, sample_subject, sample_object
        return None

    async def synthesize(self, system: str, user: str) -> str | None:
        """Optional generic JSON-completion hook used by ``surriti.cognition``.

        Default returns ``None`` -- callers (cognition modules) treat
        that as "skip the LLM step, fall back to heuristics". Real
        adapters override this to delegate to their underlying
        ``_complete(system, user)``.
        """
        del system, user
        return None


class DummyLLMClient(LLMClient):
    """Heuristic, offline LLM stub.

    - ``extract``: pulls capitalised tokens as entities and fabricates a
      single ``MENTIONS_WITH`` fact connecting consecutive entities.
    - ``find_contradictions``: marks an existing fact as contradicted when
      it shares >=2 significant tokens with the new fact AND contains a
      negation/transition cue (``not``, ``no longer``, ``moved``, ``changed``).
    """

    _entity_re = re.compile(r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b")

    async def extract(
        self,
        content: str,
        *,
        group_id: str = "",
        entity_types: dict[str, type] | None = None,
        custom_instructions: str | None = None,
        context: str | None = None,
    ) -> ExtractionResult:
        # The dummy extractor ignores ``context`` -- it's heuristic-only.
        text = content or ""
        all_names: list[str] = []
        seen: set[str] = set()
        for match in self._entity_re.finditer(text):
            name = match.group(1).strip()
            if name.lower() in _STOPWORDS:
                continue
            if name not in seen:
                seen.add(name)
                all_names.append(name)

        entities = [ExtractedEntity(name=n) for n in all_names]

        # Build per-sentence facts so the original wording (and any
        # negation/transition cues) is preserved on the EntityEdge.fact.
        facts: list[ExtractedFact] = []
        for sentence in re.split(r"(?<=[.!?])\s+|;\s*", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            present = [n for n in all_names if n in sentence]
            for left, right in zip(present, present[1:], strict=False):
                facts.append(
                    ExtractedFact(
                        subject=left,
                        predicate="related_to",
                        object=right,
                        fact=sentence if sentence.endswith(".") else sentence + ".",
                    )
                )
        return ExtractionResult(entities=entities, facts=facts)

    async def find_contradictions(
        self,
        new_fact: str,
        existing_facts: list[str],
        *,
        candidates: list[ContradictionCandidate] | None = None,
        new_fact_struct: ExtractedFact | None = None,
    ) -> list[int]:
        cues = ("not ", "no longer", "moved", "changed", "stopped", "former")
        has_transition_cue = any(c in new_fact.lower() for c in cues)

        # --- Layer A: structured candidate overlap (preferred when available) ---
        if candidates:
            # Extract subject/predicate/object from the new fact struct or
            # from the raw fact text as a fallback.
            new_subject = (new_fact_struct.subject if new_fact_struct else "").lower()
            new_predicate = (new_fact_struct.predicate if new_fact_struct else "").lower()
            new_object = (new_fact_struct.object if new_fact_struct else "").lower()

            new_tokens = {t.lower() for t in re.findall(r"\w+", new_fact) if len(t) > 2}

            contradicted: list[int] = []
            for idx, cand in enumerate(candidates):
                if idx >= len(existing_facts):
                    break
                # Subject overlap is a strong contradiction signal when
                # combined with a transition cue.
                subj_overlap = (
                    new_subject and cand.subject.lower() == new_subject
                )
                # Object overlap (same entity, different relationship).
                obj_overlap = (
                    new_object and cand.object.lower() == new_object
                )
                # Token overlap (fallback when structured fields are empty).
                existing_tokens = {
                    t.lower() for t in re.findall(r"\w+", cand.fact) if len(t) > 2
                }
                token_shared = len(new_tokens & existing_tokens)

                if has_transition_cue and (
                    subj_overlap
                    or obj_overlap
                    or token_shared >= 1  # lowered from 2
                ):
                    contradicted.append(idx)
            if contradicted:
                return contradicted

        # --- Layer B: pure text heuristic (legacy path) ---
        new_tokens = {t.lower() for t in re.findall(r"\w+", new_fact) if len(t) > 2}
        contradicted: list[int] = []
        for idx, existing in enumerate(existing_facts):
            existing_tokens = {
                t.lower() for t in re.findall(r"\w+", existing) if len(t) > 2
            }
            shared = new_tokens & existing_tokens
            if len(shared) >= 1 and has_transition_cue:  # lowered from 2
                contradicted.append(idx)
        return contradicted


@dataclass
class ScriptedResponse:
    """Pre-baked extraction response for :class:`ScriptedLLMClient`.

    ``frame`` is consumed by :meth:`ScriptedLLMClient.classify_relation_frame`
    -- supplying one lets a test exercise the dynamic-frame-classifier
    code path without a real LLM. Frame responses are queued separately
    from extract responses and dispensed in order of arrival.
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    contradictions: list[int] = field(default_factory=list)
    frame: "Any | None" = None


class ScriptedLLMClient(LLMClient):
    """LLM stub that replays a queue of pre-recorded responses.

    Useful for **prompt-style tests**: assert that ``Surriti`` correctly
    handles whatever a real model might return — empty extraction, custom
    entity labels, multi-fact episodes, edge invalidation, etc. — without
    needing network access.

    Example
    -------
    >>> client = ScriptedLLMClient([
    ...     ScriptedResponse(
    ...         entities=[ExtractedEntity(name="Alice", labels=["Person"])],
    ...         facts=[ExtractedFact("Alice", "works_at", "Acme",
    ...                             "Alice works at Acme.")],
    ...     ),
    ...     ScriptedResponse(contradictions=[0]),  # second extract call
    ... ])
    """

    def __init__(self, responses: list[ScriptedResponse]) -> None:
        self._responses = list(responses)
        self._extract_calls: list[dict[str, object]] = []
        self._contradiction_calls: list[dict[str, object]] = []
        self._classify_calls: list[dict[str, object]] = []
        # Queue of pre-baked frames pulled from the responses on first use.
        self._frame_queue: list[Any] = [
            r.frame for r in self._responses if r.frame is not None
        ]
        self._index = 0

    @property
    def classify_calls(self) -> list[dict[str, object]]:
        return self._classify_calls

    async def classify_relation_frame(
        self,
        *,
        predicate: str,
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ):
        self._classify_calls.append(
            {
                "predicate": predicate,
                "source_span": source_span,
                "sample_subject": sample_subject,
                "sample_object": sample_object,
            }
        )
        if not self._frame_queue:
            return None
        return self._frame_queue.pop(0)

    @property
    def extract_calls(self) -> list[dict[str, object]]:
        return self._extract_calls

    @property
    def contradiction_calls(self) -> list[dict[str, object]]:
        return self._contradiction_calls

    def _next(self) -> ScriptedResponse:
        if self._index >= len(self._responses):
            return ScriptedResponse()
        resp = self._responses[self._index]
        self._index += 1
        return resp

    async def extract(
        self,
        content: str,
        *,
        group_id: str = "",
        entity_types: dict[str, type] | None = None,
        custom_instructions: str | None = None,
        context: str | None = None,
    ) -> ExtractionResult:
        self._extract_calls.append(
            {
                "content": content,
                "context": context,
                "group_id": group_id,
                "entity_types": list((entity_types or {}).keys()),
                "custom_instructions": custom_instructions,
            }
        )
        resp = self._next()
        return ExtractionResult(entities=list(resp.entities), facts=list(resp.facts))

    async def find_contradictions(
        self,
        new_fact: str,
        existing_facts: list[str],
        *,
        candidates: list[ContradictionCandidate] | None = None,
        new_fact_struct: ExtractedFact | None = None,
    ) -> list[int]:
        self._contradiction_calls.append(
            {
                "new_fact": new_fact,
                "existing_facts": list(existing_facts),
                "candidates": list(candidates) if candidates else [],
                "new_fact_struct": new_fact_struct,
            }
        )
        # Re-use the next scripted response's `contradictions` payload, but do
        # not advance `_index` so contradictions can be paired with extracts.
        if self._index < len(self._responses):
            return list(self._responses[self._index - 1].contradictions) if self._index else []
        return []
