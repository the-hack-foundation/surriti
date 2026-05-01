"""Production LLM client adapters for OpenAI and Anthropic.

Each adapter implements :class:`surriti.llm.LLMClient` by asking the model
to return a strict JSON document. Both ``openai`` and ``anthropic`` are
optional dependencies::

    pip install "surriti[openai]"
    pip install "surriti[anthropic]"
    pip install "surriti[all]"

Example
-------
>>> from surriti import Surriti
>>> from surriti.llm_clients import OpenAILLMClient
>>> memory = Surriti.from_env(llm_client=OpenAILLMClient(model="gpt-4o-mini"))
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from surriti.errors import SurritiConfigError, SurritiLLMError
from surriti.llm import (
    ContradictionCandidate,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
    LLMClient,
)
from surriti.relation_frames import RelationFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """\
You are a knowledge-graph extractor. Your input has two clearly fenced \
sections:

  CONTEXT (read-only, do NOT extract):
    <prior episodes for reference -- use only to resolve pronouns and
     to recognise entities; never re-emit a fact whose source text
     lives only in this section>

  CURRENT EPISODE (extract from this only):
    <the new text -- every fact you emit must come from THIS section>

If the CONTEXT block is omitted there is no prior context to consider.

Return STRICT JSON with two arrays:

{
  "entities": [
    {"name": "<a real name from the input>", "labels": ["Person"], "summary": "..."}
  ],
  "facts": [
    {"subject": "<entity name>",
     "predicate": "snake_case_verb",
     "relation_phrase": "<verbatim verb phrase from the source>",
     "object":  "<entity name>",
     "fact":    "A complete natural-language sentence.",
     "operation": "assert",
     "temporal": false,
     "singleton": false,
     "domain": null,
     "qualifiers": {},
     "argument_roles": {},
     "source_span": "<verbatim slice of CURRENT EPISODE>",
     "replaces": []}
  ]
}

WHAT COUNTS AS A FACT (extract these from CURRENT EPISODE):
- Self-introductions and properties of the speaker ("my name is X",
  "I'm X", "I am 5 months old", "I work at Acme", "I live in Berlin",
  "I like pizza", "my birthday is October 14").
- Properties of other named entities ("Alice works at Acme").
- Compound sentences split into one fact per claim.
- VALUES become entities too: dates, ages, places, companies must
  appear in `entities` AND be the `object` of the fact. NEVER use a
  literal placeholder like "speaker" or "value" as subject or object.
- When in doubt, extract.

WHAT TO SKIP (return no fact, but mention any named entities):
- Pure interjections ("hi", "thanks", "ok", "hmm").
- Pure questions ("where do I work?", "what's my name?").
- Vague placeholder objects: if the object would be a meaningless
  filler like "world", "everywhere", "thing", "something", "nothing",
  "someone", DROP the fact entirely. Such words are not entities.

FACT METADATA -- general rubric, NOT a closed list of predicates:
- `operation` (default "assert"). Use "terminate" when the input
  explicitly negates a previously true state ("I quit X", "I no
  longer X", "X is not true anymore", "I stopped X"). The fact then
  describes the PRIOR state being closed. Use "correct" when the
  input explicitly replaces a previously stated value with a new one
  for the same slot ("actually it's X, not Y"; "I meant X"). Use
  "qualify" when the input adds a scoped variant of an existing
  state without overriding it ("I live in Florida in the winter"
  alongside a year-round residence). Use "noop" only to ignore the
  fact. Otherwise omit or "assert".
- `temporal` = true when the fact describes a CURRENT state of the
  subject that can change over time (where they live, who they work
  for, what they do, how they feel, what they own, what they prefer
  right now).
- `singleton` = true when the subject can hold only ONE such value
  true at a time (people live in one place, hold one current job
  title, have one current age, have one current employer).
- `domain` = a short free-form bucket label (one or two words) you
  pick to group facts that obviously describe the same slot, so the
  engine can scope contradictions. Use the SAME label across facts
  that share a slot. Examples of labels you might invent: "employment",
  "residence", "naming", "preference", "identity". Free text -- pick
  whatever fits.
- `relation_phrase` = the verbatim verb phrase from the source ("is
  the wife of", "moved to", "no longer works at"). Lets the engine
  canonicalize alias predicates without losing the original wording.
- `qualifiers` = optional dict of conditions that scope the claim
  ({"season": "winter"}, {"role": "weekday"}, {"location": "remote"}).
  Two facts that share subject+predicate+object but differ in
  qualifiers represent distinct slots and coexist.
- `argument_roles` = optional dict naming the semantic role each
  argument plays ({"subject": "employee", "object": "employer"};
  {"object_role_relative_to_subject": "wife"}). Used by the
  direction-repair pass for symmetric/inverse-pair frames.
- `source_span` = the verbatim slice of CURRENT EPISODE that produced
  the fact. Forwarded to the LLM frame classifier on cold-start so a
  brand-new predicate gets typed correctly.
- `replaces` = optional list of prior fact descriptions this one
  closes (free text like "X works_at OldCo"). Helps `terminate` /
  `correct` find their target when wording differs.

WORKED EXAMPLES (general patterns; the predicate names are illustrative
only -- the real ones come from the input):

  "I work at Acme" ->
    {"subject":"<speaker>", "predicate":"works_at",
     "relation_phrase":"work at", "object":"Acme",
     "fact":"... works at Acme.",
     "operation":"assert", "temporal":true, "singleton":true,
     "domain":"employment",
     "argument_roles":{"subject":"employee","object":"employer"},
     "source_span":"I work at Acme"}

  "I quit my job at Acme" ->
    {"subject":"<speaker>", "predicate":"works_at",
     "relation_phrase":"quit my job at", "object":"Acme",
     "fact":"... quit working at Acme.",
     "operation":"terminate", "temporal":true, "singleton":true,
     "domain":"employment",
     "source_span":"I quit my job at Acme"}

  "I live in Florida during the winter" ->
    {"subject":"<speaker>", "predicate":"lives_in",
     "relation_phrase":"live in", "object":"Florida",
     "fact":"... lives in Florida during the winter.",
     "operation":"qualify", "temporal":true, "singleton":true,
     "domain":"residence",
     "qualifiers":{"season":"winter"},
     "source_span":"I live in Florida during the winter"}

  "I love jazz now" ->
    {"subject":"<speaker>", "predicate":"likes",
     "relation_phrase":"love", "object":"jazz",
     "fact":"... likes jazz.",
     "operation":"assert", "temporal":true, "singleton":false,
     "domain":"preference",
     "source_span":"I love jazz now"}

HARD RULES (violations make the output unusable):
- Extract facts ONLY from CURRENT EPISODE. CONTEXT is read-only.
- NEVER invent entities, predicates, or relations not supported by
  the input. Do NOT use placeholder names like Alice, Bob, Acme, Foo,
  Bar unless they appear in the text.
- NEVER emit a self-loop fact (subject == object). For naming, the
  subject is the SPEAKER and the object is the new name.
- Tokens that look like internal metadata -- bracketed labels
  (`[chat]`, `[turn-a]`), bare UUIDs -- are NOT entities. Ignore them.
- Use the EXACT entity name strings inside facts (subject/object).
- Predicates are snake_case verbs. Avoid `related_to`.
- Each fact's `fact` is a complete natural-language sentence.
- Return ONLY the JSON object, no commentary, no markdown fences.
"""


FRAME_CLASSIFICATION_SYSTEM = """\
You classify a never-seen-before relation predicate into a generic
frame so a temporal knowledge graph can reason over it without any
domain-specific code. Return STRICT JSON with these keys (no others,
no markdown):

  {
    "canonical_name":   "snake_case_verb_or_phrase",
    "aliases":          ["other", "phrasings"],
    "directionality":   "directed" | "symmetric" | "inverse_pair" | "unknown",
    "temporal_kind":    "state" | "event" | "timeless" | "recurring" | "unknown",
    "cardinality":      "one_current" | "many_current" | "many_historical" | "timeless" | "unknown",
    "contradiction_policy": "replace" | "coexist" | "negate" | "uncertain",
    "inverse_name":     "snake_case_inverse_or_null",
    "subject_role":     "role_label_or_null",
    "object_role":      "role_label_or_null",
    "confidence":       0.0 to 1.0
  }

GUIDANCE:
- `directionality`: "symmetric" iff swapping subject and object is
  semantically identical ("sibling_of", "married_to"). "inverse_pair"
  iff there is a natural inverse predicate ("parent_of"/"child_of");
  set `inverse_name` accordingly. Otherwise "directed".
- `temporal_kind`: "state" for ongoing facts that can change over
  time (residence, job); "timeless" for facts that never change
  (birthplace, parentage); "event" for point-in-time happenings;
  "recurring" for repeating activities.
- `cardinality`: "one_current" iff at most one such fact can be
  simultaneously true for a subject (current employer, current
  residence). "many_current" if multiple coexist (friendships,
  hobbies). "many_historical" for events that accumulate. "timeless"
  for immutable facts.
- `contradiction_policy`: "replace" iff a new value supersedes the
  prior one (always pair with `one_current`). "coexist" for
  many_current/timeless facts. "negate" when the predicate carries
  explicit truth flips. "uncertain" when conflicting claims should
  be flagged for human resolution rather than auto-merged.
- `confidence` reflects how sure you are about the classification
  itself, not the underlying fact.

Return JSON only.
"""


CONTRADICTION_SYSTEM = """\
You decide which prior facts are invalidated by a new fact. Return STRICT \
JSON: {"invalidated_indexes": [<int>, ...]}.

A prior fact is invalidated ONLY when ALL of these are true:
1. It has the SAME subject as the new fact.
2. It is in the SAME relation domain as the new fact (employment vs
   employment, residence vs residence, naming vs naming, preference vs
   preference). Different domains never contradict each other.
3. The new fact materially supersedes it (e.g. "X works at A" then
   "X now works at B" or "X no longer works at A"; "X lives in P" then
   "X moved to Q").

Examples that are NOT contradictions (return `[]`):
- new: `Michael is_brother_of Michael`, prior: `Michael works_with Mark`
  (different domains: family vs employment).
- new: `Alice likes pizza`, prior: `Alice lives_in Berlin`
  (different domains).
- new: `Bob is_named Robert`, prior: `Bob works_at Acme`
  (different domains).

When in doubt, return an empty list.
"""


def _build_extraction_user(
    content: str,
    *,
    group_id: str,
    entity_types: dict[str, type] | None,
    custom_instructions: str | None,
    context: str | None = None,
) -> str:
    parts: list[str] = []
    if context and context.strip():
        parts.append(
            "CONTEXT (read-only, do NOT extract; use only for pronoun/"
            "entity resolution):\n" + context.strip()
        )
    parts.append(
        f"CURRENT EPISODE (group_id={group_id!r}; extract from this only):\n"
        + content.strip()
    )
    if entity_types:
        parts.append("Allowed entity types: " + ", ".join(entity_types.keys()))
    if custom_instructions:
        parts.append("Additional instructions: " + custom_instructions)
    return "\n\n".join(parts)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _parse_extraction(raw: str) -> ExtractionResult:
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise SurritiLLMError(f"LLM did not return valid JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise SurritiLLMError(f"LLM JSON was not an object: {data!r}")

    entities: list[ExtractedEntity] = []
    for ent in data.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        raw_name = ent.get("name")
        if isinstance(raw_name, list):
            names = [str(x).strip() for x in raw_name if x and str(x).strip()]
        elif raw_name:
            names = [str(raw_name).strip()]
        else:
            continue
        for nm in names:
            entities.append(
                ExtractedEntity(
                    name=nm,
                    summary=str(ent.get("summary") or ""),
                    labels=list(ent.get("labels") or ["Entity"]),
                )
            )

    facts: list[ExtractedFact] = []

    def _as_strs(v: Any) -> list[str]:
        """Normalise a JSON value (str | list | other) into a list of strings."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]
        return [str(v).strip()]

    for f in data.get("facts") or []:
        if not isinstance(f, dict):
            continue
        subjects   = _as_strs(f.get("subject"))
        objects    = _as_strs(f.get("object"))
        predicate  = (str(f.get("predicate") or "related_to")).strip() or "related_to"
        fact_text_raw = f.get("fact")
        if not subjects or not objects:
            continue
        op_raw = (str(f.get("operation") or "assert")).strip().lower()
        operation = (
            op_raw
            if op_raw in {"assert", "terminate", "correct", "qualify", "noop"}
            else "assert"
        )
        temporal = bool(f.get("temporal") or False)
        singleton = bool(f.get("singleton") or False)
        domain_raw = f.get("domain")
        domain = str(domain_raw).strip().lower() or None if isinstance(domain_raw, str) else None
        replaces_raw = f.get("replaces") or []
        replaces = [str(x).strip() for x in replaces_raw if x and str(x).strip()] if isinstance(replaces_raw, list) else []
        try:
            confidence = float(f.get("confidence")) if f.get("confidence") is not None else 1.0
        except (TypeError, ValueError):
            confidence = 1.0
        # Generalized claim metadata -- all optional, safe defaults so
        # older extractor outputs continue to parse.
        relation_phrase_raw = f.get("relation_phrase")
        relation_phrase = (
            str(relation_phrase_raw).strip()
            if isinstance(relation_phrase_raw, str) and relation_phrase_raw.strip()
            else None
        )
        qualifiers_raw = f.get("qualifiers")
        qualifiers = dict(qualifiers_raw) if isinstance(qualifiers_raw, dict) else {}
        roles_raw = f.get("argument_roles")
        argument_roles = (
            {str(k): str(val) for k, val in roles_raw.items()}
            if isinstance(roles_raw, dict)
            else {}
        )
        source_span_raw = f.get("source_span")
        source_span = (
            str(source_span_raw).strip()
            if isinstance(source_span_raw, str) and source_span_raw.strip()
            else None
        )
        for subject in subjects:
            for obj in objects:
                if not subject or not obj:
                    continue
                fact_text = (
                    str(fact_text_raw).strip()
                    if fact_text_raw
                    else f"{subject} {predicate} {obj}."
                )
                facts.append(
                    ExtractedFact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        fact=fact_text,
                        valid_at=f.get("valid_at") or None,
                        invalid_at=f.get("invalid_at") or None,
                        operation=operation,
                        temporal=temporal,
                        singleton=singleton,
                        domain=domain,
                        replaces=list(replaces),
                        confidence=confidence,
                        relation_phrase=relation_phrase,
                        qualifiers=dict(qualifiers),
                        argument_roles=dict(argument_roles),
                        source_span=source_span,
                    )
                )

    return ExtractionResult(entities=entities, facts=facts)


def _parse_frame_classification(
    raw: str, *, fallback_predicate: str
) -> RelationFrame | None:
    """Parse the JSON returned by ``FRAME_CLASSIFICATION_SYSTEM`` into a
    :class:`RelationFrame`. Returns ``None`` when the payload is
    unusable (missing canonical name, malformed JSON, etc.) so the
    engine cleanly falls back to per-fact heuristics.
    """
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    canonical = (str(data.get("canonical_name") or fallback_predicate)).strip().lower()
    if not canonical:
        return None
    aliases_raw = data.get("aliases") or []
    aliases = (
        [str(a).strip().lower() for a in aliases_raw if a and str(a).strip()]
        if isinstance(aliases_raw, list)
        else []
    )
    valid = {
        "directionality": {"directed", "symmetric", "inverse_pair", "unknown"},
        "temporal_kind": {"state", "event", "timeless", "recurring", "unknown"},
        "cardinality": {
            "one_current", "many_current", "many_historical",
            "timeless", "unknown",
        },
        "contradiction_policy": {"replace", "coexist", "negate", "uncertain"},
    }
    def _enum(key: str, default: str) -> str:
        value = (str(data.get(key) or default)).strip().lower() or default
        return value if value in valid[key] else default
    inverse_raw = data.get("inverse_name")
    inverse = (
        str(inverse_raw).strip().lower()
        if isinstance(inverse_raw, str) and inverse_raw.strip()
        else None
    )
    subject_role_raw = data.get("subject_role")
    object_role_raw = data.get("object_role")
    try:
        confidence = float(data.get("confidence")) if data.get("confidence") is not None else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    try:
        return RelationFrame(
            canonical_name=canonical,
            aliases=aliases,
            directionality=_enum("directionality", "unknown"),
            temporal_kind=_enum("temporal_kind", "unknown"),
            cardinality=_enum("cardinality", "unknown"),
            contradiction_policy=_enum("contradiction_policy", "uncertain"),
            inverse_name=inverse,
            subject_role=str(subject_role_raw).strip() or None
                if isinstance(subject_role_raw, str) else None,
            object_role=str(object_role_raw).strip() or None
                if isinstance(object_role_raw, str) else None,
            confidence=max(0.0, min(1.0, confidence)),
        )
    except Exception:
        return None


def _parse_contradictions(raw: str, n_existing: int) -> list[int]:
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        return []
    idxs = data.get("invalidated_indexes") if isinstance(data, dict) else None
    if not isinstance(idxs, list):
        return []
    return [int(i) for i in idxs if isinstance(i, int) and 0 <= i < n_existing]


def _build_contradiction_user(
    new_fact: str,
    existing_facts: list[str],
    *,
    candidates: list[ContradictionCandidate] | None = None,
    new_fact_struct: ExtractedFact | None = None,
) -> str:
    """Render the contradiction-detection user prompt.

    When structured ``candidates`` (and optionally ``new_fact_struct``)
    are provided, the prompt includes per-candidate subject/predicate/
    object/domain lines so the model can apply the same-domain rule
    without guessing from natural-language text alone.
    """

    parts: list[str] = []
    if new_fact_struct is not None:
        parts.append(
            "NEW FACT:\n"
            f"  text:      {new_fact}\n"
            f"  subject:   {new_fact_struct.subject}\n"
            f"  predicate: {new_fact_struct.predicate}\n"
            f"  object:    {new_fact_struct.object}\n"
            f"  domain:    {new_fact_struct.domain or '<none>'}\n"
            f"  operation: {new_fact_struct.operation}"
        )
    else:
        parts.append(f"NEW FACT: {new_fact}")

    if candidates:
        rendered = []
        for idx, cand in enumerate(candidates):
            rendered.append(
                f"  {idx}.\n"
                f"     text:      {cand.fact}\n"
                f"     subject:   {cand.subject}\n"
                f"     predicate: {cand.predicate}\n"
                f"     object:    {cand.object}\n"
                f"     domain:    {cand.domain or '<none>'}"
            )
        parts.append("PRIOR FACTS (indexed):\n" + "\n".join(rendered))
    else:
        parts.append(
            "PRIOR FACTS (indexed):\n"
            + "\n".join(f"  {i}. {f}" for i, f in enumerate(existing_facts))
        )
    return "\n\n".join(parts)


def _build_frame_classification_user(
    *,
    predicate: str,
    source_span: str = "",
    sample_subject: str = "",
    sample_object: str = "",
) -> str:
    """Render the user message for ``FRAME_CLASSIFICATION_SYSTEM``.

    Compact on purpose: the system prompt carries the entire schema; the
    user message just supplies the unknown predicate plus a single
    grounded example so the model can pick the right axes.
    """
    lines = [f"PREDICATE: {predicate}"]
    if source_span:
        lines.append(f"SOURCE SPAN: {source_span}")
    if sample_subject or sample_object:
        lines.append(
            f"EXAMPLE TRIPLE: ({sample_subject or '<subject>'}) "
            f"-[{predicate}]-> ({sample_object or '<object>'})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAILLMClient(LLMClient):
    """LLM client backed by OpenAI's chat completions API.

    Uses JSON mode (``response_format={"type": "json_object"}``) so the
    model is forced to return parseable JSON.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body
        if client is not None:
            self._client = client
            return
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise SurritiConfigError(
                "Install 'surriti[openai]' to use OpenAILLMClient."
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SurritiConfigError(
                "OPENAI_API_KEY is not set and no api_key argument was provided."
            )
        self._client = AsyncOpenAI(api_key=key)

    async def _complete(self, system: str, user: str) -> str:
        try:
            kwargs: dict = dict(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise SurritiLLMError(f"OpenAI API call failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    async def extract(
        self,
        content: str,
        *,
        group_id: str = "",
        entity_types: dict[str, type] | None = None,
        custom_instructions: str | None = None,
        context: str | None = None,
    ) -> ExtractionResult:
        user = _build_extraction_user(
            content,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_instructions,
            context=context,
        )
        raw = await self._complete(EXTRACTION_SYSTEM, user)
        return _parse_extraction(raw)

    async def find_contradictions(
        self,
        new_fact: str,
        existing_facts: list[str],
        *,
        candidates: list[ContradictionCandidate] | None = None,
        new_fact_struct: ExtractedFact | None = None,
    ) -> list[int]:
        if not existing_facts:
            return []
        user = _build_contradiction_user(
            new_fact,
            existing_facts,
            candidates=candidates,
            new_fact_struct=new_fact_struct,
        )
        raw = await self._complete(CONTRADICTION_SYSTEM, user)
        return _parse_contradictions(raw, len(existing_facts))

    async def classify_relation_frame(
        self,
        *,
        predicate: str,
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ) -> RelationFrame | None:
        user = _build_frame_classification_user(
            predicate=predicate,
            source_span=source_span,
            sample_subject=sample_subject,
            sample_object=sample_object,
        )
        try:
            raw = await self._complete(FRAME_CLASSIFICATION_SYSTEM, user)
        except SurritiLLMError:
            return None
        return _parse_frame_classification(raw, fallback_predicate=predicate)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicLLMClient(LLMClient):
    """LLM client backed by Anthropic's Messages API.

    Anthropic does not have a built-in JSON mode flag; we instead instruct
    the model to return JSON only and parse the response, stripping any
    accidental code fences.
    """

    def __init__(
        self,
        *,
        model: str = "claude-3-5-haiku-latest",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise SurritiConfigError(
                "Install 'surriti[anthropic]' to use AnthropicLLMClient."
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SurritiConfigError(
                "ANTHROPIC_API_KEY is not set and no api_key argument was provided."
            )
        self._client = AsyncAnthropic(api_key=key)

    async def _complete(self, system: str, user: str) -> str:
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise SurritiLLMError(f"Anthropic API call failed: {exc}") from exc
        chunks = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks).strip()

    async def extract(
        self,
        content: str,
        *,
        group_id: str = "",
        entity_types: dict[str, type] | None = None,
        custom_instructions: str | None = None,
        context: str | None = None,
    ) -> ExtractionResult:
        user = _build_extraction_user(
            content,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_instructions,
            context=context,
        )
        raw = await self._complete(EXTRACTION_SYSTEM, user)
        return _parse_extraction(raw)

    async def find_contradictions(
        self,
        new_fact: str,
        existing_facts: list[str],
        *,
        candidates: list[ContradictionCandidate] | None = None,
        new_fact_struct: ExtractedFact | None = None,
    ) -> list[int]:
        if not existing_facts:
            return []
        user = _build_contradiction_user(
            new_fact,
            existing_facts,
            candidates=candidates,
            new_fact_struct=new_fact_struct,
        )
        raw = await self._complete(CONTRADICTION_SYSTEM, user)
        return _parse_contradictions(raw, len(existing_facts))

    async def classify_relation_frame(
        self,
        *,
        predicate: str,
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ) -> RelationFrame | None:
        user = _build_frame_classification_user(
            predicate=predicate,
            source_span=source_span,
            sample_subject=sample_subject,
            sample_object=sample_object,
        )
        try:
            raw = await self._complete(FRAME_CLASSIFICATION_SYSTEM, user)
        except SurritiLLMError:
            return None
        return _parse_frame_classification(raw, fallback_predicate=predicate)


__all__ = [
    "AnthropicLLMClient",
    "OpenAILLMClient",
]
