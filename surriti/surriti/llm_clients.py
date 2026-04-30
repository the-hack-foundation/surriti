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
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
    LLMClient,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """\
You are a knowledge-graph extractor. Read the user's text and return STRICT \
JSON with two arrays:

{
  "entities": [
    {"name": "<a real name from the input>", "labels": ["Person"], "summary": "..."}
  ],
  "facts": [
    {"subject": "<entity name>", "predicate": "snake_case_verb",
     "object":  "<entity name>",
     "fact":    "A complete natural-language sentence."}
  ]
}

WHAT COUNTS AS A FACT (extract these):
- Self-introductions: "my name is X" / "I'm X" / "call me X" -> a fact
  using the speaker as subject and predicate `is_named`.
- Properties of the speaker: "I am 5 months old" -> `is_age`;
  "I am a baby" -> `is_a`; "I work at Acme" -> `works_at`;
  "I live in Berlin" -> `lives_in`; "I like pizza" -> `likes`;
  "I am learning to crawl" -> `is_learning`;
  "my birthday is October 14" -> `has_birthday`;
  "I will be 33" -> `is_age`.
- Properties of other named entities: "Alice works at Acme" -> a fact.
- Compound sentences: "my name is Auley and I am 5 months old" -> TWO
  facts (one `is_named`, one `is_age`).
- VALUES become entities too: dates ("October 14"), ages ("33"),
  places ("Berlin"), companies ("Acme") must appear in the `entities`
  array AND be used as the `object` of the fact. NEVER use a literal
  placeholder like "speaker" or "value" as subject or object.
- When in doubt, extract.

WHAT TO SKIP (return no fact for these, but mention any named entities):
- Pure interjections with no claim: "hi", "thanks", "ok", "lol", "hmm".
- Pure questions: "where do I work?", "what's my name?".

HARD RULES (violations make the output unusable):
- NEVER invent entities, predicates, or relations not supported by the
  input. Do NOT use placeholder names like Alice, Bob, Acme, Foo, Bar
  unless they appear in the text.
- NEVER emit a self-loop fact (subject == object). "Michael knows
  Michael" is FORBIDDEN. For naming, the subject is the SPEAKER (see the
  speaker context appended below) and the object is the new name --
  they are different entities.
- Tokens that look like internal metadata -- bracketed labels (`[chat]`,
  `[turn-a]`), bare UUIDs -- are NOT entities. Ignore them entirely.
- Use the EXACT entity name strings inside facts (subject/object).
- Predicates are snake_case verbs (is_named, is_a, is_age, works_at,
  lives_in, likes, knows, is_brother_of, ...). Avoid `related_to`.
- Each fact's `fact` is a complete natural-language sentence.
- Return ONLY the JSON object, no commentary, no markdown fences.
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
) -> str:
    parts = [f"Episode (group_id={group_id!r}):", content.strip()]
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
                    )
                )

    return ExtractionResult(entities=entities, facts=facts)


def _parse_contradictions(raw: str, n_existing: int) -> list[int]:
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        return []
    idxs = data.get("invalidated_indexes") if isinstance(data, dict) else None
    if not isinstance(idxs, list):
        return []
    return [int(i) for i in idxs if isinstance(i, int) and 0 <= i < n_existing]


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
    ) -> ExtractionResult:
        user = _build_extraction_user(
            content,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_instructions,
        )
        raw = await self._complete(EXTRACTION_SYSTEM, user)
        return _parse_extraction(raw)

    async def find_contradictions(
        self, new_fact: str, existing_facts: list[str]
    ) -> list[int]:
        if not existing_facts:
            return []
        user = (
            f"NEW FACT: {new_fact}\n\nPRIOR FACTS (indexed):\n"
            + "\n".join(f"  {i}. {f}" for i, f in enumerate(existing_facts))
        )
        raw = await self._complete(CONTRADICTION_SYSTEM, user)
        return _parse_contradictions(raw, len(existing_facts))


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
    ) -> ExtractionResult:
        user = _build_extraction_user(
            content,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_instructions,
        )
        raw = await self._complete(EXTRACTION_SYSTEM, user)
        return _parse_extraction(raw)

    async def find_contradictions(
        self, new_fact: str, existing_facts: list[str]
    ) -> list[int]:
        if not existing_facts:
            return []
        user = (
            f"NEW FACT: {new_fact}\n\nPRIOR FACTS (indexed):\n"
            + "\n".join(f"  {i}. {f}" for i, f in enumerate(existing_facts))
        )
        raw = await self._complete(CONTRADICTION_SYSTEM, user)
        return _parse_contradictions(raw, len(existing_facts))


__all__ = [
    "AnthropicLLMClient",
    "OpenAILLMClient",
]
