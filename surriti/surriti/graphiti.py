"""Top-level Surriti class - mirrors the public surface of ``graphiti.Graphiti``."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from surriti.driver import SurrealDriver
from surriti.edges import CommunityEdge, EntityEdge, EpisodicEdge, make_fact_key
from surriti.embedder import DummyEmbedder, EmbedderClient
from surriti.llm import (
    DummyLLMClient,
    ExtractedEntity,
    ExtractedFact,
    LLMClient,
)
from surriti.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from surriti.relation_frames import (
    RelationFrame,
    RelationFrameRegistry,
    make_slot_key,
    normalize_symmetric,
    qualifier_hash,
)
from surriti.rerankers import CrossEncoderClient
from surriti.search import (
    SearchConfig,
    SearchResults,
    hybrid_search,
    search_communities,
    search_episodes,
    search_nodes,
)
from surriti.search_filters import SearchFilters
from surriti.temporal import invalidate_edges, resolve_contradictions
from surriti.utils import parse_community, parse_edge, parse_entity, parse_episode
from surriti.validators import IDENTITY_PREDICATES, repair_fact

logger = logging.getLogger(__name__)


def _entity_name_key(name: str) -> str:
    """High-confidence entity key: case-insensitive, whitespace-normalized only."""

    return " ".join(str(name or "").casefold().split())


@dataclass
class AddEpisodeResults:
    episode: EpisodicNode
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    invalidated_edges: list[EntityEdge] = field(default_factory=list)
    communities: list[CommunityNode] = field(default_factory=list)
    community_edges: list[CommunityEdge] = field(default_factory=list)


@dataclass
class AddBulkEpisodeResults:
    episodes: list[EpisodicNode] = field(default_factory=list)
    episodic_edges: list[EpisodicEdge] = field(default_factory=list)
    nodes: list[EntityNode] = field(default_factory=list)
    edges: list[EntityEdge] = field(default_factory=list)
    invalidated_edges: list[EntityEdge] = field(default_factory=list)
    communities: list[CommunityNode] = field(default_factory=list)
    community_edges: list[CommunityEdge] = field(default_factory=list)


@dataclass
class AddTripletResults:
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    invalidated_edges: list[EntityEdge] = field(default_factory=list)


@dataclass
class RawEpisode:
    """Lightweight container for :meth:`Surriti.add_episode_bulk` items."""

    name: str
    content: str
    source: EpisodeType = EpisodeType.message
    source_description: str = ""
    reference_time: datetime | None = None
    group_id: str | None = None
    uuid: str | None = None


@dataclass
class MemoryContext:
    """Result of :meth:`Surriti.recall` -- a query-focused memory bundle.

    Render any subset directly into a prompt:
    ``profiles`` are the dossier paragraphs for entities the query
    mentions, ``facts`` are the most relevant edges (already
    ego-filtered when entities resolved), and ``episodes`` /
    ``communities`` are populated only at ``depth="deep"``.
    ``resolved_entities`` lets callers explain or audit *why* a given
    profile / fact appears.
    """

    query: str
    profiles: list[EntityNode] = field(default_factory=list)
    facts: list[EntityEdge] = field(default_factory=list)
    episodes: list[EpisodicNode] = field(default_factory=list)
    communities: list[CommunityNode] = field(default_factory=list)
    resolved_entities: list[dict] = field(default_factory=list)
    traits: list[EntityNode] = field(default_factory=list)
    """Synthesized trait entities tied to the resolved subjects.
    Populated by the cognitive layer; empty when ``cognition=False``."""
    goals: list[EntityNode] = field(default_factory=list)
    """Active goal entities tied to the resolved subjects."""
    prediction: dict | None = None
    """Per-group prediction bundle (likely topics / preferences /
    questions) emitted by the cognitive layer. Only populated at
    ``depth='deep'``."""


class Surriti:
    """Graphiti-compatible facade backed by SurrealDB.

    Parameters
    ----------
    driver:
        A connected (or to-be-connected) :class:`SurrealDriver`.
    llm_client:
        Implementation of :class:`~surriti.llm.LLMClient`. Defaults to the
        offline :class:`DummyLLMClient` so the API is usable without keys.
    embedder:
        Implementation of :class:`~surriti.embedder.EmbedderClient`. Defaults
        to :class:`DummyEmbedder`.
    """

    def __init__(
        self,
        driver: SurrealDriver,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        *,
        relation_frames: RelationFrameRegistry | None = None,
        seed_default_frames: bool = True,
        alias_resolution: bool = True,
        alias_resolution_threshold: float = 0.86,
        alias_resolution_llm: bool = True,
        profile_refresh: str = "async",
        profile_summary_max_facts: int = 30,
        cognition: "CognitionConfig | bool" = True,
    ) -> None:
        from surriti.cognition import CognitionConfig

        self.driver = driver
        self.llm = llm_client or DummyLLMClient()
        self.embedder = embedder or DummyEmbedder(embedding_dim=driver.embedding_dim)
        self.cross_encoder = cross_encoder
        # Cognitive abstraction layer. ``True`` enables the default
        # config; ``False`` disables it entirely; pass a ``CognitionConfig``
        # for fine-grained tuning. The scheduler itself is created lazily
        # in ``connect()`` so a never-connected ``Surriti`` stays cheap.
        if isinstance(cognition, bool):
            self._cognition_config = CognitionConfig(enabled=cognition)
        else:
            self._cognition_config = cognition
        self._cognition: "CognitionScheduler | None" = None
        # Alias resolution / dossier knobs. ``alias_resolution`` gates the
        # whole layered pipeline; ``profile_refresh`` controls when
        # ``profiles.refresh_entity_profiles`` runs after add_episode.
        # All defaults are tuned to "good for production" -- opt out only
        # for cost/latency-sensitive paths.
        if profile_refresh not in ("sync", "async", "off"):
            raise ValueError(
                "profile_refresh must be one of 'sync', 'async', 'off'"
            )
        self.alias_resolution_enabled = alias_resolution
        self.alias_resolution_threshold = float(alias_resolution_threshold)
        self.alias_resolution_llm = alias_resolution_llm
        self.profile_refresh_mode = profile_refresh
        self.profile_summary_max_facts = int(profile_summary_max_facts)
        # Generalized relation-frame layer. Defaults are seeded so brand
        # new deployments get spouse-direction collapse, current-location
        # supersession, and identity handling without paying an LLM
        # round-trip on first encounter. Pass ``relation_frames=...`` to
        # supply a pre-built registry (useful for sharing across
        # ``Surriti`` instances) or ``seed_default_frames=False`` to
        # opt out entirely. When no registry is passed and the LLM
        # client implements ``classify_relation_frame``, we wire it in
        # automatically so unknown predicates dynamically mint frames.
        if relation_frames is None:
            self.relation_frames = RelationFrameRegistry(
                seed_defaults=seed_default_frames,
                llm_classifier=self._llm_frame_classifier,
            )
        else:
            self.relation_frames = relation_frames

    async def _llm_frame_classifier(
        self,
        *,
        predicate: str,
        source_span: str = "",
        sample_subject: str = "",
        sample_object: str = "",
    ) -> RelationFrame | None:
        """Bridge between the registry's classifier hook and the LLM
        adapter. Returns ``None`` (and the registry falls through) when
        the adapter does not override the default no-op.
        """
        return await self.llm.classify_relation_frame(
            predicate=predicate,
            source_span=source_span,
            sample_subject=sample_subject,
            sample_object=sample_object,
        )

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_env(
        cls,
        *,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        relation_frames: RelationFrameRegistry | None = None,
        seed_default_frames: bool = True,
        alias_resolution: bool = True,
        alias_resolution_threshold: float = 0.86,
        alias_resolution_llm: bool = True,
        profile_refresh: str = "async",
        profile_summary_max_facts: int = 30,
        cognition: "CognitionConfig | bool" = True,
    ) -> "Surriti":
        """Build a Surriti instance from environment variables.

        Reads:

        - ``SURRITI_SURREAL_URL``  (default ``ws://localhost:8000/rpc``)
        - ``SURRITI_SURREAL_NS``   (default ``surriti``)
        - ``SURRITI_SURREAL_DB``   (default ``surriti``)
        - ``SURRITI_SURREAL_USER`` / ``SURRITI_SURREAL_PASS``
        - ``SURRITI_EMBEDDING_DIM`` (default ``768``)

        The returned instance is *not* connected. Use it inside
        ``async with`` or call ``await surriti.connect()`` first.
        """

        driver = SurrealDriver.from_env()
        return cls(
            driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            relation_frames=relation_frames,
            seed_default_frames=seed_default_frames,
            alias_resolution=alias_resolution,
            alias_resolution_threshold=alias_resolution_threshold,
            alias_resolution_llm=alias_resolution_llm,
            profile_refresh=profile_refresh,
            profile_summary_max_facts=profile_summary_max_facts,
            cognition=cognition,
        )

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> "Surriti":
        """Connect the underlying driver and apply the schema. Idempotent."""

        if hasattr(self.driver, "connect"):
            await self.driver.connect()
        if hasattr(self.driver, "init_schema"):
            await self.driver.init_schema()
        if self._cognition is None:
            from surriti.cognition import CognitionScheduler

            self._cognition = CognitionScheduler(
                driver=self.driver,
                llm=self.llm,
                embedder=self.embedder,
                config=self._cognition_config,
            )
            self._cognition.start()
        return self

    async def close(self) -> None:
        """Close the underlying driver. Safe to call multiple times."""

        if self._cognition is not None:
            try:
                await self._cognition.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("cognition shutdown failed")
            self._cognition = None
        if hasattr(self.driver, "close"):
            await self.driver.close()

    async def __aenter__(self) -> "Surriti":
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------------------------------------------------------------ schema
    async def build_indices_and_constraints(self) -> None:
        """Initialise SurrealDB schema. Idempotent."""

        await self.driver.init_schema()

    # ------------------------------------------------------------------ ingest
    async def add_episode(
        self,
        *,
        name: str,
        episode_body: str,
        source: EpisodeType = EpisodeType.message,
        source_description: str = "",
        reference_time: datetime | None = None,
        group_id: str = "",
        uuid: str | None = None,
        update_communities: bool = False,
        excluded_entity_types: list[str] | None = None,
        entity_types: dict[str, type] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, type] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: Any | None = None,
        saga_previous_episode_uuid: str | None = None,
        speaker_id: str | None = None,
        speaker_name: str | None = None,
        source_type: str = "user",
    ) -> AddEpisodeResults:
        """Process a new episode end-to-end.

        Steps: persist episode -> LLM extraction -> upsert entities ->
        create/dedupe RELATES_TO edges -> contradiction resolution ->
        link MENTIONS edges -> optionally refresh communities.

        Parameters mirror Graphiti's ``add_episode``:

        - ``uuid``: pre-assigned episode UUID (otherwise auto-generated).
        - ``update_communities``: rebuild communities for ``group_id`` after ingest.
        - ``excluded_entity_types``: drop extracted entities labelled with any of these.
        - ``entity_types``: name -> Pydantic model. Supplies labels and an
          optional ``description`` attribute used by the LLM. The ``DummyLLM``
          uses these as a hint for label assignment via case-insensitive name
          containment.
        - ``previous_episode_uuids``: include text from these episodes as
          extra context to the LLM extractor.
        - ``custom_extraction_instructions``: free-form string passed to the
          LLM extractor.
        """

        ref_time = reference_time or datetime.now(timezone.utc)

        # Graphiti supplies recent episodes as extraction context by default.
        # Surriti keeps the prompt compact by passing only prior content,
        # never episode names/metadata, so small models do not extract turn
        # labels. As of the temporal-state refactor we also pass the prior
        # text in a SEPARATE ``context`` channel rather than concatenating
        # it onto the current episode -- this stops small models from
        # re-extracting facts that were already persisted from earlier
        # turns.
        if previous_episode_uuids is None:
            ctx = await self._fetch_recent_episode_contents(
                reference_time=ref_time,
                group_id=group_id,
                source=source,
            )
        else:
            ctx = await self._fetch_episode_contents(previous_episode_uuids)

        episode = EpisodicNode(
            uuid=uuid or str(uuid4()),
            name=name,
            content=episode_body,
            source=source,
            source_description=source_description,
            reference_time=ref_time,
            group_id=group_id,
        )
        await self._save_episode(episode)

        # Ensure the speaker has a canonical User entity in this tenant and
        # tell the extractor to anchor first-person pronouns to it.
        if speaker_id:
            await self.upsert_user(
                group_id=group_id,
                user_id=speaker_id,
                display_name=speaker_name,
            )
            if speaker_name:
                speaker_hint = (
                    f"(Speaker context: \"I\"/\"me\"/\"my\"/\"mine\" "
                    f"refer to \"{speaker_name}\" (stable id "
                    f"\"{speaker_id}\"). Use \"{speaker_name}\" as the "
                    f"subject for facts about the speaker. The OBJECT "
                    f"must be the actual value mentioned in the text -- "
                    f"never the literal word \"speaker\" or "
                    f"\"{speaker_name}\" again. Examples: "
                    f"\"I work at Acme\" -> "
                    f"`{speaker_name} -[works_at]-> Acme`; "
                    f"\"my birthday is October 14\" -> "
                    f"`{speaker_name} -[has_birthday]-> October 14`; "
                    f"\"I am 33 years old\" -> "
                    f"`{speaker_name} -[is_age]-> 33`. Add the value "
                    f"(Acme, October 14, 33) to the entities list too. "
                    f"\n\n"
                    f"CRITICAL — third-party subjects: when the "
                    f"sentence's grammatical subject is a NAMED entity "
                    f"(a person, pet, place, object) rather than a "
                    f"first-person pronoun, KEEP that named subject -- "
                    f"DO NOT substitute \"{speaker_name}\". The speaker "
                    f"swap applies ONLY to first-person pronouns "
                    f"(I/me/my/mine). Likewise, resolve third-person "
                    f"pronouns (he/she/they/him/her/them/his/hers) to "
                    f"the most recently mentioned named entity from "
                    f"CONTEXT, not to the speaker. Anti-examples: "
                    f"\"Pixel is allergic to chicken\" -> "
                    f"`Pixel -[allergic_to]-> chicken` (NOT "
                    f"`{speaker_name} -[allergic_to]-> chicken`); "
                    f"\"Mango hates thunderstorms\" -> "
                    f"`Mango -[hates]-> thunderstorms` (NOT "
                    f"`{speaker_name} -[hates]-> thunderstorms`); "
                    f"\"My sister Ava lives in Denver\" -> "
                    f"`Ava -[lives_in]-> Denver` AND "
                    f"`{speaker_name} -[has_sister]-> Ava` (two facts; "
                    f"Ava is the subject of the residence claim, NOT "
                    f"{speaker_name}); \"The vet cleared him\" "
                    f"following a sentence about Pixel -> "
                    f"`Pixel -[was_cleared_by]-> vet` (the pronoun "
                    f"\"him\" resolves to Pixel, NOT to "
                    f"{speaker_name}).)"
                )
            else:
                speaker_hint = (
                    f"(Speaker context: \"I\"/\"me\"/\"my\" refer to the "
                    f"entity with stable id \"{speaker_id}\". Use "
                    f"\"{speaker_id}\" as the subject for facts about the "
                    f"speaker. The OBJECT must be the actual value "
                    f"mentioned in the text -- never the literal word "
                    f"\"speaker\" or \"{speaker_id}\" again. Examples: "
                    f"\"my name is Auley\" -> "
                    f"`{speaker_id} -[is_named]-> Auley`; "
                    f"\"I am 5 months old\" -> "
                    f"`{speaker_id} -[is_age]-> 5 months`; "
                    f"\"I am a baby\" -> `{speaker_id} -[is_a]-> baby`. "
                    f"Add the value (Auley, 5 months, baby) to the "
                    f"entities list too."
                    f"\n\nCRITICAL -- third-party subjects: when the "
                    f"sentence's grammatical subject is a NAMED entity "
                    f"(a person, pet, place, object) rather than a "
                    f"first-person pronoun, KEEP that named subject -- "
                    f"DO NOT substitute \"{speaker_id}\". Pronouns "
                    f"he/she/they/him/her resolve to the most recently "
                    f"mentioned named entity from CONTEXT, not to the "
                    f"speaker. Anti-examples: \"Pixel is allergic to "
                    f"chicken\" -> `Pixel -[is_allergic_to]-> chicken` "
                    f"(NOT `{speaker_id} -[is_allergic_to]-> chicken`); "
                    f"\"Mango hates thunderstorms\" -> "
                    f"`Mango -[hates]-> thunderstorms`; \"My sister Ava "
                    f"lives in Denver\" -> `Ava -[lives_in]-> Denver` "
                    f"AND `{speaker_id} -[has_sister]-> Ava` (two facts; "
                    f"Ava is the subject of the residence claim, NOT "
                    f"{speaker_id}); \"The vet cleared him\" following "
                    f"a sentence about Pixel -> "
                    f"`Pixel -[was_cleared_by]-> vet` (the pronoun "
                    f"\"him\" resolves to Pixel, NOT to "
                    f"{speaker_id}).)"
                )
            custom_extraction_instructions = (
                f"{custom_extraction_instructions}\n\n{speaker_hint}"
                if custom_extraction_instructions
                else speaker_hint
            )

        extraction = await self.llm.extract(
            episode_body,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_extraction_instructions,
            context=ctx or None,
        )
        logger.debug(
            "extract -> entities=%r facts=%r",
            [(e.name, e.labels) for e in extraction.entities],
            [(f.subject, f.predicate, f.object) for f in extraction.facts],
        )

        # Apply entity type label hinting (DummyLLM doesn't, so we do it here too).
        if entity_types:
            type_names = list(entity_types.keys())
            for ent in extraction.entities:
                lname = ent.name.lower()
                hits = [t for t in type_names if t.lower() in lname]
                if hits:
                    ent.labels = sorted(set((ent.labels or []) + hits))
        if edge_types:
            allowed = set(edge_types)
            extraction.facts = [
                f for f in extraction.facts if not f.predicate or f.predicate in allowed
            ]
        if edge_type_map:
            extraction.facts = self._filter_facts_by_edge_type_map(
                extraction.entities, extraction.facts, edge_type_map
            )
        if excluded_entity_types:
            excluded = set(excluded_entity_types)
            extraction.entities = [
                e for e in extraction.entities if not (excluded & set(e.labels or []))
            ]
            keep_names = {e.name for e in extraction.entities}
            extraction.facts = [
                f for f in extraction.facts
                if f.subject in keep_names and f.object in keep_names
            ]

        entities = await self._upsert_entities(
            extraction.entities,
            group_id=group_id,
            episode_uuid=episode.uuid,
            episode_context=episode_body,
        )
        entity_by_key = {_entity_name_key(e.name): e for e in entities}
        name_to_entity = {e.name: e for e in entities}
        for ext in extraction.entities:
            ent = entity_by_key.get(_entity_name_key(ext.name))
            if ent is not None:
                name_to_entity[ext.name] = ent

        # Auto-resolve the speaker so the LLM can use either the stable id
        # (`default`) or the display name (`Michael`) as a subject without
        # having to list it in the entities array. Without this, every
        # `Michael -[works_at]-> Acme` from a turn that didn't echo the
        # speaker name silently becomes an unresolved-entity drop.
        if speaker_id and speaker_id not in name_to_entity:
            speaker_ents = await self._upsert_entities(
                [ExtractedEntity(name=speaker_id, labels=["User"])],
                group_id=group_id,
            )
            name_to_entity[speaker_id] = speaker_ents[0]
            if speaker_ents[0].uuid not in {e.uuid for e in entities}:
                entities.append(speaker_ents[0])
        if speaker_name and speaker_name not in name_to_entity:
            speaker_ents = await self._upsert_entities(
                [ExtractedEntity(name=speaker_name, labels=["Person"])],
                group_id=group_id,
            )
            name_to_entity[speaker_name] = speaker_ents[0]
            if speaker_ents[0].uuid not in {e.uuid for e in entities}:
                entities.append(speaker_ents[0])

        # Predicates that *legitimately* connect an entity to itself
        # (identity/aliasing). Imported from the shared validator so the
        # add-episode loop and the standalone validator agree.
        _IDENTITY_PREDICATES = IDENTITY_PREDICATES

        edges: list[EntityEdge] = []
        invalidated_all: list[EntityEdge] = []
        for fact in extraction.facts:
            # Run the deterministic post-extraction repair pass first.
            # This normalises predicates, drops banned placeholder
            # objects ("lives_in world"), and rewrites identity
            # self-loops to the speaker's stable id when possible.
            original_subject = fact.subject
            repaired = repair_fact(
                fact,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
            )
            if repaired is None:
                logger.debug(
                    "Validator dropped fact %r -[%s]-> %r",
                    original_subject, fact.predicate, fact.object,
                )
                continue
            fact = repaired

            # If the validator rewrote the subject to the speaker_id
            # (identity self-loop repair) we need to make sure that
            # entity exists in this episode's name->entity map.
            if (
                speaker_id
                and fact.subject == speaker_id
                and speaker_id != original_subject
                and fact.subject not in name_to_entity
            ):
                speaker_ents = await self._upsert_entities(
                    [ExtractedEntity(name=speaker_id, labels=["User"])],
                    group_id=group_id,
                )
                name_to_entity[speaker_id] = speaker_ents[0]
                if speaker_ents[0].uuid not in {e.uuid for e in entities}:
                    entities.append(speaker_ents[0])

            subject = name_to_entity.get(fact.subject)
            obj = name_to_entity.get(fact.object)
            if subject is None or obj is None:
                logger.debug(
                    "Skipping fact: unresolved entities subj=%r(found=%s) "
                    "obj=%r(found=%s) names_known=%r",
                    fact.subject, subject is not None,
                    fact.object, obj is not None,
                    list(name_to_entity.keys()),
                )
                continue
            predicate = fact.predicate
            if subject.uuid == obj.uuid and predicate not in _IDENTITY_PREDICATES:
                # Identity-predicate self-loops without a repair path
                # are kept by the validator; everything else with
                # subject==object after repair is unsalvageable.
                logger.debug(
                    "Skipping post-repair self-loop %r -[%s]-> %r",
                    fact.subject, fact.predicate, fact.object,
                )
                continue
            op = (fact.operation or "assert").lower()
            if op == "noop":
                continue
            if op == "terminate":
                # Locate any active edge matching (subject, predicate, obj)
                # in this group and close it. No new edge is inserted.
                terminated = await self._terminate_matching_edge(
                    group_id=group_id,
                    subject_uuid=subject.uuid,
                    object_uuid=obj.uuid,
                    predicate=fact.predicate,
                    invalid_at=ref_time,
                )
                invalidated_all.extend(terminated)
                continue
            # "assert", "correct", and "qualify" all insert a new edge.
            # "correct" is treated as singleton-asserted regardless of
            # the LLM's flag, so the prior matching value is closed
            # deterministically. "qualify" inserts without closing peers
            # -- the qualifier hash naturally puts it in a distinct slot.
            if op == "correct":
                fact.singleton = True
            edge, invalidated = await self._add_fact_edge(
                fact=fact,
                subject=subject,
                obj=obj,
                episode=episode,
                group_id=group_id,
                source_type=source_type,
            )
            edges.append(edge)
            invalidated_all.extend(invalidated)

        # Track entity_edges on the episode for downstream lookups.
        if edges:
            episode.entity_edges = [e.uuid for e in edges]
            await self.driver.query(
                "UPDATE episode SET entity_edges = $ee WHERE uuid = $u;",
                {"ee": episode.entity_edges, "u": episode.uuid},
            )

        episodic_edges = await self._link_mentions(
            episode=episode, entities=entities, group_id=group_id
        )

        communities: list[CommunityNode] = []
        community_edges: list[CommunityEdge] = []
        if update_communities:
            communities, community_edges = await self.build_communities(
                group_id=group_id
            )

        # Touched-only profile refresh. ``async`` schedules a fire-and-forget
        # task so the caller's add_episode latency stays unchanged; ``sync``
        # blocks for callers that want guaranteed-fresh dossiers (tests,
        # eval runs); ``off`` skips entirely. Either way, we only refresh
        # entities this episode actually touched.
        if entities and self.profile_refresh_mode != "off":
            from surriti.profiles import refresh_entity_profiles

            entity_uuids = [e.uuid for e in entities if e.uuid]
            if entity_uuids:
                coro = refresh_entity_profiles(
                    driver=self.driver,
                    embedder=self.embedder,
                    llm=self.llm,
                    group_id=group_id,
                    entity_uuids=entity_uuids,
                    max_facts=self.profile_summary_max_facts,
                )
                if self.profile_refresh_mode == "sync":
                    await coro
                else:
                    asyncio.create_task(coro)

        # Fire-and-forget cognitive abstraction pass. The scheduler
        # debounces and batches; failures inside the pass are logged
        # but never raised. Wrap defensively so an unexpected scheduler
        # bug can never break ingest.
        if self._cognition is not None and self._cognition.enabled:
            try:
                self._cognition.notify(group_id, episode.uuid)
            except Exception:  # noqa: BLE001
                logger.exception("cognition notify raised; ignoring")

        return AddEpisodeResults(
            episode=episode,
            episodic_edges=episodic_edges,
            nodes=entities,
            edges=edges,
            invalidated_edges=invalidated_all,
            communities=communities,
            community_edges=community_edges,
        )

    async def add_triplet(
        self,
        source_node: "EntityNode | str | None" = None,
        edge: "EntityEdge | None" = None,
        target_node: "EntityNode | str | None" = None,
        *,
        # Convenience kwargs (Surriti-specific shorthand)
        subject_name: str | None = None,
        predicate: str | None = None,
        object_name: str | None = None,
        fact: str | None = None,
        group_id: str = "",
        valid_at: datetime | None = None,
    ) -> AddTripletResults:
        """Insert a single (subject, predicate, object) fact.

        Two calling conventions are supported:

        1. **Graphiti-style** (preferred):
           ``add_triplet(source_node=EntityNode(...), edge=EntityEdge(...), target_node=EntityNode(...))``.
        2. **Convenience**: ``add_triplet(subject_name=..., predicate=..., object_name=..., fact=...)``.

        Existing entities are reused when their ``(group_id, name)`` matches.
        """

        if isinstance(source_node, EntityNode) and isinstance(target_node, EntityNode) and edge is not None:
            grp = source_node.group_id or edge.group_id or group_id
            entities = await self._upsert_entities(
                [
                    ExtractedEntity(name=source_node.name, summary=source_node.summary, labels=source_node.labels),
                    ExtractedEntity(name=target_node.name, summary=target_node.summary, labels=target_node.labels),
                ],
                group_id=grp,
            )
            subject, obj = entities[0], entities[1]
            ref_time = edge.valid_at or valid_at or datetime.now(timezone.utc)
            edge_obj, invalidated = await self._add_fact_edge(
                fact=ExtractedFact(
                    subject=subject.name,
                    predicate=edge.name,
                    object=obj.name,
                    fact=edge.fact or f"{subject.name} {edge.name} {obj.name}.",
                    valid_at=ref_time.isoformat(),
                ),
                subject=subject,
                obj=obj,
                episode=None,
                group_id=grp,
            )
            return AddTripletResults(nodes=entities, edges=[edge_obj], invalidated_edges=invalidated)

        # Convenience path
        if subject_name is None or predicate is None or object_name is None:
            # Allow positional convenience: add_triplet("Alice", "loves", "Bob")
            if isinstance(source_node, str) and isinstance(target_node, str) and isinstance(edge, str):
                subject_name, predicate, object_name = source_node, edge, target_node
            else:
                raise TypeError(
                    "add_triplet requires either (source_node, edge, target_node) objects or "
                    "the (subject_name, predicate, object_name) convenience kwargs"
                )

        ref_time = valid_at or datetime.now(timezone.utc)
        entities = await self._upsert_entities(
            [ExtractedEntity(name=subject_name), ExtractedEntity(name=object_name)],
            group_id=group_id,
        )
        subject, obj = entities[0], entities[1]
        fact_text = fact or f"{subject_name} {predicate} {object_name}."
        edge_obj, invalidated = await self._add_fact_edge(
            fact=ExtractedFact(
                subject=subject_name,
                predicate=predicate,
                object=object_name,
                fact=fact_text,
                valid_at=ref_time.isoformat(),
            ),
            subject=subject,
            obj=obj,
            episode=None,
            group_id=group_id,
        )
        return AddTripletResults(nodes=entities, edges=[edge_obj], invalidated_edges=invalidated)

    async def add_episode_bulk(
        self,
        episodes: "list[dict[str, Any] | RawEpisode]",
        *,
        group_id: str = "",
        update_communities: bool = False,
    ) -> AddBulkEpisodeResults:
        """Process a batch of episodes sequentially.

        Returns a flat :class:`AddBulkEpisodeResults` aggregating the per-episode
        outputs. Order is preserved.
        """

        agg = AddBulkEpisodeResults()
        for ep in episodes:
            if isinstance(ep, RawEpisode):
                kwargs: dict[str, Any] = dict(
                    name=ep.name,
                    episode_body=ep.content,
                    source=ep.source,
                    source_description=ep.source_description,
                    reference_time=ep.reference_time,
                    group_id=ep.group_id or group_id,
                    uuid=ep.uuid,
                )
            else:
                kwargs = dict(
                    name=ep["name"],
                    episode_body=ep.get("episode_body", ep.get("content", "")),
                    source=ep.get("source", EpisodeType.message),
                    source_description=ep.get("source_description", ""),
                    reference_time=ep.get("reference_time"),
                    group_id=ep.get("group_id", group_id),
                    uuid=ep.get("uuid"),
                )
            res = await self.add_episode(**kwargs)
            agg.episodes.append(res.episode)
            agg.episodic_edges.extend(res.episodic_edges)
            agg.nodes.extend(res.nodes)
            agg.edges.extend(res.edges)
            agg.invalidated_edges.extend(res.invalidated_edges)

        if update_communities:
            agg.communities, agg.community_edges = await self.build_communities(
                group_id=group_id
            )
        return agg

    # ------------------------------------------------------------------ delete / cleanup
    async def remove_episode(self, episode_uuid: str) -> None:
        """Delete an episode along with its MENTIONS edges and any RELATES_TO
        edges it was the only source for. Edges that have other supporting
        episodes simply have this UUID dropped from their ``episodes`` array.
        """

        from surriti.search import _unwrap

        edge_rows = await self.driver.query(
            "SELECT * FROM relates_to WHERE $ep IN episodes;",
            {"ep": episode_uuid},
        )
        edges = _unwrap(edge_rows)
        sole_source_uuids = [e["uuid"] for e in edges if list(e.get("episodes") or []) == [episode_uuid]]
        shared_uuids = [e["uuid"] for e in edges if e["uuid"] not in sole_source_uuids]

        if sole_source_uuids:
            await self.driver.query(
                "DELETE relates_to WHERE uuid IN $u;", {"u": sole_source_uuids}
            )
        if shared_uuids:
            await self.driver.query(
                """
                UPDATE relates_to
                SET episodes = array::filter(episodes, |$x| $x != $ep)
                WHERE uuid IN $u;
                """,
                {"u": shared_uuids, "ep": episode_uuid},
            )
        await self.driver.query(
            "DELETE mentions WHERE in = type::record('episode', $ep);",
            {"ep": episode_uuid},
        )
        await self.driver.query(
            "DELETE episode WHERE uuid = $ep;", {"ep": episode_uuid}
        )

    async def delete_group(self, group_id: str) -> None:
        """Remove every record (nodes and edges) associated with ``group_id``."""

        for table in ("mentions", "relates_to", "has_member", "episode", "entity", "community"):
            await self.driver.query(
                f"DELETE {table} WHERE group_id = $g;", {"g": group_id}
            )

    async def remove_node(self, entity_uuid: str) -> None:
        """Delete an entity and any edges it participates in."""

        await self.driver.query(
            """
            DELETE relates_to WHERE in = type::record('entity', $u) OR out = type::record('entity', $u);
            DELETE mentions WHERE out = type::record('entity', $u);
            DELETE has_member WHERE out = type::record('entity', $u);
            DELETE entity WHERE uuid = $u;
            """,
            {"u": entity_uuid},
        )

    # ------------------------------------------------------------------ communities
    async def build_communities(
        self, *, group_id: str
    ) -> tuple[list[CommunityNode], list[CommunityEdge]]:
        """Compute communities via connected components over RELATES_TO edges.

        This is intentionally a simple, deterministic baseline (Graphiti uses
        the Leiden algorithm + LLM summaries). For each component we create a
        :class:`CommunityNode` named after the most-connected entity, and a
        :class:`CommunityEdge` (``has_member``) per member.
        """

        from surriti.search import _unwrap

        # Wipe existing communities for this group.
        await self.driver.query(
            "DELETE has_member WHERE group_id = $g;", {"g": group_id}
        )
        await self.driver.query(
            "DELETE community WHERE group_id = $g;", {"g": group_id}
        )
        node_rows = _unwrap(
            await self.driver.query(
                "SELECT * FROM entity WHERE group_id = $g;", {"g": group_id}
            )
        )
        edge_rows = _unwrap(
            await self.driver.query(
                """
                SELECT
                    record::id(in) AS source_uuid,
                    record::id(out) AS target_uuid
                FROM relates_to
                WHERE group_id = $g AND invalid_at IS NONE;
                """,
                {"g": group_id},
            )
        )
        if not node_rows:
            return [], []

        # Union-Find
        parent: dict[str, str] = {n["uuid"]: n["uuid"] for n in node_rows}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        degree: dict[str, int] = defaultdict(int)
        for e in edge_rows:
            s = e.get("source_uuid")
            t = e.get("target_uuid")
            if s in parent and t in parent:
                union(s, t)
                degree[s] += 1
                degree[t] += 1

        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for n in node_rows:
            clusters[find(n["uuid"])].append(n)

        communities: list[CommunityNode] = []
        community_edges: list[CommunityEdge] = []

        for members in clusters.values():
            if len(members) < 2:
                continue  # singletons aren't communities
            head = max(members, key=lambda m: degree.get(m["uuid"], 0))
            summary_names = ", ".join(sorted(m["name"] for m in members))
            community = CommunityNode(
                name=f"{head['name']} community",
                summary=f"Members: {summary_names}",
                group_id=group_id,
                name_embedding=await self.embedder.create(head["name"]),
            )
            await self.driver.query(
                """
                CREATE type::record("community", $uuid) CONTENT {
                    uuid: $uuid, group_id: $group_id, name: $name,
                    summary: $summary, name_embedding: $emb, created_at: $created_at
                };
                """,
                {
                    "uuid": community.uuid,
                    "group_id": group_id,
                    "name": community.name,
                    "summary": community.summary,
                    "emb": community.name_embedding,
                    "created_at": community.created_at,
                },
            )
            communities.append(community)
            for m in members:
                ce = CommunityEdge(
                    group_id=group_id,
                    source_node_uuid=community.uuid,
                    target_node_uuid=m["uuid"],
                )
                await self.driver.query(
                    """
                    RELATE (type::record("community", $c))->has_member->(type::record("entity", $e))
                    CONTENT { uuid: $uuid, group_id: $group_id, created_at: $created_at };
                    """,
                    {
                        "c": community.uuid,
                        "e": m["uuid"],
                        "uuid": ce.uuid,
                        "group_id": group_id,
                        "created_at": ce.created_at,
                    },
                )
                community_edges.append(ce)
        return communities, community_edges

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results: int | None = None,
        search_filter: SearchFilters | None = None,
        driver: Any | None = None,
        *,
        group_id: str | None = None,
        config: SearchConfig | None = None,
        return_results: bool | None = None,
        # Convenience kwargs that map to SearchConfig fields
        limit: int | None = None,
        depth: str | None = None,
        rerank_strategy: str | None = None,
        only_valid: bool | None = None,
    ) -> SearchResults | list[EntityEdge]:
        """Hybrid (vector + BM25) search over EntityEdge facts.

        Surriti-native calls return :class:`SearchResults`. Graphiti-style
        calls that use ``group_ids`` / ``num_results`` / ``search_filter`` /
        ``center_node_uuid`` return the edge list unless ``return_results`` is
        explicitly set.

        Convenience kwargs (mapped to SearchConfig):

        * ``limit`` – maximum results to return
        * ``depth`` – search depth: ``"fast"`` (limit=10), ``"normal"`` (limit=25),
          ``"deep"`` (limit=50, includes community search)
        * ``rerank_strategy`` – reranker name: ``"rrf"``, ``"mmr"``,
          ``"cross_encoder"``, ``"node_distance"``, ``"episode_mentions"``
        * ``only_valid`` – when True, exclude edges whose ``invalid_at`` /
          ``expired_at`` are in the past
        """

        from surriti.search import Reranker

        graphiti_style = any(
            value is not None
            for value in (center_node_uuid, group_ids, num_results, search_filter, driver)
        )
        embedding = await self.embedder.create(query) if query else None
        cfg = config or SearchConfig()
        if num_results is not None:
            cfg.limit = num_results
        if limit is not None:
            cfg.limit = limit
        if center_node_uuid is not None:
            cfg.focal_uuid = center_node_uuid
        if search_filter is not None:
            cfg.filters = search_filter
        if cfg.cross_encoder is None:
            cfg.cross_encoder = self.cross_encoder
        # Map convenience kwargs
        if only_valid is not None:
            cfg.only_valid = only_valid
        if rerank_strategy is not None:
            try:
                cfg.reranker = Reranker(rerank_strategy)
            except ValueError:
                pass  # keep default reranker
        if depth is not None:
            depth_map = {"fast": 10, "normal": 25, "deep": 50}
            cfg.limit = depth_map.get(depth, cfg.limit)
        effective_group_id = group_id
        if effective_group_id is None and group_ids:
            effective_group_id = group_ids[0]
        results = await hybrid_search(
            driver or self.driver,
            query=query,
            query_embedding=embedding,
            group_id=effective_group_id,
            config=cfg,
        )
        if return_results is None:
            return_results = not graphiti_style
        return results if return_results else results.edges

    async def search_(
        self,
        query: str,
        config: SearchConfig | None = None,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
        driver: Any | None = None,
        *,
        group_id: str | None = None,
        search_config: SearchConfig | None = None,
        filters: SearchFilters | None = None,
    ) -> SearchResults:
        """Advanced search returning edges + nodes + episodes + communities.

        Mirrors Graphiti's ``search_()``. Honors all reranker strategies and
        ``SearchFilters`` (date ranges, edge types, labels, properties).
        """

        cfg = config or search_config or SearchConfig(include_nodes=True, include_episodes=True)
        effective_filter = filters or search_filter
        if effective_filter is not None:
            cfg.filters = effective_filter
        if center_node_uuid is not None:
            cfg.focal_uuid = center_node_uuid
        if cfg.cross_encoder is None:
            cfg.cross_encoder = self.cross_encoder
        effective_group_id = group_id
        if effective_group_id is None and group_ids:
            effective_group_id = group_ids[0]

        embedding = await self.embedder.create(query) if query else None
        results = await hybrid_search(
            driver or self.driver,
            query=query,
            query_embedding=embedding,
            group_id=effective_group_id,
            config=cfg,
        )
        if cfg.include_nodes:
            results.nodes = await search_nodes(
                driver or self.driver,
                query=query,
                query_embedding=embedding,
                group_id=effective_group_id,
                limit=cfg.limit,
                filters=cfg.filters,
            )
        if cfg.include_episodes:
            results.episodes = await search_episodes(
                driver or self.driver,
                query=query,
                group_id=effective_group_id,
                limit=cfg.limit,
            )
        if cfg.include_communities:
            results.communities = await search_communities(
                driver or self.driver,
                query=query,
                query_embedding=embedding,
                group_id=effective_group_id,
                limit=cfg.limit,
            )
        return results

    async def recall(
        self,
        query: str,
        *,
        group_id: str,
        depth: str = "normal",
        as_of: datetime | None = None,
        limit: int = 20,
        include_invalid: bool = False,
        memory_classes: list[str] | None = None,
        # Convenience kwargs for test / API compatibility
        include_edges: bool = False,
        include_entities: bool = False,
    ) -> "MemoryContext":
        """Build a query-focused memory context.

        ``recall`` is the read-side counterpart of ``add_episode``: it
        resolves the entities mentioned in ``query`` (alias-aware,
        no-create), pulls their dossiers, and fetches an ego-filtered
        slice of facts. The result is a structured bundle the caller can
        render directly into a prompt without juggling search options.

        ``depth``:

        * ``"fast"`` -- profiles + top-``limit`` facts; one DB roundtrip.
        * ``"normal"`` -- adds a hybrid edge search restricted to the
          resolved entities' ego graph.
        * ``"deep"`` -- ``normal`` plus a free-text search over episodes
          and communities; useful for "tell me everything about" prompts.

        ``as_of`` is reserved for time-travel queries; current build
        ignores it and returns the latest valid state.

        ``include_invalid``: when ``True``, invalidated / superseded edges
        are included in the result alongside active ones.  Callers should
        label them as historical so the LLM does not treat them as current
        truth.  Useful for queries like "where did X previously work?"
        where the *old* fact is exactly what is wanted.

        ``include_edges`` / ``include_entities``: compatibility flags.
        ``include_edges`` has no effect (edges are always returned as
        ``facts``).  ``include_entities`` is a no-op for now; entity
        resolution is always performed.
        """

        from surriti.entity_resolution import resolve_entity_mentions
        from surriti.llm import ExtractedEntity
        from surriti.search import _unwrap

        del as_of  # reserved for future bitemporal querying

        if depth not in ("fast", "normal", "deep"):
            raise ValueError("depth must be one of 'fast', 'normal', 'deep'")

        # 1. Cheap query→entities extraction. We avoid a full LLM extract()
        #    here -- recall is on the hot read path. Instead we do an
        #    alias-aware lookup against the existing entity index using a
        #    bag-of-words sweep: every distinct token in the query is
        #    tested as a potential mention. Spurious mentions get filtered
        #    out by the resolver (none of the four stages match) and cost
        #    only one bulk SELECT.
        tokens = [t for t in (query or "").replace(",", " ").split() if len(t) > 1]
        # Also try multi-word windows up to length 3 so "Drexel University"
        # resolves as one mention rather than two.
        mentions: list[ExtractedEntity] = []
        seen: set[str] = set()
        words = tokens
        for n in (3, 2, 1):
            for i in range(0, max(0, len(words) - n + 1)):
                phrase = " ".join(words[i : i + n])
                key = phrase.casefold()
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(ExtractedEntity(name=phrase, labels=["Entity"]))

        resolved = []
        if mentions:
            resolved = await resolve_entity_mentions(
                driver=self.driver,
                embedder=self.embedder,
                llm=self.llm,
                mentions=mentions,
                group_id=group_id,
                episode_context=query,
                threshold=self.alias_resolution_threshold,
                use_llm=False,  # tiebreak is too expensive for the hot path
                create_missing=False,
            )

        ego_uuids = [r.canonical_uuid for r in resolved if r.canonical_uuid]
        # Deduplicate while preserving order.
        ego_uuids = list(dict.fromkeys(ego_uuids))

        # 2. Profiles for the ego entities.
        profiles: list[EntityNode] = []
        if ego_uuids:
            rows = _unwrap(
                await self.driver.query(
                    "SELECT * FROM entity WHERE group_id = $g AND uuid IN $u;",
                    {"g": group_id, "u": ego_uuids},
                )
            )
            profiles = [parse_entity(r) for r in rows]

        # 3. Facts. Free-text + vector hybrid; ego_filter clamps to the
        #    resolved entities when present.
        embedding = await self.embedder.create(query) if query else None
        from surriti.search_filters import SearchFilters as _SF
        _filters = (
            _SF(edge_memory_classes=list(memory_classes))
            if memory_classes
            else None
        )
        cfg = SearchConfig(
            limit=limit,
            candidate_limit=max(limit * 4, 40),
            only_valid=not include_invalid,
            filters=_filters,
            decay_aware=True,
        )
        edge_results = await hybrid_search(
            self.driver,
            query=query,
            query_embedding=embedding,
            group_id=group_id,
            config=cfg,
            ego_filter=ego_uuids if ego_uuids else None,
        )
        facts = edge_results.edges

        # Fallback: when caller asked for specific memory_classes but the
        # hybrid search produced nothing (e.g. empty query string used for
        # an always-pin sweep of subjective edges), do a direct SELECT
        # scoped to those classes. This is the read-side guarantee that
        # preference / style / constraint facts are always retrievable
        # regardless of free-text overlap with the user's question.
        if memory_classes and not facts:
            classes_lc = [str(c).strip().lower() for c in memory_classes if c]
            rows = _unwrap(
                await self.driver.query(
                    """
                    SELECT * FROM relates_to
                    WHERE group_id = $g
                        AND status = "active"
                        AND invalid_at IS NONE
                        AND (attributes.memory_class IN $classes
                             OR (attributes.memory_class IS NONE
                                 AND "objective" IN $classes))
                    ORDER BY created_at DESC
                    LIMIT $lim;
                    """,
                    {"g": group_id, "classes": classes_lc, "lim": limit},
                )
            )
            facts = [parse_edge(r) for r in rows]

        episodes: list[EpisodicNode] = []
        communities: list[CommunityNode] = []
        prediction: dict | None = None
        if depth == "deep":
            episodes = await search_episodes(
                self.driver, query=query, group_id=group_id, limit=limit
            )
            communities = await search_communities(
                self.driver,
                query=query,
                query_embedding=embedding,
                group_id=group_id,
                limit=limit,
            )
            try:
                pred_rows = _unwrap(
                    await self.driver.query(
                        "SELECT payload FROM community WHERE group_id = $g AND kind = 'prediction' LIMIT 1;",
                        {"g": group_id},
                    )
                )
                if pred_rows:
                    payload = pred_rows[0].get("payload")
                    if isinstance(payload, dict):
                        prediction = payload
            except Exception:
                logger.exception("recall: prediction load failed")

        # Trait + goal sidecar fetch. We read the cached ``traits`` /
        # ``goals_active`` arrays denormalised onto each resolved
        # subject, then resolve them to actual ``entity`` rows in one
        # bulk SELECT. Both lists stay empty when the cognitive layer
        # is disabled or has not yet ratified anything.
        traits: list[EntityNode] = []
        goals: list[EntityNode] = []
        try:
            wanted_trait_uuids: list[str] = []
            wanted_goal_uuids: list[str] = []
            for p in profiles:
                wanted_trait_uuids.extend(getattr(p, "traits", []) or [])
                wanted_goal_uuids.extend(getattr(p, "goals_active", []) or [])
            wanted_trait_uuids = list(dict.fromkeys(wanted_trait_uuids))
            wanted_goal_uuids = list(dict.fromkeys(wanted_goal_uuids))
            sidecar_uuids = wanted_trait_uuids + wanted_goal_uuids
            if sidecar_uuids:
                rows = _unwrap(
                    await self.driver.query(
                        "SELECT * FROM entity WHERE group_id = $g AND uuid IN $u;",
                        {"g": group_id, "u": sidecar_uuids},
                    )
                )
                by_uuid = {str(r.get("uuid")): parse_entity(r) for r in rows}
                traits = [by_uuid[u] for u in wanted_trait_uuids if u in by_uuid]
                goals = [by_uuid[u] for u in wanted_goal_uuids if u in by_uuid]
        except Exception:
            logger.exception("recall: trait/goal sidecar load failed")

        return MemoryContext(
            query=query,
            profiles=profiles,
            facts=facts,
            episodes=episodes,
            communities=communities,
            traits=traits,
            goals=goals,
            prediction=prediction,
            resolved_entities=[
                {
                    "mention": r.mention.name,
                    "uuid": r.canonical_uuid,
                    "name": r.canonical_name,
                    "resolution": r.resolution,
                    "confidence": r.confidence,
                }
                for r in resolved
                if r.canonical_uuid is not None
            ],
        )

    async def get_nodes_and_edges_by_episode(
        self, episode_uuids: list[str]
    ) -> SearchResults:
        """Fetch every entity mentioned by, and every edge sourced from, the given episodes."""

        from surriti.search import _unwrap

        if not episode_uuids:
            return SearchResults()

        # Edges whose `episodes` array intersects the requested set.
        edge_rows = await self.driver.query(
            """
            SELECT * FROM relates_to
            WHERE episodes ANYINSIDE $eps;
            """,
            {"eps": episode_uuids},
        )
        edges = [parse_edge(r) for r in _unwrap(edge_rows)]

        # Entities mentioned via the `mentions` edge from these episodes.
        node_rows = await self.driver.query(
            """
            SELECT * FROM (
                SELECT out AS rec FROM mentions
                WHERE in IN $ep_ids
            ).rec.*;
            """,
            {"ep_ids": [f"episode:{u}" for u in episode_uuids]},
        )
        nodes_unwrapped = _unwrap(node_rows)
        # Fallback path for SDK quirks: directly fetch entity rows by UUID.
        if not nodes_unwrapped:
            edge_endpoints = {e.source_node_uuid for e in edges} | {
                e.target_node_uuid for e in edges
            }
            if edge_endpoints:
                row = await self.driver.query(
                    "SELECT * FROM entity WHERE uuid IN $ids;",
                    {"ids": list(edge_endpoints)},
                )
                nodes_unwrapped = _unwrap(row)
        nodes = [parse_entity(r) for r in nodes_unwrapped]

        ep_rows = await self.driver.query(
            "SELECT * FROM episode WHERE uuid IN $eps;", {"eps": episode_uuids}
        )
        episodes = [parse_episode(r) for r in _unwrap(ep_rows)]
        return SearchResults(edges=edges, nodes=nodes, episodes=episodes)

    async def retrieve_episodes(
        self,
        reference_time: datetime | None = None,
        last_n: int = 10,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
        *,
        group_id: str | None = None,
    ) -> list[EpisodicNode]:
        """Retrieve the most recent episodes (newest first).

        Compatible with Graphiti's signature. Either ``group_ids`` (list) or
        the convenience ``group_id=`` (single string) is accepted.
        """

        from surriti.search import _unwrap

        if group_id is not None and not group_ids:
            group_ids = [group_id]

        clauses = []
        params: dict[str, Any] = {"n": last_n}
        if reference_time is not None:
            clauses.append("reference_time <= $ref")
            params["ref"] = reference_time
        if group_ids:
            clauses.append("group_id IN $groups")
            params["groups"] = group_ids
        if source is not None:
            clauses.append("source = $source")
            params["source"] = source.value
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await self.driver.query(
            f"SELECT * FROM episode {where} ORDER BY reference_time DESC LIMIT $n;",
            params,
        )
        return [parse_episode(r) for r in _unwrap(rows)]

    # ----------------------------------------------- direct lookups & persistence
    async def get_entity_node(self, uuid: str) -> EntityNode | None:
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                "SELECT * FROM entity WHERE uuid = $u LIMIT 1;", {"u": uuid}
            )
        )
        return parse_entity(rows[0]) if rows else None

    async def get_entity_edge(self, uuid: str) -> EntityEdge | None:
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                "SELECT * FROM relates_to WHERE uuid = $u LIMIT 1;", {"u": uuid}
            )
        )
        return parse_edge(rows[0]) if rows else None

    async def get_episode(self, uuid: str) -> EpisodicNode | None:
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                "SELECT * FROM episode WHERE uuid = $u LIMIT 1;", {"u": uuid}
            )
        )
        return parse_episode(rows[0]) if rows else None

    async def save_node(self, node: EntityNode) -> EntityNode:
        """Insert or update an EntityNode. Generates the embedding if missing."""

        if node.name_embedding is None:
            node.name_embedding = await self.embedder.create(node.name)
        await self.driver.query(
            """
            UPSERT type::record("entity", $uuid) MERGE {
                uuid: $uuid, group_id: $group_id, name: $name,
                summary: $summary, labels: $labels, attributes: $attributes,
                name_embedding: $emb, created_at: $created_at
            };
            """,
            {
                "uuid": node.uuid,
                "group_id": node.group_id,
                "name": node.name,
                "summary": node.summary,
                "labels": node.labels,
                "attributes": node.attributes,
                "emb": node.name_embedding,
                "created_at": node.created_at,
            },
        )
        return node

    async def save_edge(self, edge: EntityEdge) -> EntityEdge:
        """Update an EntityEdge in place. The RELATE link is not changed."""

        if edge.fact_embedding is None and edge.fact:
            edge.fact_embedding = await self.embedder.create(edge.fact)
        await self.driver.query(
            """
            UPDATE relates_to MERGE {
                name: $name, fact: $fact, fact_embedding: $emb,
                episodes: $episodes, valid_at: $valid_at,
                invalid_at: $invalid_at, expired_at: $expired_at,
                attributes: $attributes
            } WHERE uuid = $uuid;
            """,
            {
                "uuid": edge.uuid,
                "name": edge.name,
                "fact": edge.fact,
                "emb": edge.fact_embedding,
                "episodes": edge.episodes,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
                "expired_at": edge.expired_at,
                "attributes": edge.attributes,
            },
        )
        return edge

    async def remove_edge(self, edge_uuid: str) -> None:
        await self.driver.query(
            "DELETE relates_to WHERE uuid = $u;", {"u": edge_uuid}
        )

    async def _fetch_episode_contents(self, episode_uuids: list[str]) -> str:
        from surriti.search import _unwrap

        if not episode_uuids:
            return ""
        rows = _unwrap(
            await self.driver.query(
                "SELECT content FROM episode WHERE uuid IN $u;",
                {"u": episode_uuids},
            )
        )
        # NOTE: never include the episode `name` -- it's internal metadata
        # (e.g. "chat", "turn-a") and small models will treat it as an
        # entity. Join contents only.
        return "\n---\n".join(
            (r.get("content") or "").strip() for r in rows if r.get("content")
        )

    async def _fetch_recent_episode_contents(
        self,
        *,
        reference_time: datetime,
        group_id: str,
        source: EpisodeType,
        limit: int = 10,
    ) -> str:
        episodes = await self.retrieve_episodes(
            reference_time=reference_time,
            last_n=limit,
            group_id=group_id,
            source=source,
        )
        # Graphiti provides prior episodes in chronological order to the LLM.
        ordered = sorted(episodes, key=lambda ep: ep.reference_time)
        return "\n---\n".join(ep.content.strip() for ep in ordered if ep.content)

    def _filter_facts_by_edge_type_map(
        self,
        entities: list[ExtractedEntity],
        facts: list[ExtractedFact],
        edge_type_map: dict[tuple[str, str], list[str]],
    ) -> list[ExtractedFact]:
        labels_by_name = {
            ent.name: set(ent.labels or ["Entity"]) | {"Entity"} for ent in entities
        }
        filtered: list[ExtractedFact] = []
        for fact in facts:
            subj_labels = labels_by_name.get(fact.subject, {"Entity"})
            obj_labels = labels_by_name.get(fact.object, {"Entity"})
            allowed: set[str] = set()
            map_had_match = False
            for (source_label, target_label), predicates in edge_type_map.items():
                if source_label in subj_labels and target_label in obj_labels:
                    map_had_match = True
                    allowed.update(predicates)
            if not map_had_match:
                continue
            # In Graphiti, an empty predicate list means unrestricted generic
            # Entity->Entity extraction.
            if not allowed or fact.predicate in allowed:
                filtered.append(fact)
        return filtered

    # ------------------------------------------------------------------ internals
    async def _save_episode(self, episode: EpisodicNode) -> None:
        await self.driver.query(
            """
            CREATE type::record("episode", $uuid) CONTENT {
                uuid: $uuid,
                group_id: $group_id,
                name: $name,
                source: $source,
                source_description: $source_description,
                content: $content,
                reference_time: $reference_time,
                created_at: $created_at,
                entity_edges: []
            };
            """,
            {
                "uuid": episode.uuid,
                "group_id": episode.group_id,
                "name": episode.name,
                "source": episode.source.value,
                "source_description": episode.source_description,
                "content": episode.content,
                "reference_time": episode.reference_time,
                "created_at": episode.created_at,
            },
        )

    async def upsert_user(
        self,
        *,
        group_id: str,
        user_id: str | None = None,
        display_name: str | None = None,
        summary: str = "",
    ) -> EntityNode:
        """UPSERT the canonical ``User`` entity for a tenant.

        ``group_id`` is the multi-tenant boundary; ``user_id`` is the stable
        identifier used as the entity's ``name`` (defaults to ``group_id``,
        which is the common case where each tenant *is* one user). The
        friendly name lives in ``attributes.display_name`` so it doesn't
        pollute the ``(group_id, name)`` unique index. Idempotent: repeated
        calls update ``display_name`` / ``summary`` and return the existing
        node.
        """

        uid = user_id or group_id
        if not uid:
            raise ValueError("upsert_user requires a non-empty user_id or group_id")

        from surriti.search import _unwrap

        rows = _unwrap(await self.driver.query(
            "SELECT * FROM entity WHERE group_id = $g AND name = $n LIMIT 1;",
            {"g": group_id, "n": uid},
        ))
        if rows:
            existing = parse_entity(rows[0])
            attrs = dict(existing.attributes or {})
            changed = False
            if display_name is not None and attrs.get("display_name") != display_name:
                attrs["display_name"] = display_name
                changed = True
            new_summary = summary or existing.summary
            new_labels = sorted(set((existing.labels or []) + ["User"]))
            if new_labels != (existing.labels or []):
                changed = True
            if changed or new_summary != existing.summary:
                await self.driver.query(
                    """
                    UPDATE type::record("entity", $uuid) SET
                        summary = $summary,
                        labels = $labels,
                        attributes = $attributes;
                    """,
                    {
                        "uuid": existing.uuid,
                        "summary": new_summary,
                        "labels": new_labels,
                        "attributes": attrs,
                    },
                )
                existing.attributes = attrs
                existing.summary = new_summary
                existing.labels = new_labels
            return existing

        embedding = await self.embedder.create(uid)
        node = EntityNode(
            name=uid,
            summary=summary,
            labels=["User"],
            name_embedding=embedding,
            group_id=group_id,
            attributes={"display_name": display_name} if display_name else {},
        )
        try:
            await self.driver.query(
                """
                CREATE type::record("entity", $uuid) CONTENT {
                    uuid: $uuid,
                    group_id: $group_id,
                    name: $name,
                    summary: $summary,
                    labels: $labels,
                    attributes: $attributes,
                    name_embedding: $emb,
                    created_at: $created_at
                };
                """,
                {
                    "uuid": node.uuid,
                    "group_id": node.group_id,
                    "name": node.name,
                    "summary": node.summary,
                    "labels": node.labels,
                    "attributes": node.attributes,
                    "emb": node.name_embedding,
                    "created_at": node.created_at,
                },
            )
            return node
        except Exception as exc:
            if "entity_name_uniq" not in str(exc):
                raise
            # Race: another caller created the User concurrently.
            fallback = _unwrap(await self.driver.query(
                "SELECT * FROM entity WHERE group_id = $g AND name = $n LIMIT 1;",
                {"g": group_id, "n": uid},
            ))
            if not fallback:
                raise
            return parse_entity(fallback[0])

    async def _upsert_entities(
        self,
        extracted: list[ExtractedEntity],
        *,
        group_id: str,
        episode_uuid: str | None = None,
        episode_context: str = "",
    ) -> list[EntityNode]:
        """Insert new entities; reuse existing ones by ``(group_id, name)``.

        When ``alias_resolution`` is on (the default), each extracted
        mention is first run through :func:`resolve_entity_mentions`,
        which short-circuits on alias hits, exact-name matches, and
        confident semantic matches against the entity name HNSW index
        before falling back to the legacy create / case-merge path.

        Idempotent under any input duplication and tolerant to a race
        against the ``entity_name_uniq`` index: if a CREATE collides we
        re-SELECT and reuse the now-existing row instead of bubbling the
        error up. Tenant isolation is enforced by ``entity_name_uniq``
        on ``(group_id, name)`` -- entities with the same name in
        different ``group_id``s are deliberately distinct nodes.
        """

        if not extracted:
            return []

        # ------------------------------------------------------------------
        # Stage 0 -- canonical resolution (alias / exact / semantic / LLM).
        # Surface forms that resolve to an existing entity skip the rest of
        # the pipeline and reuse the canonical row directly. Mentions that
        # remain "new" fall through to the legacy create logic below.
        # ------------------------------------------------------------------
        from surriti.entity_resolution import (
            ResolvedEntity,
            normalize_alias,
            resolve_entity_mentions,
        )

        resolved_by_mention: dict[str, ResolvedEntity] = {}
        if self.alias_resolution_enabled:
            resolved_list = await resolve_entity_mentions(
                driver=self.driver,
                embedder=self.embedder,
                llm=self.llm,
                mentions=extracted,
                group_id=group_id,
                episode_context=episode_context,
                threshold=self.alias_resolution_threshold,
                use_llm=self.alias_resolution_llm,
                episode_uuid=episode_uuid,
            )
            for r in resolved_list:
                resolved_by_mention[r.mention.name] = r

        # 1. Dedupe high-confidence case/whitespace variants in this batch
        #    within the tenant. Fuzzy name matches stay manual.
        seen: dict[str, ExtractedEntity] = {}
        for ext in extracted:
            key = _entity_name_key(ext.name)
            if key and key not in seen:
                seen[key] = ext
        deduped = list(seen.values())

        from surriti.search import _unwrap

        # 2. Bulk lookup every existing entity in this tenant so exact
        #    casefold matches reuse the canonical row and old duplicates
        #    can be collapsed without crossing group/user boundaries.
        rows = _unwrap(await self.driver.query(
            "SELECT * FROM entity WHERE group_id = $g;",
            {"g": group_id},
        ))
        existing_by_key: dict[str, EntityNode] = {}
        duplicates_by_key: dict[str, list[EntityNode]] = defaultdict(list)
        for row in rows:
            node = parse_entity(row)
            key = _entity_name_key(node.name)
            if not key:
                continue
            if key not in existing_by_key:
                existing_by_key[key] = node
            else:
                duplicates_by_key[key].append(node)
        for key, aliases in duplicates_by_key.items():
            await self._merge_entity_case_duplicates(
                canonical=existing_by_key[key],
                aliases=aliases,
                group_id=group_id,
            )

        # Fold semantic / LLM resolutions into the existing-row map keyed
        # by the *mention's* casefold key so the rest of the pipeline
        # transparently reuses them.
        if resolved_by_mention:
            uuid_to_node = {n.uuid: n for n in existing_by_key.values()}
            for ext in extracted:
                hit = resolved_by_mention.get(ext.name)
                if hit is None or hit.canonical_uuid is None:
                    continue
                key = _entity_name_key(ext.name)
                if not key or key in existing_by_key:
                    continue
                node = uuid_to_node.get(hit.canonical_uuid) or hit.existing
                if node is not None:
                    existing_by_key[key] = node

        results: list[EntityNode] = []
        # Batch-embed all NEW entities up front (one round-trip instead of N).
        missing = [e for e in deduped if _entity_name_key(e.name) not in existing_by_key]
        embeddings: dict[str, list[float]] = {}
        if missing:
            vectors = await self.embedder.create_batch([e.name for e in missing])
            embeddings = {e.name: v for e, v in zip(missing, vectors)}
        for ext in deduped:
            key = _entity_name_key(ext.name)
            if key in existing_by_key:
                results.append(existing_by_key[key])
                continue
            embedding = embeddings.get(ext.name)
            node = EntityNode(
                name=ext.name,
                summary=ext.summary,
                labels=ext.labels,
                name_embedding=embedding,
                group_id=group_id,
            )
            try:
                await self.driver.query(
                    """
                    CREATE type::record("entity", $uuid) CONTENT {
                        uuid: $uuid,
                        group_id: $group_id,
                        name: $name,
                        summary: $summary,
                        labels: $labels,
                        attributes: {},
                        name_embedding: $emb,
                        canonical_name: $canonical_name,
                        aliases: $aliases,
                        created_at: $created_at
                    };
                    """,
                    {
                        "uuid": node.uuid,
                        "group_id": node.group_id,
                        "name": node.name,
                        "summary": node.summary,
                        "labels": node.labels,
                        "emb": node.name_embedding,
                        "canonical_name": node.name,
                        "aliases": [node.name],
                        "created_at": node.created_at,
                    },
                )
                results.append(node)
                existing_by_key[key] = node
            except Exception as exc:
                # Race against entity_name_uniq: a concurrent ingest (or a
                # duplicate that slipped past dedupe) created
                # (group_id, name) between our SELECT and CREATE. Re-fetch
                # the existing row and reuse it.
                if "entity_name_uniq" not in str(exc):
                    raise
                logger.debug(
                    "entity_name_uniq race for (%s, %s); reusing existing.",
                    group_id, ext.name,
                )
                fallback = _unwrap(await self.driver.query(
                    "SELECT * FROM entity WHERE group_id = $g AND name = $n LIMIT 1;",
                    {"g": group_id, "n": ext.name},
                ))
                if not fallback:
                    fallback = _unwrap(await self.driver.query(
                        "SELECT * FROM entity WHERE group_id = $g;",
                        {"g": group_id},
                    ))
                    fallback = [
                        row for row in fallback
                        if _entity_name_key(row.get("name", "")) == key
                    ][:1]
                if not fallback:
                    raise
                results.append(parse_entity(fallback[0]))
        return results

    async def _merge_entity_case_duplicates(
        self,
        *,
        canonical: EntityNode,
        aliases: list[EntityNode],
        group_id: str,
    ) -> None:
        alias_ids = [a.uuid for a in aliases if a.uuid != canonical.uuid]
        if not alias_ids:
            return

        await self.driver.query(
            """
            UPDATE relates_to
            SET in = type::record("entity", $canonical)
            WHERE group_id = $group_id AND record::id(in) IN $aliases;
            UPDATE relates_to
            SET out = type::record("entity", $canonical)
            WHERE group_id = $group_id AND record::id(out) IN $aliases;
            UPDATE mentions
            SET out = type::record("entity", $canonical)
            WHERE group_id = $group_id AND record::id(out) IN $aliases;
            DELETE entity WHERE group_id = $group_id AND uuid IN $aliases;
            """,
            {"group_id": group_id, "canonical": canonical.uuid, "aliases": alias_ids},
        )

    async def _add_fact_edge(
        self,
        *,
        fact: ExtractedFact,
        subject: EntityNode,
        obj: EntityNode,
        episode: EpisodicNode | None,
        group_id: str,
        source_type: str = "user",
    ) -> tuple[EntityEdge, list[EntityEdge]]:
        valid_at = _parse_iso(fact.valid_at) or (
            episode.reference_time if episode else datetime.now(timezone.utc)
        )
        invalid_at = _parse_iso(fact.invalid_at)

        # ------------------------------------------------------------------
        # Relation-frame resolution. When a frame matches the predicate
        # (alias-aware, group-scoped first), it overrides the per-fact
        # ``singleton``/``domain`` heuristics with data-driven policy:
        #
        # * ``frame.canonical_name`` becomes the stored predicate name so
        #   ``wife_of``/``husband_of``/``married_to`` collapse onto a
        #   single ``spouse_of`` edge across episodes and aliases.
        # * ``frame.directionality == "symmetric"`` triggers lex-min
        #   normalization of (subject, object) so direction-equivalent
        #   restatements dedupe.
        # * ``frame.cardinality == "one_current"`` drives the singleton
        #   slot closer regardless of ``source_type`` -- the policy is
        #   now data, not a hardcoded user-only gate.
        # When no frame resolves, behavior is unchanged for back-compat.
        # ------------------------------------------------------------------
        frame = await self.relation_frames.resolve(
            fact.predicate,
            group_id=group_id,
            source_span=fact.source_span or fact.fact or "",
            sample_subject=subject.name,
            sample_object=obj.name,
        )
        canonical_name = frame.canonical_name if frame else fact.predicate
        edge_name = canonical_name
        subj_uuid, obj_uuid = subject.uuid, obj.uuid
        subj_node, obj_node = subject, obj
        if frame and frame.directionality == "symmetric":
            new_subj_uuid, new_obj_uuid = normalize_symmetric(subj_uuid, obj_uuid)
            if (new_subj_uuid, new_obj_uuid) != (subj_uuid, obj_uuid):
                subj_uuid, obj_uuid = new_subj_uuid, new_obj_uuid
                subj_node, obj_node = obj, subject

        # Cardinality drives slot exclusivity. Frame metadata wins; a
        # legacy ``fact.singleton=True`` hint still triggers the closer
        # for predicates without a registered frame. Two explicit
        # opt-outs:
        #   * ``operation == "qualify"`` -- the qualified variant gets
        #     a distinct slot via its qualifier hash and must coexist.
        #   * frame ``contradiction_policy == "uncertain"`` -- the
        #     human-in-the-loop policy supersedes the deterministic
        #     closer; Layer 4 will surface the conflict instead.
        op = (fact.operation or "assert").lower()
        is_singleton_slot = (
            op != "qualify"
            and not (frame is not None and frame.contradiction_policy == "uncertain")
            and bool(
                (frame and frame.cardinality == "one_current")
                or (frame is None and fact.singleton and source_type == "user")
            )
        )

        qhash = qualifier_hash(fact.qualifiers)
        fact_text = fact.fact or f"{subj_node.name} {edge_name} {obj_node.name}."
        embedding = await self.embedder.create(fact_text)

        existing = await self._find_equivalent_edge(
            group_id=group_id,
            subject_uuid=subj_uuid,
            object_uuid=obj_uuid,
            predicate=edge_name,
            fact_text=fact_text,
            qualifier_hash_value=qhash,
        )
        if existing is not None:
            if episode and episode.uuid not in existing.episodes:
                existing.episodes.append(episode.uuid)
                await self.driver.query(
                    """
                    UPDATE relates_to
                    SET episodes = array::distinct(array::concat(episodes, $episodes))
                    WHERE uuid = $uuid;
                    """,
                    {"uuid": existing.uuid, "episodes": [episode.uuid]},
                )
            return existing, []

        edge_uuid = str(uuid4())

        # ------------------------------------------------------------------
        # Contradiction cascade -- explicit, ordered layers. Whichever
        # layer fires first decides the new edge's status and the set of
        # invalidated peers; later layers are skipped.
        #
        #   Layer 0: extractor-declared ``replaces`` -- the LLM listed
        #            prior facts this assertion supersedes (e.g. "sold
        #            X" supersedes "drives X"). Each descriptor is
        #            embedding-matched against active edges in the same
        #            group whose source is this subject; matches are
        #            terminated. Authoritative because it comes from
        #            the model that read the source text.
        #   Layer 1: deterministic frame closure (cardinality=one_current
        #            or legacy singleton hint -- runs ``_close_singleton_slot``).
        #   Layer 2: explicit ``operation`` from the extractor -- ``terminate``
        #            and the closing half of ``correct`` are handled in the
        #            caller (``add_episode``); ``qualify`` is handled by the
        #            slot-key construction above (distinct qualifier hash).
        #   Layer 3: LLM semantic contradiction detection -- only when no
        #            frame governs the predicate, or the frame's policy is
        #            ``uncertain``. ``coexist`` short-circuits to no-op.
        #   Layer 4: needs_resolution -- when policy is ``uncertain`` AND
        #            the LLM returned no contradictions AND there is at
        #            least one active same-slot peer with a different
        #            object, mark the new edge ``needs_resolution`` and
        #            stamp every peer with the same ``conflict_group_id``
        #            so ``Surriti.get_conflicts()`` can surface the group.
        # ------------------------------------------------------------------

        # Layer 0: extractor-declared replaces. Cheap, deterministic,
        # and authoritative when the LLM populates the field.
        replaces_closed: list[EntityEdge] = []
        if fact.replaces:
            replaces_closed = await self._apply_replaces(
                group_id=group_id,
                subject_uuid=subj_uuid,
                replace_descriptors=fact.replaces,
                exclude_edge_uuid=None,
                invalid_at=valid_at,
                superseded_by=edge_uuid,
            )

        # Layer 1: deterministic singleton closure (cardinality-driven).
        singleton_closed: list[EntityEdge] = []
        if is_singleton_slot:
            singleton_closed = await self._close_singleton_slot(
                group_id=group_id,
                subject_uuid=subj_uuid,
                predicate=edge_name,
                keep_object_uuid=obj_uuid,
                invalid_at=valid_at,
                superseded_by=edge_uuid,
                qualifier_hash_value=qhash,
                memory_class=(fact.memory_class or "objective").strip().lower() or "objective",
            )

        edge_status = "active"
        conflict_group_id: str | None = None

        # Layer 3: LLM semantic contradiction. Skipped when:
        # * the deterministic closer already handled this slot, OR
        # * the frame's policy explicitly says peers coexist.
        if singleton_closed:
            invalidated = list(singleton_closed)
        elif frame is not None and frame.contradiction_policy == "coexist":
            invalidated = []
        else:
            invalidated = await resolve_contradictions(
                self.driver,
                llm=self.llm,
                new_fact=fact_text,
                new_fact_embedding=embedding,
                new_valid_at=valid_at,
                group_id=group_id,
                new_fact_struct=fact,
                new_edge_uuid=edge_uuid,
                new_subject_uuid=subj_uuid,
                new_object_uuid=obj_uuid,
            )

        # Merge Layer 0 results, deduping by uuid so a peer terminated
        # by both ``replaces`` and another layer is only listed once.
        if replaces_closed:
            seen_uuids = {e.uuid for e in invalidated}
            for e in replaces_closed:
                if e.uuid not in seen_uuids:
                    invalidated.append(e)
                    seen_uuids.add(e.uuid)

        # Layer 4: needs_resolution. When the frame says "uncertain" and
        # Layer 3 did not pick a winner, surface every same-slot peer in
        # one ``conflict_group_id`` so the caller can resolve manually.
        if (
            not invalidated
            and frame is not None
            and frame.contradiction_policy == "uncertain"
        ):
            peers = await self._find_active_slot_peers(
                group_id=group_id,
                subject_uuid=subj_uuid,
                predicate=edge_name,
                exclude_object_uuid=obj_uuid,
                qualifier_hash_value=qhash,
            )
            if peers:
                conflict_group_id = str(uuid4())
                edge_status = "needs_resolution"
                await self._mark_conflict_group(
                    [p.uuid for p in peers], conflict_group_id
                )

        edge = EntityEdge(
            uuid=edge_uuid,
            group_id=group_id,
            source_node_uuid=subj_uuid,
            target_node_uuid=obj_uuid,
            name=edge_name,
            fact=fact_text,
            fact_embedding=embedding,
            episodes=[episode.uuid] if episode else [],
            valid_at=valid_at,
            invalid_at=invalid_at,
            status=edge_status,
            source_type=source_type,
            confidence=fact.confidence,
            temporal=fact.temporal or (frame is not None and frame.temporal_kind == "state"),
            singleton=is_singleton_slot,
            domain=fact.domain,
            supersedes=list({e.uuid for e in (singleton_closed + replaces_closed)}),
            fact_key=make_fact_key(
                group_id, subj_uuid, edge_name, obj_uuid, qualifier_hash=qhash
            ),
            relation_frame_id=frame.uuid if frame else None,
            canonical_name=canonical_name,
            qualifiers=dict(fact.qualifiers or {}),
            roles=dict(fact.argument_roles or {}),
            conflict_group_id=conflict_group_id,
            memory_class=(fact.memory_class or "objective").strip().lower() or "objective",
            attributes={
                "memory_class": (fact.memory_class or "objective").strip().lower()
                or "objective"
            },
        )

        await self.driver.query(
            """
            RELATE (type::record("entity", $src))->relates_to->(type::record("entity", $tgt))
            CONTENT {
                uuid: $uuid,
                group_id: $group_id,
                name: $name,
                fact: $fact,
                fact_embedding: $emb,
                episodes: $episodes,
                valid_at: $valid_at,
                invalid_at: $invalid_at,
                status: $status,
                polarity: $polarity,
                source_type: $source_type,
                confidence: $confidence,
                temporal: $temporal,
                singleton: $singleton,
                domain: $domain,
                supersedes: $supersedes,
                fact_key: $fact_key,
                relation_frame_id: $relation_frame_id,
                canonical_name: $canonical_name,
                qualifiers: $qualifiers,
                roles: $roles,
                conflict_group_id: $conflict_group_id,
                derived: $derived,
                derived_from: $derived_from,
                attributes: $attributes,
                created_at: $created_at
            };
            """,
            {
                "src": subj_uuid,
                "tgt": obj_uuid,
                "uuid": edge.uuid,
                "group_id": edge.group_id,
                "name": edge.name,
                "fact": edge.fact,
                "emb": edge.fact_embedding,
                "episodes": edge.episodes,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
                "status": edge.status,
                "polarity": edge.polarity,
                "source_type": edge.source_type,
                "confidence": edge.confidence,
                "temporal": edge.temporal,
                "singleton": edge.singleton,
                "domain": edge.domain,
                "supersedes": edge.supersedes,
                "fact_key": edge.fact_key,
                "relation_frame_id": edge.relation_frame_id,
                "canonical_name": edge.canonical_name,
                "qualifiers": edge.qualifiers,
                "roles": edge.roles,
                "conflict_group_id": edge.conflict_group_id,
                "derived": edge.derived,
                "derived_from": edge.derived_from,
                "attributes": dict(edge.attributes or {}),
                "created_at": edge.created_at,
            },
        )
        return edge, invalidated

    async def _close_singleton_slot(
        self,
        *,
        group_id: str,
        subject_uuid: str,
        predicate: str,
        keep_object_uuid: str,
        invalid_at: datetime,
        superseded_by: str,
        qualifier_hash_value: str = "",
        memory_class: str = "objective",
    ) -> list[EntityEdge]:
        """Close active edges that share the singleton slot with the new fact.

        Generic mechanism (no hardcoded predicate list): every active edge
        with the same ``(group_id, subject, predicate, qualifier_hash)``
        and a DIFFERENT object is invalidated and marked
        ``status="superseded"``. ``qualifier_hash_value`` keeps qualified
        variants (e.g. ``lives_in(Florida, season=winter)``) in distinct
        slots from the unqualified or differently-qualified ones.
        """

        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND in = type::record("entity", $src)
                    AND name = $name
                    AND status = "active"
                    AND invalid_at IS NONE;
                """,
                {
                    "group_id": group_id,
                    "src": subject_uuid,
                    "name": predicate,
                },
            )
        )
        to_close: list[EntityEdge] = []
        for row in rows:
            target = row.get("target_node_uuid") or row.get("out")
            target_uuid = target.split(":", 1)[-1] if isinstance(target, str) else target
            if target_uuid == keep_object_uuid:
                continue
            # Skip rows that occupy a different qualifier-scoped slot
            # (e.g. ``lives_in(Florida, season=winter)`` must not close
            # ``lives_in(Vermont, season=summer)``). The fact_key trailing
            # segment carries the qualifier hash when one was set, so
            # comparing on it cleanly partitions slots without an extra
            # column. Legacy rows (4 segments) carry no hash and only
            # match unqualified claims.
            row_key = row.get("fact_key") or ""
            parts = row_key.split("::")
            row_qhash = parts[4] if len(parts) >= 5 else ""
            if row_qhash != qualifier_hash_value:
                continue
            # Kind-aware closure: a "preference" assertion must NOT
            # supersede a coexisting "objective" claim with the same
            # predicate (and vice versa). Subjective + objective facts
            # live in independent slots even when the predicate matches.
            row_attrs = row.get("attributes") or {}
            row_class = str(row_attrs.get("memory_class") or "objective").strip().lower() or "objective"
            new_class = (memory_class or "objective").strip().lower() or "objective"
            if row_class != new_class:
                continue
            to_close.append(parse_edge(row))
        if to_close:
            await invalidate_edges(
                self.driver,
                [e.uuid for e in to_close],
                invalid_at=invalid_at,
                superseded_by=superseded_by,
            )
            for e in to_close:
                e.invalid_at = invalid_at
                e.status = "superseded"
                e.superseded_by = superseded_by
        return to_close

    async def _apply_replaces(
        self,
        *,
        group_id: str,
        subject_uuid: str,
        replace_descriptors: list[str],
        exclude_edge_uuid: str | None,
        invalid_at: datetime,
        superseded_by: str,
        similarity_limit: int = 5,
    ) -> list[EntityEdge]:
        """Terminate active edges that the extractor declared superseded.

        ``replace_descriptors`` are free-text descriptions emitted by
        the extractor in :attr:`ExtractedFact.replaces` (e.g.
        ``"<speaker> drives Honda Civic"``). For each descriptor we do
        a hybrid (vector + fulltext) search against active edges in
        the same group whose source is the same subject, then
        terminate every match. This honours the model's explicit
        supersession signal -- the layer-3 LLM-as-judge pass only
        triggers on undeclared contradictions, so transfer-of-state
        events ("sold X" supersedes "drives X") that the model
        already reasoned about don't depend on a second LLM hop to
        be applied.
        """
        if not replace_descriptors:
            return []
        from surriti.temporal import find_similar_edges

        seen: dict[str, EntityEdge] = {}
        for descriptor in replace_descriptors:
            text = (descriptor or "").strip()
            if not text:
                continue
            try:
                embedding = await self.embedder.create(text)
            except Exception:  # noqa: BLE001
                embedding = None
            try:
                candidates = await find_similar_edges(
                    self.driver,
                    fact=text,
                    fact_embedding=embedding,
                    group_id=group_id,
                    limit=similarity_limit,
                )
            except Exception:  # noqa: BLE001
                logger.exception("replaces lookup failed for descriptor %r", text)
                continue
            for edge in candidates:
                if edge.uuid == exclude_edge_uuid:
                    continue
                if edge.uuid in seen:
                    continue
                if edge.source_node_uuid != subject_uuid:
                    continue
                if edge.invalid_at is not None or edge.status != "active":
                    continue
                seen[edge.uuid] = edge

        to_close = list(seen.values())
        if to_close:
            await invalidate_edges(
                self.driver,
                [e.uuid for e in to_close],
                invalid_at=invalid_at,
                superseded_by=superseded_by,
            )
            for e in to_close:
                e.invalid_at = invalid_at
                e.status = "superseded"
                e.superseded_by = superseded_by
        return to_close

    async def _terminate_matching_edge(
        self,
        *,
        group_id: str,
        subject_uuid: str,
        object_uuid: str,
        predicate: str,
        invalid_at: datetime,
    ) -> list[EntityEdge]:
        """Close every active edge matching (subject, predicate, object).

        Used to implement ``ExtractedFact.operation == "terminate"`` --
        the user said the prior fact is no longer true, so we close it
        without inserting anything new.
        """

        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND in = type::record("entity", $src)
                    AND out = type::record("entity", $tgt)
                    AND name = $name
                    AND status = "active"
                    AND invalid_at IS NONE;
                """,
                {
                    "group_id": group_id,
                    "src": subject_uuid,
                    "tgt": object_uuid,
                    "name": predicate,
                },
            )
        )
        edges = [parse_edge(r) for r in rows]
        if edges:
            await invalidate_edges(
                self.driver,
                [e.uuid for e in edges],
                invalid_at=invalid_at,
            )
            for e in edges:
                e.invalid_at = invalid_at
                e.status = "superseded"
        return edges

    # ------------------------------------------------------------------
    # Relation-frame public API
    # ------------------------------------------------------------------

    def register_frame(
        self, frame: RelationFrame, *, group_id: str | None = None
    ) -> RelationFrame:
        """Register a :class:`RelationFrame` for use by the ingest pipeline.

        Pass ``group_id=None`` (default) to register globally; pass a
        specific tenant id to scope the frame to that tenant only.
        """
        return self.relation_frames.register(frame, group_id=group_id)

    def get_frame(
        self, predicate: str, *, group_id: str = ""
    ) -> RelationFrame | None:
        """Return the registered :class:`RelationFrame` for ``predicate``,
        or ``None`` if no frame is registered."""
        return self.relation_frames.get(predicate, group_id=group_id)

    async def get_conflicts(
        self, *, group_id: str, limit: int = 100
    ) -> list[EntityEdge]:
        """Return active edges that the contradiction engine could not
        resolve confidently (``status == "needs_resolution"``).

        Use this surface to expose unresolved-conflict groups in your
        application and let the user pick the canonical answer.
        """
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id AND status = "needs_resolution"
                LIMIT $limit;
                """,
                {"group_id": group_id, "limit": limit},
            )
        )
        return [parse_edge(r) for r in rows]

    async def _find_active_slot_peers(
        self,
        *,
        group_id: str,
        subject_uuid: str,
        predicate: str,
        exclude_object_uuid: str,
        qualifier_hash_value: str = "",
    ) -> list[EntityEdge]:
        """Return active edges in the same ``(subject, predicate, qualifier)``
        slot whose target is NOT ``exclude_object_uuid``. Powers the
        Layer-4 ``needs_resolution`` writer in :meth:`_add_fact_edge`.
        """
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND in = type::record("entity", $src)
                    AND name = $name
                    AND status = "active"
                    AND invalid_at IS NONE;
                """,
                {"group_id": group_id, "src": subject_uuid, "name": predicate},
            )
        )
        peers: list[EntityEdge] = []
        for row in rows:
            target = row.get("target_node_uuid") or row.get("out")
            target_uuid = target.split(":", 1)[-1] if isinstance(target, str) else target
            if target_uuid == exclude_object_uuid:
                continue
            row_key = row.get("fact_key") or ""
            parts = row_key.split("::")
            row_qhash = parts[4] if len(parts) >= 5 else ""
            if row_qhash != qualifier_hash_value:
                continue
            peers.append(parse_edge(row))
        return peers

    async def _mark_conflict_group(
        self, edge_uuids: list[str], conflict_group_id: str
    ) -> None:
        """Stamp ``conflict_group_id`` onto every listed edge so the
        whole group can be retrieved together by ``get_conflicts``."""
        if not edge_uuids:
            return
        await self.driver.query(
            """
            UPDATE relates_to
            SET conflict_group_id = $cg
            WHERE uuid IN $uuids;
            """,
            {"uuids": edge_uuids, "cg": conflict_group_id},
        )

    def merge_frames(
        self,
        *,
        source: str,
        target: str,
        group_id: str | None = None,
        strategy: str = "alias",
    ) -> RelationFrame:
        """Fold the ``source`` frame's canonical name + aliases into ``target``.

        After this call the registry resolves any prior alias of
        ``source`` to ``target``. Historical edges keep their stored
        ``canonical_name`` value (no DB rewrite) -- this preserves
        provenance and stays cheap. Pass ``group_id`` to scope the merge
        to one tenant; otherwise both frames are looked up in the
        global catalog.

        ``strategy`` is reserved for future modes (``"replace"``,
        ``"split"``); only ``"alias"`` is implemented today.
        """
        if strategy != "alias":
            raise ValueError(
                f"Unsupported merge strategy {strategy!r}; only 'alias' is implemented."
            )
        src_frame = self.relation_frames.get(source, group_id=group_id or "")
        tgt_frame = self.relation_frames.get(target, group_id=group_id or "")
        if src_frame is None or tgt_frame is None:
            raise KeyError(
                f"merge_frames: unknown frame(s) source={source!r} target={target!r}"
            )
        if src_frame is tgt_frame:
            return tgt_frame
        existing = {a.lower() for a in tgt_frame.aliases}
        new_aliases = list(tgt_frame.aliases)
        for cand in [src_frame.canonical_name, *src_frame.aliases]:
            key = (cand or "").strip().lower()
            if key and key != tgt_frame.canonical_name.lower() and key not in existing:
                new_aliases.append(key)
                existing.add(key)
        tgt_frame.aliases = new_aliases
        # Re-register so the registry's alias index picks up the new keys.
        self.relation_frames.register(tgt_frame, group_id=group_id)
        return tgt_frame

    async def current_profile(
        self,
        *,
        subject_uuid: str,
        group_id: str = "",
        limit: int = 200,
    ) -> dict[str, list[EntityEdge]]:
        """Return all currently-true facts for ``subject_uuid`` grouped
        by canonical relation name.

        Convenience over :meth:`get_current_facts` for building
        "what do you know about me right now?" surfaces. Edges that
        carry a ``canonical_name`` (from a resolved frame) are bucketed
        by that name; edges without one fall back to their raw
        ``name`` so unregistered predicates still surface.
        """
        edges = await self.get_current_facts(
            subject_uuid=subject_uuid, group_id=group_id, limit=limit
        )
        grouped: dict[str, list[EntityEdge]] = defaultdict(list)
        for edge in edges:
            key = edge.canonical_name or edge.name
            grouped[key].append(edge)
        return dict(grouped)

    async def get_current_fact(
        self,
        *,
        subject_uuid: str,
        predicate: str,
        group_id: str = "",
    ) -> EntityEdge | None:
        """Return the single live edge for a (subject, predicate) slot, or None.

        Generic current-state resolver: walks ``relates_to`` for the most
        recent ``active`` edge with the given subject and predicate. No
        canonical-predicate translation -- whatever string the extractor
        used IS the predicate.
        """

        edges = await self.get_current_facts(
            subject_uuid=subject_uuid,
            group_id=group_id,
            predicate=predicate,
            limit=1,
        )
        return edges[0] if edges else None

    async def get_current_facts(
        self,
        *,
        subject_uuid: str,
        group_id: str = "",
        predicate: str | None = None,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[EntityEdge]:
        """Return all live edges for a subject, optionally scoped to predicate or domain.

        Useful for "what do you know about me currently?" recall without
        relying on hybrid search to surface every active fact.
        """

        from surriti.search import _unwrap

        clauses = [
            "group_id = $group_id",
            'in = type::record("entity", $src)',
            'status = "active"',
            "invalid_at IS NONE",
        ]
        params: dict[str, Any] = {
            "group_id": group_id,
            "src": subject_uuid,
            "limit": int(limit),
        }
        if predicate is not None:
            clauses.append("name = $name")
            params["name"] = predicate
        if domain is not None:
            clauses.append("domain = $domain")
            params["domain"] = domain
        surql = (
            "SELECT * FROM relates_to WHERE "
            + " AND ".join(clauses)
            + " ORDER BY valid_at DESC LIMIT $limit;"
        )
        rows = _unwrap(await self.driver.query(surql, params))
        return [parse_edge(r) for r in rows]

    async def get_facts_as_of(
        self,
        *,
        subject_uuid: str,
        as_of: datetime,
        group_id: str = "",
        predicate: str | None = None,
        domain: str | None = None,
        limit: int = 200,
    ) -> list[EntityEdge]:
        """Return edges that were valid at the given timestamp.

        An edge is valid "as of" ``as_of`` when its ``valid_at`` is at
        or before ``as_of`` AND its ``invalid_at`` is either unset or
        strictly after ``as_of``. The query is generic and uses no
        hardcoded predicate vocabulary -- pass ``predicate`` or
        ``domain`` to scope the result.
        """

        from surriti.search import _unwrap

        clauses = [
            "group_id = $group_id",
            'in = type::record("entity", $src)',
            "(valid_at IS NONE OR valid_at <= $as_of)",
            "(invalid_at IS NONE OR invalid_at > $as_of)",
        ]
        params: dict[str, Any] = {
            "group_id": group_id,
            "src": subject_uuid,
            "as_of": as_of,
            "limit": int(limit),
        }
        if predicate is not None:
            clauses.append("name = $name")
            params["name"] = predicate
        if domain is not None:
            clauses.append("domain = $domain")
            params["domain"] = domain
        surql = (
            "SELECT * FROM relates_to WHERE "
            + " AND ".join(clauses)
            + " ORDER BY valid_at DESC LIMIT $limit;"
        )
        rows = _unwrap(await self.driver.query(surql, params))
        return [parse_edge(r) for r in rows]

    async def get_state_as_of(
        self,
        *,
        subject_uuid: str,
        as_of: datetime,
        group_id: str = "",
        predicate: str | None = None,
        domain: str | None = None,
    ) -> dict[tuple[str, str], EntityEdge]:
        """Return the latest valid edge per ``(predicate, object)`` slot.

        Convenience reducer for visualizer "as of" rendering: collapses
        the raw list from :meth:`get_facts_as_of` to one edge per slot,
        keeping the one with the most recent ``valid_at``.
        """

        edges = await self.get_facts_as_of(
            subject_uuid=subject_uuid,
            as_of=as_of,
            group_id=group_id,
            predicate=predicate,
            domain=domain,
        )
        latest: dict[tuple[str, str], EntityEdge] = {}
        for edge in edges:
            key = (edge.name, edge.target_node_uuid)
            existing = latest.get(key)
            if existing is None:
                latest[key] = edge
                continue
            ev = edge.valid_at or datetime.min.replace(tzinfo=timezone.utc)
            xv = existing.valid_at or datetime.min.replace(tzinfo=timezone.utc)
            if ev > xv:
                latest[key] = edge
        return latest

    async def _find_equivalent_edge(
        self,
        *,
        group_id: str,
        subject_uuid: str,
        object_uuid: str,
        predicate: str,
        fact_text: str,
        qualifier_hash_value: str = "",
        require_text_match: bool = False,
    ) -> EntityEdge | None:
        """Find an existing active edge for the same canonical triple.

        Equivalence is determined by the deterministic ``fact_key``
        (group_id + subject + predicate + object [+ qualifier_hash]) --
        not by the natural-language ``fact`` string. Two extractions
        that produce the same triple from differently-worded source
        text describe the same world fact and must collapse onto a
        single edge so the new episode just appends to its supporting
        list. Temporal changes ("X moved", "Y sold the car") are not
        masked because the extractor surfaces them as
        ``operation="terminate"`` or via cross-predicate contradiction
        detection in :meth:`_add_fact_edge`, NOT via a fact-text
        diff heuristic.

        ``require_text_match`` and ``fact_text`` are kept for backward
        compatibility with callers but are otherwise unused.
        """

        from surriti.search import _unwrap

        del require_text_match, fact_text  # legacy parameters, see docstring

        key = make_fact_key(
            group_id, subject_uuid, predicate, object_uuid,
            qualifier_hash=qualifier_hash_value,
        )
        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND fact_key = $key
                    AND invalid_at IS NONE
                LIMIT 10;
                """,
                {"group_id": group_id, "key": key},
            )
        )
        if rows:
            return parse_edge(rows[0])

        # Legacy fallback for rows written before ``fact_key`` existed.
        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND in = type::record("entity", $src)
                    AND out = type::record("entity", $tgt)
                    AND name = $name
                    AND invalid_at IS NONE
                LIMIT 10;
                """,
                {
                    "group_id": group_id,
                    "src": subject_uuid,
                    "tgt": object_uuid,
                    "name": predicate,
                },
            )
        )
        return parse_edge(rows[0]) if rows else None

    async def _link_mentions(
        self,
        *,
        episode: EpisodicNode,
        entities: list[EntityNode],
        group_id: str,
    ) -> list[EpisodicEdge]:
        edges: list[EpisodicEdge] = []
        for ent in entities:
            ee = EpisodicEdge(
                group_id=group_id,
                source_node_uuid=episode.uuid,
                target_node_uuid=ent.uuid,
            )
            await self.driver.query(
                """
                RELATE (type::record("episode", $ep))->mentions->(type::record("entity", $en))
                CONTENT {
                    uuid: $uuid,
                    group_id: $group_id,
                    created_at: $created_at
                };
                """,
                {
                    "ep": episode.uuid,
                    "en": ent.uuid,
                    "uuid": ee.uuid,
                    "group_id": ee.group_id,
                    "created_at": ee.created_at,
                },
            )
            edges.append(ee)
        return edges


    # ------------------------------------------------------------------
    # Self-awareness API
    # ------------------------------------------------------------------

    async def add_self_episode(
        self,
        *,
        episode_type: EpisodeType,
        content: str,
        group_id: str = "",
        name: str | None = None,
        reference_time: datetime | None = None,
    ) -> AddEpisodeResults:
        """Store a self-referential episode for operational self-awareness.

        Self-episodes are stored in the universal memory graph alongside
        world/user facts, but are flagged with a special source type so the
        cognition layer can process them differently (self-model extraction,
        pattern detection, belief refinement).

        Supported types:
        - ``self_observation`` — explicit reflection ("I was too verbose")
        - ``self_correction`` — noticing a mistake
        - ``self_success`` — noticing a win
        - ``self_pattern`` — recurring behavioral trend

        Parameters
        ----------
        episode_type : EpisodeType
            One of the self-episode types from EpisodeType enum.
        content : str
            The self-observation content.
        group_id : str
            Tenant/group ID for isolation.
        name : str | None
            Optional episode name. Defaults to f"self_{episode_type.value}".
        reference_time : datetime | None
            Optional timestamp. Defaults to now.

        Returns
        -------
        AddEpisodeResults with the stored episode and any derived entities/edges.
        """
        if episode_type not in (
            EpisodeType.self_observation,
            EpisodeType.self_correction,
            EpisodeType.self_success,
            EpisodeType.self_pattern,
        ):
            raise ValueError(
                f"Invalid episode_type {episode_type!r}. "
                f"Must be one of: self_observation, self_correction, "
                f"self_success, self_pattern"
            )

        episode_name = name or f"self_{episode_type.value}"
        ref_time = reference_time or datetime.now(timezone.utc)

        episode = EpisodicNode(
            uuid=str(uuid4()),
            name=episode_name,
            content=content,
            source=episode_type,
            source_description=f"self_{episode_type.value}",
            reference_time=ref_time,
            group_id=group_id,
        )
        await self._save_episode(episode)

        # Extract self-referential entities/facts via LLM
        extraction = await self.llm.extract(
            content,
            group_id=group_id,
            custom_instructions=(
                "This is a SELF-REFERENTIAL episode about the AI assistant's "
                "own behavior, not about the user or the world. Extract facts "
                "about the assistant's behavior, patterns, or self-assessment. "
                "Do NOT extract facts about external entities or world knowledge."
            ),
        )

        entities: list[EntityNode] = []
        edges: list[EntityEdge] = []

        # Create/update a "self" entity for this group
        self_entity_name = f"assistant_{group_id}" if group_id else "assistant"
        self_entity = await self._get_entity_by_name(
            name=self_entity_name,
            group_id=group_id,
        )
        if self_entity is None:
            self_entity_uuid = str(uuid4())
            self_entity = EntityNode(
                uuid=self_entity_uuid,
                name=self_entity_name,
                summary=f"Self-referential entity for group {group_id or 'default'}",
                labels=["SelfEntity", "Assistant"],
                group_id=group_id,
            )
            await self.driver.query(
                f"""
                CREATE type::record("entity", $uuid) CONTENT {{
                    uuid: $uuid,
                    group_id: $group_id,
                    name: $name,
                    summary: $summary,
                    labels: $labels,
                    created_at: $created_at
                }};
                """,
                {
                    "uuid": self_entity_uuid,
                    "group_id": group_id,
                    "name": self_entity_name,
                    "summary": self_entity.summary,
                    "labels": self_entity.labels,
                    "created_at": self_entity.created_at.isoformat(),
                },
            )
        entities.append(self_entity)

        # Create fact edges for each extracted self-fact
        for fact in extraction.facts:
            edge, invalidated = await self._add_fact_edge(
                fact=fact,
                subject=EntityNode(name=self_entity_name, labels=["SelfEntity"]),
                obj=EntityNode(name=fact.object, labels=[]),
                episode=episode,
                group_id=group_id,
                source_type="assistant",
            )
            edges.append(edge)
            if invalidated:
                edges.extend(invalidated)

        # Link episode to self-entity
        episodic_edges = await self._link_mentions(
            episode=episode,
            entities=[self_entity],
            group_id=group_id,
        )

        return AddEpisodeResults(
            episode=episode,
            episodic_edges=episodic_edges,
            nodes=entities,
            edges=edges,
        )

    async def get_self_model(
        self,
        *,
        group_id: str,
        include_traits: bool = True,
        include_patterns: bool = True,
        include_beliefs: bool = True,
    ) -> dict[str, Any]:
        """Return the current self-model for a group.

        The self-model is a synthesized view of the assistant's own
        behavior patterns, traits, beliefs, and goals derived from
        self-episodes stored via :meth:`add_self_episode`.

        Parameters
        ----------
        group_id : str
            Tenant/group ID.
        include_traits : bool
            Include synthesized trait entities.
        include_patterns : bool
            Include interaction pattern analysis.
        include_beliefs : bool
            Include belief edges (subjective self-assessments).

        Returns
        -------
        Dict with keys: traits, patterns, beliefs, goals, summary.
        """
        from surriti.search import _unwrap

        result: dict[str, Any] = {
            "group_id": group_id,
            "traits": [],
            "patterns": [],
            "beliefs": [],
            "goals": [],
            "summary": "",
        }

        # 1. Trait entities tied to the self-entity
        if include_traits:
            self_entity = await self._get_entity_by_name(
                name=f"assistant_{group_id}" if group_id else "assistant",
                group_id=group_id,
            )
            if self_entity:
                trait_edges = await self.driver.query(
                    """
                    SELECT * FROM relates_to
                    WHERE group_id = $group_id
                        AND in = type::record("entity", $src)
                        AND name = "has_trait"
                        AND status = "active"
                        AND invalid_at IS NONE;
                    """,
                    {"group_id": group_id, "src": self_entity.uuid},
                )
                result["traits"] = [
                    {"fact": _unwrap(row).get("fact", ""), "confidence": _unwrap(row).get("confidence", 1.0)}
                    for row in trait_edges
                ]

        # 2. Interaction patterns from procedural cognition
        if include_patterns:
            episodes = await self.driver.query(
                """
                SELECT name, content, interaction_pattern, created_at
                FROM episode
                WHERE group_id = $group_id
                    AND source LIKE "self_%"
                ORDER BY created_at DESC
                LIMIT 50;
                """,
                {"group_id": group_id},
            )
            patterns = defaultdict(int)
            for ep in episodes:
                row = _unwrap(ep)
                pattern = row.get("interaction_pattern")
                if pattern:
                    patterns[pattern] += 1
            result["patterns"] = [
                {"pattern": k, "count": v, "confidence": v / max(len(episodes), 1)}
                for k, v in sorted(patterns.items(), key=lambda x: -x[1])
            ]

        # 3. Belief edges (subjective self-assessments)
        if include_beliefs:
            belief_edges = await self.driver.query(
                """
                SELECT * FROM relates_to
                WHERE group_id = $group_id
                    AND (attributes->>'$.is_belief' = true OR is_belief = true)
                    AND status = "active"
                    AND invalid_at IS NONE;
                """,
                {"group_id": group_id},
            )
            result["beliefs"] = [
                {"fact": _unwrap(row).get("fact", ""), "source_type": "self"}
                for row in belief_edges
            ]

        # 4. Goal entities
        if include_beliefs:
            result["goals"] = []  # Populated by cognition pass

        # 5. Summary
        total_self_episodes = len(
            await self.driver.query(
                """
                SELECT count() as cnt FROM episode
                WHERE group_id = $group_id
                    AND source LIKE "self_%";
                """,
                {"group_id": group_id},
            )
        )
        result["summary"] = (
            f"Self-model based on {total_self_episodes} self-episodes. "
            f"{len(result['traits'])} traits identified. "
            f"{len(result['patterns'])} interaction patterns detected."
        )

        return result

    async def _get_entity_by_name(
        self,
        *,
        name: str,
        group_id: str,
    ) -> EntityNode | None:
        """Find an entity by name in a group."""
        from surriti.search import _unwrap

        rows = _unwrap(
            await self.driver.query(
                """
                SELECT * FROM entity
                WHERE group_id = $group_id
                    AND name = $name
                LIMIT 1;
                """,
                {"group_id": group_id, "name": name},
            )
        )
        if rows:
            return parse_entity(rows[0])
        return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# Graphiti-compatible class name for consumers migrating imports.
Graphiti = Surriti
