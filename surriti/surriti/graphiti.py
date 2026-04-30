"""Top-level Surriti class - mirrors the public surface of ``graphiti.Graphiti``."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from surriti.driver import SurrealDriver
from surriti.edges import CommunityEdge, EntityEdge, EpisodicEdge
from surriti.embedder import DummyEmbedder, EmbedderClient
from surriti.llm import (
    DummyLLMClient,
    ExtractedEntity,
    ExtractedFact,
    LLMClient,
)
from surriti.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
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
from surriti.temporal import resolve_contradictions
from surriti.utils import parse_community, parse_edge, parse_entity, parse_episode

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.driver = driver
        self.llm = llm_client or DummyLLMClient()
        self.embedder = embedder or DummyEmbedder(embedding_dim=driver.embedding_dim)
        self.cross_encoder = cross_encoder

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_env(
        cls,
        *,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
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
        return cls(driver, llm_client=llm_client, embedder=embedder, cross_encoder=cross_encoder)

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> "Surriti":
        """Connect the underlying driver and apply the schema. Idempotent."""

        if hasattr(self.driver, "connect"):
            await self.driver.connect()
        if hasattr(self.driver, "init_schema"):
            await self.driver.init_schema()
        return self

    async def close(self) -> None:
        """Close the underlying driver. Safe to call multiple times."""

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
        custom_extraction_instructions: str | None = None,
        speaker_id: str | None = None,
        speaker_name: str | None = None,
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
                    f"(Speaker context: \"I\"/\"me\"/\"my\" refer to "
                    f"\"{speaker_name}\" (stable id \"{speaker_id}\"). "
                    f"Use \"{speaker_name}\" as the subject for facts "
                    f"about the speaker. The OBJECT must be the actual "
                    f"value mentioned in the text -- never the literal "
                    f"word \"speaker\" or \"{speaker_name}\" again. "
                    f"Examples: \"I work at Acme\" -> "
                    f"`{speaker_name} -[works_at]-> Acme`; "
                    f"\"my birthday is October 14\" -> "
                    f"`{speaker_name} -[has_birthday]-> October 14`; "
                    f"\"I am 33 years old\" -> "
                    f"`{speaker_name} -[is_age]-> 33`. Add the value "
                    f"(Acme, October 14, 33) to the entities list too.)"
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
                    f"entities list too.)"
                )
            custom_extraction_instructions = (
                f"{custom_extraction_instructions}\n\n{speaker_hint}"
                if custom_extraction_instructions
                else speaker_hint
            )

        # Optionally include previous episode text as extra context.
        context_text = episode_body
        if previous_episode_uuids:
            ctx = await self._fetch_episode_contents(previous_episode_uuids)
            if ctx:
                context_text = f"{ctx}\n---\n{episode_body}"

        extraction = await self.llm.extract(
            context_text,
            group_id=group_id,
            entity_types=entity_types,
            custom_instructions=custom_extraction_instructions,
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

        entities = await self._upsert_entities(extraction.entities, group_id=group_id)
        name_to_entity = {e.name: e for e in entities}

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
        # (identity/aliasing). Everything else with subject == object is LLM
        # garbage we drop as defence-in-depth on top of the system prompt.
        _IDENTITY_PREDICATES = {"is_named", "is_called", "is_self", "is_aka"}

        edges: list[EntityEdge] = []
        invalidated_all: list[EntityEdge] = []
        for fact in extraction.facts:
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
            predicate = (fact.predicate or "").lower()
            if subject.uuid == obj.uuid:
                # Speaker-aware repair: small models often emit
                # "Auley is_named Auley" when the speaker's stable id is
                # "default". Rewrite the subject to the speaker's id so
                # the naming edge connects two distinct entities.
                if (
                    speaker_id
                    and predicate in _IDENTITY_PREDICATES
                    and fact.subject != speaker_id
                    and speaker_id != fact.object
                ):
                    speaker_ents = await self._upsert_entities(
                        [ExtractedEntity(name=speaker_id, labels=["User"])],
                        group_id=group_id,
                    )
                    subject = speaker_ents[0]
                    name_to_entity[speaker_id] = subject
                    if subject.uuid not in {e.uuid for e in entities}:
                        entities.append(subject)
                    logger.debug(
                        "Rewrote self-loop identity fact subject %r -> %r",
                        fact.subject, speaker_id,
                    )
                elif predicate in _IDENTITY_PREDICATES and not speaker_id:
                    # Identity predicate but no speaker context to repair
                    # with -- keep the self-loop as a fallback so we don't
                    # lose the naming entirely.
                    pass
                else:
                    logger.debug(
                        "Skipping self-loop fact %r -[%s]-> %r",
                        fact.subject, fact.predicate, fact.object,
                    )
                    continue
            edge, invalidated = await self._add_fact_edge(
                fact=fact,
                subject=subject,
                obj=obj,
                episode=episode,
                group_id=group_id,
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
        *,
        group_id: str | None = None,
        config: SearchConfig | None = None,
    ) -> SearchResults:
        """Hybrid (vector + BM25) search over EntityEdge facts."""

        embedding = await self.embedder.create(query) if query else None
        cfg = config or SearchConfig()
        if cfg.cross_encoder is None:
            cfg.cross_encoder = self.cross_encoder
        return await hybrid_search(
            self.driver,
            query=query,
            query_embedding=embedding,
            group_id=group_id,
            config=cfg,
        )

    async def search_(
        self,
        query: str,
        *,
        group_id: str | None = None,
        config: SearchConfig | None = None,
        filters: SearchFilters | None = None,
    ) -> SearchResults:
        """Advanced search returning edges + nodes + episodes + communities.

        Mirrors Graphiti's ``search_()``. Honors all reranker strategies and
        ``SearchFilters`` (date ranges, edge types, labels, properties).
        """

        cfg = config or SearchConfig(include_nodes=True, include_episodes=True)
        if filters is not None:
            cfg.filters = filters
        if cfg.cross_encoder is None:
            cfg.cross_encoder = self.cross_encoder

        embedding = await self.embedder.create(query) if query else None
        results = await hybrid_search(
            self.driver,
            query=query,
            query_embedding=embedding,
            group_id=group_id,
            config=cfg,
        )
        if cfg.include_nodes:
            results.nodes = await search_nodes(
                self.driver,
                query=query,
                query_embedding=embedding,
                group_id=group_id,
                limit=cfg.limit,
                filters=cfg.filters,
            )
        if cfg.include_episodes:
            results.episodes = await search_episodes(
                self.driver,
                query=query,
                group_id=group_id,
                limit=cfg.limit,
            )
        if cfg.include_communities:
            results.communities = await search_communities(
                self.driver,
                query=query,
                query_embedding=embedding,
                group_id=group_id,
                limit=cfg.limit,
            )
        return results

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
        self, extracted: list[ExtractedEntity], *, group_id: str
    ) -> list[EntityNode]:
        """Insert new entities; reuse existing ones by ``(group_id, name)``.

        Idempotent under any input duplication and tolerant to a race against
        the ``entity_name_uniq`` index: if a CREATE collides we re-SELECT
        and reuse the now-existing row instead of bubbling the error up.
        Tenant isolation is enforced by ``entity_name_uniq`` on
        ``(group_id, name)`` — entities with the same name in different
        ``group_id``s are deliberately distinct nodes.
        """

        if not extracted:
            return []

        # 1. Dedupe input by name (preserve first occurrence) so the LLM
        #    emitting "Michael" twice in one extraction doesn't trip the
        #    unique index on the second CREATE.
        seen: dict[str, ExtractedEntity] = {}
        for ext in extracted:
            if ext.name and ext.name not in seen:
                seen[ext.name] = ext
        deduped = list(seen.values())

        from surriti.search import _unwrap

        # 2. Bulk lookup of every existing entity in this tenant.
        rows = _unwrap(await self.driver.query(
            "SELECT * FROM entity WHERE group_id = $g AND name IN $names;",
            {"g": group_id, "names": [e.name for e in deduped]},
        ))
        existing: dict[str, EntityNode] = {r["name"]: parse_entity(r) for r in rows}

        results: list[EntityNode] = []
        for ext in deduped:
            if ext.name in existing:
                results.append(existing[ext.name])
                continue
            embedding = await self.embedder.create(ext.name)
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
                        "created_at": node.created_at,
                    },
                )
                results.append(node)
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
                    raise
                results.append(parse_entity(fallback[0]))
        return results

    async def _add_fact_edge(
        self,
        *,
        fact: ExtractedFact,
        subject: EntityNode,
        obj: EntityNode,
        episode: EpisodicNode | None,
        group_id: str,
    ) -> tuple[EntityEdge, list[EntityEdge]]:
        valid_at = _parse_iso(fact.valid_at) or (
            episode.reference_time if episode else datetime.now(timezone.utc)
        )
        invalid_at = _parse_iso(fact.invalid_at)
        fact_text = fact.fact or f"{subject.name} {fact.predicate} {obj.name}."
        embedding = await self.embedder.create(fact_text)

        edge = EntityEdge(
            uuid=str(uuid4()),
            group_id=group_id,
            source_node_uuid=subject.uuid,
            target_node_uuid=obj.uuid,
            name=fact.predicate,
            fact=fact_text,
            fact_embedding=embedding,
            episodes=[episode.uuid] if episode else [],
            valid_at=valid_at,
            invalid_at=invalid_at,
        )

        invalidated = await resolve_contradictions(
            self.driver,
            llm=self.llm,
            new_fact=fact_text,
            new_fact_embedding=embedding,
            new_valid_at=valid_at,
            group_id=group_id,
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
                attributes: {},
                created_at: $created_at
            };
            """,
            {
                "src": subject.uuid,
                "tgt": obj.uuid,
                "uuid": edge.uuid,
                "group_id": edge.group_id,
                "name": edge.name,
                "fact": edge.fact,
                "emb": edge.fact_embedding,
                "episodes": edge.episodes,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
                "created_at": edge.created_at,
            },
        )
        return edge, invalidated

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


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
