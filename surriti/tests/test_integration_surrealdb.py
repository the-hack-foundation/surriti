"""Integration tests against a live SurrealDB instance.

Skipped unless ``SURRITI_TEST_SURREAL_URL`` is set (defaults to ``ws://localhost:8000/rpc``)
*and* the server is reachable. Bring it up with::

    docker compose up -d
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from surriti import (
    ComparisonOperator,
    DateFilter,
    DummyCrossEncoder,
    DummyEmbedder,
    EpisodeType,
    Reranker,
    SearchConfig,
    SearchFilters,
    SurrealDriver,
    Surriti,
)


URL = os.environ.get("SURRITI_TEST_SURREAL_URL", "ws://localhost:8000/rpc")


def _server_reachable(url: str) -> bool:
    # Crude TCP probe so we can skip gracefully when the container is down.
    try:
        host = url.split("//", 1)[1].split("/", 1)[0]
        host, port = host.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(URL), reason="SurrealDB not reachable at " + URL
)


@pytest.fixture
async def memory():
    driver = SurrealDriver(
        url=URL,
        namespace="surriti_it",
        database="run_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        username="root",
        password="root",
        embedding_dim=64,
    )
    await driver.connect()
    m = Surriti(driver, embedder=DummyEmbedder(64), cross_encoder=DummyCrossEncoder())
    await m.build_indices_and_constraints()
    await driver.clear()
    try:
        yield m
    finally:
        await driver.clear()
        await driver.close()


# ---------------------------------------------------------------- ingest


async def test_add_episode_persists_episode_entities_and_edges(memory):
    res = await memory.add_episode(
        name="meeting",
        episode_body="Alice met Bob at Acme Corp.",
        source=EpisodeType.text,
        group_id="g1",
    )
    names = {n.name for n in res.nodes}
    assert {"Alice", "Bob", "Acme Corp"}.issubset(names)
    assert res.edges, "expected at least one fact edge"
    assert len(res.episodic_edges) == len(res.nodes)


async def test_add_triplet_creates_two_entities_and_one_edge(memory):
    res = await memory.add_triplet(
        subject_name="Carol",
        predicate="loves",
        object_name="ChocolateChip",
        fact="Carol loves ChocolateChip cookies.",
        group_id="g_triplet",
    )
    assert len(res.nodes) == 2
    assert len(res.edges) == 1
    assert res.edges[0].name == "loves"


async def test_add_episode_bulk(memory):
    bulk = [
        {"name": "ep1", "episode_body": "Alice works at Acme.", "source": EpisodeType.text},
        {"name": "ep2", "episode_body": "Bob joined Acme.", "source": EpisodeType.text},
    ]
    out = await memory.add_episode_bulk(bulk, group_id="g_bulk")
    assert len(out.episodes) == 2
    eps = await memory.retrieve_episodes(group_ids=["g_bulk"], last_n=10)
    assert len(eps) == 2


# ---------------------------------------------------------------- search


async def test_basic_search_returns_relevant_edge(memory):
    await memory.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp in San Francisco.",
        source=EpisodeType.text,
        group_id="g_search",
    )
    results = await memory.search("Alice Acme", group_id="g_search")
    assert results.edges
    assert any("Acme" in e.fact for e in results.edges)


async def test_search_advanced_returns_nodes_and_episodes(memory):
    await memory.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        source=EpisodeType.text,
        group_id="g_adv",
    )
    cfg = SearchConfig(include_nodes=True, include_episodes=True, limit=5)
    results = await memory.search_("Alice", group_id="g_adv", config=cfg)
    # Episodes use BM25 over `content`; node search uses BM25 over name/summary.
    assert any(n.name == "Alice" for n in results.nodes)


async def test_search_with_mmr_reranker(memory):
    for body in (
        "Alice met Bob at Acme.",
        "Alice met Bob at Globex.",
        "Carol met Dave at Initech.",
    ):
        await memory.add_episode(
            name=body[:5],
            episode_body=body,
            source=EpisodeType.text,
            group_id="g_mmr",
        )
    cfg = SearchConfig(reranker=Reranker.mmr, mmr_lambda=0.5, limit=3)
    res = await memory.search("Alice Bob", group_id="g_mmr", config=cfg)
    assert res.edges
    # MMR should preserve all results when the candidate pool >= limit.
    assert len(res.edges) == 3


async def test_search_with_cross_encoder(memory):
    await memory.add_episode(
        name="e1",
        episode_body="Alice met Bob at Acme.",
        source=EpisodeType.text,
        group_id="g_xe",
    )
    cfg = SearchConfig(reranker=Reranker.cross_encoder, limit=3)
    res = await memory.search("Acme Alice met", group_id="g_xe", config=cfg)
    assert res.edges


async def test_search_with_focal_reranker(memory):
    res1 = await memory.add_episode(
        name="e1",
        episode_body="Alice manages Bob and Charlie at Acme.",
        source=EpisodeType.text,
        group_id="g_focal",
    )
    alice_uuid = next(n.uuid for n in res1.nodes if n.name == "Alice")
    cfg = SearchConfig(focal_uuid=alice_uuid, limit=5)
    res = await memory.search("manages", group_id="g_focal", config=cfg)
    assert res.edges


async def test_search_filters_edge_types(memory):
    await memory.add_triplet(
        subject_name="X", predicate="owns", object_name="Y", group_id="g_flt"
    )
    await memory.add_triplet(
        subject_name="X", predicate="loves", object_name="Z", group_id="g_flt"
    )
    filters = SearchFilters(edge_types=["owns"])
    cfg = SearchConfig(filters=filters, only_valid=False, limit=10)
    res = await memory.search("X", group_id="g_flt", config=cfg)
    assert res.edges
    assert all(e.name == "owns" for e in res.edges)


async def test_search_filters_date_window(memory):
    past_ep = await memory.add_episode(
        name="old",
        episode_body="Alice met Bob.",
        source=EpisodeType.text,
        reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        group_id="g_date",
    )
    await memory.add_episode(
        name="new",
        episode_body="Alice met Charlie.",
        source=EpisodeType.text,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        group_id="g_date",
    )
    filters = SearchFilters(
        valid_at=[[DateFilter(date=datetime(2025, 1, 1, tzinfo=timezone.utc), op=ComparisonOperator.gte)]]
    )
    cfg = SearchConfig(filters=filters, only_valid=False, limit=10)
    res = await memory.search("Alice", group_id="g_date", config=cfg)
    assert res.edges
    assert all(e.valid_at and e.valid_at.year >= 2025 for e in res.edges)


# ---------------------------------------------------------------- temporal


async def test_contradiction_invalidates_prior_edge(memory):
    await memory.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        source=EpisodeType.text,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        group_id="g_temp",
    )
    res2 = await memory.add_episode(
        name="e2",
        episode_body="Alice no longer works at Acme Corp; Alice moved to Globex.",
        source=EpisodeType.text,
        reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        group_id="g_temp",
    )
    assert res2.invalidated_edges, "expected at least one prior fact to be invalidated"

    # Default search hides invalidated edges
    res = await memory.search("Alice Acme", group_id="g_temp")
    assert all((e.invalid_at is None) for e in res.edges)


# ---------------------------------------------------------------- retrieval & deletion


async def test_retrieve_episodes_returns_recent_first(memory):
    for i, ts in enumerate([2020, 2022, 2026]):
        await memory.add_episode(
            name=f"y{ts}",
            episode_body=f"event-{i}",
            source=EpisodeType.text,
            reference_time=datetime(ts, 6, 1, tzinfo=timezone.utc),
            group_id="g_ret",
        )
    eps = await memory.retrieve_episodes(group_id="g_ret", last_n=2)
    assert len(eps) == 2
    assert eps[0].reference_time.year >= eps[1].reference_time.year


async def test_get_nodes_and_edges_by_episode(memory):
    res = await memory.add_episode(
        name="e1",
        episode_body="Alice met Bob at Acme Corp.",
        source=EpisodeType.text,
        group_id="g_byep",
    )
    fetched = await memory.get_nodes_and_edges_by_episode([res.episode.uuid])
    assert fetched.edges
    assert fetched.nodes


async def test_remove_episode_drops_edges_with_no_other_source(memory):
    res = await memory.add_episode(
        name="solo",
        episode_body="Dave works at Initech.",
        source=EpisodeType.text,
        group_id="g_rm",
    )
    await memory.remove_episode(res.episode.uuid)
    leftover = await memory.search("Dave Initech", group_id="g_rm", config=SearchConfig(only_valid=False))
    assert not leftover.edges


async def test_delete_group_clears_everything(memory):
    await memory.add_episode(
        name="x",
        episode_body="Eve met Frank.",
        source=EpisodeType.text,
        group_id="g_clear",
    )
    await memory.delete_group("g_clear")
    eps = await memory.retrieve_episodes(group_id="g_clear", last_n=10)
    assert eps == []


# ---------------------------------------------------------------- communities


async def test_build_communities_groups_connected_entities(memory):
    await memory.add_triplet(subject_name="A", predicate="knows", object_name="B", group_id="g_com")
    await memory.add_triplet(subject_name="B", predicate="knows", object_name="C", group_id="g_com")
    await memory.add_triplet(subject_name="X", predicate="knows", object_name="Y", group_id="g_com")
    communities, edges = await memory.build_communities(group_id="g_com")
    assert len(communities) == 2  # {A,B,C} and {X,Y}
    assert len(edges) == 5  # 3 + 2 members


# ---------------------------------------------------------------- new feature integration


async def test_save_node_get_entity_node_round_trip(memory):
    from surriti import EntityNode

    n = EntityNode(name="Zara", group_id="g_save", summary="A person.", labels=["Person"])
    await memory.save_node(n)
    fetched = await memory.get_entity_node(n.uuid)
    assert fetched is not None
    assert fetched.name == "Zara"
    assert "Person" in fetched.labels


async def test_save_node_updates_existing(memory):
    from surriti import EntityNode

    n = EntityNode(name="Yves", group_id="g_save2", summary="v1")
    await memory.save_node(n)
    n.summary = "v2"
    await memory.save_node(n)
    fetched = await memory.get_entity_node(n.uuid)
    assert fetched.summary == "v2"


async def test_save_edge_updates_fact(memory):
    res = await memory.add_triplet(
        subject_name="Mia", predicate="visits", object_name="Paris",
        fact="Mia visits Paris.", group_id="g_se",
    )
    edge = res.edges[0]
    edge.fact = "Mia visits Paris in summer."
    await memory.save_edge(edge)
    fetched = await memory.get_entity_edge(edge.uuid)
    assert fetched is not None
    assert "summer" in fetched.fact


async def test_remove_edge_deletes_relation(memory):
    res = await memory.add_triplet(
        subject_name="Lin", predicate="owns", object_name="Bike", group_id="g_re",
    )
    eid = res.edges[0].uuid
    await memory.remove_edge(eid)
    assert await memory.get_entity_edge(eid) is None


async def test_get_episode_returns_persisted_episode(memory):
    res = await memory.add_episode(
        name="capture", episode_body="A short note.",
        source=EpisodeType.text, group_id="g_ep",
    )
    fetched = await memory.get_episode(res.episode.uuid)
    assert fetched is not None
    assert fetched.name == "capture"


async def test_retrieve_episodes_with_reference_time_and_source(memory):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, src in enumerate([EpisodeType.text, EpisodeType.message, EpisodeType.text]):
        await memory.add_episode(
            name=f"r{i}",
            episode_body=f"body {i}",
            source=src,
            reference_time=base + timedelta(days=i),
            group_id="g_rt",
        )
    # Filter by source - only text episodes
    eps = await memory.retrieve_episodes(
        reference_time=base + timedelta(days=10),
        last_n=10, group_ids=["g_rt"], source=EpisodeType.text,
    )
    assert len(eps) == 2
    assert all(e.source is EpisodeType.text for e in eps)


async def test_retrieve_episodes_multi_group(memory):
    await memory.add_episode(
        name="a", episode_body="x", source=EpisodeType.text,
        group_id="g_mga",
    )
    await memory.add_episode(
        name="b", episode_body="y", source=EpisodeType.text,
        group_id="g_mgb",
    )
    eps = await memory.retrieve_episodes(group_ids=["g_mga", "g_mgb"], last_n=10)
    groups = {e.group_id for e in eps}
    assert groups == {"g_mga", "g_mgb"}


async def test_add_episode_with_uuid_kwarg_uses_supplied_id(memory):
    fixed = "11111111-1111-1111-1111-111111111111"
    res = await memory.add_episode(
        name="fixed", episode_body="x", source=EpisodeType.text,
        group_id="g_uuid", uuid=fixed,
    )
    assert res.episode.uuid == fixed
    assert (await memory.get_episode(fixed)).name == "fixed"


async def test_add_episode_update_communities_kwarg_runs(memory):
    # Seed two triplets so a community exists
    await memory.add_triplet(subject_name="P", predicate="knows", object_name="Q", group_id="g_uc")
    res = await memory.add_episode(
        name="ep", episode_body="P met Q at the cafe.",
        source=EpisodeType.text, group_id="g_uc",
        update_communities=True,
    )
    # update_communities should have produced at least one community node
    assert len(res.communities) >= 1


async def test_add_episode_previous_episode_uuids_threads_context(memory):
    first = await memory.add_episode(
        name="first", episode_body="Alice met Bob.",
        source=EpisodeType.text, group_id="g_prev",
    )
    res = await memory.add_episode(
        name="second", episode_body="They went to dinner.",
        source=EpisodeType.text, group_id="g_prev",
        previous_episode_uuids=[first.episode.uuid],
    )
    # Smoke test - just verify no error and episode persisted
    assert res.episode.name == "second"


async def test_add_triplet_with_entity_node_objects(memory):
    from surriti import EntityEdge, EntityNode

    src = EntityNode(name="Sam", group_id="g_obj", labels=["Person"])
    tgt = EntityNode(name="Berlin", group_id="g_obj", labels=["Place"])
    e = EntityEdge(
        group_id="g_obj",
        source_node_uuid=src.uuid,
        target_node_uuid=tgt.uuid,
        name="lives_in",
        fact="Sam lives in Berlin.",
    )
    res = await memory.add_triplet(source_node=src, edge=e, target_node=tgt)
    assert len(res.nodes) == 2
    assert len(res.edges) == 1
    assert res.edges[0].name == "lives_in"


async def test_search_recipes_edge_rrf_returns_results(memory):
    from surriti import search_recipes as recipes

    await memory.add_episode(
        name="r1", episode_body="Alice loves Acme Corp.",
        source=EpisodeType.text, group_id="g_rec",
    )
    res = await memory.search_("Alice", group_id="g_rec", config=recipes.EDGE_HYBRID_SEARCH_RRF)
    assert res.edges



# ---------------------------------------------------------------- multi-tenant + idempotency

async def test_duplicate_extraction_does_not_crash_against_unique_index(memory):
    """LLM duplicates the same entity in one extraction; the pipeline must
    dedupe so the ``entity_name_uniq`` index is not violated."""

    from surriti.llm import (
        ExtractedEntity,
        ExtractedFact,
        ScriptedLLMClient,
        ScriptedResponse,
    )

    memory.llm = ScriptedLLMClient([
        ScriptedResponse(
            entities=[
                ExtractedEntity(name="Michael", labels=["Person"]),
                ExtractedEntity(name="Michael", labels=["Person"]),
            ],
            facts=[],
        )
    ])

    res = await memory.add_episode(
        name="dup", episode_body="hello, my name is Michael",
        source=EpisodeType.message, group_id="dup_tenant",
    )
    assert sum(1 for n in res.nodes if n.name == "Michael") == 1


async def test_multi_tenant_same_name_isolated(memory):
    """Same entity name in two different tenants must coexist as distinct rows
    and never appear in each other's search results."""

    from surriti.llm import (
        ExtractedEntity,
        ScriptedLLMClient,
        ScriptedResponse,
    )

    memory.llm = ScriptedLLMClient([
        ScriptedResponse(entities=[ExtractedEntity(name="Michael", labels=["Person"])]),
        ScriptedResponse(entities=[ExtractedEntity(name="Michael", labels=["Person"])]),
    ])

    await memory.add_episode(
        name="a", episode_body="hi from alice",
        source=EpisodeType.message, group_id="alice-uuid",
    )
    await memory.add_episode(
        name="b", episode_body="hi from bob",
        source=EpisodeType.message, group_id="bob-uuid",
    )

    from surriti.search import _unwrap

    rows = _unwrap(await memory.driver.query(
        "SELECT * FROM entity WHERE name = 'Michael';", {}
    ))
    michaels = [r for r in rows if r.get("name") == "Michael"]
    assert len(michaels) == 2
    assert {m["group_id"] for m in michaels} == {"alice-uuid", "bob-uuid"}


async def test_repeated_add_episode_reuses_entity(memory):
    """Same tenant ingests the same Michael-bearing episode twice → one entity,
    two episodes, both linked via mentions edges."""

    from surriti.llm import (
        ExtractedEntity,
        ScriptedLLMClient,
        ScriptedResponse,
    )

    memory.llm = ScriptedLLMClient([
        ScriptedResponse(entities=[ExtractedEntity(name="Michael", labels=["Person"])]),
        ScriptedResponse(entities=[ExtractedEntity(name="Michael", labels=["Person"])]),
    ])

    await memory.add_episode(
        name="t1", episode_body="hi I'm Michael",
        source=EpisodeType.message, group_id="rep",
    )
    await memory.add_episode(
        name="t2", episode_body="hi again, Michael",
        source=EpisodeType.message, group_id="rep",
    )

    from surriti.search import _unwrap

    rows = _unwrap(await memory.driver.query(
        "SELECT * FROM entity WHERE group_id = 'rep' AND name = 'Michael';", {}
    ))
    assert len(rows) == 1


async def test_upsert_user_creates_and_updates_canonical_user_entity(memory):
    u1 = await memory.upsert_user(group_id="user-42")
    u2 = await memory.upsert_user(group_id="user-42", display_name="Michael")
    u3 = await memory.upsert_user(group_id="user-42", display_name="Michael")
    assert u1.uuid == u2.uuid == u3.uuid
    assert "User" in u2.labels

    from surriti.search import _unwrap

    rows = _unwrap(await memory.driver.query(
        "SELECT * FROM entity WHERE group_id = 'user-42' AND name = 'user-42';", {}
    ))
    assert len(rows) == 1
    assert rows[0]["attributes"].get("display_name") == "Michael"


async def test_add_episode_with_speaker_id_creates_user_entity(memory):
    from surriti.llm import (
        ExtractedEntity,
        ScriptedLLMClient,
        ScriptedResponse,
    )

    memory.llm = ScriptedLLMClient([
        ScriptedResponse(entities=[ExtractedEntity(name="Michael", labels=["Person"])]),
    ])

    await memory.add_episode(
        name="intro", episode_body="hello, my name is Michael",
        source=EpisodeType.message, group_id="user-99",
        speaker_id="user-99", speaker_name="Michael",
    )

    from surriti.search import _unwrap

    rows = _unwrap(await memory.driver.query(
        "SELECT * FROM entity WHERE group_id = 'user-99' AND name = 'user-99';", {}
    ))
    assert len(rows) == 1
    assert "User" in rows[0]["labels"]
    assert rows[0]["attributes"].get("display_name") == "Michael"
