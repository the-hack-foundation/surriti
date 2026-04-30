"""Pure unit tests for surriti's standalone helpers (no driver, no LLM IO).

These tests exercise:
- SearchFilters predicate helpers (date/property/edge/label filtering)
- Reranker primitives (RRF, MMR, cross-encoder, episode_mentions)
- ScriptedLLMClient queue & call recording
- search_recipes module exports the expected presets
- AddEpisodeResults / AddBulkEpisodeResults / RawEpisode / AddTripletResults shapes
- DummyCrossEncoder ranking
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from surriti import (
    AddBulkEpisodeResults,
    AddEpisodeResults,
    AddTripletResults,
    ComparisonOperator,
    DateFilter,
    DummyCrossEncoder,
    DummyEmbedder,
    EntityEdge,
    EntityNode,
    EpisodicEdge,
    EpisodicNode,
    PropertyFilter,
    RawEpisode,
    Reranker,
    ScriptedLLMClient,
    ScriptedResponse,
    SearchConfig,
    SearchFilters,
)
from surriti import search_recipes
from surriti.llm import ExtractedEntity, ExtractedFact
from surriti.rerankers import (
    cross_encoder_rerank,
    episode_mentions_rerank,
    mmr_rerank,
    rrf,
)
from surriti.search_filters import edge_passes_filters, node_passes_filters


# --------------------------------------------------------------- search_filters
NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


def test_edge_passes_filters_no_filter():
    assert edge_passes_filters({"name": "owns"}, None) is True


def test_edge_passes_filters_edge_type_match_and_miss():
    flt = SearchFilters(edge_types=["owns"])
    assert edge_passes_filters({"name": "owns"}, flt) is True
    assert edge_passes_filters({"name": "loves"}, flt) is False


def test_edge_passes_filters_edge_uuid():
    flt = SearchFilters(edge_uuids=["u1"])
    assert edge_passes_filters({"uuid": "u1"}, flt) is True
    assert edge_passes_filters({"uuid": "u2"}, flt) is False


def test_edge_passes_filters_valid_at_window_or_of_and():
    # Window 1: valid_at >= 2025-01-01 AND <= 2025-12-31
    # Window 2: valid_at >= 2027-01-01
    flt = SearchFilters(
        valid_at=[
            [
                DateFilter(date=datetime(2025, 1, 1, tzinfo=timezone.utc), op=ComparisonOperator.gte),
                DateFilter(date=datetime(2025, 12, 31, tzinfo=timezone.utc), op=ComparisonOperator.lte),
            ],
            [DateFilter(date=datetime(2027, 1, 1, tzinfo=timezone.utc), op=ComparisonOperator.gte)],
        ]
    )
    inside_w1 = {"valid_at": datetime(2025, 6, 1, tzinfo=timezone.utc)}
    outside = {"valid_at": datetime(2026, 6, 1, tzinfo=timezone.utc)}
    inside_w2 = {"valid_at": datetime(2027, 6, 1, tzinfo=timezone.utc)}
    assert edge_passes_filters(inside_w1, flt) is True
    assert edge_passes_filters(outside, flt) is False
    assert edge_passes_filters(inside_w2, flt) is True


def test_edge_passes_filters_is_null():
    flt = SearchFilters(
        invalid_at=[[DateFilter(date=NOW, op=ComparisonOperator.is_null)]]
    )
    assert edge_passes_filters({"invalid_at": None}, flt) is True
    assert edge_passes_filters({"invalid_at": NOW}, flt) is False


def test_edge_passes_filters_property():
    flt = SearchFilters(
        property_filters=[PropertyFilter(name="confidence", op=ComparisonOperator.gte, value=0.5)]
    )
    assert edge_passes_filters({"attributes": {"confidence": 0.7}}, flt) is True
    assert edge_passes_filters({"attributes": {"confidence": 0.3}}, flt) is False


def test_node_passes_filters_label_subset():
    flt = SearchFilters(node_labels=["Person"])
    assert node_passes_filters({"labels": ["Person", "Entity"]}, flt) is True
    assert node_passes_filters({"labels": ["Place"]}, flt) is False


# ------------------------------------------------------------------- rerankers
def test_rrf_orders_consistently_appearing_items_higher():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    order = sorted(fused, key=lambda k: fused[k], reverse=True)
    # 'a' and 'b' both appear in both rankings; 'c' and 'd' only one.
    assert set(order[:2]) == {"a", "b"}


def test_rrf_empty():
    assert rrf([]) == {}


@pytest.mark.asyncio
async def test_mmr_picks_query_relevant_when_lambda_high():
    emb = DummyEmbedder(64)
    q = await emb.create("alice apple")
    e1 = await emb.create("alice apple")
    e2 = await emb.create("zebra elephant")
    cands = [
        {"uuid": "1", "fact_embedding": e2},  # irrelevant
        {"uuid": "2", "fact_embedding": e1},  # most relevant
    ]
    out = mmr_rerank(
        candidates=cands,
        query_embedding=q,
        embedding_field="fact_embedding",
        lambda_mult=1.0,  # pure relevance
        limit=2,
    )
    assert out[0]["uuid"] == "2"


def test_mmr_short_circuits_on_no_query():
    out = mmr_rerank(
        candidates=[{"uuid": "x"}], query_embedding=None,
        embedding_field="fact_embedding", limit=5,
    )
    assert out == [{"uuid": "x"}]


def test_episode_mentions_rerank_prioritises_more_episodes():
    cands = [
        {"uuid": "1", "episodes": ["a"]},
        {"uuid": "2", "episodes": ["a", "b", "c"]},
        {"uuid": "3", "episodes": []},
    ]
    out = episode_mentions_rerank(cands, limit=3)
    assert [c["uuid"] for c in out] == ["2", "1", "3"]


@pytest.mark.asyncio
async def test_cross_encoder_rerank_uses_provided_client():
    cands = [{"uuid": "1", "fact": "alice apple"}, {"uuid": "2", "fact": "bob banana"}]
    out = await cross_encoder_rerank(
        candidates=cands,
        query="bob",
        text_field="fact",
        cross_encoder=DummyCrossEncoder(),
        limit=2,
    )
    assert out[0]["fact"] == "bob banana"


@pytest.mark.asyncio
async def test_dummy_cross_encoder_ranks_token_overlap():
    enc = DummyCrossEncoder()
    ranked = await enc.rank("alice apple", ["alice apple pie", "zebra elephant"])
    assert ranked[0][0] == "alice apple pie"


# ---------------------------------------------------------------- search_config
def test_search_config_defaults():
    cfg = SearchConfig()
    assert cfg.reranker is Reranker.rrf
    assert cfg.use_vector and cfg.use_fulltext
    assert cfg.only_valid is True


def test_search_recipes_have_expected_rerankers():
    assert search_recipes.EDGE_HYBRID_SEARCH_RRF.reranker is Reranker.rrf
    assert search_recipes.EDGE_HYBRID_SEARCH_MMR.reranker is Reranker.mmr
    assert search_recipes.EDGE_HYBRID_SEARCH_CROSS_ENCODER.reranker is Reranker.cross_encoder
    assert search_recipes.EDGE_HYBRID_SEARCH_EPISODE_MENTIONS.reranker is Reranker.episode_mentions
    assert search_recipes.COMBINED_HYBRID_SEARCH_RRF.include_nodes
    assert search_recipes.COMBINED_HYBRID_SEARCH_RRF.include_episodes
    assert search_recipes.COMBINED_HYBRID_SEARCH_RRF.include_communities
    nd = search_recipes.edge_hybrid_search_node_distance("focal-uuid")
    assert nd.reranker is Reranker.node_distance and nd.focal_uuid == "focal-uuid"


def test_search_recipes_are_independent_instances():
    a = search_recipes.EDGE_HYBRID_SEARCH_RRF
    b = search_recipes.EDGE_HYBRID_SEARCH_MMR
    a.limit = 99
    assert b.limit != 99


# -------------------------------------------------------- result dataclass shapes
def test_add_episode_results_shape():
    ep = EpisodicNode(name="x")
    res = AddEpisodeResults(episode=ep, episodic_edges=[], nodes=[], edges=[])
    assert res.invalidated_edges == []
    assert res.communities == []
    assert res.community_edges == []


def test_add_bulk_episode_results_default_empty():
    res = AddBulkEpisodeResults()
    for fld in ("episodes", "episodic_edges", "nodes", "edges", "invalidated_edges", "communities", "community_edges"):
        assert getattr(res, fld) == []


def test_add_triplet_results_includes_invalidated():
    res = AddTripletResults(nodes=[], edges=[])
    assert res.invalidated_edges == []


def test_raw_episode_defaults():
    r = RawEpisode(name="n", content="c")
    assert r.source.value == "message"
    assert r.reference_time is None


# ------------------------------------------------------------ scripted LLM
@pytest.mark.asyncio
async def test_scripted_llm_replays_responses_and_records_calls():
    client = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice")],
                facts=[ExtractedFact("Alice", "loves", "Bob", "Alice loves Bob.")],
            ),
            ScriptedResponse(entities=[ExtractedEntity(name="Carol")]),
        ]
    )
    r1 = await client.extract("first", group_id="g", custom_instructions="be terse")
    r2 = await client.extract("second")
    r3 = await client.extract("third")  # exhausted -> empty

    assert [e.name for e in r1.entities] == ["Alice"]
    assert r1.facts[0].fact == "Alice loves Bob."
    assert [e.name for e in r2.entities] == ["Carol"]
    assert r3.entities == [] and r3.facts == []

    # Calls recorded with the parameters threaded through.
    calls = client.extract_calls
    assert len(calls) == 3
    assert calls[0]["custom_instructions"] == "be terse"
    assert calls[0]["group_id"] == "g"


@pytest.mark.asyncio
async def test_scripted_llm_threads_entity_types():
    client = ScriptedLLMClient([ScriptedResponse()])
    await client.extract("x", entity_types={"Person": object, "Place": object})
    assert sorted(client.extract_calls[0]["entity_types"]) == ["Person", "Place"]
