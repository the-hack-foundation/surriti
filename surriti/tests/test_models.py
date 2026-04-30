from datetime import datetime, timezone

import pytest

from surriti.edges import EntityEdge
from surriti.embedder import DummyEmbedder, cosine_similarity
from surriti.llm import DummyLLMClient
from surriti.nodes import EntityNode, EpisodeType, EpisodicNode
from surriti.search import _rrf_merge


def test_episode_defaults():
    ep = EpisodicNode(name="hello", content="hi", source=EpisodeType.text)
    assert ep.uuid
    assert ep.source is EpisodeType.text
    assert isinstance(ep.created_at, datetime)


def test_entity_edge_temporal_fields():
    edge = EntityEdge(
        group_id="g",
        source_node_uuid="a",
        target_node_uuid="b",
        name="knows",
        fact="A knows B",
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert edge.valid_at.year == 2026
    assert edge.invalid_at is None
    assert edge.expired_at is None


@pytest.mark.asyncio
async def test_dummy_embedder_dimensions_and_similarity():
    emb = DummyEmbedder(embedding_dim=128)
    a = await emb.create("apple banana")
    b = await emb.create("apple banana")
    c = await emb.create("zebra elephant")
    assert len(a) == 128
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, c) < 0.5


@pytest.mark.asyncio
async def test_dummy_llm_extraction_and_contradiction():
    llm = DummyLLMClient()
    extracted = await llm.extract("Alice met Bob at Acme Corp.")
    names = [e.name for e in extracted.entities]
    assert "Alice" in names and "Bob" in names

    contradictions = await llm.find_contradictions(
        "Alice no longer works at Acme Corp",
        ["Alice works at Acme Corp", "Bob lives in Paris"],
    )
    assert contradictions == [0]


def test_rrf_merge_orders_by_combined_rank():
    fused = _rrf_merge([["a", "b", "c"], ["b", "a", "d"]])
    # 'b' appears at rank 2 then rank 1 -> highest combined score expected.
    ordered = sorted(fused, key=lambda u: fused[u], reverse=True)
    assert ordered[0] in {"a", "b"}
    assert "c" in fused and "d" in fused
