"""End-to-end test using the in-memory driver from ``surriti.testing``.

This test does not require a running SurrealDB instance.
"""

from __future__ import annotations

import pytest

from surriti.embedder import DummyEmbedder
from surriti.graphiti import Surriti
from surriti.testing import InMemoryDriver as FakeSurrealDriver


@pytest.mark.asyncio
async def test_add_episode_creates_entities_and_edges():
    driver = FakeSurrealDriver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))

    result = await surriti.add_episode(
        name="ep1",
        episode_body="Alice met Bob at Acme Corp.",
        group_id="g1",
    )

    names = {n.name for n in result.nodes}
    assert {"Alice", "Bob", "Acme Corp"}.issubset(names)
    assert len(result.edges) >= 1
    assert len(result.episodic_edges) == len(result.nodes)
    assert len(driver.records["episode"]) == 1
    assert len(driver.records["entity"]) == len(result.nodes)


@pytest.mark.asyncio
async def test_contradiction_invalidates_prior_edge():
    driver = FakeSurrealDriver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))

    await surriti.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        group_id="g1",
    )
    second = await surriti.add_episode(
        name="e2",
        episode_body="Alice no longer works at Acme Corp; Alice moved to Globex.",
        group_id="g1",
    )
    # The second episode should have invalidated at least one previous fact.
    assert len(second.invalidated_edges) >= 1
    invalidated = [r for r in driver.records["relates_to"] if r.get("invalid_at")]
    assert invalidated, "expected at least one edge to be marked invalid"


@pytest.mark.asyncio
async def test_search_returns_facts():
    driver = FakeSurrealDriver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await surriti.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        group_id="g1",
    )

    results = await surriti.search("Alice Acme", group_id="g1")
    assert results.edges, "search should surface at least one edge"
    assert any("Acme" in e.fact for e in results.edges)
