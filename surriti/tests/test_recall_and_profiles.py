"""Tests for ``Surriti.recall`` and touched-only profile refresh."""

from __future__ import annotations

import pytest

from surriti.embedder import DummyEmbedder
from surriti.graphiti import MemoryContext, Surriti
from surriti.testing import InMemoryDriver


@pytest.mark.asyncio
async def test_profile_refresh_runs_synchronously_when_configured():
    driver = InMemoryDriver()
    surriti = Surriti(
        driver,
        embedder=DummyEmbedder(embedding_dim=64),
        profile_refresh="sync",
    )

    await surriti.add_episode(
        name="ep",
        episode_body="Alice met Bob at Acme Corp.",
        group_id="g",
    )

    # Every entity touched by the episode has its profile populated and
    # mention_count incremented.
    entities = driver.records["entity"]
    assert entities, "expected entities to have been created"
    for e in entities:
        assert e.get("profile_summary"), f"missing profile summary for {e!r}"
        assert e.get("mention_count", 0) >= 1


@pytest.mark.asyncio
async def test_profile_refresh_off_skips_writes():
    driver = InMemoryDriver()
    surriti = Surriti(
        driver,
        embedder=DummyEmbedder(embedding_dim=64),
        profile_refresh="off",
    )

    await surriti.add_episode(
        name="ep",
        episode_body="Alice met Bob.",
        group_id="g",
    )

    for e in driver.records["entity"]:
        assert not e.get("profile_summary")
        assert e.get("mention_count", 0) == 0


@pytest.mark.asyncio
async def test_recall_returns_memory_context_with_profiles():
    driver = InMemoryDriver()
    surriti = Surriti(
        driver,
        embedder=DummyEmbedder(embedding_dim=64),
        profile_refresh="sync",
    )

    await surriti.add_episode(
        name="ep",
        episode_body="Alice works at Acme Corp with Bob.",
        group_id="g",
    )

    ctx = await surriti.recall("Tell me about Alice", group_id="g", depth="fast")
    assert isinstance(ctx, MemoryContext)
    assert any(p.name.casefold() == "alice" for p in ctx.profiles)


@pytest.mark.asyncio
async def test_recall_invalid_depth_rejected():
    driver = InMemoryDriver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    with pytest.raises(ValueError):
        await surriti.recall("x", group_id="g", depth="bogus")
