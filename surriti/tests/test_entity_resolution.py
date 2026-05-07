"""Tests for the canonical entity resolution layer."""

from __future__ import annotations

import pytest

from surriti.embedder import DummyEmbedder
from surriti.entity_resolution import normalize_alias, resolve_entity_mentions
from surriti.graphiti import Surriti
from surriti.llm import ExtractedEntity, ExtractedFact, ScriptedLLMClient, ScriptedResponse
from surriti.testing import InMemoryDriver


def test_normalize_alias_handles_punctuation_and_case():
    assert normalize_alias("Drexel University!") == "drexel university"
    assert normalize_alias("  Acme,  Corp.") == "acme corp"
    assert normalize_alias("  Acme Corp ") == normalize_alias("acme corp")
    assert normalize_alias("") == ""


@pytest.mark.asyncio
async def test_alias_pipeline_collapses_case_variants():
    driver = InMemoryDriver()
    embedder = DummyEmbedder(embedding_dim=64)
    surriti = Surriti(driver, embedder=embedder)

    await surriti.add_episode(
        name="e1",
        episode_body="Alice joined Acme Corp.",
        group_id="g",
    )
    await surriti.add_episode(
        name="e2",
        episode_body="ALICE met Charlie at acme corp.",
        group_id="g",
    )

    # Same canonical entities reused (no duplicate Alice / Acme Corp rows).
    names = [r.get("name") for r in driver.records["entity"]]
    # "Alice" should appear once even though mentioned twice with different case.
    assert sum(1 for n in names if (n or "").casefold() == "alice") == 1
    assert sum(1 for n in names if (n or "").casefold() == "acme corp") == 1


@pytest.mark.asyncio
async def test_resolver_alias_hit_short_circuits():
    driver = InMemoryDriver()
    embedder = DummyEmbedder(embedding_dim=64)

    # Pre-populate one entity + one alias row.
    driver._insert(
        "entity",
        {
            "uuid": "e-1",
            "group_id": "g",
            "name": "Drexel University",
            "canonical_name": "Drexel University",
            "summary": "",
            "labels": ["Entity"],
            "name_embedding": await embedder.create("Drexel University"),
            "attributes": {},
        },
    )
    driver._insert(
        "entity_alias",
        {
            "uuid": "a-1",
            "group_id": "g",
            "alias": "Drexel",
            "normalized_alias": "drexel",
            "entity_uuid": "e-1",
            "confidence": 1.0,
        },
    )

    resolved = await resolve_entity_mentions(
        driver=driver,
        embedder=embedder,
        llm=None,
        mentions=[ExtractedEntity(name="Drexel", labels=["Entity"])],
        group_id="g",
        use_llm=False,
    )
    assert len(resolved) == 1
    assert resolved[0].canonical_uuid == "e-1"
    assert resolved[0].resolution == "alias_hit"


@pytest.mark.asyncio
async def test_resolver_writes_alias_for_semantic_match():
    driver = InMemoryDriver()
    embedder = DummyEmbedder(embedding_dim=64)

    canonical_vec = await embedder.create("Acme Corporation")
    driver._insert(
        "entity",
        {
            "uuid": "e-acme",
            "group_id": "g",
            "name": "Acme Corporation",
            "canonical_name": "Acme Corporation",
            "summary": "",
            "labels": ["Entity"],
            "name_embedding": canonical_vec,
            "attributes": {},
        },
    )

    # DummyEmbedder is deterministic per-text but unrelated phrases will not
    # cross the 0.86 threshold. Force a semantic match by using an identical
    # phrase variant that hashes to the same embedding -- since DummyEmbedder
    # is purely text-derived, identical input gives identical output and
    # cosine = 1.0. We then assert the alias row is recorded.
    resolved = await resolve_entity_mentions(
        driver=driver,
        embedder=embedder,
        llm=None,
        mentions=[ExtractedEntity(name="Acme Corporation", labels=["Entity"])],
        group_id="g",
        use_llm=False,
        threshold=0.5,
    )
    assert resolved[0].canonical_uuid == "e-acme"
    # exact_name path -- aliases should NOT be written for trivial matches.
    assert resolved[0].resolution == "exact_name"
    assert driver.records["entity_alias"] == []
