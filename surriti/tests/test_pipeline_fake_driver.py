"""End-to-end test using the in-memory driver from ``surriti.testing``.

This test does not require a running SurrealDB instance.
"""

from __future__ import annotations

import pytest

from surriti.embedder import DummyEmbedder
from surriti.graphiti import Surriti
from surriti.llm import (
    ExtractedEntity,
    ExtractedFact,
    ScriptedLLMClient,
    ScriptedResponse,
)
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


# ---------------------------------------------------------------------------
# Regression: idempotent extraction + multi-tenant + User node
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_extraction_does_not_crash_unique_index():
    """The LLM may emit the same entity name twice in one extraction. The
    pipeline must dedupe and survive the unique ``(group_id, name)`` index.
    """

    duplicated = ScriptedResponse(
        entities=[
            ExtractedEntity(name="Michael", labels=["Person"]),
            # Same name again — used to trip entity_name_uniq on CREATE.
            ExtractedEntity(name="Michael", labels=["Person"]),
            ExtractedEntity(name="Acme Corp", labels=["Organization"]),
        ],
        facts=[
            ExtractedFact("Michael", "works_at", "Acme Corp",
                          "Michael works at Acme Corp."),
        ],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([duplicated]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    result = await surriti.add_episode(
        name="dup-ep",
        episode_body="Michael works at Acme Corp.",
        group_id="tenant-A",
    )

    michaels = [r for r in driver.records["entity"] if r["name"] == "Michael"]
    assert len(michaels) == 1, "expected exactly one Michael row"
    names = [n.name for n in result.nodes]
    assert names.count("Michael") == 1
    assert "Acme Corp" in names
    assert len(result.edges) == 1


@pytest.mark.asyncio
async def test_repeat_add_episode_reuses_entity():
    """Two episodes that mention the same entity should share one entity row."""

    resp = ScriptedResponse(
        entities=[ExtractedEntity(name="Michael", labels=["Person"])],
        facts=[],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([resp, resp]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    await surriti.add_episode(name="ep1", episode_body="hi", group_id="t")
    await surriti.add_episode(name="ep2", episode_body="hi again", group_id="t")

    michaels = [r for r in driver.records["entity"] if r["name"] == "Michael"]
    assert len(michaels) == 1
    assert len(driver.records["episode"]) == 2


@pytest.mark.asyncio
async def test_multi_tenant_same_name_isolated():
    """Same entity name in two tenants must coexist as distinct rows."""

    resp = ScriptedResponse(
        entities=[ExtractedEntity(name="Michael", labels=["Person"])],
        facts=[],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([resp, resp]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    await surriti.add_episode(name="a", episode_body="hi", group_id="alice-uuid")
    await surriti.add_episode(name="b", episode_body="hi", group_id="bob-uuid")

    michaels = [r for r in driver.records["entity"] if r["name"] == "Michael"]
    assert len(michaels) == 2
    assert {m["group_id"] for m in michaels} == {"alice-uuid", "bob-uuid"}


@pytest.mark.asyncio
async def test_upsert_user_is_idempotent():
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))

    u1 = await surriti.upsert_user(group_id="tenant-A")
    u2 = await surriti.upsert_user(group_id="tenant-A", display_name="Michael")
    u3 = await surriti.upsert_user(group_id="tenant-A", display_name="Michael")

    assert u1.uuid == u2.uuid == u3.uuid
    assert u2.attributes.get("display_name") == "Michael"
    assert "User" in u2.labels
    rows = [r for r in driver.records["entity"] if r["name"] == "tenant-A"]
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_add_episode_with_speaker_id_creates_user_entity():
    resp = ScriptedResponse(
        entities=[ExtractedEntity(name="Michael", labels=["Person"])],
        facts=[
            ExtractedFact("user-42", "is_named", "Michael",
                          "user-42 is named Michael."),
        ],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    llm = ScriptedLLMClient([resp])
    surriti = Surriti(
        driver,
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    await surriti.add_episode(
        name="t1",
        episode_body="my name is Michael",
        group_id="user-42",
        speaker_id="user-42",
        speaker_name="Michael",
    )

    user_rows = [r for r in driver.records["entity"] if r["name"] == "user-42"]
    assert len(user_rows) == 1
    assert "User" in user_rows[0]["labels"]
    assert user_rows[0]["attributes"].get("display_name") == "Michael"

    # The speaker hint must reach the extractor.
    call = llm.extract_calls[0]
    assert "speaker" in (call["custom_instructions"] or "").lower()
    assert "user-42" in (call["custom_instructions"] or "")


# ---------------------------------------------------------------------------
# Quality regressions: self-loop guard, episode-name leak, prompt rules
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_self_loop_facts_are_dropped():
    """Small models occasionally hallucinate `Michael -[knows]-> Michael`.
    The pipeline must drop those (except identity predicates like
    `is_named`) before they hit the DB."""

    resp = ScriptedResponse(
        entities=[ExtractedEntity(name="Michael", labels=["Person"])],
        facts=[
            ExtractedFact("Michael", "knows", "Michael",
                          "Michael knows Michael."),
            ExtractedFact("Michael", "is_named", "Michael",
                          "Michael is named Michael."),
        ],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([resp]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    result = await surriti.add_episode(
        name="ep1",
        episode_body="hello, my name is Michael",
        group_id="t",
    )

    # Only the identity predicate should survive.
    predicates = [e.name for e in result.edges]
    assert "knows" not in predicates
    assert predicates == ["is_named"]


@pytest.mark.asyncio
async def test_previous_episode_context_does_not_leak_episode_name():
    """`_fetch_episode_contents` must NOT prefix prior content with the
    episode name (e.g. `[turn-a]`); small models read brackets as
    entities."""

    resp = ScriptedResponse(entities=[], facts=[])
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    llm = ScriptedLLMClient([resp, resp])
    surriti = Surriti(
        driver,
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    first = await surriti.add_episode(
        name="turn-a",
        episode_body="hello, my name is Michael",
        group_id="t",
    )
    await surriti.add_episode(
        name="turn-a",
        episode_body="anything",
        group_id="t",
        previous_episode_uuids=[first.episode.uuid],
    )

    # Second call's content must contain the prior body but NOT the
    # episode `name` ("turn-a") in any bracketed form.
    second_call_content = llm.extract_calls[1]["content"]
    assert "hello, my name is Michael" in second_call_content
    assert "[turn-a]" not in second_call_content
    assert "turn-a" not in second_call_content


def test_extraction_system_prompt_has_hard_rules():
    """Snapshot-style guard so future edits don't silently weaken the
    system prompt that small models depend on."""

    from surriti.llm_clients import EXTRACTION_SYSTEM, CONTRADICTION_SYSTEM

    assert "subject" in EXTRACTION_SYSTEM and "object" in EXTRACTION_SYSTEM
    assert "is_named" in EXTRACTION_SYSTEM  # identity-predicate exception
    assert "FORBIDDEN" in EXTRACTION_SYSTEM or "NEVER" in EXTRACTION_SYSTEM
    # Positive examples for small models -- the prompt MUST teach what to
    # extract, not just what to skip.
    assert "is_a" in EXTRACTION_SYSTEM
    assert "is_age" in EXTRACTION_SYSTEM
    assert "my name is" in EXTRACTION_SYSTEM.lower()
    # Self-loop ban must be unconditional in the prompt (the speaker is the
    # subject for naming).
    assert "self-loop" in EXTRACTION_SYSTEM.lower()
    # Bracketed/UUID metadata must be called out as not-an-entity.
    assert "metadata" in EXTRACTION_SYSTEM.lower()

    assert "SAME subject" in CONTRADICTION_SYSTEM
    assert "domain" in CONTRADICTION_SYSTEM.lower()
    assert "is_brother_of" in CONTRADICTION_SYSTEM


@pytest.mark.asyncio
async def test_self_loop_identity_fact_is_repaired_with_speaker_id():
    """When the LLM emits ``Auley is_named Auley`` but the speaker's stable
    id is ``default``, the pipeline should rewrite the subject to
    ``default`` so the naming edge connects two distinct entities (instead
    of becoming a useless self-loop)."""

    resp = ScriptedResponse(
        entities=[ExtractedEntity(name="Auley", labels=["Person"])],
        facts=[
            ExtractedFact("Auley", "is_named", "Auley",
                          "Auley is named Auley."),
        ],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([resp]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    result = await surriti.add_episode(
        name="chat",
        episode_body="my name is Auley",
        group_id="default",
        speaker_id="default",
    )

    assert len(result.edges) == 1
    edge = result.edges[0]
    by_uuid = {n.uuid: n for n in result.nodes}
    assert by_uuid[edge.source_node_uuid].name == "default"
    assert by_uuid[edge.target_node_uuid].name == "Auley"
    assert edge.name == "is_named"
