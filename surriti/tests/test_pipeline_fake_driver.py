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
from surriti.nodes import EpisodeType
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


@pytest.mark.asyncio
async def test_recall_reinforces_returned_edges():
    driver = FakeSurrealDriver()
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    await surriti.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        group_id="g1",
    )

    ctx = await surriti.recall("Alice Acme", group_id="g1")

    assert ctx.facts
    recalled = [r for r in driver.records["relates_to"] if r.get("last_recalled_at")]
    assert recalled
    assert all(int(r.get("recall_count") or 0) >= 1 for r in recalled)


@pytest.mark.asyncio
async def test_add_self_episode_persists_object_entities_before_edges():
    response = ScriptedResponse(
        entities=[ExtractedEntity(name="verbosity", labels=["Behavior"])],
        facts=[
            ExtractedFact(
                "assistant",
                "has_pattern",
                "verbosity",
                "Assistant has a verbosity pattern.",
            )
        ],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([response]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    result = await surriti.add_self_episode(
        episode_type=EpisodeType.self_observation,
        content="I was too verbose in the last answer.",
        group_id="g1",
    )

    entity_ids = {r["uuid"] for r in driver.records["entity"]}
    assert {e.name for e in result.nodes} >= {"assistant_g1", "verbosity"}
    assert result.edges
    for edge in result.edges:
        assert edge.source_node_uuid in entity_ids
        assert edge.target_node_uuid in entity_ids


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
async def test_case_variant_entities_merge_within_tenant_only():
    resp1 = ScriptedResponse(
        entities=[
            ExtractedEntity(name="Michael", labels=["Person"]),
            ExtractedEntity(name="Florida", labels=["Place"]),
        ],
        facts=[ExtractedFact("Michael", "lives_in", "Florida", "Michael lives in Florida.")],
    )
    resp2 = ScriptedResponse(
        entities=[
            ExtractedEntity(name="Michael", labels=["Person"]),
            ExtractedEntity(name="florida", labels=["Place"]),
        ],
        facts=[ExtractedFact("Michael", "lives_in", "florida", "Michael lives in florida.")],
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(
        driver,
        llm_client=ScriptedLLMClient([resp1, resp2]),
        embedder=DummyEmbedder(embedding_dim=64),
    )

    first = await surriti.add_episode(name="a", episode_body="a", group_id="tenant-A")
    second = await surriti.add_episode(name="b", episode_body="b", group_id="tenant-A")

    florida = next(n for n in first.nodes if n.name == "Florida")
    assert any(n.name == "Florida" and n.uuid == florida.uuid for n in second.nodes)
    assert not [r for r in driver.records["entity"] if r["group_id"] == "tenant-A" and r["name"] == "florida"]

    other = ScriptedResponse(entities=[ExtractedEntity(name="florida", labels=["Place"])], facts=[])
    surriti.llm = ScriptedLLMClient([other])
    await surriti.add_episode(name="c", episode_body="c", group_id="tenant-B")

    rows = [r for r in driver.records["entity"] if r["name"].lower() == "florida"]
    assert len(rows) == 2
    assert {r["group_id"] for r in rows} == {"tenant-A", "tenant-B"}


@pytest.mark.asyncio
async def test_existing_case_duplicate_entities_are_cleaned_up_within_tenant():
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    surriti = Surriti(driver, embedder=DummyEmbedder(embedding_dim=64))
    driver.records["entity"].extend([
        {"uuid": "canonical-florida", "group_id": "tenant-A", "name": "Florida", "labels": ["Place"], "summary": "", "attributes": {}},
        {"uuid": "alias-florida", "group_id": "tenant-A", "name": "florida", "labels": ["Place"], "summary": "", "attributes": {}},
        {"uuid": "other-florida", "group_id": "tenant-B", "name": "florida", "labels": ["Place"], "summary": "", "attributes": {}},
    ])
    driver.records["relates_to"].append({
        "uuid": "edge-1",
        "group_id": "tenant-A",
        "name": "lives_in",
        "source_node_uuid": "person-1",
        "target_node_uuid": "alias-florida",
        "in": "person-1",
        "out": "alias-florida",
    })

    nodes = await surriti._upsert_entities(
        [ExtractedEntity(name="FLORIDA", labels=["Place"])],
        group_id="tenant-A",
    )

    assert nodes[0].uuid == "canonical-florida"
    assert [r["name"] for r in driver.records["entity"] if r["group_id"] == "tenant-A"] == ["Florida"]
    assert any(r["uuid"] == "other-florida" for r in driver.records["entity"])
    assert driver.records["relates_to"][0]["target_node_uuid"] == "canonical-florida"


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

    # The prior body must NOT contaminate `content` (current episode);
    # it now travels through the dedicated `context` channel so the model
    # is told explicitly not to re-extract it.
    second_call_content = llm.extract_calls[1]["content"]
    second_call_context = llm.extract_calls[1].get("context") or ""
    assert second_call_content == "anything"
    assert "hello, my name is Michael" in second_call_context
    # The episode `name` must never appear in either channel.
    assert "[turn-a]" not in second_call_content
    assert "[turn-a]" not in second_call_context
    assert "turn-a" not in second_call_content
    assert "turn-a" not in second_call_context


def test_extraction_system_prompt_has_hard_rules():
    """Snapshot-style guard so future edits don't silently weaken the
    system prompt that small models depend on."""

    from surriti.llm_clients import EXTRACTION_SYSTEM, CONTRADICTION_SYSTEM

    assert "subject" in EXTRACTION_SYSTEM and "object" in EXTRACTION_SYSTEM
    # Two-channel prompt structure for the temporal-state engine.
    assert "CURRENT EPISODE" in EXTRACTION_SYSTEM
    assert "CONTEXT" in EXTRACTION_SYSTEM
    # Generic per-fact metadata rubric (no hardcoded predicate vocabulary).
    assert "operation" in EXTRACTION_SYSTEM
    assert "singleton" in EXTRACTION_SYSTEM
    assert "temporal" in EXTRACTION_SYSTEM
    assert "domain" in EXTRACTION_SYSTEM
    assert "terminate" in EXTRACTION_SYSTEM
    assert "FORBIDDEN" in EXTRACTION_SYSTEM or "NEVER" in EXTRACTION_SYSTEM
    # Self-loop ban must be unconditional in the prompt (the speaker is the
    # subject for naming).
    assert "self-loop" in EXTRACTION_SYSTEM.lower()

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


# ---------------------------------------------------------------------------
# Generic temporal-state engine: per-fact metadata drives invalidation.
# ---------------------------------------------------------------------------


def _surriti():
    return Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=None,
        embedder=DummyEmbedder(embedding_dim=64),
    )


@pytest.mark.asyncio
async def test_singleton_assert_closes_prior_active_with_different_object():
    """A second ``singleton=True`` assert on the same (subject, predicate)
    slot must close the prior active edge deterministically -- no LLM
    contradiction call required."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="engineer")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="engineer",
                    fact="michael works as an engineer.",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="blogger")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="blogger",
                    fact="michael works as a food blogger.",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    r1 = await s.add_episode(name="t1", episode_body="...", group_id="g")
    r2 = await s.add_episode(name="t2", episode_body="...", group_id="g")

    assert len(r2.invalidated_edges) == 1
    assert r2.invalidated_edges[0].uuid == r1.edges[0].uuid
    assert r2.invalidated_edges[0].status == "superseded"
    assert r2.invalidated_edges[0].superseded_by == r2.edges[0].uuid
    assert r2.edges[0].supersedes == [r1.edges[0].uuid]


@pytest.mark.asyncio
async def test_non_singleton_assert_does_not_close_prior():
    """Non-singleton facts (e.g. ``likes``) must coexist; the closer must
    NOT touch them even when subject + predicate match."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="alice"), ExtractedEntity(name="pizza")],
                facts=[ExtractedFact(
                    subject="alice", predicate="likes", object="pizza",
                    fact="alice likes pizza.",
                    operation="assert", singleton=False, domain="preference",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="alice"), ExtractedEntity(name="sushi")],
                facts=[ExtractedFact(
                    subject="alice", predicate="likes", object="sushi",
                    fact="alice likes sushi.",
                    operation="assert", singleton=False, domain="preference",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    await s.add_episode(name="t1", episode_body="...", group_id="g")
    r2 = await s.add_episode(name="t2", episode_body="...", group_id="g")

    assert r2.invalidated_edges == []


@pytest.mark.asyncio
async def test_terminate_operation_invalidates_prior_edge_no_insert():
    """``operation="terminate"`` closes the matching active edge but
    inserts no new edge."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="engineer")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="engineer",
                    fact="michael works as an engineer.",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="engineer")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="engineer",
                    fact="michael no longer works as an engineer.",
                    operation="terminate",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    r1 = await s.add_episode(name="t1", episode_body="...", group_id="g")
    r2 = await s.add_episode(name="t2", episode_body="...", group_id="g")

    assert r2.edges == []
    assert len(r2.invalidated_edges) == 1
    assert r2.invalidated_edges[0].uuid == r1.edges[0].uuid


@pytest.mark.asyncio
async def test_noop_operation_skips_fact_entirely():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="alice"), ExtractedEntity(name="bob")],
                facts=[ExtractedFact(
                    subject="alice", predicate="met", object="bob",
                    operation="noop",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )
    r = await s.add_episode(name="t1", episode_body="...", group_id="g")
    assert r.edges == []
    assert r.invalidated_edges == []


@pytest.mark.asyncio
async def test_source_type_assistant_skips_singleton_closer():
    """Only ``source_type="user"`` triggers the deterministic singleton
    closer. Assistant/tool/system facts are advisory and must not
    silently nuke prior user truth."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="engineer")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="engineer",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="blogger")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="blogger",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    await s.add_episode(name="t1", episode_body="...", group_id="g", source_type="user")
    r2 = await s.add_episode(
        name="t2", episode_body="...", group_id="g", source_type="assistant"
    )
    assert r2.invalidated_edges == []


@pytest.mark.asyncio
async def test_get_current_facts_returns_only_active_edges():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="engineer")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="engineer",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="michael"), ExtractedEntity(name="blogger")],
                facts=[ExtractedFact(
                    subject="michael", predicate="works_as", object="blogger",
                    operation="assert", singleton=True, domain="employment",
                )],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )

    r1 = await s.add_episode(name="t1", episode_body="...", group_id="g")
    r2 = await s.add_episode(name="t2", episode_body="...", group_id="g")
    michael_uuid = r1.edges[0].source_node_uuid

    current = await s.get_current_facts(subject_uuid=michael_uuid, group_id="g")
    assert len(current) == 1
    assert current[0].uuid == r2.edges[0].uuid
    assert current[0].status == "active"

    one = await s.get_current_fact(
        subject_uuid=michael_uuid, predicate="works_as", group_id="g"
    )
    assert one is not None and one.uuid == r2.edges[0].uuid


@pytest.mark.asyncio
async def test_context_kwarg_is_passed_separately_from_current_episode():
    """Sanity: previous episode body is passed via the dedicated
    ``context`` kwarg, never concatenated onto ``content``."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(entities=[ExtractedEntity(name="alice")]),
            ScriptedResponse(entities=[ExtractedEntity(name="bob")]),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )
    first = await s.add_episode(name="e1", episode_body="alice content", group_id="g")
    await s.add_episode(
        name="e2", episode_body="bob content", group_id="g",
        previous_episode_uuids=[first.episode.uuid],
    )

    second = llm.extract_calls[1]
    assert second["content"] == "bob content"
    assert "alice content" in (second.get("context") or "")
    assert "alice content" not in second["content"]


# ---------------------------------------------------------------------------
# New surface: validators, fact_key dedupe, as-of queries, structured
# contradiction candidates.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_repair_fact_drops_lives_in_world_filler():
    """Banned placeholder objects ("world", "earth") for location
    predicates are dropped before they reach the graph."""

    from surriti.validators import repair_fact

    bad = ExtractedFact("Michael", "lives_in", "world", "Michael lives in world.")
    assert repair_fact(bad) is None

    good = ExtractedFact("Michael", "lives_in", "Berlin", "Michael lives in Berlin.")
    assert repair_fact(good) is good


@pytest.mark.asyncio
async def test_repair_fact_rewrites_identity_self_loop_with_speaker_id():
    from surriti.validators import repair_fact

    fact = ExtractedFact("Michael", "is_named", "Michael", "Michael is named Michael.")
    out = repair_fact(fact, speaker_id="default")
    assert out is not None
    assert out.subject == "default"
    assert out.object == "Michael"


@pytest.mark.asyncio
async def test_repair_fact_drops_unrepaired_self_loops():
    from surriti.validators import repair_fact

    bad = ExtractedFact("Alice", "knows", "Alice", "Alice knows Alice.")
    assert repair_fact(bad) is None


@pytest.mark.asyncio
async def test_lives_in_world_dropped_at_episode_ingest():
    """Integration: the validator runs inside add_episode."""

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Michael"), ExtractedEntity(name="world")],
                facts=[
                    ExtractedFact(
                        "Michael", "lives_in", "world",
                        "Michael lives in world.",
                    ),
                ],
            ),
        ]
    )
    s = Surriti(
        FakeSurrealDriver(enforce_entity_name_uniq=True),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )
    result = await s.add_episode(name="e1", episode_body="x", group_id="g")
    # The placeholder fact must not produce a ``relates_to`` edge.
    assert all(e.name != "lives_in" for e in result.edges)


@pytest.mark.asyncio
async def test_fact_key_is_set_on_insert():
    """Every persisted edge gets a deterministic fact_key."""

    from surriti.edges import make_fact_key

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[ExtractedFact("Alice", "works_at", "Acme", "Alice works at Acme.")],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    result = await s.add_episode(name="e1", episode_body="x", group_id="g")

    assert result.edges
    edge = result.edges[0]
    expected = make_fact_key("g", edge.source_node_uuid, "works_at", edge.target_node_uuid)
    assert edge.fact_key == expected
    rows = driver.records["relates_to"]
    assert any(r.get("fact_key") == expected for r in rows)


@pytest.mark.asyncio
async def test_repeat_episode_dedupes_via_fact_key():
    """Re-emitting the same triple in a later episode reuses the existing
    edge instead of creating a duplicate."""

    fact = ExtractedFact("Alice", "works_at", "Acme", "Alice works at Acme.")
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[fact],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[fact],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    r1 = await s.add_episode(name="e1", episode_body="x", group_id="g")
    r2 = await s.add_episode(name="e2", episode_body="y", group_id="g")

    relates = driver.records["relates_to"]
    assert len(relates) == 1, "duplicate triple must reuse the existing edge"
    # The single row should now reference both episodes.
    assert {ep for ep in relates[0].get("episodes", [])} >= {
        r1.episode.uuid, r2.episode.uuid,
    }


@pytest.mark.asyncio
async def test_get_facts_as_of_returns_only_valid_at_timestamp():
    """An edge that becomes invalid at T should appear in queries before
    T and disappear in queries at/after T."""

    from datetime import datetime, timedelta, timezone

    fact1 = ExtractedFact(
        "Alice", "works_at", "Acme", "Alice works at Acme.",
        temporal=True, singleton=True, domain="employment",
    )
    fact2 = ExtractedFact(
        "Alice", "works_at", "Globex", "Alice works at Globex.",
        temporal=True, singleton=True, domain="employment",
    )
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[fact1],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Globex")],
                facts=[fact2],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    r1 = await s.add_episode(
        name="e1", episode_body="x", group_id="g", reference_time=t0,
    )
    await s.add_episode(
        name="e2", episode_body="y", group_id="g", reference_time=t1,
    )

    alice_uuid = next(n.uuid for n in r1.nodes if n.name == "Alice")

    # Before t1: only Acme should be valid.
    before = await s.get_facts_as_of(
        subject_uuid=alice_uuid,
        as_of=t0 + timedelta(days=10),
        group_id="g",
        predicate="works_at",
    )
    targets_before = {e.target_node_uuid for e in before}
    acme_uuid = next(n.uuid for n in r1.nodes if n.name == "Acme")
    assert acme_uuid in targets_before
    assert all(e.target_node_uuid == acme_uuid for e in before)

    # At/after t1: only Globex should be valid (Acme closed by singleton).
    after = await s.get_facts_as_of(
        subject_uuid=alice_uuid,
        as_of=t1 + timedelta(days=1),
        group_id="g",
        predicate="works_at",
    )
    assert after, "expected at least one edge valid after the switch"
    assert acme_uuid not in {e.target_node_uuid for e in after}


@pytest.mark.asyncio
async def test_get_state_as_of_collapses_to_one_edge_per_slot():
    from datetime import datetime, timezone

    fact = ExtractedFact(
        "Alice", "works_at", "Acme", "Alice works at Acme.",
        temporal=True, singleton=True, domain="employment",
    )
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[fact],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    s = Surriti(driver, llm_client=llm, embedder=DummyEmbedder(embedding_dim=64))
    r = await s.add_episode(
        name="e1", episode_body="x", group_id="g",
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    alice_uuid = next(n.uuid for n in r.nodes if n.name == "Alice")
    state = await s.get_state_as_of(
        subject_uuid=alice_uuid,
        as_of=datetime(2026, 12, 1, tzinfo=timezone.utc),
        group_id="g",
    )
    assert state, "expected at least one slot in the as-of state map"
    # Each key is (predicate, target_uuid).
    assert all(isinstance(k, tuple) and len(k) == 2 for k in state.keys())


@pytest.mark.asyncio
async def test_find_contradictions_receives_structured_candidates():
    """The temporal layer forwards a structured candidate list to the
    LLM client, on top of the legacy fact-string list."""

    from surriti.llm import ContradictionCandidate

    captured: list[dict] = []

    class Spy(ScriptedLLMClient):
        async def find_contradictions(self, new_fact, existing_facts, **kwargs):
            captured.append({
                "new_fact": new_fact,
                "existing_facts": list(existing_facts),
                "candidates": list(kwargs.get("candidates") or []),
                "new_fact_struct": kwargs.get("new_fact_struct"),
            })
            return []

    llm = Spy(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[ExtractedFact(
                    "Alice", "works_at", "Acme", "Alice works at Acme.",
                    temporal=True, domain="employment",
                )],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Globex")],
                facts=[ExtractedFact(
                    "Alice", "works_at", "Globex", "Alice works at Globex.",
                    temporal=True, domain="employment",
                )],
            ),
        ]
    )
    driver = FakeSurrealDriver(enforce_entity_name_uniq=True)
    # Disable default relation frames so ``works_at`` is *not* recognized
    # as a one_current/replace slot. Without a frame the engine routes
    # the second assertion through the LLM contradiction pass, which is
    # what this test asserts.
    s = Surriti(
        driver,
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
        seed_default_frames=False,
    )
    await s.add_episode(name="e1", episode_body="x", group_id="g")
    await s.add_episode(name="e2", episode_body="y", group_id="g")

    # The first contradiction call (for episode 2) should carry structured
    # candidates with subject/predicate/object/domain populated.
    assert captured, "expected at least one find_contradictions call"
    last = captured[-1]
    assert last["new_fact_struct"] is not None
    assert last["new_fact_struct"].predicate == "works_at"
    assert last["candidates"], "structured candidates should be forwarded"
    assert all(isinstance(c, ContradictionCandidate) for c in last["candidates"])
    assert last["candidates"][0].domain == "employment"
