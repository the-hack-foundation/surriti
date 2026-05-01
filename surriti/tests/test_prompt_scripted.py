"""Prompt-style tests using a deterministic ScriptedLLMClient.

These tests treat the LLM as a black box and verify:
  - Surriti correctly threads context (group_id, custom_instructions,
    entity_types, previous_episode_uuids) into the LLM extract call.
  - The structured ScriptedResponse drives entity/edge persistence.
  - excluded_entity_types removes entities and dependent facts.
  - entity_types apply labels by name containment.
  - find_contradictions output marks prior facts as invalidated.

They run against the in-memory FakeSurrealDriver so no DB or network is needed.
"""

from __future__ import annotations

import pytest

from surriti import (
    DummyEmbedder,
    EpisodeType,
    ExtractedEntity,
    ExtractedFact,
    RawEpisode,
    ScriptedLLMClient,
    ScriptedResponse,
    Surriti,
)

from tests.test_pipeline_fake_driver import FakeSurrealDriver


def _make(llm: ScriptedLLMClient) -> Surriti:
    return Surriti(
        FakeSurrealDriver(),
        llm_client=llm,
        embedder=DummyEmbedder(embedding_dim=64),
    )


@pytest.mark.asyncio
async def test_scripted_extraction_drives_entities_and_edges():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme Corp")],
                facts=[ExtractedFact("Alice", "works_at", "Acme Corp", "Alice works at Acme Corp.")],
            )
        ]
    )
    s = _make(llm)
    res = await s.add_episode(
        name="ep", episode_body="anything", group_id="g1",
        custom_extraction_instructions="prefer-canonical-names",
    )

    assert {n.name for n in res.nodes} == {"Alice", "Acme Corp"}
    assert len(res.edges) == 1
    assert res.edges[0].name == "works_at"
    assert res.edges[0].fact == "Alice works at Acme Corp."
    # extract_calls captured the threaded context
    call = llm.extract_calls[0]
    assert call["group_id"] == "g1"
    assert call["custom_instructions"] == "prefer-canonical-names"
    assert call["content"] == "anything"


@pytest.mark.asyncio
async def test_excluded_entity_types_filters_facts():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Alice", labels=["Person"]),
                    ExtractedEntity(name="Acme Corp", labels=["Organization"]),
                ],
                facts=[ExtractedFact("Alice", "works_at", "Acme Corp", "Alice works at Acme Corp.")],
            )
        ]
    )
    s = _make(llm)
    res = await s.add_episode(
        name="ep", episode_body="x", group_id="g1",
        excluded_entity_types=["Organization"],
    )
    names = {n.name for n in res.nodes}
    assert "Acme Corp" not in names
    assert "Alice" in names
    # Fact dropped because object entity excluded
    assert res.edges == []


@pytest.mark.asyncio
async def test_entity_types_apply_label_by_name_containment():
    llm = ScriptedLLMClient(
        [ScriptedResponse(entities=[ExtractedEntity(name="Person Alice")])]
    )
    s = _make(llm)
    res = await s.add_episode(
        name="ep", episode_body="x", group_id="g1",
        entity_types={"Person": object},
    )
    assert "Person" in res.nodes[0].labels


@pytest.mark.asyncio
async def test_previous_episode_uuids_thread_context_into_extractor():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(entities=[ExtractedEntity(name="Alice")]),
            ScriptedResponse(entities=[ExtractedEntity(name="Bob")]),
        ]
    )
    s = _make(llm)
    first = await s.add_episode(name="e1", episode_body="alice content", group_id="g")
    await s.add_episode(
        name="e2", episode_body="bob content", group_id="g",
        previous_episode_uuids=[first.episode.uuid],
    )

    second_call = llm.extract_calls[1]
    # Prior episode body now travels in the dedicated `context` channel
    # (read-only for the extractor); the current episode goes in `content`.
    assert second_call["content"] == "bob content"
    assert "alice content" in (second_call.get("context") or "")


@pytest.mark.asyncio
async def test_scripted_contradiction_invalidates_previous_edge():
    """Surriti uses LLMClient.find_contradictions to flag stale facts.

    We override the ScriptedLLMClient's find_contradictions so the second
    episode's incoming fact contradicts the first.
    """

    llm = ScriptedLLMClient(
        [
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Acme")],
                facts=[ExtractedFact("Alice", "works_at", "Acme", "Alice works at Acme.")],
            ),
            ScriptedResponse(
                entities=[ExtractedEntity(name="Alice"), ExtractedEntity(name="Globex")],
                facts=[ExtractedFact("Alice", "works_at", "Globex", "Alice works at Globex.")],
                contradictions=[0],
            ),
        ]
    )

    async def find_contradictions(new_fact, prior_facts):  # noqa: ARG001
        # Replays the response queue's contradiction hints. The first call
        # to extract popped the first response; subsequent contradiction
        # calls correspond to the *next* extract result. To keep this test
        # deterministic, contradict any prior fact mentioning "Acme" when the
        # incoming fact mentions "Globex".
        if "Globex" in new_fact and any("Acme" in p for p in prior_facts):
            return list(range(len(prior_facts)))
        return []

    llm.find_contradictions = find_contradictions  # type: ignore[method-assign]
    s = _make(llm)

    await s.add_episode(name="e1", episode_body="x", group_id="g")
    second = await s.add_episode(name="e2", episode_body="x", group_id="g")

    assert len(second.invalidated_edges) >= 1


@pytest.mark.asyncio
async def test_add_episode_bulk_with_raw_episodes():
    llm = ScriptedLLMClient(
        [
            ScriptedResponse(entities=[ExtractedEntity(name="Alice")]),
            ScriptedResponse(entities=[ExtractedEntity(name="Bob")]),
        ]
    )
    s = _make(llm)
    out = await s.add_episode_bulk(
        [
            RawEpisode(name="a", content="x", source=EpisodeType.text),
            RawEpisode(name="b", content="y", source=EpisodeType.text),
        ],
        group_id="g",
    )
    assert len(out.episodes) == 2
    names = {n.name for n in out.nodes}
    assert names == {"Alice", "Bob"}
    assert len(llm.extract_calls) == 2


@pytest.mark.asyncio
async def test_scripted_runs_out_returns_empty():
    llm = ScriptedLLMClient([])  # nothing scripted
    s = _make(llm)
    res = await s.add_episode(name="ep", episode_body="hi", group_id="g")
    assert res.nodes == []
    assert res.edges == []
    # Episode itself is still saved
    assert res.episode.name == "ep"
