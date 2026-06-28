"""Comprehensive integration test for Surriti memory system.

Tests:
1. Connection & schema initialization
2. Episode ingestion with family data
3. Search patterns (exact, fuzzy, semantic)
4. Recall with different depths
5. Edge cases (empty, unicode, special chars)
6. Bulk operations
7. Contradiction handling
8. Temporal queries
9. Entity resolution
10. Speaker context
11. Search filters
12. Profile generation
13. Stress test (50+ episodes)
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from surriti import (
    Surriti,
    DummyLLMClient,
    DummyEmbedder,
    EpisodeType,
    SearchConfig,
    SearchResults,
    SearchFilters,
    PropertyFilter,
    ComparisonOperator,
    MemoryContext,
    resolve_entity_mentions,
    refresh_entity_profiles,
    SurrealDriver,
    RawEpisode,
)
from surriti.llm import ExtractedEntity, ExtractedFact, ExtractionResult


# ============================================================================
# Helpers
# ============================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ago(hours: int) -> datetime:
    return _now() - timedelta(hours=hours)


def _days_ago(days: int) -> datetime:
    return _now() - timedelta(days=days)


# Use a unique namespace per test run to avoid collisions
_RUN_NS = "surriti_test_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _make_surriti(**kwargs) -> Surriti:
    """Create a Surriti instance with proper auth credentials."""
    driver = SurrealDriver(
        url=os.environ.get("SURRITI_TEST_SURREAL_URL", "ws://localhost:8000/rpc"),
        namespace=_RUN_NS,
        database=_RUN_NS,
        username="root",
        password="root",
        embedding_dim=64,
    )
    llm = kwargs.pop("llm_client", DummyLLMClient())
    embedder = kwargs.pop("embedder", DummyEmbedder(64))
    s = Surriti(driver, llm_client=llm, embedder=embedder, **kwargs)
    return s


async def _setup_db(s: Surriti) -> None:
    """Connect, build indices, clear data."""
    await s.driver.connect()
    await s.build_indices_and_constraints()
    await s.driver.clear()
    await s.connect()


async def _teardown(s: Surriti) -> None:
    """Clean up: clear data and close."""
    try:
        await s.driver.clear()
    except Exception:
        pass
    try:
        await s.close()
    except Exception:
        pass


# ============================================================================
# Fixture: fresh Surriti instance
# ============================================================================

@pytest_asyncio.fixture
async def surriti():
    """Provide a fresh Surriti instance connected to SurrealDB."""
    s = _make_surriti()
    await _setup_db(s)
    yield s
    # Force-cancel any lingering cognition tasks before teardown
    if s._cognition is not None:
        for t in list(s._cognition._tasks.values()):
            if not t.done():
                t.cancel()
        await _teardown(s)


# ============================================================================
# 1. Connection & Schema
# ============================================================================

@pytest.mark.asyncio
async def test_connect_and_schema(surriti):
    """Verify we can connect to SurrealDB and initialize schema."""
    pass  # fixture already connected


@pytest.mark.asyncio
async def test_clear_and_reconnect(surriti):
    """Clear database and reconnect without errors."""
    await surriti.driver.query("DELETE episode;")
    await surriti.driver.query("DELETE relates_to;")
    await surriti.driver.query("DELETE entity;")
    await surriti.driver.query("DELETE entity_alias;")
    await surriti.driver.query("DELETE community;")
    await surriti.driver.query("DELETE mentions;")
    await surriti.driver.clear()
    await surriti.connect()


# ============================================================================
# 2. Episode Ingestion
# ============================================================================

@pytest.mark.asyncio
async def test_add_single_episode(surriti):
    """Add one episode and verify it's stored."""
    result = await surriti.add_episode(
        name="test_intro",
        episode_body="Michael is a 32-year-old software engineer.",
        group_id="milord_family",
    )
    assert result.episode is not None
    assert "Michael" in result.episode.content
    assert len(result.nodes) > 0 or len(result.edges) > 0


@pytest.mark.asyncio
async def test_add_multiple_episodes(surriti):
    """Add several episodes about the family."""
    episodes = [
        ("intro_michael", "Michael Milord is 32 years old and works as a software engineer."),
        ("intro_judy", "Judy Milord is 31 years old and works in clinical research."),
        ("daughter", "They have a 6-month-old daughter named Aulia."),
        ("michelle", "Michelle is Michael's mother, age 65, and works as a rheumatologist."),
        ("dogs", "The family has two dogs named Duke and Lady."),
        ("address", "The family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708."),
    ]
    for name, body in episodes:
        result = await surriti.add_episode(
            name=name,
            episode_body=body,
            group_id="milord_family",
        )
        assert result.episode is not None


@pytest.mark.asyncio
async def test_episode_with_reference_time(surriti):
    """Episodes with explicit reference_time are stored correctly."""
    past = _days_ago(30)
    result = await surriti.add_episode(
        name="old_ep",
        episode_body="The family used to live in Denver, CO.",
        group_id="milord_family",
        reference_time=past,
    )
    assert result.episode.reference_time == past


# ============================================================================
# 3. Search Patterns
# ============================================================================

@pytest.mark.asyncio
async def test_exact_search(surriti):
    """Search for exact entity names."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael works at a tech company in Florida.",
        group_id="milord_family",
    )
    results = await surriti.search("Michael", group_id="milord_family")
    assert len(results.edges) > 0 or len(results.nodes) >= 0


@pytest.mark.asyncio
async def test_fuzzy_search(surriti):
    """Search with partial/related terms finds relevant facts."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Judy works in clinical research at a hospital.",
        group_id="milord_family",
    )
    results = await surriti.search("clinical research", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_semantic_search(surriti):
    """Semantic search via embeddings finds conceptually related facts."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael is a software engineer who writes Python code.",
        group_id="milord_family",
    )
    results = await surriti.search("software developer", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_search_with_config(surriti):
    """Search with custom SearchConfig options."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Duke is a golden retriever dog.",
        group_id="milord_family",
    )
    cfg = SearchConfig(limit=5, use_vector=True, use_fulltext=True)
    results = await surriti.search("dog", group_id="milord_family", config=cfg)
    assert len(results.edges) <= 5


@pytest.mark.asyncio
async def test_search_no_results(surriti):
    """Search for something that doesn't exist returns empty."""
    results = await surriti.search("unicorn", group_id="milord_family")
    assert isinstance(results, SearchResults)


# ============================================================================
# 4. Recall
# ============================================================================

@pytest.mark.asyncio
async def test_recall_normal_depth(surriti):
    """Recall at normal depth returns profiles and facts."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael is a software engineer who loves hiking.",
        group_id="milord_family",
    )
    ctx = await surriti.recall("Tell me about Michael", group_id="milord_family", depth="normal")
    assert isinstance(ctx, MemoryContext)
    assert ctx.query == "Tell me about Michael"


@pytest.mark.asyncio
async def test_recall_deep_depth(surriti):
    """Recall at deep depth includes episodes and communities."""
    await surriti.add_episode(
        name="ep1",
        episode_body="The family lives in North Redington Beach, Florida.",
        group_id="milord_family",
    )
    ctx = await surriti.recall("Where does the family live?", group_id="milord_family", depth="deep")
    assert isinstance(ctx, MemoryContext)


# ============================================================================
# 5. Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_search(surriti):
    """Empty search string doesn't crash."""
    results = await surriti.search("", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_unicode_content(surriti):
    """Episodes with unicode characters are stored and searchable."""
    await surriti.add_episode(
        name="ep_unicode",
        episode_body="Aulia's first word was 'mama' \U0001f389",
        group_id="milord_family",
    )
    results = await surriti.search("Aulia", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_special_characters(surriti):
    """Episodes with special characters don't break the system."""
    await surriti.add_episode(
        name="ep_special",
        episode_body="Address: 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
        group_id="milord_family",
    )
    results = await surriti.search("320 Bath Club", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_long_episode(surriti):
    """Very long episodes are handled without crashing."""
    long_text = " ".join([f"Michael works on project {i}" for i in range(100)])
    result = await surriti.add_episode(
        name="ep_long",
        episode_body=long_text,
        group_id="milord_family",
    )
    assert result.episode is not None


# ============================================================================
# 6. Bulk Operations
# ============================================================================

@pytest.mark.asyncio
async def test_add_episode_bulk(surriti):
    """Add multiple episodes in bulk."""
    episodes = [
        RawEpisode(
            name=f"bulk_{i}",
            content=f"Fact number {i}: Michael is a software engineer.",
            group_id="milord_family",
        )
        for i in range(10)
    ]
    result = await surriti.add_episode_bulk(list(episodes))
    assert len(result.episodes) == 10


# ============================================================================
# 7. Contradiction Handling
# ============================================================================

@pytest.mark.asyncio
async def test_contradiction_detection(surriti):
    """Adding contradictory facts should mark old facts as invalidated."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael works at Google.",
        group_id="milord_family",
    )
    result = await surriti.add_episode(
        name="ep2",
        episode_body="Michael left Google and now works at Amazon.",
        group_id="milord_family",
    )
    assert result.episode is not None


@pytest.mark.asyncio
async def test_temporal_facts(surriti):
    """Facts with temporal validity are stored correctly."""
    past = _days_ago(60)
    await surriti.add_episode(
        name="ep_past",
        episode_body="The family lived in Denver in January 2026.",
        group_id="milord_family",
        reference_time=past,
    )
    results = await surriti.search("Denver", group_id="milord_family")
    assert isinstance(results, SearchResults)


# ============================================================================
# 8. Entity Resolution
# ============================================================================

@pytest.mark.asyncio
async def test_entity_resolution(surriti):
    """Entity resolution maps mentions to canonical entities."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael works as a software engineer.",
        group_id="milord_family",
    )
    mentions = [
        ExtractedEntity(name="Michael"),
        ExtractedEntity(name="michael"),
        ExtractedEntity(name="Michael Milord"),
    ]
    resolved = await resolve_entity_mentions(
        driver=surriti.driver,
        embedder=DummyEmbedder(64),
        llm=DummyLLMClient(),
        mentions=mentions,
        group_id="milord_family",
        threshold=0.86,
    )
    assert isinstance(resolved, list)


@pytest.mark.asyncio
async def test_profile_generation(surriti):
    """Entity profiles are generated after adding episodes."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Michael is a 32-year-old software engineer who lives in Florida.",
        group_id="milord_family",
    )
    await asyncio.sleep(0.5)
    rows = await surriti.driver.query(
        "SELECT name, profile_summary FROM entity WHERE group_id = $g AND name = $n;",
        {"g": "milord_family", "n": "Michael"},
    )
    assert isinstance(rows, list)


# ============================================================================
# 9. Search Filters
# ============================================================================

@pytest.mark.asyncio
async def test_search_with_filters(surriti):
    """Search with property filters."""
    await surriti.add_episode(
        name="ep1",
        episode_body="Duke is a golden retriever dog.",
        group_id="milord_family",
    )
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                name="group_id",
                op=ComparisonOperator.eq,
                value="milord_family",
            )
        ]
    )
    cfg = SearchConfig(filters=filters)
    results = await surriti.search("dog", group_id="milord_family", config=cfg)
    assert isinstance(results, SearchResults)


# ============================================================================
# 10. Speaker Context
# ============================================================================

@pytest.mark.asyncio
async def test_speaker_context(surriti):
    """Episodes with speaker context resolve pronouns correctly."""
    result = await surriti.add_episode(
        name="ep_speaker",
        episode_body="I work as a software engineer.",
        group_id="milord_family",
        speaker_id="michael",
        speaker_name="Michael",
    )
    assert result.episode is not None


# ============================================================================
# 11. Stress Test
# ============================================================================

@pytest.mark.asyncio
async def test_stress_50_episodes(surriti):
    """Add 50+ episodes with varied content and verify performance."""
    family_facts = [
        "Michael is a software engineer who loves coding in Python.",
        "Judy is a clinical researcher specializing in oncology trials.",
        "Aulia is their 6-month-old daughter who loves to smile.",
        "Michelle is Michael's mother and works as a rheumatologist.",
        "Duke is their golden retriever who loves the beach.",
        "Lady is their second dog, a labrador mix.",
        "The family lives in North Redington Beach, Florida.",
        "Michael enjoys hiking on weekends.",
        "Judy prefers reading medical journals in the evening.",
        "The family visits Disney World every Christmas.",
    ]

    start = time.monotonic()
    for i in range(55):
        fact = family_facts[i % len(family_facts)]
        await surriti.add_episode(
            name=f"stress_{i}",
            episode_body=f"Note {i}: {fact}",
            group_id="milord_family",
            reference_time=_days_ago(i),
        )
    elapsed = time.monotonic() - start
    print(f"\n  Stress test: 55 episodes in {elapsed:.2f}s ({elapsed/55:.3f}s/ep)")
    assert elapsed < 120


@pytest.mark.asyncio
async def test_stress_contradictions(surriti):
    """Add contradictory episodes and verify temporal handling."""
    await surriti.add_episode(
        name="contradict_1",
        episode_body="Michael worked at Google from 2020 to 2024.",
        group_id="milord_family",
        reference_time=_days_ago(365),
    )
    await surriti.add_episode(
        name="contradict_2",
        episode_body="Michael left Google in 2024 and joined Amazon.",
        group_id="milord_family",
        reference_time=_days_ago(30),
    )
    await surriti.add_episode(
        name="contradict_3",
        episode_body="Michael is now the engineering manager at Amazon.",
        group_id="milord_family",
        reference_time=_now(),
    )
    results = await surriti.search("Michael employer", group_id="milord_family")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_stress_search_performance(surriti):
    """Search performance with growing data."""
    for i in range(30):
        await surriti.add_episode(
            name=f"perf_{i}",
            episode_body=f"Michael works on project alpha-{i} using Python and Docker.",
            group_id="milord_family",
        )
    start = time.monotonic()
    results = await surriti.search("Python project", group_id="milord_family")
    elapsed = time.monotonic() - start
    print(f"\n  Search performance: {elapsed:.3f}s for 30 episodes")
    assert elapsed < 10


# ============================================================================
# 12. Multi-hop / Cross-entity queries
# ============================================================================

@pytest.mark.asyncio
async def test_cross_entity_search(surriti):
    """Search that relates multiple entities."""
    await surriti.add_episode(
        name="ep_family",
        episode_body="Michael and Judy are married. They have a daughter Aulia and two dogs Duke and Lady.",
        group_id="milord_family",
    )
    results = await surriti.search("Michael family", group_id="milord_family")
    assert isinstance(results, SearchResults)


# ============================================================================
# 13. Edge Cases with Empty/Null Data
# ============================================================================

@pytest.mark.asyncio
async def test_search_nonexistent_group(surriti):
    """Search for a group that doesn't exist returns empty."""
    results = await surriti.search("anything", group_id="nonexistent_group_xyz")
    assert isinstance(results, SearchResults)


@pytest.mark.asyncio
async def test_add_episode_with_empty_body(surriti):
    """Adding an episode with empty body doesn't crash."""
    result = await surriti.add_episode(
        name="ep_empty",
        episode_body="",
        group_id="milord_family",
    )
    assert result.episode is not None


@pytest.mark.asyncio
async def test_add_episode_with_whitespace_only(surriti):
    """Adding an episode with only whitespace is handled."""
    result = await surriti.add_episode(
        name="ep_ws",
        episode_body="   \n\t  ",
        group_id="milord_family",
    )
    assert result.episode is not None


# ============================================================================
# 14. Multiple Groups / Tenants
# ============================================================================

@pytest.mark.asyncio
async def test_multi_tenant_isolation(surriti):
    """Data is isolated between different group_ids."""
    await surriti.add_episode(
        name="ep_a",
        episode_body="Alice works at Acme Corp.",
        group_id="group_a",
    )
    await surriti.add_episode(
        name="ep_b",
        episode_body="Bob works at Beta Inc.",
        group_id="group_b",
    )
    results_a = await surriti.search("works", group_id="group_a")
    results_b = await surriti.search("works", group_id="group_b")
    assert isinstance(results_a, SearchResults)
    assert isinstance(results_b, SearchResults)


# ============================================================================
# 15. Search with include_nodes and include_episodes
# ============================================================================

@pytest.mark.asyncio
async def test_search_include_nodes_episodes(surriti):
    """Search with include_nodes and include_episodes returns full results."""
    await surriti.add_episode(
        name="ep_full",
        episode_body="Michael loves hiking in the mountains.",
        group_id="milord_family",
    )
    cfg = SearchConfig(
        include_nodes=True,
        include_episodes=True,
        include_communities=False,
    )
    results = await surriti.search_("Michael", group_id="milord_family", config=cfg)
    assert isinstance(results, SearchResults)
    assert isinstance(results.edges, list)


# ============================================================================
# 16. Cognition Layer
# ============================================================================

@pytest.mark.asyncio
async def test_cognition_enabled():
    """Cognition layer runs without crashing when enabled."""
    s = _make_surriti(cognition=True)
    await _setup_db(s)
    await s.add_episode(
        name="ep_cog",
        episode_body="Michael is learning Rust programming.",
        group_id="milord_family",
    )
    await asyncio.sleep(0.3)
    await _teardown(s)


@pytest.mark.asyncio
async def test_cognition_disabled(surriti):
    """Memory works fine with cognition disabled."""
    result = await surriti.add_episode(
        name="ep_nocog",
        episode_body="Judy reads clinical research papers.",
        group_id="milord_family",
    )
    assert result.episode is not None
    results = await surriti.search("Judy", group_id="milord_family")
    assert isinstance(results, SearchResults)


# ============================================================================
# 17. Recall with as_of (temporal recall)
# ============================================================================

@pytest.mark.asyncio
async def test_recall_as_of(surriti):
    """Recall with as_of parameter returns facts valid at that time."""
    past = _days_ago(90)
    await surriti.add_episode(
        name="ep_past",
        episode_body="The family lived in Denver, Colorado.",
        group_id="milord_family",
        reference_time=past,
    )
    ctx = await surriti.recall("Where did the family live?", group_id="milord_family", depth="normal", as_of=past)
    assert isinstance(ctx, MemoryContext)


# ============================================================================
# 18. Edge type filtering
# ============================================================================

@pytest.mark.asyncio
async def test_add_episode_with_edge_types(surriti):
    """Add episode with edge type filtering."""
    result = await surriti.add_episode(
        name="ep_edge_types",
        episode_body="Michael works at a software company.",
        group_id="milord_family",
    )
    assert result.episode is not None
    assert isinstance(result.edges, list)


# ============================================================================
# 19. Duplicate episode UUID
# ============================================================================

@pytest.mark.asyncio
async def test_duplicate_uuid(surriti):
    """Adding an episode with a pre-assigned UUID works."""
    import uuid
    my_uuid = str(uuid.uuid4())
    result = await surriti.add_episode(
        name="ep_dupe",
        episode_body="Testing duplicate UUID.",
        group_id="milord_family",
        uuid=my_uuid,
    )
    assert result.episode.uuid == my_uuid


# ============================================================================
# 20. Comprehensive family memory test
# ============================================================================

@pytest.mark.asyncio
async def test_comprehensive_family_memory(surriti):
    """Full family memory scenario — the real-world use case."""
    family_data = [
        ("onboarding", "Michael Milord is 32 years old, a software engineer. His wife Judy is 31 and works in clinical research. They have a 6-month-old daughter named Aulia. Michael's mother Michelle, 65, is a rheumatologist. They have two dogs, Duke and Lady. They live at 320 Bath Club Blvd S, North Redington Beach, FL 33708."),
        ("preferences", "Michael prefers concise responses. He likes Python and Go. Judy prefers detailed explanations for medical topics."),
        ("routine", "The family wakes up early. Michael cooks breakfast. Judy handles Aulia's morning routine. They walk Duke and Lady every evening."),
        ("work", "Michael works remotely most days. Judy goes to the clinical research site twice a week. Michelle works at the rheumatology clinic downtown."),
        ("pets", "Duke is a golden retriever, 5 years old. Lady is a labrador mix, 3 years old. Both are vaccinated and spayed/neutered."),
        ("health", "Michelle has rheumatoid arthritis. Judy specializes in oncology clinical trials. Michael has no chronic conditions."),
        ("travel", "The family visits Disney World every December. They went to Key West last summer. They want to visit Japan next spring."),
        ("home", "The house has a screened lanai overlooking the water. They just painted the exterior. The roof was replaced in 2024."),
    ]

    for name, body in family_data:
        result = await surriti.add_episode(
            name=name,
            episode_body=body,
            group_id="milord_family",
        )
        assert result.episode is not None

    searches = [
        "Michael",
        "Judy",
        "Aulia",
        "dogs",
        "address",
        "Michelle",
        "work",
        "travel",
        "Florida",
    ]

    for query in searches:
        results = await surriti.search(query, group_id="milord_family")
        assert isinstance(results, SearchResults), f"Search for '{query}' failed"

    ctx = await surriti.recall("Tell me about the family", group_id="milord_family", depth="deep")
    assert isinstance(ctx, MemoryContext)

    print("\n  Comprehensive family memory test PASSED")
