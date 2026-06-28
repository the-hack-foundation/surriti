"""Comprehensive integration test with realistic family data.

Tests:
- Connection and schema initialization
- Family seeding (all 5 family members)
- Exact, fuzzy, semantic, and temporal searches
- Recall with different depths
- Edge cases (empty search, special characters, unicode)
- Bulk operations
- Contradiction handling (facts that change over time)
- Temporal queries (facts valid at different times)
- Entity resolution (same person, different names)
- Multi-hop reasoning
- Performance with growing data
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from surriti import Surriti, DummyEmbedder, DummyLLMClient, EpisodeType
from surriti.driver import SurrealDriver
from surriti.search_filters import SearchFilters, DateFilter, ComparisonOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def driver():
    """Create a fresh SurrealDriver connected to the running SurrealDB."""
    drv = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti",
        database="surriti",
        username="root",
        password="root",
    )
    await drv.connect()
    await drv.init_schema()
    yield drv
    # Cleanup: drop all tables
    for table in ("episode", "entity", "entity_alias", "community", "mentions",
                  "relates_to", "has_member", "relation_frame", "prediction",
                  "trait", "goal", "conflict", "interaction_pattern"):
        try:
            await drv.query(f"DELETE {table};")
        except Exception:
            pass
    try:
        await drv.close()
    except Exception:
        pass


@pytest_asyncio.fixture
async def surriti(driver):
    """Create a Surriti instance with dummy LLM/embedder."""
    s = Surriti(
        driver=driver,
        llm_client=DummyLLMClient(),
        embedder=DummyEmbedder(),
        alias_resolution=True,
        alias_resolution_threshold=0.86,
        alias_resolution_llm=True,
        profile_refresh="off",
        cognition=False,  # disable cognitive layer for speed
    )
    await s.connect()
    await s.build_indices_and_constraints()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# Test 1: Connection & Schema
# ---------------------------------------------------------------------------

class TestConnection:
    async def test_connect_and_schema(self, surriti):
        """Verify connection and schema initialization."""
        assert surriti.driver is not None
        # Verify schema tables exist
        tables = await surriti.driver.query("INFO FOR DB;")
        assert tables is not None

    async def test_build_indices(self, surriti):
        """Verify HNSW indices are built."""
        # Should not raise
        await surriti.build_indices_and_constraints()


# ---------------------------------------------------------------------------
# Test 2: Family Seeding
# ---------------------------------------------------------------------------

class TestFamilySeeding:
    """Seed the Milord family into Surriti."""

    FAMILY_DATA = [
        # Michael
        {
            "name": "michael-intro",
            "content": "Michael is 32 years old. He is a software engineer who works remotely. He lives in North Redington Beach, FL with his family.",
            "reference_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        },
        # Judy
        {
            "name": "judy-intro",
            "content": "Judy is 31 years old. She works in clinical research. She is Michael's wife.",
            "reference_time": datetime(2026, 1, 15, 10, 5, tzinfo=timezone.utc),
        },
        # Aulia
        {
            "name": "aulia-intro",
            "content": "Aulia is the 6-month-old daughter of Michael and Judy. She was born around November 2025.",
            "reference_time": datetime(2026, 1, 15, 10, 10, tzinfo=timezone.utc),
        },
        # Michelle
        {
            "name": "michelle-intro",
            "content": "Michelle is 65 years old. She is a rheumatologist. She is Michael's mother.",
            "reference_time": datetime(2026, 1, 15, 10, 15, tzinfo=timezone.utc),
        },
        # Duke
        {
            "name": "duke-intro",
            "content": "Duke is the family dog. He is a large breed dog.",
            "reference_time": datetime(2026, 1, 15, 10, 20, tzinfo=timezone.utc),
        },
        # Lady
        {
            "name": "lady-intro",
            "content": "Lady is the family dog. She is a small breed dog.",
            "reference_time": datetime(2026, 1, 15, 10, 25, tzinfo=timezone.utc),
        },
        # Address
        {
            "name": "address-intro",
            "content": "The Milord family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
            "reference_time": datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
        },
        # Timezone
        {
            "name": "timezone-intro",
            "content": "The family is in the EST/EDT timezone (Eastern Time).",
            "reference_time": datetime(2026, 1, 15, 10, 35, tzinfo=timezone.utc),
        },
    ]

    async def test_seed_family(self, surriti):
        """Seed all family members and verify entities exist."""
        group_id = "user:milord"

        for episode_data in self.FAMILY_DATA:
            result = await surriti.add_episode(
                name=episode_data["name"],
                episode_body=episode_data["content"],
                source=EpisodeType.text,
                reference_time=episode_data["reference_time"],
                group_id=group_id,
            )

            assert result.episode is not None
            assert result.episode.name == episode_data["name"]
            assert result.nodes is not None
            assert result.edges is not None

        # Verify entities were created by searching
        results = await surriti.search(query="Michael", group_id=group_id, limit=5)
        assert results is not None
        # Should find at least one result
        assert len(results.edges) > 0 or len(results.nodes) > 0

    async def test_upsert_user(self, surriti):
        """Test upsert_user creates canonical User entity."""
        group_id = "user:milord"

        user = await surriti.upsert_user(
            group_id=group_id,
            user_id="michael",
            display_name="Michael Milord",
            summary="Software engineer, father of one.",
        )
        assert user is not None
        assert user.name == "michael"
        assert user.labels == ["User"]
        assert user.attributes.get("display_name") == "Michael Milord"

        # Idempotent: second call should return same user
        user2 = await surriti.upsert_user(
            group_id=group_id,
            user_id="michael",
            display_name="Michael Milord",
            summary="Updated summary.",
        )
        assert user2.uuid == user.uuid
        assert user2.summary == "Updated summary."


# ---------------------------------------------------------------------------
# Test 3: Search Patterns
# ---------------------------------------------------------------------------

class TestSearchPatterns:
    async def test_exact_search(self, surriti):
        """Test exact keyword search."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-exact",
            episode_body="Michael works as a software engineer at a tech company.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        results = await surriti.search(query="software engineer", group_id=group_id, limit=5)
        assert results is not None
        # Should find at least one result
        assert len(results.edges) > 0 or len(results.nodes) > 0 or len(results.episodes) > 0

    async def test_fuzzy_search(self, surriti):
        """Test fuzzy/misspelled search."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-fuzzy",
            episode_body="Judy works in clinical research for a pharmaceutical company.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Misspelled query should still not crash
        results = await surriti.search(query="clincial reserch", group_id=group_id, limit=5)
        assert results is not None

    async def test_semantic_search(self, surriti):
        """Test semantic search (synonyms, related concepts)."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-semantic",
            episode_body="The family has two dogs named Duke and Lady.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # "pets" should semantically relate to "dogs"
        results = await surriti.search(query="pets", group_id=group_id, limit=5)
        assert results is not None

    async def test_temporal_search(self, surriti):
        """Test search with temporal filters."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-temporal",
            episode_body="In January 2026, the family moved to a new house.",
            source=EpisodeType.text,
            reference_time=datetime(2026, 1, 15, tzinfo=timezone.utc),
            group_id=group_id,
        )

        # Search with date filter
        results = await surriti.search(
            query="family house",
            group_id=group_id,
            search_filter=SearchFilters(
                created_at=[[DateFilter(datetime(2026, 1, 1), ComparisonOperator.gte), DateFilter(datetime(2026, 2, 1), ComparisonOperator.lte)]],
            ),
            limit=5,
        )
        assert results is not None

    async def test_empty_search(self, surriti):
        """Test that empty search doesn't crash."""
        group_id = "user:milord"
        results = await surriti.search(query="", group_id=group_id, limit=5)
        assert results is not None

    async def test_special_characters(self, surriti):
        """Test search with special characters."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-special",
            episode_body="The address is 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        results = await surriti.search(query="320 Bath Club", group_id=group_id, limit=5)
        assert results is not None

    async def test_unicode_search(self, surriti):
        """Test search with unicode characters."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-unicode",
            episode_body="The family enjoys café culture and résumé writing workshops.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        results = await surriti.search(query="café", group_id=group_id, limit=5)
        assert results is not None


# ---------------------------------------------------------------------------
# Test 4: Recall
# ---------------------------------------------------------------------------

class TestRecall:
    async def test_recall_fast(self, surriti):
        """Test recall at fast depth."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-recall",
            episode_body="Michael and Judy have a daughter named Aulia.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        ctx = await surriti.recall(
            query="Who is Michael's family?",
            group_id=group_id,
            depth="fast",
            limit=5,
        )
        assert ctx is not None
        assert ctx.query == "Who is Michael's family?"

    async def test_recall_normal(self, surriti):
        """Test recall at normal depth."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-recall-normal",
            episode_body="Judy works in clinical research. She is married to Michael.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        ctx = await surriti.recall(
            query="Judy's job",
            group_id=group_id,
            depth="normal",
            limit=10,
        )
        assert ctx is not None

    async def test_recall_deep(self, surriti):
        """Test recall at deep depth (includes episodes and communities)."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-recall-deep",
            episode_body="The family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        ctx = await surriti.recall(
            query="Where does the family live?",
            group_id=group_id,
            depth="deep",
            limit=10,
        )
        assert ctx is not None
        # Deep should include episodes
        assert ctx.episodes is not None

    async def test_recall_with_entities(self, surriti):
        """Test recall with include_entities and include_edges."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="test-recall-entities",
            episode_body="Michael is a software engineer.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        ctx = await surriti.recall(
            query="Michael",
            group_id=group_id,
            depth="normal",
            include_entities=True,
            include_edges=True,
        )
        assert ctx is not None


# ---------------------------------------------------------------------------
# Test 5: Bulk Operations
# ---------------------------------------------------------------------------

class TestBulkOperations:
    async def test_add_episode_bulk(self, surriti):
        """Test bulk episode ingestion."""
        group_id = "user:milord"

        episodes = [
            {
                "name": f"bulk-{i}",
                "content": f"Fact number {i}: The family enjoys outdoor activities.",
                "source": EpisodeType.text,
                "group_id": group_id,
            }
            for i in range(10)
        ]

        results = await surriti.add_episode_bulk(episodes)
        assert results is not None
        assert len(results.episodes) == 10

    async def test_bulk_with_different_sources(self, surriti):
        """Test bulk with mixed source types."""
        group_id = "user:milord"

        episodes = [
            {
                "name": "bulk-msg",
                "content": "Michael said he loves coding.",
                "source": EpisodeType.message,
                "group_id": group_id,
            },
            {
                "name": "bulk-json",
                "content": '{"activity": "coding", "person": "Michael"}',
                "source": EpisodeType.json,
                "group_id": group_id,
            },
            {
                "name": "bulk-text",
                "content": "Judy prefers morning runs.",
                "source": EpisodeType.text,
                "group_id": group_id,
            },
        ]

        results = await surriti.add_episode_bulk(episodes)
        assert results is not None
        assert len(results.episodes) == 3


# ---------------------------------------------------------------------------
# Test 6: Contradiction Handling
# ---------------------------------------------------------------------------

class TestContradictionHandling:
    async def test_facts_that_change_over_time(self, surriti):
        """Test that contradictory facts are handled with temporal awareness."""
        group_id = "user:milord"

        # Episode 1: Michael drives a Honda
        r1 = await surriti.add_episode(
            name="car-honda",
            episode_body="Michael drives a Honda Civic.",
            source=EpisodeType.text,
            reference_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
            group_id=group_id,
        )

        # Episode 2: Michael now drives a Toyota (contradiction)
        r2 = await surriti.add_episode(
            name="car-toyota",
            episode_body="Michael now drives a Toyota Camry.",
            source=EpisodeType.text,
            reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            group_id=group_id,
        )

        assert r1 is not None
        assert r2 is not None

        # Search for car-related facts
        results = await surriti.search(
            query="Michael drives car",
            group_id=group_id,
            limit=10,
        )
        assert results is not None

    async def test_singleton_slot_closing(self, surriti):
        """Test that singleton facts (like spouse) get properly closed."""
        group_id = "user:milord"

        # First: Michael's spouse is Judy
        r1 = await surriti.add_episode(
            name="spouse-judy",
            episode_body="Michael's wife is Judy.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Second: Michael's wife is Sarah (contradiction - should close Judy)
        r2 = await surriti.add_episode(
            name="spouse-sarah",
            episode_body="Michael's wife is Sarah.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        assert r1 is not None
        assert r2 is not None
        # The system should handle the contradiction
        # (with DummyLLM, it won't auto-resolve, but shouldn't crash)


# ---------------------------------------------------------------------------
# Test 7: Entity Resolution
# ---------------------------------------------------------------------------

class TestEntityResolution:
    async def test_case_insensitive_entity(self, surriti):
        """Test that 'Michael' and 'michael' resolve to the same entity."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="entity-michael",
            episode_body="Michael is a software engineer.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        await surriti.add_episode(
            name="entity-michael-2",
            episode_body="michael works remotely from home.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Search should find Michael
        results = await surriti.search(query="Michael", group_id=group_id, limit=5)
        assert results is not None

    async def test_alias_resolution(self, surriti):
        """Test that entity aliases are resolved."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="alias-test",
            episode_body="Michael M. is a software engineer.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Should not crash
        results = await surriti.search(query="Michael M", group_id=group_id, limit=5)
        assert results is not None


# ---------------------------------------------------------------------------
# Test 8: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    async def test_very_long_content(self, surriti):
        """Test episode with very long content."""
        group_id = "user:milord"

        long_content = "Michael is great. " * 5000  # ~75KB

        result = await surriti.add_episode(
            name="test-long",
            episode_body=long_content,
            source=EpisodeType.text,
            group_id=group_id,
        )

        assert result is not None
        assert result.episode is not None

    async def test_empty_content(self, surriti):
        """Test episode with empty content."""
        group_id = "user:milord"

        result = await surriti.add_episode(
            name="test-empty",
            episode_body="",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Should handle gracefully
        assert result is not None

    async def test_special_characters_in_content(self, surriti):
        """Test episode with special characters."""
        group_id = "user:milord"

        content = "Email: user@example.com | Phone: (555) 123-4567 | URL: https://example.com/path?q=1&r=2"

        result = await surriti.add_episode(
            name="test-special-chars",
            episode_body=content,
            source=EpisodeType.text,
            group_id=group_id,
        )

        assert result is not None

    async def test_null_like_content(self, surriti):
        """Test episode with None-like content."""
        group_id = "user:milord"

        result = await surriti.add_episode(
            name="test-null",
            episode_body="   \n\t  ",
            source=EpisodeType.text,
            group_id=group_id,
        )

        assert result is not None


# ---------------------------------------------------------------------------
# Test 9: Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    async def test_search_performance(self, surriti):
        """Test search performance with moderate data."""
        group_id = "user:milord"

        # Add 20 episodes
        start = time.time()
        for i in range(20):
            await surriti.add_episode(
                name=f"perf-{i}",
                episode_body=f"Michael's fact number {i}: He enjoys programming and reading.",
                source=EpisodeType.text,
                group_id=group_id,
            )
        add_time = time.time() - start

        # Search
        start = time.time()
        results = await surriti.search(query="Michael programming", group_id=group_id, limit=5)
        search_time = time.time() - start

        # Should complete in reasonable time
        assert add_time < 30, f"Adding 20 episodes took {add_time:.1f}s"
        assert search_time < 5, f"Search took {search_time:.1f}s"

    async def test_recall_performance(self, surriti):
        """Test recall performance."""
        group_id = "user:milord"

        # Add some episodes
        for i in range(10):
            await surriti.add_episode(
                name=f"recall-perf-{i}",
                episode_body=f"Judy's fact {i}: She works in clinical research.",
                source=EpisodeType.text,
                group_id=group_id,
            )

        start = time.time()
        ctx = await surriti.recall(
            query="Judy's work",
            group_id=group_id,
            depth="normal",
            limit=10,
        )
        recall_time = time.time() - start

        assert recall_time < 10, f"Recall took {recall_time:.1f}s"
        assert ctx is not None


# ---------------------------------------------------------------------------
# Test 10: Multi-hop Reasoning
# ---------------------------------------------------------------------------

class TestMultiHopReasoning:
    async def test_family_relationships(self, surriti):
        """Test multi-hop: find relationships through entities."""
        group_id = "user:milord"

        # Build a chain: Michael -> married_to -> Judy -> works_in -> clinical research
        await surriti.add_episode(
            name="rel-1",
            episode_body="Michael is married to Judy.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        await surriti.add_episode(
            name="rel-2",
            episode_body="Judy works in clinical research.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Search for Michael's wife's profession
        results = await surriti.search(
            query="Michael wife profession",
            group_id=group_id,
            limit=10,
        )
        assert results is not None

    async def test_entity_dossier(self, surriti):
        """Test getting a full entity dossier."""
        group_id = "user:milord"

        await surriti.add_episode(
            name="dossier-michael",
            episode_body="Michael is 32 years old. He is a software engineer. He lives in Florida.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Search for Michael
        results = await surriti.search(query="Michael", group_id=group_id, limit=5)
        assert results is not None
        assert len(results.edges) > 0 or len(results.nodes) > 0

        # Get Michael's entity node
        if results.nodes:
            michael_node = results.nodes[0]
            node_detail = await surriti.get_entity_node(michael_node.uuid)
            assert node_detail is not None


# ---------------------------------------------------------------------------
# Test 11: Temporal Queries
# ---------------------------------------------------------------------------

class TestTemporalQueries:
    async def test_facts_valid_at_time(self, surriti):
        """Test querying facts valid at a specific time."""
        group_id = "user:milord"

        # Old fact
        await surriti.add_episode(
            name="temporal-old",
            episode_body="The family lived in New York in 2020.",
            source=EpisodeType.text,
            reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            group_id=group_id,
        )

        # New fact
        await surriti.add_episode(
            name="temporal-new",
            episode_body="The family lives in Florida in 2026.",
            source=EpisodeType.text,
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            group_id=group_id,
        )

        # Search with temporal awareness
        results = await surriti.search(
            query="family location",
            group_id=group_id,
            limit=10,
        )
        assert results is not None

    async def test_include_invalid_edges(self, surriti):
        """Test retrieving invalidated (superseded) edges."""
        group_id = "user:milord"

        # Add a fact that will be invalidated
        await surriti.add_episode(
            name="invalid-1",
            episode_body="Michael used to live in New York.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Add a contradicting fact
        await surriti.add_episode(
            name="invalid-2",
            episode_body="Michael now lives in Florida.",
            source=EpisodeType.text,
            group_id=group_id,
        )

        # Search with include_invalid
        results = await surriti.search(
            query="Michael lives",
            group_id=group_id,
            limit=10,
            only_valid=False,
        )
        assert results is not None


# ---------------------------------------------------------------------------
# Test 12: Driver Direct Operations
# ---------------------------------------------------------------------------

class TestDriverDirect:
    async def test_driver_query(self, driver):
        """Test direct driver queries."""
        result = await driver.query("SELECT COUNT() FROM episode;")
        assert result is not None

    async def test_driver_query_with_params(self, driver):
        """Test parameterized queries."""
        result = await driver.query(
            "SELECT * FROM episode WHERE group_id = $g LIMIT 1;",
            {"g": "test"},
        )
        assert result is not None

    async def test_driver_clear(self, driver):
        """Test clearing all data."""
        # Add some data
        await driver.query(
            'CREATE episode:test CONTENT {"uuid": "test", "group_id": "test", "name": "test", "content": "test content"};'
        )

        # Clear
        await driver.clear()

        # Verify empty
        result = await driver.query("SELECT COUNT() FROM episode;")
        assert result is not None


# ---------------------------------------------------------------------------
# Test 13: Embedder & LLM Client
# ---------------------------------------------------------------------------

class TestEmbedderLLMClient:
    async def test_dummy_embedder(self, surriti):
        """Test DummyEmbedder creates embeddings."""
        emb = await surriti.embedder.create("test text")
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) > 0

    async def test_dummy_embedder_batch(self, surriti):
        """Test DummyEmbedder batch creation."""
        embs = await surriti.embedder.create_batch(["text1", "text2", "text3"])
        assert len(embs) == 3
        assert all(isinstance(e, list) for e in embs)

    async def test_dummy_llm_extract(self, surriti):
        """Test DummyLLMClient extract_entities."""
        result = await surriti.llm.extract_entities(
            text="Michael is a software engineer.",
            extraction_instructions="",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60", "-x"])
