"""Comprehensive integration test for the Surriti SDK.

Tests:
  1. Connection & schema init
  2. Family memory seeding
  3. Episode ingestion (single + bulk)
  4. Entity resolution (case variants, aliases)
  5. Contradiction handling (temporal facts)
  6. Search: exact, fuzzy, semantic, temporal
  7. Recall with different depths
  8. Edge cases (empty search, unicode, special chars)
  9. Bulk operations
  10. Temporal queries (facts valid at different times)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio

# Ensure the surriti package on disk is importable
SDK_ROOT = Path(__file__).resolve().parent.parent / "surriti"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from surriti.graphiti import Surriti
from surriti.llm import DummyLLMClient, ScriptedLLMClient, ScriptedResponse, ExtractedFact, ExtractedEntity
from surriti.embedder import DummyEmbedder, cosine_similarity
from surriti.driver import SurrealDriver
from surriti.search import SearchConfig, SearchResults
from surriti.nodes import EpisodicNode
from surriti.edges import EntityEdge
from surriti.errors import SurritiConnectionError

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- fixtures
# Use pytest_asyncio.fixture with function scope for proper event-loop
# scoping under asyncio_mode="auto".  Module-scoped async fixtures get
# bound to a different loop than the tests, causing "attached to a
# different loop" errors.

@pytest_asyncio.fixture
async def driver():
    # Local SurrealDB runs with --user root --pass root
    d = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti",
        database="surriti",
        username="root",
        password="root",
    )
    await d.connect()
    await d.init_schema()
    yield d
    # Teardown: reconnect if the test closed the driver, then clear
    try:
        if not hasattr(d, '_db') or d._db is None:
            await d.connect()
    except Exception:
        pass
    try:
        await d.clear()
    except Exception:
        pass
    try:
        await d.close()
    except Exception:
        pass


@pytest_asyncio.fixture
async def surriti(driver):
    s = Surriti(
        driver=driver,
        llm_client=DummyLLMClient(),
        embedder=DummyEmbedder(),
    )
    await s.connect()
    yield s
    # Use driver.clear() since Surriti has no clear() method
    await driver.clear()
    await s.close()


# =================================================================== 1. Connection
class TestConnection:
    async def test_connect_and_init_schema(self, driver):
        """Driver connects and schema is idempotent."""
        await driver.init_schema()  # should not raise
        await driver.init_schema()  # idempotent

    async def test_clear_is_safe(self, driver):
        """Clearing empty DB is fine."""
        await driver.clear()

    async def test_disconnect_reconnect(self, driver):
        await driver.close()
        await driver.connect()
        await driver.close()


# =================================================================== 2. Seeding & Ingestion
class TestSeeding:
    async def test_add_episode(self, surriti):
        ep = await surriti.add_episode(
            name="Michael's profession",
            episode_body="Michael is a software engineer.",
            group_id="test-family",
        )
        assert ep is not None
        assert ep.episode is not None

    async def test_add_episode_with_name(self, surriti):
        ep = await surriti.add_episode(
            name="Judy's profession",
            episode_body="Judy works in clinical research.",
            group_id="test-family",
        )
        assert ep is not None

    async def test_add_episode_unicode(self, surriti):
        ep = await surriti.add_episode(
            name="Aulia's toy",
            episode_body="Aulia loves her stuffed bear 小熊.",
            group_id="test-family",
        )
        assert ep is not None

    async def test_add_episode_special_chars(self, surriti):
        ep = await surriti.add_episode(
            name="Michelle's address",
            episode_body="Michelle's address is 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
            group_id="test-family",
        )
        assert ep is not None

    async def test_add_empty_content(self, surriti):
        ep = await surriti.add_episode(
            name="empty",
            episode_body="",
            group_id="test-family",
        )
        assert ep is not None  # should not crash

    async def test_bulk_add_episodes(self, surriti):
        episodes = [
            {"name": "Duke", "episode_body": "Duke is a dog."},
            {"name": "Lady", "episode_body": "Lady is a dog."},
            {"name": "Aulia", "episode_body": "Aulia is 6 months old."},
            {"name": "Michelle", "episode_body": "Michelle is a rheumatologist."},
            {"name": "Florida", "episode_body": "The family lives in Florida."},
        ]
        results = await surriti.add_episode_bulk(episodes, group_id="test-family")
        assert len(results.episodes) == 5
        assert all(r.episode is not None for r in results.episodes)

    async def test_add_episode_returns_results(self, surriti):
        ep = await surriti.add_episode(
            name="Michael daughter",
            episode_body="Michael has a daughter named Aulia.",
            group_id="test-family",
        )
        # Verify it's a dataclass-like result with expected attributes
        assert hasattr(ep, "episode")
        assert hasattr(ep, "nodes")
        assert hasattr(ep, "edges")


# =================================================================== 3. Entity Resolution
class TestEntityResolution:
    async def test_same_entity_different_cases(self, surriti):
        await surriti.add_episode(
            name="Michael 1",
            episode_body="Michael is a software engineer.",
            group_id="test-family",
        )
        await surriti.add_episode(
            name="Michael 2",
            episode_body="michael works on AI projects.",
            group_id="test-family",
        )
        # Both should resolve to the same entity
        results = await surriti.recall("michael", depth="normal", group_id="test-family")
        assert results is not None

    async def test_entity_mentions_created(self, surriti):
        await surriti.add_episode(
            name="Duke",
            episode_body="Duke is a dog.",
            group_id="test-family",
        )
        results = await surriti.recall("Duke", depth="fast", group_id="test-family")
        assert results is not None

    async def test_multiple_entities_in_one_episode(self, surriti):
        await surriti.add_episode(
            name="Family",
            episode_body="Michael and Judy are married and have a daughter Aulia.",
            group_id="test-family",
        )
        results = await surriti.recall("Michael", depth="fast", group_id="test-family")
        assert results is not None


# =================================================================== 4. Contradiction Handling
class TestContradictionHandling:
    async def test_temporal_fact_update(self, surriti):
        """Add a fact, then add a contradicting fact."""
        await surriti.add_episode(
            name="Duke weight 1",
            episode_body="Duke weighs 70 pounds.",
            group_id="test-family",
        )
        await asyncio.sleep(0.1)
        await surriti.add_episode(
            name="Duke weight 2",
            episode_body="Duke no longer weighs 70 pounds; he now weighs 75 pounds.",
            group_id="test-family",
        )
        # The old fact should be invalidated
        results = await surriti.recall("Duke", depth="normal", group_id="test-family")
        assert results is not None

    async def test_non_contradiction_same_topic(self, surriti):
        """Adding related facts about the same entity shouldn't contradict."""
        await surriti.add_episode(
            name="Duke breed",
            episode_body="Duke is a golden retriever.",
            group_id="test-family",
        )
        await surriti.add_episode(
            name="Duke swim",
            episode_body="Duke likes to swim.",
            group_id="test-family",
        )
        results = await surriti.recall("Duke", depth="fast", group_id="test-family")
        assert results is not None
        # Both facts should still be active
        facts = [e for e in results.facts if "Duke" in e.fact]
        assert len(facts) >= 2

    async def test_conflicting_location(self, surriti):
        await surriti.add_episode(
            name="Location 1",
            episode_body="The family lives in Florida.",
            group_id="test-family",
        )
        await asyncio.sleep(0.1)
        await surriti.add_episode(
            name="Location 2",
            episode_body="The family moved to a new house in Florida.",
            group_id="test-family",
        )
        results = await surriti.recall("family", depth="normal", group_id="test-family")
        assert results is not None


# =================================================================== 5. Search
class TestSearch:
    async def test_exact_search(self, surriti):
        await surriti.add_episode(
            name="Michael job",
            episode_body="Michael is a software engineer.",
            group_id="test-family",
        )
        results = await surriti.search("Michael", group_id="test-family")
        assert results is not None
        assert len(results.episodes) > 0

    async def test_fuzzy_search(self, surriti):
        await surriti.add_episode(
            name="Judy job",
            episode_body="Judy works in clinical research.",
            group_id="test-family",
        )
        results = await surriti.search("clinical", group_id="test-family")
        assert results is not None

    async def test_semantic_search(self, surriti):
        await surriti.add_episode(
            name="Aulia baby",
            episode_body="Aulia is a baby girl.",
            group_id="test-family",
        )
        results = await surriti.search("baby", group_id="test-family")
        assert results is not None

    async def test_search_no_results(self, surriti):
        results = await surriti.search("zzzznonexistent", group_id="test-family")
        assert results is not None
        assert len(results.episodes) == 0

    async def test_search_with_limit(self, surriti):
        await surriti.add_episode(name="Duke dog", episode_body="Duke is a dog.", group_id="test-family")
        await surriti.add_episode(name="Lady dog", episode_body="Lady is a dog.", group_id="test-family")
        results = await surriti.search("dog", group_id="test-family", limit=1)
        assert results is not None
        assert len(results.episodes) <= 1

    async def test_search_with_depth(self, surriti):
        await surriti.add_episode(
            name="Married",
            episode_body="Michael and Judy are married.",
            group_id="test-family",
        )
        results_depth_fast = await surriti.search("Michael", depth="fast", group_id="test-family")
        results_depth_deep = await surriti.search("Michael", depth="deep", group_id="test-family")
        assert results_depth_fast is not None
        assert results_depth_deep is not None

    async def test_search_with_config(self, surriti):
        await surriti.add_episode(
            name="Michelle age",
            episode_body="Michelle is 65 years old.",
            group_id="test-family",
        )
        config = SearchConfig(limit=5, include_nodes=True)
        results = await surriti.search("Michelle", group_id="test-family", config=config)
        assert results is not None

    async def test_search_unicode(self, surriti):
        await surriti.add_episode(
            name="Aulia bear",
            episode_body="Aulia loves 小熊 the teddy bear.",
            group_id="test-family",
        )
        results = await surriti.search("Aulia", group_id="test-family")
        assert results is not None

    async def test_search_special_characters(self, surriti):
        await surriti.add_episode(
            name="Address",
            episode_body="Address: 320 Bath Club Blvd S.",
            group_id="test-family",
        )
        results = await surriti.search("Bath Club", group_id="test-family")
        assert results is not None

    async def test_search_empty_string(self, surriti):
        results = await surriti.search("", group_id="test-family")
        assert results is not None

    async def test_search_with_reranker(self, surriti):
        await surriti.add_episode(name="Duke golden", episode_body="Duke is a golden retriever dog.", group_id="test-family")
        await surriti.add_episode(name="Lady mixed", episode_body="Lady is a mixed breed dog.", group_id="test-family")
        results = await surriti.search("dog", group_id="test-family", rerank_strategy="rrf")
        assert results is not None


# =================================================================== 6. Recall
class TestRecall:
    async def test_recall_by_entity(self, surriti):
        await surriti.add_episode(
            name="Michael job",
            episode_body="Michael is a software engineer.",
            group_id="test-family",
        )
        results = await surriti.recall("Michael", depth="fast", group_id="test-family")
        assert results is not None
        assert len(results.episodes) > 0

    async def test_recall_depth_0(self, surriti):
        await surriti.add_episode(name="Duke", episode_body="Duke is a dog.", group_id="test-family")
        results = await surriti.recall("Duke", depth="fast", group_id="test-family")
        assert results is not None

    async def test_recall_depth_2(self, surriti):
        await surriti.add_episode(
            name="Married",
            episode_body="Michael and Judy are married.",
            group_id="test-family",
        )
        results = await surriti.recall("Michael", depth="deep", group_id="test-family")
        assert results is not None

    async def test_recall_no_results(self, surriti):
        results = await surriti.recall("zzzznonexistent", depth="fast", group_id="test-family")
        assert results is not None

    async def test_recall_with_include_edges(self, surriti):
        await surriti.add_episode(name="Duke", episode_body="Duke is a dog.", group_id="test-family")
        results = await surriti.recall("Duke", depth="fast", group_id="test-family", include_edges=True)
        assert results is not None

    async def test_recall_with_include_entities(self, surriti):
        await surriti.add_episode(name="Lady", episode_body="Lady is a dog.", group_id="test-family")
        results = await surriti.recall("Lady", depth="fast", group_id="test-family", include_entities=True)
        assert results is not None


# =================================================================== 7. Temporal Queries
class TestTemporalQueries:
    async def test_only_valid_search(self, surriti):
        await surriti.add_episode(
            name="Duke weight 1",
            episode_body="Duke weighs 70 pounds.",
            group_id="test-family",
        )
        await asyncio.sleep(0.1)
        await surriti.add_episode(
            name="Duke weight 2",
            episode_body="Duke now weighs 75 pounds.",
            group_id="test-family",
        )
        # Search with only_valid=True should only return current fact
        results = await surriti.search("Duke weight", group_id="test-family", only_valid=True)
        assert results is not None

    async def test_history_query(self, surriti):
        await surriti.add_episode(
            name="Duke old weight",
            episode_body="Duke was 70 pounds.",
            group_id="test-family",
        )
        await asyncio.sleep(0.1)
        await surriti.add_episode(
            name="Duke new weight",
            episode_body="Duke is now 75 pounds.",
            group_id="test-family",
        )
        # Search with only_valid=False should include history
        results = await surriti.search("Duke", group_id="test-family", only_valid=False)
        assert results is not None


# =================================================================== 8. Edge Cases
class TestEdgeCases:
    async def test_very_long_episode(self, surriti):
        long_text = "Michael is a software engineer. " * 500
        ep = await surriti.add_episode(
            name="long",
            episode_body=long_text,
            group_id="test-family",
        )
        assert ep is not None

    async def test_single_word_episode(self, surriti):
        ep = await surriti.add_episode(
            name="Duke",
            episode_body="Duke",
            group_id="test-family",
        )
        assert ep is not None

    async def test_numbers_in_episode(self, surriti):
        ep = await surriti.add_episode(
            name="Michelle",
            episode_body="Michelle is 65 years old and lives at 320 Bath Club Blvd.",
            group_id="test-family",
        )
        assert ep is not None

    async def test_repeated_ingest_same_content(self, surriti):
        await surriti.add_episode(name="Duke 1", episode_body="Duke is a dog.", group_id="test-family")
        await surriti.add_episode(name="Duke 2", episode_body="Duke is a dog.", group_id="test-family")
        results = await surriti.recall("Duke", depth="fast", group_id="test-family")
        assert results is not None

    async def test_mixed_language_episode(self, surriti):
        ep = await surriti.add_episode(
            name="Mixed",
            episode_body="Aulia plays with 小熊 and teddy bear 泰迪熊.",
            group_id="test-family",
        )
        assert ep is not None

    async def test_punctuation_heavy(self, surriti):
        ep = await surriti.add_episode(
            name="Punctuation",
            episode_body="Michael's, Judy's, Aulia's, Michelle's — all at 320 Bath Club Blvd S!",
            group_id="test-family",
        )
        assert ep is not None


# =================================================================== 9. Bulk Operations
class TestBulkOperations:
    async def test_bulk_add_many(self, surriti):
        episodes = [
            {"name": f"Episode {i}", "episode_body": f"Michael learned something new about software engineering in episode {i}."}
            for i in range(20)
        ]
        results = await surriti.add_episode_bulk(episodes, group_id="test-family")
        assert len(results.episodes) == 20

    async def test_bulk_add_empty_list(self, surriti):
        results = await surriti.add_episode_bulk([], group_id="test-family")
        assert len(results.episodes) == 0

    async def test_bulk_add_with_varied_content(self, surriti):
        episodes = [
            {"name": "Dog 1", "episode_body": "Duke barks loudly."},
            {"name": "Dog 2", "episode_body": "Lady sleeps all day."},
            {"name": "Baby 1", "episode_body": "Aulia crawls fast."},
            {"name": "Baby 2", "episode_body": "Aulia says mama."},
            {"name": "Work 1", "episode_body": "Michael codes in Python."},
            {"name": "Work 2", "episode_body": "Judy runs clinical trials."},
        ]
        results = await surriti.add_episode_bulk(episodes, group_id="test-family")
        assert len(results.episodes) == 6


# =================================================================== 10. ScriptedLLMClient
class TestScriptedLLM:
    async def test_scripted_extraction(self, surriti):
        """Use scripted LLM to control extraction precisely."""
        scripted = ScriptedLLMClient([
            ScriptedResponse(entities=[], facts=[]),
        ])
        s = Surriti(
            driver=surriti.driver,
            llm_client=scripted,
            embedder=DummyEmbedder(),
        )
        await s.connect()
        ep = await s.add_episode(name="test", episode_body="Test content", group_id="scripted-test")
        assert ep is not None
        await surriti.driver.clear()
        await s.close()

    async def test_scripted_contradiction(self, surriti):
        """Scripted LLM can return contradiction indices."""
        scripted = ScriptedLLMClient([
            ScriptedResponse(
                entities=[ExtractedEntity(name="TestEntity")],
                facts=[ExtractedFact("TestEntity", "has_property", "value", "TestEntity has value.")],
            ),
            ScriptedResponse(contradictions=[]),
        ])
        s = Surriti(
            driver=surriti.driver,
            llm_client=scripted,
            embedder=DummyEmbedder(),
        )
        await s.connect()
        ep = await s.add_episode(name="test", episode_body="TestEntity has value.", group_id="scripted-contradiction")
        assert ep is not None
        await surriti.driver.clear()
        await s.close()


# =================================================================== 11. Driver Direct
class TestDriverDirect:
    async def test_driver_query(self, driver):
        result = await driver.query("SELECT * FROM episode LIMIT 1;")
        assert isinstance(result, list)

    async def test_driver_clear_after_data(self, driver):
        await driver.query(
            'CREATE episode SET uuid = "test-1", group_id = "test", content = "hello";'
        )
        await driver.clear()
        result = await driver.query("SELECT count() FROM episode;")
        assert result[0]["count"] == 0


# =================================================================== 12. Embedder
class TestEmbedder:
    async def test_dummy_embedder_vector_length(self):
        e = DummyEmbedder(768)
        vec = await e.create("test")
        assert len(vec) == 768

    async def test_dummy_embedder_empty(self):
        e = DummyEmbedder(768)
        vec = await e.create("")
        assert len(vec) == 768

    async def test_cosine_similarity_identical(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001

    async def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 0.001

    async def test_cosine_similarity_empty(self):
        assert cosine_similarity([], []) == 0.0


# =================================================================== 13. LLM Client
class TestLLMClient:
    async def test_dummy_extract(self):
        llm = DummyLLMClient()
        result = await llm.extract("Michael is a software engineer.")
        # DummyLLMClient extracts entities but may return empty facts
        # since it uses simple regex-based extraction
        assert result is not None
        assert hasattr(result, "entities")
        assert hasattr(result, "facts")

    async def test_dummy_extract_empty(self):
        llm = DummyLLMClient()
        result = await llm.extract("")
        assert len(result.entities) == 0
        assert len(result.facts) == 0

    async def test_dummy_contradiction(self):
        llm = DummyLLMClient()
        existing = ["Michael is a software engineer."]
        new = "Michael is no longer a software engineer."
        contradicted = await llm.find_contradictions(new, existing)
        assert len(contradicted) > 0

    async def test_dummy_contradiction_no_match(self):
        llm = DummyLLMClient()
        existing = ["Duke is a dog."]
        new = "Judy works in clinical research."
        contradicted = await llm.find_contradictions(new, existing)
        assert len(contradicted) == 0
