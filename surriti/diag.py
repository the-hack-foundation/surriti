"""Diagnostic script to understand why search/recall returns empty results."""
import asyncio
from datetime import datetime, timezone

async def main():
    from surriti.driver import SurrealDriver
    from surriti.embedder import DummyEmbedder
    from surriti.llm import DummyLLMClient
    from surriti import Surriti

    driver = SurrealDriver("ws://localhost:8000")
    await driver.connect()
    await driver.clear()

    # Add an episode
    s = Surriti(driver=driver, llm_client=DummyLLMClient(), embedder=DummyEmbedder())
    await s.connect()
    
    ep = await s.add_episode(
        name="Michael job",
        episode_body="Michael is a software engineer.",
        group_id="test-family",
    )
    print(f"Added episode: {ep}")
    print(f"Episode uuid: {ep.uuid if hasattr(ep, 'uuid') else 'N/A'}")

    # Check what's in the DB
    episodes = await driver.query('SELECT * FROM episode WHERE group_id = "test-family";')
    print(f"\nEpisodes in DB: {episodes}")
    
    edges = await driver.query('SELECT * FROM relates_to WHERE group_id = "test-family";')
    print(f"\nEdges in DB: {edges}")
    
    entities = await driver.query('SELECT * FROM entity WHERE group_id = "test-family";')
    print(f"\nEntities in DB: {entities}")
    
    # Try search
    results = await s.search("Michael", group_id="test-family")
    print(f"\nSearch results: edges={len(results.edges)}, episodes={len(results.episodes)}")
    print(f"  edges: {results.edges}")
    
    # Try recall
    ctx = await s.recall("Michael", depth="fast", group_id="test-family")
    print(f"\nRecall results: facts={len(ctx.facts)}, profiles={len(ctx.profiles)}")
    print(f"  facts: {ctx.facts}")
    print(f"  profiles: {ctx.profiles}")
    
    await s.close()
    await driver.close()

asyncio.run(main())
