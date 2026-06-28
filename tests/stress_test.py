"""Stress test for Surriti — 50+ episodes, contradictions, performance."""
import asyncio
import time
from datetime import datetime, timezone

from surriti import Surriti, DummyLLMClient, DummyEmbedder
from surriti.driver import SurrealDriver


async def main():
    driver = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti",
        database="surriti",
        username="root",
        password="root",
        embedding_dim=768,
    )
    s = Surriti(driver=driver, llm_client=DummyLLMClient(), embedder=DummyEmbedder(embedding_dim=768), cognition=False)
    await s.connect()

    # Clear any prior stress test data
    await s.delete_group("stress-test")

    # ── Phase 1: Add 60 varied episodes ──
    print("=== Phase 1: Adding 60 episodes ===")
    episodes = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    family_names = ["Michael", "Judy", "Aulia", "Michelle", "Duke", "Lady"]
    topics = [
        "Michael is a software engineer at a fintech startup",
        "Judy works in clinical research at a hospital",
        "Aulia is 6 months old and loves tummy time",
        "Michelle is a rheumatologist who enjoys gardening",
        "Duke is a golden retriever who loves swimming",
        "Lady is a tabby cat who sleeps 16 hours a day",
        "The family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708",
        "Michael prefers black coffee in the morning",
        "Judy is learning to play the piano",
        "The family has a garden with tomatoes and basil",
        "Duke was adopted from a shelter in 2020",
        "Lady was a stray found near the beach",
        "Michael's birthday is in March",
        "Judy's birthday is in July",
        "The family vacations in Florida during winter",
        "Michelle retired from full-time practice in 2024",
        "Duke has a mild allergy to chicken",
        "Lady prefers tuna over salmon",
        "Michael codes in Python and Go",
        "Judy publishes research papers on clinical trials",
    ]

    t0 = time.time()
    for i in range(60):
        name = f"stress-episode-{i}"
        topic = topics[i % len(topics)]
        # Vary the content slightly
        extra = f" Additional note {i} about {topic.split()[0]}."
        body = topic + extra
        # Stagger timestamps
        ts = base_time + __import__("datetime").timedelta(days=i)
        ep = await s.add_episode(
            name=name,
            episode_body=body,
            group_id="stress-test",
            reference_time=ts,
        )
        episodes.append(ep)

    print(f"  Added {len(episodes)} episodes in {(time.time() - t0):.2f}s")

    # ── Phase 2: Contradictory episodes ──
    print("\n=== Phase 2: Adding contradictory episodes ===")
    contradictions = [
        ("contradiction-1", "Duke is a labrador retriever.", "Duke is a golden retriever."),
        ("contradiction-2", "Michael lives in New York.", "Michael lives in Florida."),
        ("contradiction-3", "Judy retired in 2023.", "Judy works in clinical research."),
        ("contradiction-4", "Lady is a dog.", "Lady is a tabby cat."),
        ("contradiction-5", "Michelle loves hiking.", "Michelle prefers gardening."),
    ]

    for uuid, body, _ in contradictions:
        await s.add_episode(
            name=uuid,
            episode_body=body,
            group_id="stress-test",
            reference_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
    print(f"  Added {len(contradictions)} contradictory episodes")

    # ── Phase 3: Search performance ──
    print("\n=== Phase 3: Search performance ===")
    search_terms = ["Michael", "Judy", "Duke", "Florida", "coffee", "piano", "shelter", "nonexistent"]

    for term in search_terms:
        t0 = time.time()
        results = await s.search(term, group_id="stress-test")
        elapsed = time.time() - t0
        print(f"  search('{term}'): {len(results.edges)} edges in {elapsed:.3f}s")

    # ── Phase 4: Recall with different depths ──
    print("\n=== Phase 4: Recall with different depths ===")
    for depth in ["fast", "normal", "deep"]:
        t0 = time.time()
        results = await s.recall("Michael", depth=depth, group_id="stress-test")
        elapsed = time.time() - t0
        print(f"  recall('Michael', depth={depth}): {len(results.facts)} facts in {elapsed:.3f}s")

    # ── Phase 5: Temporal queries ──
    print("\n=== Phase 5: Temporal queries ===")
    t0 = time.time()
    results = await s.search("Duke", group_id="stress-test", only_valid=True)
    elapsed = time.time() - t0
    print(f"  search('Duke', only_valid=True): {len(results.edges)} edges in {elapsed:.3f}s")

    t0 = time.time()
    results = await s.search("Duke", group_id="stress-test")
    elapsed = time.time() - t0
    print(f"  search('Duke', all): {len(results.edges)} edges in {elapsed:.3f}s")

    # ── Phase 6: Bulk operations ──
    print("\n=== Phase 6: Bulk operations ===")
    bulk_episodes = []
    for i in range(20):
        bulk_episodes.append({
            "name": f"bulk-{i}",
            "episode_body": f"Bulk episode {i} about family activities.",
            "group_id": "stress-test",
        })
    t0 = time.time()
    bulk_result = await s.add_episode_bulk(bulk_episodes, group_id="stress-test")
    elapsed = time.time() - t0
    print(f"  Added {len(bulk_result.episodes)} bulk episodes in {elapsed:.2f}s")

    # ── Phase 7: Edge cases ──
    print("\n=== Phase 7: Edge cases ===")
    edge_cases = [
        ("edge-empty", ""),
        ("edge-unicode", "日本語のテスト。Привет мир. 🎉"),
        ("edge-long", "A" * 10000),
        ("edge-special", "Test with <special> & \"characters\"'s"),
    ]
    for name, body in edge_cases:
        try:
            ep = await s.add_episode(name=name, episode_body=body, group_id="stress-test")
            print(f"  {name}: OK (ep={ep is not None})")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── Summary ──
    print("\n=== Stress Test Complete ===")
    print(f"  Total episodes added: {len(episodes) + len(contradictions) + len(bulk_result.episodes) + len(edge_cases)}")
    print(f"  All phases completed successfully")

    await s.close()


if __name__ == "__main__":
    asyncio.run(main())
