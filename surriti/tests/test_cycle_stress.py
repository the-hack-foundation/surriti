"""Stress test for Surriti — Cycle NNN.

Pushes the system with 50+ episodes, contradictions, multi-hop queries,
and performance measurements.
"""

import asyncio
import time
import random
from datetime import datetime, timezone, timedelta

from surriti import (
    Surriti,
    DummyLLMClient,
    DummyEmbedder,
    SearchConfig,
)


FAMILY_EPISODES = [
    # Michael
    {"body": "Michael is 32 years old and works as a software engineer.", "tags": ["michael", "age", "profession"]},
    {"body": "Michael lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708.", "tags": ["michael", "address"]},
    {"body": "Michael enjoys cooking Italian food on weekends.", "tags": ["michael", "hobby"]},
    {"body": "Michael plays guitar in his free time.", "tags": ["michael", "hobby"]},
    {"body": "Michael graduated from university with a degree in Computer Science.", "tags": ["michael", "education"]},
    {"body": "Michael drives a black Tesla Model 3.", "tags": ["michael", "car"]},
    {"body": "Michael prefers dark roast coffee over tea.", "tags": ["michael", "preference"]},
    {"body": "Michael works remotely from home most days.", "tags": ["michael", "work_style"]},
    {"body": "Michael has been married to Judy since 2018.", "tags": ["michael", "marriage"]},
    {"body": "Michael's birthday is in March.", "tags": ["michael", "birthday"]},
    # Judy
    {"body": "Judy is 31 years old and works in clinical research.", "tags": ["judy", "age", "profession"]},
    {"body": "Judy works at a hospital in St. Petersburg, Florida.", "tags": ["judy", "workplace"]},
    {"body": "Judy loves gardening and grows vegetables in their backyard.", "tags": ["judy", "hobby"]},
    {"body": "Judy is studying for a certification in clinical data management.", "tags": ["judy", "education"]},
    {"body": "Judy prefers herbal tea over coffee.", "tags": ["judy", "preference"]},
    {"body": "Judy is learning to play the piano.", "tags": ["judy", "hobby"]},
    {"body": "Judy's favorite book is 'The Immortal Life of Henrietta Lacks'.", "tags": ["judy", "preference"]},
    {"body": "Judy drives a silver Honda CR-V.", "tags": ["judy", "car"]},
    {"body": "Judy was born in October.", "tags": ["judy", "birthday"]},
    {"body": "Judy and Michael met at a friend's barbecue in 2016.", "tags": ["judy", "relationship"]},
    # Aulia
    {"body": "Aulia is 6 months old, the daughter of Michael and Judy.", "tags": ["aulia", "age", "family"]},
    {"body": "Aulia loves her stuffed bunny toy.", "tags": ["aulia", "toy"]},
    {"body": "Aulia sleeps well during the day but is active at night.", "tags": ["aulia", "sleep"]},
    {"body": "Aulia's favorite food is breast milk.", "tags": ["aulia", "food"]},
    {"body": "Aulia smiles when she hears her father sing.", "tags": ["aulia", "behavior"]},
    # Michelle
    {"body": "Michelle is 65 years old and works as a rheumatologist.", "tags": ["michelle", "age", "profession"]},
    {"body": "Michelle is Michael's mother.", "tags": ["michelle", "family"]},
    {"body": "Michelle lives in Tampa, Florida.", "tags": ["michelle", "location"]},
    {"body": "Michelle loves gardening and has a large rose garden.", "tags": ["michelle", "hobby"]},
    {"body": "Michelle visits the family regularly on weekends.", "tags": ["michelle", "visits"]},
    {"body": "Michelle makes the best lasagna in the family.", "tags": ["michelle", "cooking"]},
    {"body": "Michelle was born in June.", "tags": ["michelle", "birthday"]},
    {"body": "Michelle drives a white Toyota Camry.", "tags": ["michelle", "car"]},
    # Dogs
    {"body": "Duke is a golden retriever and the family's male dog.", "tags": ["duke", "dog", "breed"]},
    {"body": "Lady is a female dog, also a golden retriever.", "tags": ["lady", "dog", "breed"]},
    {"body": "Duke and Lady both love going to the beach.", "tags": ["duke", "lady", "beach"]},
    {"body": "Duke weighs about 70 pounds.", "tags": ["duke", "weight"]},
    {"body": "Lady is more timid than Duke.", "tags": ["lady", "personality"]},
    {"body": "The dogs eat Royal Canin dog food twice a day.", "tags": ["duke", "lady", "food"]},
    {"body": "Duke and Lady were adopted from a local shelter.", "tags": ["duke", "lady", "adoption"]},
    # Location
    {"body": "The family home is at 320 Bath Club Blvd S, North Redington Beach, FL 33708.", "tags": ["address", "location"]},
    {"body": "North Redington Beach is on the Gulf Coast of Florida.", "tags": ["location", "geography"]},
    {"body": "The nearest grocery store is Publix on Ulmerton Road.", "tags": ["location", "shopping"]},
    {"body": "The family's nearest hospital is St. Petersburg General.", "tags": ["location", "healthcare"]},
    # Events
    {"body": "The family went to Disney World for vacation in January 2026.", "tags": ["event", "travel"]},
    {"body": "Michael and Judy celebrated their anniversary last month.", "tags": ["event", "anniversary"]},
    {"body": "Aulia's first tooth came in at 5 months old.", "tags": ["event", "milestone"]},
    {"body": "The family adopted Duke when they moved to North Redington Beach.", "tags": ["event", "adoption"]},
    # Contradictions (deliberate)
    {"body": "Michael used to work at Acme Corp before joining TechStart Inc.", "tags": ["michael", "work_history", "contradiction"]},
    {"body": "Judy used to work at a pharmaceutical company before the hospital.", "tags": ["judy", "work_history", "contradiction"]},
    {"body": "The family previously lived in Tampa before moving to North Redington Beach.", "tags": ["location", "history", "contradiction"]},
    # Multi-hop facts
    {"body": "Judy's favorite restaurant is the same one Michelle recommends.", "tags": ["judy", "michelle", "restaurant"]},
    {"body": "Duke was named after a character in a movie Michael loves.", "tags": ["duke", "michael", "naming"]},
    {"body": "Lady was named because she was the second dog, like Lady and the Tramp.", "tags": ["lady", "naming"]},
]


async def run_stress_test():
    print("=" * 60)
    print("SURRITI STRESS TEST — Cycle NNN")
    print("=" * 60)

    from surriti.driver import SurrealDriver
    import os

    kwargs = {
        "url": os.environ.get("SURRITI_SURREAL_URL", "ws://localhost:8000/rpc"),
        "namespace": os.environ.get("SURRITI_SURREAL_NS", "surriti"),
        "database": os.environ.get("SURRITI_SURREAL_DB", "surriti"),
        "username": os.environ.get("SURRITI_SURREAL_USER", "root"),
        "password": os.environ.get("SURRITI_SURREAL_PASS", "root"),
        "embedding_dim": int(os.environ.get("SURRITI_EMBEDDING_DIM", "768")),
    }

    driver = SurrealDriver(**kwargs)
    llm = DummyLLMClient()
    embedder = DummyEmbedder(embedding_dim=kwargs["embedding_dim"])

    async with Surriti(driver, llm_client=llm, embedder=embedder) as memory:
        # --- Phase 1: Bulk ingest 50+ episodes ---
        print("\n--- Phase 1: Bulk Ingest 50+ Episodes ---")
        episodes = []
        for i, ep in enumerate(FAMILY_EPISODES):
            episodes.append({
                "name": f"stress_ep_{i:03d}",
                "episode_body": ep["body"],
                "group_id": "milord-family",
            })

        start = time.time()
        results = await memory.add_episode_bulk(episodes)
        ingest_time = time.time() - start
        print(f"  ✓ Ingested {len(results.episodes)} episodes in {ingest_time:.2f}s")
        print(f"  ✓ Extracted {len(results.nodes)} entities")
        print(f"  ✓ Created {len(results.edges)} edges")
        print(f"  ✓ Throughput: {len(results.episodes)/ingest_time:.1f} eps/sec")

        # --- Phase 2: Contradiction stress ---
        print("\n--- Phase 2: Contradiction Stress ---")
        contradiction_eps = [
            {"name": "contradiction_1", "episode_body": "Michael now works at Google as a senior software engineer.", "group_id": "milord-family"},
            {"name": "contradiction_2", "episode_body": "Judy quit the hospital and now works remotely for a biotech startup.", "group_id": "milord-family"},
            {"name": "contradiction_3", "episode_body": "The family sold their Tesla and bought a Ford F-150.", "group_id": "milord-family"},
            {"name": "contradiction_4", "episode_body": "Duke was rescued from a shelter in Miami, not locally.", "group_id": "milord-family"},
            {"name": "contradiction_5", "episode_body": "Michelle moved to Orlando, not Tampa.", "group_id": "milord-family"},
        ]
        start = time.time()
        cr = await memory.add_episode_bulk(contradiction_eps)
        ct = time.time() - start
        print(f"  ✓ Added {len(cr.episodes)} contradiction episodes in {ct:.2f}s")

        # Verify contradictions are tracked
        search_results = await memory.search("Michael employer")
        edges = search_results.edges if hasattr(search_results, 'edges') else search_results
        print(f"  ✓ Found {len(edges)} facts about Michael's employer (expect multiple, some invalidated)")

        # --- Phase 3: Multi-hop reasoning queries ---
        print("\n--- Phase 3: Multi-hop Reasoning Queries ---")
        queries = [
            ("Michael's wife", "Should find Judy"),
            ("Judy's mother-in-law", "Should find Michelle"),
            ("Family dogs breed", "Should find golden retriever"),
            ("Where does the family live", "Should find address"),
            ("Michelle's profession", "Should find rheumatologist"),
            ("Aulia's parents", "Should find Michael and Judy"),
            ("Family car", "Should find Tesla or Ford"),
            ("Duke's owner", "Should find Michael or family"),
        ]
        for query, expected in queries:
            start = time.time()
            results = await memory.search(query, config=SearchConfig(limit=5))
            edges = results.edges if hasattr(results, 'edges') else results
            elapsed = time.time() - start
            print(f"  ✓ '{query}' → {len(edges)} results in {elapsed*1000:.0f}ms (expect: {expected})")

        # --- Phase 4: Performance with growing data ---
        print("\n--- Phase 4: Performance Scaling ---")
        perf_queries = [
            "family",
            "Michael",
            "Judy",
            "Aulia",
            "Michelle",
            "dog",
            "beach",
            "Florida",
            "work",
            "hobby",
        ]
        latencies = []
        for q in perf_queries:
            start = time.time()
            for _ in range(10):
                r = await memory.search(q, config=SearchConfig(limit=3))
            elapsed = (time.time() - start) / 10
            latencies.append(elapsed)
            edges = r.edges if hasattr(r, 'edges') else r
            print(f"  ✓ '{q}': avg {elapsed*1000:.0f}ms per query ({len(edges)} results)")

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        print(f"\n  Summary: avg={avg_latency*1000:.0f}ms, min={min_latency*1000:.0f}ms, max={max_latency*1000:.0f}ms")

        # --- Phase 5: Edge cases ---
        print("\n--- Phase 5: Edge Cases ---")
        edge_cases = [
            ("", "empty string"),
            ("!!!", "special chars only"),
            ("🎉🎊🎈", "unicode emojis"),
            ("Michael and Judy and Aulia and Michelle and Duke and Lady", "very long query"),
            ("xyznonexistent12345", "nonexistent entity"),
        ]
        for query, desc in edge_cases:
            try:
                results = await memory.search(query, config=SearchConfig(limit=3))
                edges = results.edges if hasattr(results, 'edges') else results
                print(f"  ✓ '{desc}': {len(edges)} results (no crash)")
            except Exception as e:
                print(f"  ✗ '{desc}': {type(e).__name__}: {str(e)[:80]}")

        # --- Phase 6: Schema table counts ---
        print("\n--- Phase 6: Final Schema State ---")
        from surriti.schema import ALL_TABLES
        for table in ALL_TABLES:
            result = await driver.query(f"SELECT count() FROM {table}")
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                count = result[0].get("count", result[0].get("EXPR", 0))
            else:
                count = 0
            print(f"  {table}: {count}")

    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_stress_test())
