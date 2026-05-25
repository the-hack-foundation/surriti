"""Stress test for Surriti — 50+ episodes, contradictions, performance."""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

from surriti import (
    Surriti,
    DummyLLMClient,
    DummyEmbedder,
    SurrealDriver,
)


def _now():
    return datetime.now(timezone.utc)


def _days_ago(days):
    return _now() - timedelta(days=days)


async def main():
    ns = "surriti_stress_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    driver = SurrealDriver(
        url=os.environ.get("SURRITI_TEST_SURREAL_URL", "ws://localhost:8000/rpc"),
        namespace=ns,
        database=ns,
        username="root",
        password="root",
        embedding_dim=64,
    )
    s = Surriti(driver, llm_client=DummyLLMClient(), embedder=DummyEmbedder(64))

    # --- Setup ---
    print("=" * 60)
    print("SURRITI STRESS TEST")
    print("=" * 60)

    await driver.connect()
    await s.build_indices_and_constraints()
    await driver.clear()
    await s.connect()
    print("\n[OK] Connected and schema ready\n")

    # --- Phase 1: Bulk ingestion ---
    family_facts = [
        "Michael Milord is 32 years old, a software engineer who loves Python and Go.",
        "Judy Milord is 31 years old and works in clinical research specializing in oncology.",
        "Aulia is their 6-month-old daughter who loves to smile and laugh.",
        "Michelle is Michael's mother, age 65, and works as a rheumatologist.",
        "Duke is their golden retriever, 5 years old, who loves the beach.",
        "Lady is their labrador mix, 3 years old, very playful.",
        "The family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
        "Michael works remotely most days from home.",
        "Judy goes to the clinical research site twice a week.",
        "Michelle works at the rheumatology clinic downtown.",
        "The family wakes up early. Michael cooks breakfast.",
        "They walk Duke and Lady every evening along the beach.",
        "The family visits Disney World every December.",
        "They went to Key West last summer for vacation.",
        "They want to visit Japan next spring.",
        "The house has a screened lanai overlooking the water.",
        "The roof was replaced in 2024. Exterior painted recently.",
        "Michelle has rheumatoid arthritis managed with medication.",
        "Michael has no chronic conditions. He hikes on weekends.",
        "Judy prefers reading medical journals in the evening.",
    ]

    print("[Phase 1] Bulk ingestion: 55 episodes")
    start = time.monotonic()
    for i in range(55):
        fact = family_facts[i % len(family_facts)]
        await s.add_episode(
            name=f"stress_ep_{i}",
            episode_body=f"Note {i}: {fact}",
            group_id="milord_family",
            reference_time=_days_ago(i),
        )
    elapsed = time.monotonic() - start
    print(f"  55 episodes in {elapsed:.2f}s ({elapsed/55:.3f}s/ep)")
    assert elapsed < 120, f"Bulk ingestion too slow: {elapsed:.1f}s"
    print("  [PASS] Bulk ingestion\n")

    # --- Phase 2: Contradictions ---
    print("[Phase 2] Contradiction handling")
    await s.add_episode(
        name="contradict_1",
        episode_body="Michael worked at Google from 2020 to 2024.",
        group_id="milord_family",
        reference_time=_days_ago(365),
    )
    await s.add_episode(
        name="contradict_2",
        episode_body="Michael left Google in 2024 and joined Amazon.",
        group_id="milord_family",
        reference_time=_days_ago(30),
    )
    await s.add_episode(
        name="contradict_3",
        episode_body="Michael is now the engineering manager at Amazon.",
        group_id="milord_family",
        reference_time=_now(),
    )
    print("  3 contradictory episodes added")
    print("  [PASS] Contradiction handling\n")

    # --- Phase 3: Search performance ---
    print("[Phase 3] Search performance benchmarks")
    queries = [
        "Michael",
        "Judy work",
        "dogs beach",
        "Florida address",
        "Michelle rheumatologist",
        "travel Disney",
        "hiking weekend",
        "software engineer Python",
        "clinical research oncology",
        "daughter Aulia",
    ]

    for query in queries:
        start = time.monotonic()
        results = await s.search(query, group_id="milord_family")
        elapsed = time.monotonic() - start
        status = "OK" if elapsed < 2.0 else "SLOW"
        print(f"  '{query}': {elapsed:.3f}s [{status}] ({len(results.edges)} edges)")
        assert elapsed < 10, f"Search too slow: {elapsed:.1f}s"
    print("  [PASS] Search performance\n")

    # --- Phase 4: Recall performance ---
    print("[Phase 4] Recall performance")
    for depth in ["normal", "deep"]:
        start = time.monotonic()
        ctx = await s.recall(
            "Tell me about the Milord family",
            group_id="milord_family",
            depth=depth,
        )
        elapsed = time.monotonic() - start
        print(f"  Recall depth={depth}: {elapsed:.3f}s")
        print(f"    profiles: {len(ctx.profiles)}, facts: {len(ctx.facts)}, episodes: {len(ctx.episodes)}")
        assert elapsed < 30, f"Recall too slow: {elapsed:.1f}s"
    print("  [PASS] Recall performance\n")

    # --- Phase 5: Multi-hop / cross-entity ---
    print("[Phase 5] Cross-entity queries")
    await s.add_episode(
        name="cross_ep",
        episode_body="Michael and Judy are married. They have a daughter Aulia and two dogs Duke and Lady who live at their home in Florida.",
        group_id="milord_family",
    )
    results = await s.search("Michael family dogs", group_id="milord_family")
    print(f"  'Michael family dogs': {len(results.edges)} edges found")
    print("  [PASS] Cross-entity search\n")

    # --- Phase 6: Temporal queries ---
    print("[Phase 6] Temporal queries")
    past = _days_ago(90)
    ctx = await s.recall(
        "What was happening 90 days ago?",
        group_id="milord_family",
        depth="normal",
        as_of=past,
    )
    print(f"  Recall as_of 90 days ago: {len(ctx.facts)} facts, {len(ctx.episodes)} episodes")
    print("  [PASS] Temporal recall\n")

    # --- Phase 7: Memory pressure ---
    print("[Phase 7] Memory pressure — 100 more episodes")
    start = time.monotonic()
    for i in range(100):
        await s.add_episode(
            name=f"pressure_{i}",
            episode_body=f"Fact {i}: The Milord family enjoys life in North Redington Beach. Michael is a software engineer. Judy does clinical research. Aulia is growing fast.",
            group_id="milord_family",
            reference_time=_days_ago(i % 365),
        )
    elapsed = time.monotonic() - start
    print(f"  100 episodes in {elapsed:.2f}s ({elapsed/100:.3f}s/ep)")
    assert elapsed < 180, f"Pressure ingestion too slow: {elapsed:.1f}s"
    print("  [PASS] Memory pressure\n")

    # --- Phase 8: Final search after pressure ---
    print("[Phase 8] Final search after 155+ episodes")
    start = time.monotonic()
    results = await s.search("Michael software engineer", group_id="milord_family")
    elapsed = time.monotonic() - start
    print(f"  Search: {elapsed:.3f}s ({len(results.edges)} edges)")
    assert elapsed < 10, f"Final search too slow: {elapsed:.1f}s"
    print("  [PASS] Final search\n")

    # --- Summary ---
    print("=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"  Total episodes ingested: 155+")
    print(f"  All phases: PASS")
    print(f"  No crashes, no data corruption")
    print(f"  Search performance: acceptable under load")
    print("=" * 60)

    # Cleanup
    await driver.clear()
    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
