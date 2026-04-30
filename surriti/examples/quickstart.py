"""Quickstart: ingest a few episodes and run a hybrid search.

Prerequisites
-------------
1. Run SurrealDB locally:

    surreal start --user root --pass root memory

2. Install Surriti:

    pip install -e .

3. Run::

    python -m examples.quickstart
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from surriti import Surriti, SurrealDriver
from surriti.nodes import EpisodeType


EPISODES = [
    (
        "intro",
        "Alice works at Acme Corp in San Francisco. Bob is her manager.",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
    (
        "promotion",
        "Bob was promoted to VP of Engineering at Acme Corp.",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    ),
    (
        "move",
        "Alice no longer works at Acme Corp; she moved to Globex in Seattle.",
        datetime(2026, 3, 15, tzinfo=timezone.utc),
    ),
]


async def main() -> None:
    driver = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti_demo",
        database="quickstart",
        username="root",
        password="root",
        embedding_dim=1024,
    )
    async with driver:
        surriti = Surriti(driver)
        await surriti.build_indices_and_constraints()
        await driver.clear()

        for name, body, ref in EPISODES:
            result = await surriti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.text,
                reference_time=ref,
                group_id="demo",
            )
            print(f"[{name}] entities={[n.name for n in result.nodes]} "
                  f"edges={len(result.edges)} invalidated={len(result.invalidated_edges)}")

        print("\n--- Search: 'Where does Alice work?' ---")
        results = await surriti.search("Where does Alice work?", group_id="demo")
        for edge in results.edges:
            print(f"- {edge.fact}  (valid_at={edge.valid_at}, invalid_at={edge.invalid_at})")


if __name__ == "__main__":
    asyncio.run(main())
