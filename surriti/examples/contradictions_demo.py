"""Demonstrate temporal invalidation: facts get a closing timestamp when
contradicted instead of being silently overwritten.

Uses the ScriptedLLMClient so the demo is fully deterministic.

NOTE: this script wires the in-memory test driver from
``surriti.testing`` for a zero-setup demo. Against a real SurrealDB
instance the pipeline performs a vector + lexical lookup of the prior
edge before invalidating it; the in-memory stub only does a coarse
match, so for a realistic invalidation walk-through prefer
``examples/openai_quickstart.py`` against a live database.

Run with:  python examples/contradictions_demo.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from surriti import (
    DummyEmbedder,
    ExtractedEntity,
    ExtractedFact,
    ScriptedLLMClient,
    ScriptedResponse,
    Surriti,
)


async def main() -> None:
    from surriti.testing import InMemoryDriver

    scripted = ScriptedLLMClient(
        responses=[
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Alice", labels=["Person"]),
                    ExtractedEntity(name="Acme Corp", labels=["Organization"]),
                ],
                facts=[
                    ExtractedFact(
                        subject="Alice",
                        predicate="works_at",
                        object="Acme Corp",
                        fact="Alice works at Acme Corp.",
                        valid_at="2026-01-15T00:00:00Z",
                    )
                ],
            ),
            ScriptedResponse(
                entities=[
                    ExtractedEntity(name="Alice", labels=["Person"]),
                    ExtractedEntity(name="Globex", labels=["Organization"]),
                ],
                facts=[
                    ExtractedFact(
                        subject="Alice",
                        predicate="works_at",
                        object="Globex",
                        fact="Alice works at Globex.",
                        valid_at="2026-03-01T00:00:00Z",
                    )
                ],
                contradictions=[0],
            ),
        ]
    )

    memory = Surriti(
        InMemoryDriver(),
        llm_client=scripted,
        embedder=DummyEmbedder(64),
    )

    await memory.add_episode(
        name="hire",
        episode_body="Alice joined Acme Corp.",
        group_id="demo",
        reference_time=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )
    await memory.add_episode(
        name="switch",
        episode_body="Alice left Acme and now works at Globex.",
        group_id="demo",
        reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    results = await memory.search("Where does Alice work?", group_id="demo")
    print("All recorded facts (most recent first):")
    for edge in results.edges:
        valid_to = edge.invalid_at or "present"
        print(f"  - {edge.fact}    valid {edge.valid_at} -> {valid_to}")


if __name__ == "__main__":
    asyncio.run(main())
