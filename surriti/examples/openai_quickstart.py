"""Surriti + OpenAI quickstart.

Prerequisites
-------------
1. SurrealDB running at ws://localhost:8000/rpc:

    docker run --rm -p 8000:8000 surrealdb/surrealdb:v2.1.4 \\
        start --user root --pass root memory

2. Install the OpenAI extra and set OPENAI_API_KEY:

    pip install "surriti[openai]"
    export OPENAI_API_KEY=sk-...

3. Surreal env vars (defaults shown):

    export SURRITI_SURREAL_URL=ws://localhost:8000/rpc
    export SURRITI_SURREAL_USER=root
    export SURRITI_SURREAL_PASS=root

Run with:  python examples/openai_quickstart.py
"""

from __future__ import annotations

import asyncio

from surriti import OpenAIEmbedder, Surriti, setup_logging
from surriti.llm_clients import OpenAILLMClient


async def main() -> None:
    setup_logging("INFO")

    async with Surriti.from_env(
        llm_client=OpenAILLMClient(model="gpt-4o-mini"),
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ) as memory:
        await memory.add_episode(
            name="onboarding",
            episode_body=(
                "Alice joined Acme Corp as a staff engineer in January 2026. "
                "She lives in Berlin and her manager is Bob."
            ),
            group_id="user-42",
        )

        results = await memory.search(
            "Where does Alice work and who is her manager?",
            group_id="user-42",
        )

        print("\n=== Edges ===")
        for edge in results.edges:
            print(f"  {edge.fact}")

        print("\n=== Entities ===")
        for node in results.nodes:
            print(f"  {node.name} ({', '.join(node.labels)}) - {node.summary[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
