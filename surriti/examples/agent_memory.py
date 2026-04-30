"""Pattern: use Surriti as the durable memory of an agent loop.

Each turn we (1) record the user's message as an episode, (2) search the
graph for relevant prior facts, and (3) feed them into the next prompt.
This script uses the offline DummyLLMClient so it runs without API keys
or even a SurrealDB instance — but the shape is the same in production.

Run with:  python examples/agent_memory.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from surriti import DummyEmbedder, DummyLLMClient, SearchConfig, Surriti
from surriti.driver import SurrealDriver


@dataclass
class Turn:
    user: str


CONVERSATION = [
    Turn("Hi, my name is Alice and I work at Acme Corp as a staff engineer."),
    Turn("My favourite programming language is Rust."),
    Turn("Actually I switched companies — I now work at Globex."),
    Turn("Where do I work and what do I like to code in?"),
]


async def main() -> None:
    # In production: use Surriti.from_env() with a real SurrealDB driver.
    from surriti.testing import InMemoryDriver

    driver = InMemoryDriver()
    memory = Surriti(driver, llm_client=DummyLLMClient(), embedder=DummyEmbedder(64))

    user_id = "alice"
    for i, turn in enumerate(CONVERSATION):
        print(f"\n--- Turn {i + 1} ---")
        print(f"USER: {turn.user}")

        # 1) record what the user said
        await memory.add_episode(
            name=f"turn-{i}",
            episode_body=turn.user,
            group_id=user_id,
        )

        # 2) retrieve relevant memory
        recalled = await memory.search(turn.user, group_id=user_id)

        # 3) build the next prompt's context
        if recalled.edges:
            print("MEMORY:")
            for e in recalled.edges:
                tag = "(invalidated)" if e.invalid_at else ""
                print(f"  - {e.fact} {tag}")


if __name__ == "__main__":
    asyncio.run(main())
