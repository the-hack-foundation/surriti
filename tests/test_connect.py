"""Test Surriti connection and basic operations."""
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "surriti"))

from surriti import Surriti, DummyLLMClient, DummyEmbedder
from surriti.driver import SurrealDriver
from surriti.nodes import EpisodeType


async def main():
    print("=== Step 1: Connect to SurrealDB ===")
    driver = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti",
        database="surriti",
        username="root",
        password="root",
        embedding_dim=768,
    )
    await driver.connect()
    print(f"Connected. Namespace: {driver.namespace}, DB: {driver.database}")

    print("\n=== Step 2: Initialize schema ===")
    await driver.init_schema()
    print("Schema initialized.")

    print("\n=== Step 3: Create Surriti instance ===")
    memory = Surriti(
        driver=driver,
        llm_client=DummyLLMClient(),
        embedder=DummyEmbedder(embedding_dim=768),
        cognition=False,  # Disable cognitive layer for now
    )
    print("Surriti instance created.")

    print("\n=== Step 4: Seed initial family memory ===")
    # Michael's info
    result = await memory.add_episode(
        name="michael_profile",
        episode_body="Michael Milord is 32 years old, a software engineer, and lives in North Redington Beach, FL with his wife Judy, daughter Aulia, and dogs Duke and Lady.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-michael-001",
    )
    print(f"Added Michael: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Judy's info
    result = await memory.add_episode(
        name="judy_profile",
        episode_body="Judy Milord (nee Spuza) is 31 years old, a clinical research coordinator, and is married to Michael Milord.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-judy-001",
    )
    print(f"Added Judy: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Aulia's info
    result = await memory.add_episode(
        name="aulia_profile",
        episode_body="Aulia Milord is 6 months old, daughter of Michael and Judy Milord.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-aulia-001",
    )
    print(f"Added Aulia: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Mother info
    result = await memory.add_episode(
        name="michelle_profile",
        episode_body="Michelle Spuza-Milord is 65 years old, a rheumatologist, and is Judy's mother.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-michelle-001",
    )
    print(f"Added Michelle: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Dogs
    result = await memory.add_episode(
        name="pets_profile",
        episode_body="The Milord family has two dogs named Duke and Lady.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-pets-001",
    )
    print(f"Added pets: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Address
    result = await memory.add_episode(
        name="address_profile",
        episode_body="The Milord family lives at 320 Bath Club Blvd S, North Redington Beach, FL 33708.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-address-001",
    )
    print(f"Added address: {len(result.nodes)} nodes, {len(result.edges)} edges")

    # Environment info
    result = await memory.add_episode(
        name="environment_profile",
        episode_body="Squire operates in Ubuntu 24.04 VM on ASUS GX10 with sudo access. Timezone is EST/EDT.",
        source=EpisodeType.message,
        source_description="initial_seed",
        reference_time=datetime.now(timezone.utc),
        group_id="milord-family",
        uuid="seed-env-001",
    )
    print(f"Added environment: {len(result.nodes)} nodes, {len(result.edges)} edges")

    print("\n=== Step 5: Test search ===")
    results = await memory.search("who is michael milord", group_id="milord-family")
    print(f"Search 'who is michael milord': {len(results.edges)} edges, {len(results.episodes)} episodes")
    for edge in results.edges[:5]:
        print(f"  - {edge.subject} -[{edge.predicate}]-> {edge.object}")

    results = await memory.search("where does the family live", group_id="milord-family")
    print(f"\nSearch 'where does the family live': {len(results.edges)} edges, {len(results.episodes)} episodes")
    for edge in results.edges[:5]:
        print(f"  - {edge.subject} -[{edge.predicate}]-> {edge.object}")

    results = await memory.search("tell me about the family", group_id="milord-family")
    print(f"\nSearch 'tell me about the family': {len(results.edges)} edges, {len(results.episodes)} episodes")
    for edge in results.edges[:5]:
        print(f"  - {edge.subject} -[{edge.predicate}]-> {edge.object}")

    print("\n=== Step 6: Test recall (memory context) ===")
    ctx = await memory.recall("who is judy", group_id="milord-family")
    print(f"Recall 'who is judy': {len(ctx.profiles)} profiles, {len(ctx.facts)} facts")
    for profile in ctx.profiles:
        print(f"  Profile: {profile.name}")
    for fact in ctx.facts[:5]:
        print(f"  Fact: {fact.subject} -[{fact.predicate}]-> {fact.object}")

    print("\n=== Step 7: Query raw data ===")
    # Count nodes
    node_count = await driver.query('SELECT COUNT(*) FROM entity')
    print(f"Entity nodes: {node_count}")

    edge_count = await driver.query('SELECT COUNT(*) FROM entity_edge')
    print(f"Entity edges: {edge_count}")

    episode_count = await driver.query('SELECT COUNT(*) FROM episodic')
    print(f"Episodic nodes: {episode_count}")

    print("\n=== DONE ===")
    await driver.close()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
