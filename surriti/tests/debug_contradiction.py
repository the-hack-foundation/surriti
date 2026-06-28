"""Debug contradiction detection - trace the full flow."""
import asyncio, sys, logging
sys.path.insert(0, "/home/squire/projects/surriti/surriti")

logging.basicConfig(level=logging.DEBUG)

from datetime import datetime, timezone
from surriti import Surriti, DummyLLMClient, DummyEmbedder, EpisodeType, SurrealDriver

# Patch resolve_contradictions to trace
from surriti import temporal
orig_resolve = temporal.resolve_contradictions

async def traced_resolve(driver, *, llm, new_fact, new_fact_embedding, new_valid_at, group_id,
                         similarity_limit=10, new_fact_struct=None, new_edge_uuid=None,
                         new_subject_uuid=None, new_object_uuid=None):
    print(f"\n=== resolve_contradictions called ===")
    print(f"  new_fact: {new_fact}")
    print(f"  new_fact_embedding: {new_fact_embedding[:3] if new_fact_embedding else None}...")
    print(f"  group_id: {group_id}")
    print(f"  new_edge_uuid: {new_edge_uuid}")
    print(f"  new_subject_uuid: {new_subject_uuid}")
    print(f"  new_object_uuid: {new_object_uuid}")
    print(f"  new_fact_struct: {new_fact_struct}")
    
    candidates = await temporal.find_similar_edges(
        driver, fact=new_fact, fact_embedding=new_fact_embedding,
        group_id=group_id, limit=similarity_limit,
        co_object_uuid=new_object_uuid,
    )
    print(f"  find_similar_edges returned {len(candidates)} candidates:")
    for c in candidates:
        print(f"    - uuid={c.uuid} fact={c.fact} status={c.status}")
        print(f"      source={c.source_node_uuid} target={c.target_node_uuid}")
        print(f"      is_belief={getattr(c, 'is_belief', 'N/A')} memory_class={getattr(c, 'memory_class', 'N/A')}")
    
    if not candidates:
        print("  -> No candidates, returning []")
        return []
    
    # Check belief filter
    new_is_belief = bool(getattr(new_fact_struct, "is_belief", False))
    print(f"  new_is_belief: {new_is_belief}")
    
    filtered = [
        c for c in candidates
        if bool(getattr(c, "is_belief", False)) == new_is_belief
        or (getattr(c, "memory_class", None) == "belief") == new_is_belief
    ]
    print(f"  After belief filter: {len(filtered)} candidates")
    
    if not filtered:
        print("  -> Belief filter removed all, returning []")
        return []
    
    fact_strings = [c.fact for c in filtered]
    print(f"  Facts passed to LLM: {fact_strings}")
    
    contradicted_idx = await llm.find_contradictions(
        new_fact, fact_strings,
        candidates=None,  # Dummy ignores structured
        new_fact_struct=new_fact_struct,
    )
    print(f"  LLM returned contradicted indices: {contradicted_idx}")
    
    if not contradicted_idx:
        print("  -> LLM found no contradictions, returning []")
        return []
    
    invalidated = [filtered[i] for i in contradicted_idx if 0 <= i < len(filtered)]
    print(f"  Invalidated: {len(invalidated)} edges")
    return invalidated

temporal.resolve_contradictions = traced_resolve

async def debug():
    driver = SurrealDriver(
        url="ws://localhost:8000/rpc",
        namespace="surriti_debug2",
        database="debug_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        username="root",
        password="root",
        embedding_dim=64,
    )
    await driver.connect()
    
    memory = Surriti(
        driver,
        llm_client=DummyLLMClient(),
        embedder=DummyEmbedder(embedding_dim=64),
        cognition=False,
    )
    await memory.connect()
    
    print("=== Adding episode 1 ===")
    res1 = await memory.add_episode(
        name="e1",
        episode_body="Alice works at Acme Corp.",
        source=EpisodeType.text,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        group_id="g_temp",
    )
    print(f"Episode 1 edges: {[(e.name, e.fact) for e in res1.edges]}")
    print(f"Episode 1 nodes: {[(n.name, n.uuid) for n in res1.nodes]}")
    
    print("\n=== Adding episode 2 ===")
    res2 = await memory.add_episode(
        name="e2",
        episode_body="Alice no longer works at Acme Corp; Alice moved to Globex.",
        source=EpisodeType.text,
        reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        group_id="g_temp",
    )
    print(f"\nEpisode 2 edges: {[(e.name, e.fact) for e in res2.edges]}")
    print(f"Episode 2 invalidated: {[(e.name, e.fact) for e in res2.invalidated_edges]}")
    
    # Check DB
    rows = await memory.driver.query(
        'SELECT * FROM relates_to WHERE group_id = "g_temp";'
    )
    print(f"\nAll edges in DB:")
    for r in rows:
        if isinstance(r, dict):
            print(f"  uuid={r.get('uuid','?')} fact={r.get('fact','?')} status={r.get('status','?')} invalid_at={r.get('invalid_at','?')}")

asyncio.run(debug())
