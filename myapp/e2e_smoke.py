"""E2E smoke: open WS, send 3 turns, verify event stream + memory state."""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import websockets

HTTP = "http://localhost:3000"
WS   = "ws://localhost:3000"
SESSION = "e2e-smoke"
USER    = "alice-smoke"

TURNS = [
    "I'm Alice and I work at Acme Corp.",
    "I love pizza and pasta.",
    "Where do I work?",
]


async def run_turn(ws, client: httpx.AsyncClient, content: str) -> dict:
    await client.post(f"{HTTP}/send", json={
        "session_id": SESSION, "user_id": USER, "content": content,
    })
    transcript: dict = {
        "events": [], "answer": "", "recall": None, "store": None,
        "errors": [],
    }
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        msg = json.loads(raw)
        t = msg.get("type")
        transcript["events"].append(t)
        if t == "memory_recall":
            transcript["recall"] = msg
        elif t == "memory_store":
            transcript["store"] = msg
        elif t == "chunk":
            transcript["answer"] += msg.get("text", "")
        elif t == "step_error":
            transcript["errors"].append(msg)
        elif t == "error":
            transcript["errors"].append(msg)
            break
        elif t == "done":
            break
    return transcript


async def main() -> int:
    failures: list[str] = []

    async with (
        websockets.connect(f"{WS}/ws/{SESSION}") as ws,
        httpx.AsyncClient(timeout=180) as client,
    ):
        # Confirm health
        h = (await client.get(f"{HTTP}/health")).json()
        assert h["memory"] is True, h
        print(f"[health] {h}")

        results = []
        for i, turn in enumerate(TURNS, 1):
            print(f"\n--- Turn {i}: {turn!r} ---")
            t = await run_turn(ws, client, turn)
            print(f"events: {t['events']}")
            print(f"answer: {t['answer'][:200]!r}")
            if t["recall"]:
                print(f"recall: edges={len(t['recall'].get('edges') or [])} "
                      f"nodes={len(t['recall'].get('nodes') or [])}")
            if t["store"]:
                print(f"store : ents+={t['store'].get('entities_added')} "
                      f"edges+={t['store'].get('edges_added')} "
                      f"invalidated={t['store'].get('invalidated')}")
                if t["store"].get("new_facts"):
                    for f in t["store"]["new_facts"]:
                        print(f"        fact: {f.get('source')} -[{f.get('name')}]-> "
                              f"{f.get('target')} :: {f.get('fact')!r}")
            results.append(t)

        # Per-turn assertions
        for i, t in enumerate(results, 1):
            if t["errors"]:
                failures.append(f"turn {i} errors: {t['errors']}")
            for required in ("step_start", "step_done", "chunk", "done"):
                if required not in t["events"]:
                    failures.append(f"turn {i} missing event {required!r}")

        # Final answer should reference Acme.
        final = results[-1]["answer"].lower()
        if "acme" not in final:
            failures.append(f"final answer missing 'acme': {results[-1]['answer']!r}")

        # Knowledge graph state
        mem = (await client.get(f"{HTTP}/memory/{USER}")).json()
        print(f"\n[memory] status={mem.get('status')} "
              f"entities={len(mem.get('nodes') or [])} "
              f"edges={len(mem.get('edges') or [])}")
        for n in (mem.get("nodes") or [])[:20]:
            print(f"  entity: {n.get('name')!r} labels={n.get('labels')}")
        for e in (mem.get("edges") or [])[:20]:
            print(f"  edge  : {e.get('source')!r} -[{e.get('name')}]-> "
                  f"{e.get('target')!r} :: {e.get('fact')!r}")
        if (mem.get("nodes") or []) == []:
            failures.append("no entities stored in knowledge graph")
        if (mem.get("edges") or []) == []:
            failures.append("no edges stored in knowledge graph")
        names = {(n.get("name") or "").lower() for n in mem.get("nodes") or []}
        if "alice" not in names and not any("alice" in n for n in names):
            failures.append(f"'Alice' not extracted as entity; got {names}")
        if "acme" not in " ".join(names) and not any("acme" in n for n in names):
            failures.append(f"'Acme' not extracted as entity; got {names}")

    if failures:
        print("\n❌ FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n✅ E2E smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
