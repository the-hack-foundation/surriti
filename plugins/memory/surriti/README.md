# Surriti memory provider for Hermes Agent

Persistent, temporal knowledge-graph memory for Hermes, backed by
[Surriti](../../../surriti/) running on SurrealDB. Facts are extracted by an
LLM on the server side, deduplicated, contradicted-on-update, and recalled via
hybrid vector + BM25 search.

## Architecture

```
┌─────────┐  prefetch ──▶ POST /recall  ┐
│ Hermes  │                              ├─▶ myapp (FastAPI) ─▶ Surriti ─▶ SurrealDB
│  Agent  │  sync_turn ─▶ POST /store   ┘                              │
└─────────┘                                                            └─▶ vLLM + Nomic
```

The plugin makes **no LLM calls of its own** — extraction happens server-side
in `myapp`, which already has the right vLLM/embedding wiring.

## Install

This plugin lives in the Surriti repo so it stays in sync with the service. To
install it into a Hermes profile, symlink it (recommended — gets updates) or
copy it:

```bash
# Symlink (preferred)
mkdir -p ~/.hermes/plugins/memory
ln -s /home/squire/projects/surriti/plugins/memory/surriti \
      ~/.hermes/plugins/memory/surriti

# Or copy
cp -r /home/squire/projects/surriti/plugins/memory/surriti \
      ~/.hermes/plugins/memory/
```

Install the runtime dep (only `httpx` is required):

```bash
pip install httpx
```

## Configure

Either via `hermes memory setup` (which calls `get_config_schema()` /
`save_config()`), or by hand at `$HERMES_HOME/surriti.json`:

```json
{
  "url": "http://localhost:3000",
  "user_id": "squire"
}
```

Environment variables override the file:

| Var               | Default                  |
| ----------------- | ------------------------ |
| `SURRITI_URL`     | `http://localhost:3000`  |
| `SURRITI_USER_ID` | `default`                |
| `HERMES_HOME`     | `~/.hermes`              |

Then activate it in your Hermes config:

```yaml
memory:
  provider: surriti
```

## Service requirements

The plugin assumes `myapp` is running and reachable at `SURRITI_URL`. From the
Surriti repo:

```bash
docker compose up -d surrealdb myapp
curl http://localhost:3000/health
```

`myapp` exposes the two endpoints the plugin uses:

- `POST /recall {query, user_id, limit}` → `{facts, entities}`
- `POST /store  {content, user_id, ...}` → `{episode_uuid, entities_added, ...}`

## Hooks implemented

| Hook                  | Behavior                                                                       |
| --------------------- | ------------------------------------------------------------------------------ |
| `prefetch(query)`     | Returns from cache if warm; otherwise sync `POST /recall` and cache result.    |
| `queue_prefetch(q)`   | Fires `/recall` in a daemon thread immediately; result lands in cache.         |
| `sync_turn(u, a)`     | Stores via daemon thread (non-blocking). Invalidates recall cache on success.  |
| `system_prompt_block` | Tells the model that `MEMORY` is authoritative.                                |
| `on_session_end`      | Joins any in-flight store thread so writes complete before exit.               |
| `shutdown`            | Joins pending store thread and closes the persistent HTTP client.              |

## Performance

The plugin keeps overhead negligible on Hermes' hot path:

- **Persistent `httpx.Client`** with HTTP keep-alive — reused TCP connection
  saves ~25 ms per call vs. a fresh client.
- **`queue_prefetch`** — Hermes fires this as soon as a query is known. By
  the time `prefetch` is called for the same query (typically after the user
  finishes typing or the agent finishes thinking), the recall is already done
  and the result returns in ~0 ms.
- **30 s in-memory result cache** (LRU, max 32 entries) — repeated identical
  queries hit the cache. Cleared automatically when `sync_turn` writes new
  facts so stale results never leak.

Measured against the local `myapp`:

| Path                                 | Latency |
| ------------------------------------ | ------- |
| Cold recall (first call)             | ~110 ms |
| Cache hit                            |   ~0 ms |
| `queue_prefetch` → `prefetch` (warm) |   ~0 ms |
| Subsequent cold recall (keep-alive)  |  ~80 ms |

## CLI

After installation, the plugin's CLI is auto-discovered:

```bash
hermes surriti status         # ping /health
hermes surriti config         # show active config
hermes surriti recall "..."   # preview what a query would recall
hermes surriti dump           # dump all entities and edges for the active user
hermes surriti clear --yes    # wipe the active user's memory
```

## Data residency

All data stays on the box running `myapp` and `surrealdb`. Nothing leaves the
network unless the user has pointed `VLLM_BASE_URL` / `EMBED_BASE_URL` in
`myapp` at a remote model.

## Threading notes

`sync_turn` must be non-blocking (Hermes contract). The provider spawns a
daemon thread per turn. Before spawning, it `join`s the previous thread with a
short timeout, so at most one store is in flight at a time. `on_session_end`
and `shutdown` both `join` the final thread to make sure the last turn lands
before process exit.
