# Surriti

> **Temporally-aware knowledge-graph memory for AI agents — built on SurrealDB.**

[![PyPI version](https://img.shields.io/pypi/v/surriti.svg)](https://pypi.org/project/surriti/)
[![Python](https://img.shields.io/pypi/pyversions/surriti.svg)](https://pypi.org/project/surriti/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-brightgreen.svg)](#)

Surriti gives your LLM application a **persistent memory** that *understands time*.
Drop in conversations, documents, or events and Surriti automatically extracts
entities, relationships, and **temporally-scoped facts** — so when reality
changes ("Alice quit Acme and joined Globex") your agent can tell the past
from the present.

It's a clean-room re-implementation of the
[Graphiti](https://github.com/getzep/graphiti) ideas, sitting natively on
[SurrealDB](https://surrealdb.com) (single store for graph + document +
vector + full-text).

```bash
pip install surriti                 # core (DummyLLM works offline)
pip install "surriti[openai]"       # + OpenAI LLM & embeddings
pip install "surriti[anthropic]"    # + Claude
pip install "surriti[all]"          # everything
```

---

## 30-second quickstart

```python
import asyncio
from surriti import Surriti, OpenAIEmbedder
from surriti.llm_clients import OpenAILLMClient

async def main():
    async with Surriti.from_env(
        llm_client=OpenAILLMClient(model="gpt-4o-mini"),
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ) as memory:
        await memory.add_episode(
            name="standup-2026-01-15",
            episode_body="Alice joined Acme Corp as a staff engineer.",
            group_id="user-42",
        )
        await memory.add_episode(
            name="standup-2026-03-01",
            episode_body="Alice left Acme and now works at Globex.",
            group_id="user-42",
        )

        results = await memory.search("where does Alice work?", group_id="user-42")
        for edge in results.edges:
            valid_to = edge.invalid_at or "present"
            print(f"  - {edge.fact}   (valid {edge.valid_at} → {valid_to})")

asyncio.run(main())
```

Required environment (uses `python-dotenv` if you have it):

```ini
SURRITI_SURREAL_URL=ws://localhost:8000/rpc
SURRITI_SURREAL_NS=surriti
SURRITI_SURREAL_DB=surriti
SURRITI_SURREAL_USER=root
SURRITI_SURREAL_PASS=root
OPENAI_API_KEY=sk-...
```

Spin up SurrealDB locally:

```bash
docker run --rm -p 8000:8000 surrealdb/surrealdb:latest \
  start --user root --pass root memory
```

> **Note:** Surriti 0.5.x requires **SurrealDB 3.x** and the `surrealdb>=2.0,<3` Python SDK.
> Replace `latest` with a specific `v3.x.y` tag to pin your version.

---

## Why temporal?

Most "agent memory" libraries store facts as a flat list. Surriti stores
each fact as an **edge with `valid_at` / `invalid_at` timestamps**, so when
new information contradicts an old one, the old one isn't deleted — it's
*invalidated* with a closing timestamp. Your agent can reason about both
"what is true now" and "what was true on 2026-02-01".

The same data store lets you do hybrid retrieval out of the box:

| Need | Surriti search recipe |
|---|---|
| "What does my user know?" | `EDGE_HYBRID_SEARCH_RRF` (semantic + BM25 + reranking) |
| "What happened last quarter?" | filtered `search_` with `DateFilter` on `valid_at` |
| "Find people who…" | `NODE_HYBRID_SEARCH_NODE_DISTANCE` |
| Cluster related entities | `Surriti.build_communities()` |

---

## Concepts

| Concept | Surreal table | Description |
|---|---|---|
| **EpisodicNode** | `episode`    | A raw input chunk (chat turn, doc, json blob). |
| **EntityNode**   | `entity`     | A canonical thing (person, org, product...). |
| **EpisodicEdge** | `mentions`   | "this episode mentioned this entity". |
| **EntityEdge**   | `relates_to` | A temporally-scoped fact between two entities. |
| **CommunityNode**| `community`  | A cluster discovered by Leiden. |

Everything is just `pydantic` v2 models so you can integrate with
LangGraph, LlamaIndex, raw FastAPI, etc.

---

## SDK surface

```python
from surriti import (
    Surriti, SurrealDriver,
    OpenAIEmbedder, DummyEmbedder,
    LLMClient, DummyLLMClient, ScriptedLLMClient,
    SearchConfig, SearchFilters, DateFilter, PropertyFilter, ComparisonOperator,
    SurritiError, SurritiConfigError, SurritiConnectionError,
    SurritiLLMError, SurritiSchemaError, SurritiNotFoundError,
    setup_logging,
)
from surriti.llm_clients import OpenAILLMClient, AnthropicLLMClient
from surriti.search_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
```

### Common operations

```python
# Bulk ingestion
await memory.add_episode_bulk([RawEpisode(name=..., content=..., source=...) ...])

# Direct triplet (skip LLM)
await memory.add_triplet(source, edge, target)

# Custom search
results = await memory.search_(
    query="...",
    group_ids=["user-42"],
    search_config=EDGE_HYBRID_SEARCH_RRF,
    search_filter=SearchFilters(valid_at=DateFilter(after="2026-01-01")),
)

# Communities (Leiden clustering over the entity graph)
await memory.build_communities()
```

### Bring your own LLM

Implement [`surriti.LLMClient`](surriti/llm.py) and pass it as
`llm_client=`. The two methods are `extract(...)` and
`find_contradictions(...)`. See `OpenAILLMClient` for a reference.

---

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — offline, dummy LLM
- [`examples/openai_quickstart.py`](examples/openai_quickstart.py) — full OpenAI
- [`examples/agent_memory.py`](examples/agent_memory.py) — agent-loop pattern
- [`examples/contradictions_demo.py`](examples/contradictions_demo.py) — temporal invalidation

Run any of them with `python examples/<name>.py` after installing.

---

## Development

```bash
git clone https://github.com/surriti/surriti
cd surriti
pip install -e ".[dev]"

# unit tests (no DB required)
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest tests -p asyncio -o asyncio_mode=auto

# integration tests (need SurrealDB on :8000)
docker compose -f docker-compose.yml up -d
python -m pytest tests/test_integration_surrealdb.py
```

Linting/type-checking:

```bash
ruff check surriti tests
mypy surriti
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
