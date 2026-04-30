# Changelog

All notable changes to **surriti** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — SDK polish

Surriti is now packaged as a real, pip-installable SDK.

### Added
- `Surriti.from_env()` and `SurrealDriver.from_env()` factories that read
  `SURRITI_SURREAL_URL`, `SURRITI_SURREAL_NS`, `SURRITI_SURREAL_DB`,
  `SURRITI_SURREAL_USER`, `SURRITI_SURREAL_PASS`, `SURRITI_EMBEDDING_DIM`.
- `async with Surriti(...) as memory:` context-manager — connects the
  driver, applies the schema, and cleans up on exit.
- `surriti.llm_clients.OpenAILLMClient` (uses JSON mode).
- `surriti.llm_clients.AnthropicLLMClient` (uses Messages API).
- Structured exception hierarchy: `SurritiError` → `SurritiConfigError`,
  `SurritiConnectionError`, `SurritiSchemaError`, `SurritiLLMError`,
  `SurritiNotFoundError`.
- `surriti.setup_logging(level="INFO")` helper for opt-in console logging.
- `py.typed` marker so type-checkers consume Surriti's annotations.
- New tests: `tests/test_sdk_surface.py` (14 tests).
- New examples: `openai_quickstart.py`, `agent_memory.py`,
  `contradictions_demo.py`.
- README rewrite with badges, install matrix, and concept table.
- Apache-2.0 `LICENSE` file.

### Changed
- `pyproject.toml` upgraded to a full PyPI manifest: keywords,
  classifiers, optional extras (`openai`, `anthropic`, `all`, `dev`),
  project URLs, ruff/mypy config.
- `SurrealDriver.connect()` now wraps SDK errors in
  `SurritiConnectionError`; `init_schema()` wraps DDL failures in
  `SurritiSchemaError`.
- `SurrealDriver.db` raises `SurritiConnectionError` (was `RuntimeError`)
  when accessed before `connect()`.
- `__init__.py` reorganised by category (core / models / clients /
  errors / helpers).

## [0.3.0] — Graphiti feature parity

### Added
- `add_episode_bulk`, `add_triplet`, `retrieve_episodes`.
- `get_nodes_and_edges_by_episode`, `save_*`/`remove_*` for nodes and edges.
- `build_communities` with Leiden clustering.
- Search recipes (`EDGE_HYBRID_SEARCH_RRF`,
  `NODE_HYBRID_SEARCH_NODE_DISTANCE`,
  `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`).
- Cross-encoder rerankers.
- Full `SearchFilters` with `ComparisonOperator`, `PropertyFilter`,
  `DateFilter` over nested attributes.

## [0.2.0] — Validation against real SurrealDB

### Added
- 28-test integration suite running against SurrealDB v2.1.4 in Docker.
- `docker-compose.yml`.

## [0.1.0] — Initial port

### Added
- Core nodes, edges, schema DDL, async driver wrapper.
- `Surriti` facade with `add_episode`, `search`, hybrid retrieval.
- `DummyLLMClient`, `DummyEmbedder`, `ScriptedLLMClient` for offline use.
