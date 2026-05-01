# Changelog

All notable changes to **surriti** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Generic temporal-state engine.** `ExtractedFact` carries new
  per-fact metadata that drives invalidation without any hardcoded
  predicate vocabulary:
  - `operation: Literal["assert","terminate","correct","noop"]`
  - `temporal: bool`, `singleton: bool`, `domain: str | None`,
    `replaces: list[str]`, `confidence: float`
  The LLM (or caller) tags each fact and the engine reasons over those
  flags. `terminate` closes the matching active edge with no insert;
  `correct` is treated as singleton-asserted; `noop` is skipped.
- **Deterministic singleton-slot closer.** When `fact.singleton=True`
  and `source_type=="user"`, `_add_fact_edge` closes every active edge
  on the same `(group_id, subject, predicate)` slot pointing at a
  different object before inserting the new edge — no LLM
  contradiction call needed. Skipped when `source_type` is
  `"assistant"`/`"tool"`/`"system"` so model-generated facts cannot
  silently nuke prior user truth.
- **Two-channel extraction prompt.** `LLMClient.extract` gained a
  `context: str | None` kwarg. Prior episode bodies now travel through
  a dedicated read-only `CONTEXT` fence in the prompt (clearly marked
  "do NOT extract"), instead of being concatenated onto the current
  episode. This stops small models from re-extracting facts that were
  already persisted in earlier turns.
- `EntityEdge` mirrors the new metadata fields (`status`, `polarity`,
  `source_type`, `confidence`, `temporal`, `singleton`, `domain`,
  `supersedes`, `superseded_by`) with safe defaults so existing code
  is unaffected.
- `relates_to` schema gained the matching nine fields plus a composite
  `relates_to_active_idx ON FIELDS group_id, in, name, status` for the
  singleton-closer hot path.
- `Graphiti.get_current_fact(...)` and `Graphiti.get_current_facts(...)`
  return live edges (`status="active" AND invalid_at IS NONE`),
  optionally scoped by predicate or domain — the generic answer to
  "what's currently true about this subject?" without going through
  hybrid search.
- `add_episode` accepts `source_type: str = "user"` to mark the
  provenance of facts in the episode.

### Changed
- `EXTRACTION_SYSTEM` rewritten around the per-fact metadata rubric:
  fenced `CONTEXT` (read-only) + `CURRENT EPISODE` sections; generic
  `temporal`/`singleton`/`domain` explanations (no hardcoded
  predicates); `operation` rubric covering "I quit X"/"no longer
  X"/"stopped X" → `terminate`, "actually X not Y" → `correct`; rule
  to drop facts whose object is a vague placeholder
  (`world`/`everywhere`/`thing`/`something`/`nothing`/`someone`).
- `temporal.invalidate_edges` now also sets `status="superseded"` and
  optional `superseded_by` on the closed rows.
- `search._filter_valid` excludes rows whose `status` is set and not
  `"active"`.
- `myapp/service.py`: stripped the hardcoded predicate examples from
  `EXTRACTION_INSTRUCTIONS` (predicates are now whatever snake_case
  verb fits) and tags both `add_episode` call sites with
  `source_type="user"`.

### Fixed
- `add_episode` now **auto-resolves** `speaker_id` / `speaker_name` against
  the LLM's extracted entity list. Previously, when the model emitted
  `subject="default"` (or the speaker's display name) but omitted that
  identifier from the `entities` array, every fact about the speaker was
  silently dropped as "unresolved entities". The speaker is now upserted
  on demand and the facts persist as edges.
- `OpenAILLMClient` now accepts an `extra_body` kwarg and forwards it to
  every chat completion (used to thread Qwen3 `enable_thinking=False`
  through extraction/contradiction calls so reasoning traces don't leak
  into the JSON response).
- `EXTRACTION_SYSTEM` adds a **VALUES-as-entities** rule: dates ("October
  14"), ages ("33"), places, and companies must appear in `entities` and
  be the `object` of the relevant fact (`has_birthday`, `is_age`,
  `lives_in`, `works_at`). Forbids placeholder subjects/objects like
  `"speaker"` or `"value"` and self-loops.
- Speaker hint passed by `add_episode` (both with-name and id-only
  branches) ships concrete value-as-object examples so smaller models
  reliably emit `Michael -[has_birthday]-> October 14` instead of
  `default -[has_birthday]-> default`.
- `EXTRACTION_SYSTEM` rewritten **again** with explicit "WHAT COUNTS AS A
  FACT" positive examples (`is_named`, `is_age`, `is_a`, `works_at`,
  `is_learning`, ...). Small models (Qwen2.5-3B) treated terse self-
  introductions like *"i am 5 months old"* or *"i am a baby"* as
  no-claim and returned `facts: []`. Positive examples + "When in doubt,
  extract." restore extraction on first-person utterances.
- `add_episode` now **repairs** `subject == object` identity facts
  (`is_named`/`is_called`/`is_self`/`is_aka`) when `speaker_id` is set:
  the subject is rewritten to the speaker's stable id so e.g. a hallucinated
  `Auley -[is_named]-> Auley` becomes `default -[is_named]-> Auley`. With no
  `speaker_id`, the fact is preserved as before. Defence-in-depth on top
  of the prompt's hard rule.
- `add_episode`'s speaker hint now ships concrete bracketed examples
  (`"my name is Auley" -> default -[is_named]-> Auley`, ...) when the
  speaker has no display name yet, so 3B models reliably use the stable
  id as the subject of identity facts.
- `myapp/service.py`: after every `add_episode`, scan the resulting edges
  for `is_named` from the speaker and persist the target as the User
  node's `display_name`. The next turn's speaker hint then carries the
  human-readable name without the caller having to upsert it explicitly.

### Fixed
- `_upsert_entities` no longer crashes with `Database index entity_name_uniq
  already contains [...]` when the LLM emits the same entity name twice in
  one extraction (e.g. "Michael" listed twice). The pipeline now dedupes
  the extracted entity list by name and recovers from any unique-index
  collision by re-SELECTing the existing row instead of bubbling the
  exception.
- `_fetch_episode_contents` no longer prefixes prior-episode text with
  `[<episode-name>]`. Small models (e.g. Qwen2.5-3B) used to read those
  bracketed labels as entities, producing garbage like
  `Michael -[works_with]-> turn-a`. Episode `name` is internal metadata
  and is no longer surfaced to the LLM.
- `add_episode` now drops self-loop facts (`subject == object`) before
  insert, except for identity predicates (`is_named`, `is_called`,
  `is_self`, `is_aka`). Defence-in-depth against small models that
  hallucinate `Michael -[knows]-> Michael` style filler.
- `EXTRACTION_SYSTEM` rewritten with HARD rules: no self-loops (except
  identity predicates), no invented entities/predicates, return zero
  facts on greetings/questions, treat bracketed/UUID tokens as metadata.
- `CONTRADICTION_SYSTEM` now requires both same subject AND same
  relation domain before invalidating prior facts. Cross-domain pairs
  like `is_brother_of` vs `works_with` no longer trigger spurious
  invalidations.

### Added
- `Surriti.upsert_user(group_id, user_id=None, display_name=None,
  summary="")` — UPSERTs the canonical `User` entity for a tenant. The
  entity's `name` defaults to `group_id` (treat your tenant id as the
  user's stable identifier); the friendly name lives in
  `attributes.display_name` so the `(group_id, name)` unique index stays
  clean. Idempotent.
- `Surriti.add_episode(...)` accepts new optional kwargs `speaker_id` and
  `speaker_name`. When `speaker_id` is set, the canonical `User` entity is
  upserted before extraction and a short speaker-context hint is appended
  to the extractor instructions so first-person pronouns can be resolved
  to that user.
- `surriti.testing.InMemoryDriver(enforce_entity_name_uniq=True)` — opt-in
  flag that enforces the `(group_id, name)` unique index in the in-memory
  driver, mirroring the real schema. Used by the new regression tests.

## [0.5.0] — SurrealDB 3.0 + SDK 2.0

### Breaking
- Requires **SurrealDB 3.x** server and **`surrealdb>=2.0,<3`** Python SDK.
  Older SurrealDB 1.x/2.x SDKs are no longer supported. SDK 2.0 returns the
  last statement's rows directly (no `[{"result": [...]}]` wrapper) and that
  shape is now the canonical one consumed by Surriti.
- `schema_ddl()` and `SurrealDriver` now default `embedding_dim=768` (matches
  fastembed `nomic-ai/nomic-embed-text-v1.5`). Override via constructor or
  `SURRITI_EMBEDDING_DIM` for other embedders.

### Changed
- `SurrealDriver.connect()` simplified — drops the legacy SDK 1.x
  `connect()` non-awaitable fallback. `await self._db.connect()` directly.
- `surriti.search._unwrap` rewritten to canonicalise three result shapes
  (SDK 2.0 flat list, SDK 1.x list-of-lists, legacy `[{"result": [...]}]`).
- Schema DDL updated for SurrealDB 3 syntax:
  - `SEARCH ANALYZER` → `FULLTEXT ANALYZER`.
  - `FLEXIBLE TYPE object` → `TYPE object FLEXIBLE`.
  - Multi-column FULLTEXT index split into two single-column indexes
    (`entity_name_fts` + `entity_summary_fts`).
- Renamed all `type::thing(...)` calls to `type::record(...)` in
  `graphiti.py` and `testing.py` (the v2 alias was removed in v3).

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
