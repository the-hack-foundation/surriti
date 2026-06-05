"""SurrealDB schema definitions for Surriti.

These DDL statements define the tables, fields, and indexes that mirror
Graphiti's data model. Running ``init_schema()`` is idempotent because each
``DEFINE ... IF NOT EXISTS`` statement is a no-op when the object exists.
"""

from __future__ import annotations


def schema_ddl(embedding_dim: int = 768) -> str:
    """Return the full DDL block needed to bootstrap a Surriti database.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the vectors produced by your embedder. Must match
        :class:`~surriti.embedder.EmbedderClient` output length.
    """

    return f"""
    -- Analyzers ---------------------------------------------------------
    DEFINE ANALYZER IF NOT EXISTS surriti_en
        TOKENIZERS blank,class,camel,punct
        FILTERS lowercase, ascii, snowball(english);

    -- Episode (raw input) ----------------------------------------------
    DEFINE TABLE IF NOT EXISTS episode SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid              ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id          ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS name              ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS source            ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS source_description ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS content           ON episode TYPE string;
    DEFINE FIELD IF NOT EXISTS reference_time    ON episode TYPE datetime;
    DEFINE FIELD IF NOT EXISTS created_at        ON episode TYPE datetime;
    DEFINE FIELD IF NOT EXISTS entity_edges      ON episode TYPE array<string> DEFAULT [];
    -- Cognitive layer (additive): per-episode affect tag and procedural
    -- interaction-pattern label. Both populated by ``surriti.cognition``;
    -- legacy rows simply read empty defaults.
    DEFINE FIELD IF NOT EXISTS affect            ON episode TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE FIELD IF NOT EXISTS interaction_pattern ON episode TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS cognition_processed_at ON episode TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS cognition_version      ON episode TYPE option<string>;
    DEFINE INDEX IF NOT EXISTS episode_uuid_idx     ON episode FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS episode_group_idx    ON episode FIELDS group_id;
    DEFINE INDEX IF NOT EXISTS episode_content_fts  ON episode FIELDS content
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;

    -- Entity ------------------------------------------------------------
    DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid           ON entity TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id       ON entity TYPE string;
    DEFINE FIELD IF NOT EXISTS name           ON entity TYPE string;
    DEFINE FIELD IF NOT EXISTS summary        ON entity TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS labels         ON entity TYPE array<string> DEFAULT ["Entity"];
    DEFINE FIELD IF NOT EXISTS attributes     ON entity TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE FIELD IF NOT EXISTS name_embedding ON entity TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS created_at     ON entity TYPE datetime;
    -- Dossier / profile fields. All have safe defaults so existing rows
    -- migrate forward without backfill. ``profiles.refresh_entity_profiles``
    -- materialises the derived fields after each ingest.
    DEFINE FIELD IF NOT EXISTS canonical_name    ON entity TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS aliases           ON entity TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS profile_summary   ON entity TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS profile_embedding ON entity TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS salience          ON entity TYPE float DEFAULT 0;
    DEFINE FIELD IF NOT EXISTS mention_count     ON entity TYPE int DEFAULT 0;
    DEFINE FIELD IF NOT EXISTS last_seen_at      ON entity TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS merged_into       ON entity TYPE option<string>;
    -- Cognitive layer (additive). All optional / cached / safe defaults.
    -- ``traits`` and ``goals_active`` are denormalised UUID lists kept in
    -- sync by ``surriti.cognition``; ``domain`` is the labelled cluster
    -- this entity belongs to (set by domain-aware community labelling).
    DEFINE FIELD IF NOT EXISTS traits            ON entity TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS goals_active      ON entity TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS domain            ON entity TYPE option<string>;
    DEFINE INDEX IF NOT EXISTS entity_uuid_idx     ON entity FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_group_idx    ON entity FIELDS group_id;
    DEFINE INDEX IF NOT EXISTS entity_name_uniq    ON entity FIELDS group_id, name UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_name_fts     ON entity FIELDS name
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS entity_summary_fts  ON entity FIELDS summary
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS entity_profile_fts  ON entity FIELDS profile_summary
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS entity_name_hnsw    ON entity FIELDS name_embedding
        HNSW DIMENSION {embedding_dim} DIST COSINE TYPE F32;
    DEFINE INDEX IF NOT EXISTS entity_profile_hnsw ON entity FIELDS profile_embedding
        HNSW DIMENSION {embedding_dim} DIST COSINE TYPE F32;

    -- Entity aliases (canonical-resolution layer). Each row is a
    -- known surface form of an entity in a tenant. Lookup by
    -- ``(group_id, normalized_alias)`` is the fast path before any
    -- semantic / LLM resolution happens.
    DEFINE TABLE IF NOT EXISTS entity_alias SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid                ON entity_alias TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id            ON entity_alias TYPE string;
    DEFINE FIELD IF NOT EXISTS alias               ON entity_alias TYPE string;
    DEFINE FIELD IF NOT EXISTS normalized_alias    ON entity_alias TYPE string;
    DEFINE FIELD IF NOT EXISTS entity_uuid         ON entity_alias TYPE string;
    DEFINE FIELD IF NOT EXISTS confidence          ON entity_alias TYPE float DEFAULT 1.0;
    DEFINE FIELD IF NOT EXISTS source_episode_uuid ON entity_alias TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS created_at          ON entity_alias TYPE datetime;
    DEFINE INDEX IF NOT EXISTS entity_alias_uuid_idx   ON entity_alias FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_alias_lookup     ON entity_alias FIELDS group_id, normalized_alias;
    DEFINE INDEX IF NOT EXISTS entity_alias_unique     ON entity_alias FIELDS group_id, normalized_alias UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_alias_entity_idx ON entity_alias FIELDS group_id, entity_uuid;

    -- Community ---------------------------------------------------------
    DEFINE TABLE IF NOT EXISTS community SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid           ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id       ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS name           ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS summary        ON community TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS name_embedding ON community TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS created_at     ON community TYPE datetime;
    -- Cognitive layer (additive). ``kind`` discriminates a normal
    -- entity-cluster ("cluster") from cognitive sidecars stored as
    -- community rows ("prediction"). ``domain`` carries the labelled
    -- semantic domain assigned by domain-aware clustering.
    -- ``payload`` is a free-form bag (e.g. prediction bundle).
    DEFINE FIELD IF NOT EXISTS kind           ON community TYPE string DEFAULT "cluster";
    DEFINE FIELD IF NOT EXISTS domain         ON community TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS payload        ON community TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE INDEX IF NOT EXISTS community_uuid_idx ON community FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS community_kind_idx ON community FIELDS group_id, kind;

    -- Edges -------------------------------------------------------------
    DEFINE TABLE IF NOT EXISTS mentions SCHEMAFULL TYPE RELATION FROM episode TO entity;
    DEFINE FIELD IF NOT EXISTS uuid       ON mentions TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id   ON mentions TYPE string;
    DEFINE FIELD IF NOT EXISTS created_at ON mentions TYPE datetime;
    DEFINE INDEX IF NOT EXISTS mentions_uuid_idx  ON mentions FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS mentions_group_idx ON mentions FIELDS group_id;

    DEFINE TABLE IF NOT EXISTS relates_to SCHEMAFULL TYPE RELATION FROM entity TO entity;
    DEFINE FIELD IF NOT EXISTS uuid           ON relates_to TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id       ON relates_to TYPE string;
    DEFINE FIELD IF NOT EXISTS name           ON relates_to TYPE string;
    DEFINE FIELD IF NOT EXISTS fact           ON relates_to TYPE string;
    DEFINE FIELD IF NOT EXISTS fact_embedding ON relates_to TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS episodes       ON relates_to TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS valid_at       ON relates_to TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS invalid_at     ON relates_to TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS expired_at     ON relates_to TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS attributes     ON relates_to TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE FIELD IF NOT EXISTS created_at     ON relates_to TYPE datetime;
    -- Generic temporal-state metadata: enables the singleton-slot closer
    -- and current-state queries without a hardcoded predicate vocabulary.
    DEFINE FIELD IF NOT EXISTS status         ON relates_to TYPE string DEFAULT "active";
    DEFINE FIELD IF NOT EXISTS polarity       ON relates_to TYPE string DEFAULT "positive";
    DEFINE FIELD IF NOT EXISTS source_type    ON relates_to TYPE string DEFAULT "user";
    DEFINE FIELD IF NOT EXISTS confidence     ON relates_to TYPE float DEFAULT 1.0;
    DEFINE FIELD IF NOT EXISTS temporal       ON relates_to TYPE bool DEFAULT false;
    DEFINE FIELD IF NOT EXISTS singleton      ON relates_to TYPE bool DEFAULT false;
    DEFINE FIELD IF NOT EXISTS domain         ON relates_to TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS supersedes     ON relates_to TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS superseded_by  ON relates_to TYPE option<string>;
    -- Deterministic dedupe key (group_id::subject_uuid::predicate::object_uuid).
    -- Empty default keeps backward compatibility with rows written before
    -- this field existed; ``backfill_fact_keys()`` populates them so the
    -- unique index can be enabled after migration.
    DEFINE FIELD IF NOT EXISTS fact_key       ON relates_to TYPE string DEFAULT "";
    -- Relation-frame metadata (generalized predicate layer). Optional
    -- on legacy rows; populated on insert once a frame resolves.
    DEFINE FIELD IF NOT EXISTS relation_frame_id ON relates_to TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS canonical_name    ON relates_to TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS qualifiers        ON relates_to TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE FIELD IF NOT EXISTS roles             ON relates_to TYPE object FLEXIBLE DEFAULT {{}};
    DEFINE FIELD IF NOT EXISTS conflict_group_id ON relates_to TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS derived           ON relates_to TYPE bool DEFAULT false;
    DEFINE FIELD IF NOT EXISTS derived_from      ON relates_to TYPE option<string>;
    -- Cognitive layer (additive). All optional / safe defaults so legacy
    -- rows load forward without backfill. Populated lazily by
    -- ``surriti.cognition`` (reinforcement / decay / consolidation /
    -- belief / affect passes) and read by recall + rerankers.
    DEFINE FIELD IF NOT EXISTS weight             ON relates_to TYPE float DEFAULT 1.0;
    DEFINE FIELD IF NOT EXISTS reinforcement_count ON relates_to TYPE int DEFAULT 1;
    DEFINE FIELD IF NOT EXISTS last_reinforced_at  ON relates_to TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS recall_count        ON relates_to TYPE int DEFAULT 0;
    DEFINE FIELD IF NOT EXISTS last_recalled_at    ON relates_to TYPE option<datetime>;
    DEFINE FIELD IF NOT EXISTS decay_score          ON relates_to TYPE float DEFAULT 1.0;
    DEFINE FIELD IF NOT EXISTS stability            ON relates_to TYPE string DEFAULT "episodic";
    DEFINE FIELD IF NOT EXISTS valence              ON relates_to TYPE option<float>;
    DEFINE FIELD IF NOT EXISTS intensity            ON relates_to TYPE option<float>;
    DEFINE FIELD IF NOT EXISTS consolidates         ON relates_to TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS is_belief            ON relates_to TYPE bool DEFAULT false;
    DEFINE FIELD IF NOT EXISTS belief_holder        ON relates_to TYPE option<string>;
    DEFINE INDEX IF NOT EXISTS relates_to_uuid_idx  ON relates_to FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS relates_to_group_idx ON relates_to FIELDS group_id;
    DEFINE INDEX IF NOT EXISTS relates_to_active_idx ON relates_to FIELDS group_id, in, name, status;
    DEFINE INDEX IF NOT EXISTS relates_to_canonical_idx ON relates_to FIELDS group_id, in, canonical_name, status;
    DEFINE INDEX IF NOT EXISTS relates_to_conflict_idx ON relates_to FIELDS group_id, conflict_group_id;
    DEFINE INDEX IF NOT EXISTS relates_to_fact_key_idx ON relates_to FIELDS group_id, fact_key;
    DEFINE INDEX IF NOT EXISTS relates_to_fact_fts  ON relates_to FIELDS fact
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS relates_to_fact_hnsw ON relates_to FIELDS fact_embedding
        HNSW DIMENSION {embedding_dim} DIST COSINE TYPE F32;

    -- Relation frames (per-predicate metadata that drives generalized
    -- temporal/contradiction reasoning without hardcoded predicate
    -- vocabulary). One row per canonical relation type per group.
    DEFINE TABLE IF NOT EXISTS relation_frame SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid                 ON relation_frame TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id             ON relation_frame TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS canonical_name       ON relation_frame TYPE string;
    DEFINE FIELD IF NOT EXISTS aliases              ON relation_frame TYPE array<string> DEFAULT [];
    DEFINE FIELD IF NOT EXISTS description          ON relation_frame TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS directionality       ON relation_frame TYPE string DEFAULT "unknown";
    DEFINE FIELD IF NOT EXISTS temporal_kind        ON relation_frame TYPE string DEFAULT "unknown";
    DEFINE FIELD IF NOT EXISTS cardinality          ON relation_frame TYPE string DEFAULT "unknown";
    DEFINE FIELD IF NOT EXISTS contradiction_policy ON relation_frame TYPE string DEFAULT "uncertain";
    DEFINE FIELD IF NOT EXISTS inverse_name         ON relation_frame TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS subject_role         ON relation_frame TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS object_role          ON relation_frame TYPE option<string>;
    DEFINE FIELD IF NOT EXISTS confidence           ON relation_frame TYPE float DEFAULT 0.5;
    DEFINE FIELD IF NOT EXISTS created_at           ON relation_frame TYPE datetime;
    DEFINE INDEX IF NOT EXISTS relation_frame_uuid_idx ON relation_frame FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS relation_frame_canon_idx ON relation_frame FIELDS group_id, canonical_name UNIQUE;

    DEFINE TABLE IF NOT EXISTS has_member SCHEMAFULL TYPE RELATION FROM community TO entity;
    DEFINE FIELD IF NOT EXISTS uuid       ON has_member TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id   ON has_member TYPE string;
    DEFINE FIELD IF NOT EXISTS created_at ON has_member TYPE datetime;
    DEFINE INDEX IF NOT EXISTS has_member_uuid_idx ON has_member FIELDS uuid UNIQUE;
    """


# All tables managed by Surriti. Useful for clear/reset operations.
ALL_TABLES: tuple[str, ...] = (
    "mentions",
    "relates_to",
    "has_member",
    "episode",
    "entity",
    "entity_alias",
    "community",
    "relation_frame",
)


async def backfill_fact_keys(driver) -> int:
    """Populate ``relates_to.fact_key`` for legacy rows lacking it.

    Returns the number of edges updated. Safe to run repeatedly: rows
    that already have a non-empty ``fact_key`` are skipped. Run once
    after upgrading from a Surriti version that did not write fact keys
    on insert; subsequent inserts populate the field automatically.
    """

    from surriti.search import _unwrap

    rows = _unwrap(
        await driver.query(
            """
            SELECT uuid, group_id, name,
                record::id(in)  AS subject_uuid,
                record::id(out) AS object_uuid
            FROM relates_to WHERE fact_key = "" OR fact_key IS NONE;
            """
        )
    )
    updated = 0
    for row in rows:
        key = "::".join(
            (
                str(row.get("group_id") or "").strip(),
                str(row.get("subject_uuid") or "").strip(),
                str(row.get("name") or "").strip().lower(),
                str(row.get("object_uuid") or "").strip(),
            )
        )
        await driver.query(
            "UPDATE relates_to SET fact_key = $key WHERE uuid = $uuid",
            {"key": key, "uuid": row.get("uuid")},
        )
        updated += 1
    return updated
