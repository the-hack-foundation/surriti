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
    DEFINE INDEX IF NOT EXISTS entity_uuid_idx     ON entity FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_group_idx    ON entity FIELDS group_id;
    DEFINE INDEX IF NOT EXISTS entity_name_uniq    ON entity FIELDS group_id, name UNIQUE;
    DEFINE INDEX IF NOT EXISTS entity_name_fts     ON entity FIELDS name
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS entity_summary_fts  ON entity FIELDS summary
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS entity_name_hnsw    ON entity FIELDS name_embedding
        HNSW DIMENSION {embedding_dim} DIST COSINE TYPE F32;

    -- Community ---------------------------------------------------------
    DEFINE TABLE IF NOT EXISTS community SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS uuid           ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS group_id       ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS name           ON community TYPE string;
    DEFINE FIELD IF NOT EXISTS summary        ON community TYPE string DEFAULT "";
    DEFINE FIELD IF NOT EXISTS name_embedding ON community TYPE option<array<float>>;
    DEFINE FIELD IF NOT EXISTS created_at     ON community TYPE datetime;
    DEFINE INDEX IF NOT EXISTS community_uuid_idx ON community FIELDS uuid UNIQUE;

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
    DEFINE INDEX IF NOT EXISTS relates_to_uuid_idx  ON relates_to FIELDS uuid UNIQUE;
    DEFINE INDEX IF NOT EXISTS relates_to_group_idx ON relates_to FIELDS group_id;
    DEFINE INDEX IF NOT EXISTS relates_to_fact_fts  ON relates_to FIELDS fact
        FULLTEXT ANALYZER surriti_en BM25 HIGHLIGHTS;
    DEFINE INDEX IF NOT EXISTS relates_to_fact_hnsw ON relates_to FIELDS fact_embedding
        HNSW DIMENSION {embedding_dim} DIST COSINE TYPE F32;

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
    "community",
)
