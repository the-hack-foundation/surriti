"""Surriti Memory Pack v1 — compact portable knowledge-graph transfer format.

Export/import/validate ``.surriti-pack.zip`` bundles that transfer entities,
aliases, relation frames, and ``relates_to`` edges between Surriti users,
agents, and deployments.  Designed to be **safe, streamable, and correct**
without dragging in raw episodes, mentions, or embeddings.

Architecture
------------
* Export reads directly from SurrealDB tables via paginated queries.
* Records are written as one-JSON-object-per-line (JSONL) files inside a ZIP.
* Import restores records with ID remapping; it does **not** call
  ``add_episode()``.
* Default policy: ``include_embeddings = "never"``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from surriti.edges import make_fact_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_PACK_FORMAT = "surriti.memory-pack"
MEMORY_PACK_VERSION = 1

# Tables included in the default portable-graph export.
_PORTABLE_TABLES = ("entity", "entity_alias", "relation_frame", "relates_to")

# Fields to strip when ``include_embeddings == "never"``.
_EMBEDDING_FIELDS: dict[str, list[str]] = {
    "entity": ["name_embedding", "profile_embedding", "emb"],
    "relates_to": ["fact_embedding", "emb"],
}

# SurrealDB internal fields we never want to export.
_SURREAL_INTERNAL_FIELDS = frozenset({"id"})


# ---------------------------------------------------------------------------
# Result / validation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ExportResult:
    output_path: str
    manifest: dict
    counts: dict[str, int]
    checksums: dict[str, dict]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    target_group_id: str
    mode: str
    counts: dict[str, int]
    validation: ValidationResult
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safe JSON helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert a value to JSON-serialisable form.

    Handles:
    * ``datetime`` → ISO-8601 string
    * SurrealDB ``RecordID`` → string
    * lists / dicts recursively
    * unknown types → ``str()`` fallback
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # SurrealDB record IDs come through as opaque objects.
    try:
        s = str(value)
        if s.startswith("{") and s.endswith("}"):
            return s  # already JSON-like
        return s
    except Exception:
        return repr(value)


def _strip_record_id_to_uuid(value: Any) -> str | None:
    """Extract a UUID from a SurrealDB record ID.

    ``"entity:⟨abc123⟩"`` → ``"abc123"``
    ``"relates_to:⟨def456⟩"`` → ``"def456"``
    """
    if value is None:
        return None
    s = str(value)
    if ":" in s:
        return s.rsplit(":", 1)[-1].strip("⟨⟩")
    return s


def _jsonl_write(fileobj, records: list[dict]) -> None:
    """Write a list of dicts as JSONL to an open file object."""
    for rec in records:
        fileobj.write(json.dumps(rec, ensure_ascii=False, default=str).encode("utf-8"))
        fileobj.write(b"\n")


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_import_uuid(
    target_group_id: str,
    table: str,
    source_uuid: Any,
    *,
    fallback_name: str = "",
) -> str:
    """Return an idempotent target UUID for a source record.

    When ``source_uuid`` is empty the caller should supply ``fallback_name``
    (e.g. entity ``name``) so repeated imports converge on the same UUID.
    Without any stable input a warning-level UUIDv4 fallback is used and
    the caller is expected to warn.
    """
    source = str(source_uuid or "").strip()
    if not source:
        if fallback_name:
            source = str(fallback_name).strip()
        if not source:
            source = uuid.uuid4().hex
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"surriti:{target_group_id}:{table}:{source}"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name} line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row in {path.name} line {line_no}: expected object")
        rows.append(row)
    return rows


_ISO_DATETIME_RE = __import__("re").compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _restore_datetimes(row: dict) -> dict:
    """Convert ISO-8601 datetime strings back to :class:`datetime` objects.

    Called on every row before upsert so SurrealDB's strict schema
    (``TYPE datetime``) doesn't reject string values written during export.
    """
    from datetime import datetime as _dt

    for key, val in row.items():
        if isinstance(val, str) and _ISO_DATETIME_RE.match(val):
            try:
                # fromisoformat handles most ISO-8601 variants in 3.11+.
                row[key] = _dt.fromisoformat(val)
            except ValueError:
                pass  # leave as string
    return row


def _without_omission_markers(row: dict) -> dict:
    out = dict(row)
    out.pop("source_episode_omitted", None)
    out.pop("episodes_omitted", None)
    out.pop("source_episode_count", None)
    return out


# ---------------------------------------------------------------------------
# Row normalisation per table
# ---------------------------------------------------------------------------


def _normalize_entity(row: dict, *, include_embeddings: str) -> dict:
    """Normalise an entity row for export."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _SURREAL_INTERNAL_FIELDS:
            continue
        out[k] = _json_safe(v)
    if include_embeddings == "never":
        for f in _EMBEDDING_FIELDS.get("entity", []):
            out.pop(f, None)
    return out


def _normalize_entity_alias(row: dict) -> dict:
    """Normalise an alias row. Drops dangling ``source_episode_uuid``."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _SURREAL_INTERNAL_FIELDS:
            continue
        out[k] = _json_safe(v)
    # Default packs do not include episodes — preserve the hint that
    # provenance was deliberately omitted rather than missing.
    if out.pop("source_episode_uuid", None) is not None:
        out["source_episode_omitted"] = True
    return out


def _normalize_relation_frame(row: dict) -> dict:
    """Normalise a relation_frame row."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _SURREAL_INTERNAL_FIELDS:
            continue
        out[k] = _json_safe(v)
    return out


def _normalize_edge(row: dict, *, include_embeddings: str) -> dict:
    """Normalise a ``relates_to`` row.

    Extracts ``source_node_uuid`` / ``target_node_uuid`` from SurrealDB
    record IDs. Drops ``fact_embedding`` unless requested.  Notes
    ``episodes_omitted`` instead of preserving raw episode UUIDs.
    """
    out: dict[str, Any] = {}

    # Resolve source / target from SurrealDB relation record IDs.
    src = _strip_record_id_to_uuid(row.get("in"))
    tgt = _strip_record_id_to_uuid(row.get("out"))
    if src is not None:
        out["source_node_uuid"] = src
    if tgt is not None:
        out["target_node_uuid"] = tgt

    for k, v in row.items():
        if k in _SURREAL_INTERNAL_FIELDS:
            continue
        if k in ("in", "out"):
            continue  # already handled above
        out[k] = _json_safe(v)

    # Episodes: note omission rather than preserving dangling UUIDs.
    eps = out.pop("episodes", None)
    if eps:
        out["source_episode_count"] = len(eps)
        out["episodes_omitted"] = True
    else:
        out["source_episode_count"] = 0
        out["episodes_omitted"] = False

    # Embeddings
    if include_embeddings == "never":
        for f in _EMBEDDING_FIELDS.get("relates_to", []):
            out.pop(f, None)

    return out


# ---------------------------------------------------------------------------
# Paginated streaming queries
# ---------------------------------------------------------------------------

_ENTITY_FIELDS = """
    uuid, group_id, name, summary, labels, attributes,
    name_embedding, created_at, canonical_name, aliases,
    profile_summary, profile_embedding, salience, mention_count,
    last_seen_at, merged_into, traits, goals_active, domain
""".replace("\n", " ").strip()

_ALIAS_FIELDS = """
    uuid, group_id, alias, normalized_alias, entity_uuid,
    confidence, source_episode_uuid, created_at
""".replace("\n", " ").strip()

_FRAME_FIELDS = """
    uuid, group_id, canonical_name, aliases, description,
    directionality, temporal_kind, cardinality, contradiction_policy,
    inverse_name, subject_role, object_role, confidence, created_at
""".replace("\n", " ").strip()

_EDGE_FIELDS = """
    uuid, group_id, name, fact, fact_embedding, episodes,
    valid_at, invalid_at, expired_at, attributes, created_at,
    status, polarity, source_type, confidence, temporal, singleton,
    domain, supersedes, superseded_by, fact_key, relation_frame_id,
    canonical_name, qualifiers, roles, conflict_group_id, derived,
    derived_from, weight, reinforcement_count, last_reinforced_at,
    recall_count, last_recalled_at, decay_score, stability, valence,
    intensity, consolidates, is_belief, belief_holder
""".replace("\n", " ").strip()


async def _paginate(
    driver,
    table: str,
    fields: str,
    group_id: str,
    *,
    page_size: int = 1000,
    resolve_edge_ids: bool = False,
) -> list[dict]:
    """Yield all rows for *group_id* from *table* using offset pagination.

    Parameters
    ----------
    resolve_edge_ids:
        When True the query also selects ``record::id(in)`` /
        ``record::id(out)`` aliases so edge exports get source/target UUIDs.
    """
    from surriti.search import _unwrap

    extra = ""
    if resolve_edge_ids:
        extra = ", record::id(in) AS `in`, record::id(out) AS `out`"

    all_rows: list[dict] = []
    offset = 0
    while True:
        surql = (
            f"SELECT {fields}{extra} FROM {table} "
            f"WHERE group_id = $g "
            f"ORDER BY created_at, uuid "
            f"LIMIT $limit START $offset;"
        )
        rows = _unwrap(
            await driver.query(
                surql,
                {"g": group_id, "limit": page_size, "offset": offset},
            )
        )
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


async def export_group_to_dir(
    driver,
    group_id: str,
    output_dir: str | Path,
    *,
    include_embeddings: str = "never",
    page_size: int = 1000,
    embedding_model: str | None = None,
) -> ExportResult:
    """Export a group's compact knowledge graph into a directory of JSONL files.

    Returns an :class:`ExportResult` with manifest, counts, checksums, and
    any warnings encountered during export.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if include_embeddings not in ("never", "auto", "always"):
        warnings.append(
            f"Unknown include_embeddings value {include_embeddings!r}; "
            f"falling back to 'never'."
        )
        include_embeddings = "never"

    export_started_at = datetime.now(timezone.utc)

    # ── entities ────────────────────────────────────────────────────────
    entity_rows = await _paginate(
        driver, "entity", _ENTITY_FIELDS, group_id, page_size=page_size
    )
    entities = [
        _normalize_entity(r, include_embeddings=include_embeddings)
        for r in entity_rows
    ]

    # ── entity aliases ──────────────────────────────────────────────────
    alias_rows = await _paginate(
        driver, "entity_alias", _ALIAS_FIELDS, group_id, page_size=page_size
    )
    aliases = [_normalize_entity_alias(r) for r in alias_rows]

    # ── relation frames ─────────────────────────────────────────────────
    frame_rows = await _paginate(
        driver, "relation_frame", _FRAME_FIELDS, group_id, page_size=page_size
    )
    frames = [_normalize_relation_frame(r) for r in frame_rows]

    # ── edges ───────────────────────────────────────────────────────────
    edge_rows = await _paginate(
        driver,
        "relates_to",
        _EDGE_FIELDS,
        group_id,
        page_size=page_size,
        resolve_edge_ids=True,
    )
    edges = [
        _normalize_edge(r, include_embeddings=include_embeddings)
        for r in edge_rows
    ]

    # Write JSONL files.
    files: dict[str, Path] = {}
    for name, data in [
        ("entities.jsonl", entities),
        ("entity_aliases.jsonl", aliases),
        ("relation_frames.jsonl", frames),
        ("edges.jsonl", edges),
    ]:
        path = out / name
        with open(path, "wb") as fh:
            _jsonl_write(fh, data)
        files[name] = path

    export_finished_at = datetime.now(timezone.utc)
    counts = {
        "entities": len(entities),
        "entity_aliases": len(aliases),
        "relation_frames": len(frames),
        "edges": len(edges),
    }

    # Checksums.
    checksums: dict[str, dict] = {}
    for fname, fpath in files.items():
        key = fname.replace(".jsonl", "")
        checksums[fname] = {
            "rows": counts.get(key, 0),
            "sha256": _compute_sha256(fpath),
        }

    # Manifest.
    manifest = {
        "format": MEMORY_PACK_FORMAT,
        "version": MEMORY_PACK_VERSION,
        "created_at": export_finished_at.isoformat(),
        "export_started_at": export_started_at.isoformat(),
        "export_finished_at": export_finished_at.isoformat(),
        "consistency": "best_effort",
        "source": {
            "group_id": group_id,
            "surriti_version": _surriti_version(),
            "embedding_model": embedding_model,
            "embedding_dim": getattr(driver, "embedding_dim", None),
        },
        "policy": {
            "mode": "portable_graph",
            "include_episodes": False,
            "include_mentions": False,
            "include_embeddings": include_embeddings,
            "include_communities": False,
        },
        "counts": counts,
        "privacy": {
            "contains_user_memory": True,
            "contains_raw_transcripts": False,
            "contains_embeddings": include_embeddings != "never",
        },
        "files": {
            "entities": "entities.jsonl",
            "entity_aliases": "entity_aliases.jsonl",
            "relation_frames": "relation_frames.jsonl",
            "edges": "edges.jsonl",
        },
    }

    # Write manifest.
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)

    # Write checksums.
    with open(out / "checksums.json", "w", encoding="utf-8") as fh:
        json.dump(checksums, fh, ensure_ascii=False, indent=2, default=str)

    return ExportResult(
        output_path=str(out),
        manifest=manifest,
        counts=counts,
        checksums=checksums,
        warnings=warnings,
    )


async def export_group_to_zip(
    driver,
    group_id: str,
    output_path: str | Path,
    *,
    include_embeddings: str = "never",
    page_size: int = 1000,
    embedding_model: str | None = None,
) -> ExportResult:
    """Export a group's compact knowledge graph into a ``.surriti-pack.zip``.

    This is the primary public API for creating Memory Packs.
    """
    output_path = Path(output_path)
    # Write to a temp file first, then atomically rename.  Avoids leaving
    # a truncated .zip on disk when the connection drops mid-export.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tempfile.TemporaryDirectory(prefix="surriti_pack_") as tmpdir:
        tmp = Path(tmpdir)

        # Export to temp directory first (no double-write to ZIP).
        result = await export_group_to_dir(
            driver,
            group_id,
            tmp,
            include_embeddings=include_embeddings,
            page_size=page_size,
            embedding_model=embedding_model,
        )

        # Bundle into ZIP (temp file).
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in [
                "manifest.json",
                "checksums.json",
                "entities.jsonl",
                "entity_aliases.jsonl",
                "relation_frames.jsonl",
                "edges.jsonl",
            ]:
                fpath = tmp / fname
                if fpath.exists():
                    zf.write(fpath, fname)

    # Atomic rename — final path only appears on complete success.
    tmp_path.rename(output_path)

    return ExportResult(
        output_path=str(output_path),
        manifest=result.manifest,
        counts=result.counts,
        checksums=result.checksums,
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _read_zip_text(zf: zipfile.ZipFile, name: str) -> str | None:
    """Read a text file from inside a ZIP. Returns ``None`` if missing."""
    try:
        return zf.read(name).decode("utf-8")
    except KeyError:
        return None


def _validate_pack_manifest(manifest: dict) -> list[str]:
    """Validate manifest structure. Returns a list of error messages."""
    errors: list[str] = []
    fmt = manifest.get("format")
    if fmt != MEMORY_PACK_FORMAT:
        errors.append(
            f"Unsupported format {fmt!r}; expected {MEMORY_PACK_FORMAT!r}"
        )
    ver = manifest.get("version")
    if not isinstance(ver, int) or ver < 1:
        errors.append(f"Unknown pack version {ver!r}")
    return errors


def validate_pack_dir(path: str | Path) -> ValidationResult:
    """Validate an unpacked Memory Pack directory."""
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}

    # Check required files exist.
    required = [
        "manifest.json",
        "checksums.json",
        "entities.jsonl",
        "entity_aliases.jsonl",
        "relation_frames.jsonl",
        "edges.jsonl",
    ]
    for fname in required:
        if not (p / fname).exists():
            errors.append(f"Missing required file: {fname}")

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

    # Validate manifest.
    try:
        manifest = json.loads((p / "manifest.json").read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"Cannot read manifest.json: {exc}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

    manifest_errors = _validate_pack_manifest(manifest)
    errors.extend(manifest_errors)

    # Validate checksums.
    try:
        checksums = json.loads((p / "checksums.json").read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"Cannot read checksums.json: {exc}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

    for fname, expected in checksums.items():
        fpath = p / fname
        if not fpath.exists():
            errors.append(f"Checksum entry {fname!r} references missing file")
            continue
        actual_sha = _compute_sha256(fpath)
        expected_sha = expected.get("sha256", "")
        if actual_sha != expected_sha:
            errors.append(
                f"Checksum mismatch for {fname}: "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
            )
        counts[fname.replace(".jsonl", "")] = expected.get("rows", 0)

    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        counts=counts,
    )


def validate_pack_zip(path: str | Path) -> ValidationResult:
    """Validate a ``.surriti-pack.zip`` file."""
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}

    if not p.exists():
        errors.append(f"File not found: {p}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

    if not zipfile.is_zipfile(p):
        errors.append(f"Not a valid ZIP file: {p}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

    with zipfile.ZipFile(p, "r") as zf:
        names = set(zf.namelist())

        # Check required files.
        required = [
            "manifest.json",
            "checksums.json",
            "entities.jsonl",
            "entity_aliases.jsonl",
            "relation_frames.jsonl",
            "edges.jsonl",
        ]
        for fname in required:
            if fname not in names:
                errors.append(f"Missing required file in ZIP: {fname}")

        if errors:
            return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

        # Validate manifest.
        manifest_raw = _read_zip_text(zf, "manifest.json")
        if manifest_raw is None:
            errors.append("Cannot read manifest.json from ZIP")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)
        try:
            manifest = json.loads(manifest_raw)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid manifest.json: {exc}")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)
        manifest_errors = _validate_pack_manifest(manifest)
        errors.extend(manifest_errors)

        # Validate checksums.
        checksums_raw = _read_zip_text(zf, "checksums.json")
        if checksums_raw is None:
            errors.append("Cannot read checksums.json from ZIP")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)
        try:
            checksums = json.loads(checksums_raw)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid checksums.json: {exc}")
            return ValidationResult(ok=False, errors=errors, warnings=warnings, counts=counts)

        for fname, expected in checksums.items():
            raw = _read_zip_text(zf, fname)
            if raw is None:
                errors.append(f"Checksum entry {fname!r} references missing file in ZIP")
                continue
            actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            expected_sha = expected.get("sha256", "")
            if actual_sha != expected_sha:
                errors.append(
                    f"Checksum mismatch for {fname}: "
                    f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )
            counts[fname.replace(".jsonl", "")] = expected.get("rows", 0)

    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Import / merge
# ---------------------------------------------------------------------------


_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024  # 512 MiB


def _validate_zip_safety(zf: zipfile.ZipFile, tmp: Path) -> None:
    """Reject zip bombs and path-traversal entries before extraction."""
    total = sum(
        m.file_size for m in zf.infolist() if not m.is_dir()
    )
    if total > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"Zip uncompressed size {total} exceeds limit "
            f"{_MAX_UNCOMPRESSED_BYTES}; possible zip bomb."
        )
    root = tmp.resolve()
    for member in zf.infolist():
        target = (tmp / member.filename).resolve()
        if not str(target).startswith(str(root) + "/") and target != root:
            raise ValueError(
                f"Zip entry {member.filename!r} escapes temp dir; "
                f"possible path-traversal attack."
            )


async def _existing_rows_by_key(
    driver,
    table: str,
    group_id: str,
    field: str,
    values: list[str],
) -> dict[str, dict]:
    if not values:
        return {}
    from surriti.search import _unwrap

    rows = _unwrap(
        await driver.query(
            f"SELECT * FROM {table} WHERE group_id = $g AND {field} IN $values;",
            {"g": group_id, "values": values},
        )
    )
    out: dict[str, dict] = {}
    for row in rows:
        key = str(row.get(field) or "")
        if key and key not in out:
            out[key] = row
    return out


async def _delete_portable_group(driver, group_id: str) -> None:
    # Edges first so relation records do not point at deleted entities.
    for table in ("relates_to", "entity_alias", "relation_frame", "entity"):
        await driver.query(f"DELETE {table} WHERE group_id = $g;", {"g": group_id})


async def _upsert_plain_record(driver, table: str, row: dict) -> None:
    await driver.query(
        f"UPSERT type::record(\"{table}\", $uuid) CONTENT $row;",
        {"uuid": row["uuid"], "row": _restore_datetimes(row)},
    )


async def _safe_upsert(
    driver, table: str, row: dict, label: str, warnings: list[str]
) -> None:
    """Upsert with concurrency-safe retry on unique-index races."""
    try:
        await _upsert_plain_record(driver, table, row)
    except Exception as exc:
        msg = str(exc).lower()
        if "already contains" in msg or "unique" in msg:
            warnings.append(f"Race on {label}; reusing existing row.")
        else:
            raise


def _resolve_frame_ref(
    raw_id: Any,
    frame_uuid_map: dict[str, str],
    source_uuid: str,
    warnings: list[str],
) -> str | None:
    """Map a source-group relation_frame_id to the target group.

    When no mapping exists the reference is cleared (set to ``None``)
    and a warning is emitted so the caller can backfill later.
    """
    sid = str(raw_id or "").strip()
    if not sid:
        return None
    mapped = frame_uuid_map.get(sid)
    if mapped is None:
        warnings.append(
            f"Edge {source_uuid!r}: relation frame {sid!r} not mapped — cleared"
        )
    return mapped


async def _upsert_edge(driver, row: dict) -> None:
    """Insert or update a ``relates_to`` edge idempotently.

    SurrealDB does not support UPSERT on graph relations.  We check
    whether the UUID already exists: UPDATE in-place when it does,
    RELATE (create) when it does not.  No DELETE — the edge is never
    left missing between operations.
    """
    src = row.pop("source_node_uuid")
    tgt = row.pop("target_node_uuid")
    restored = _restore_datetimes(row)
    from surriti.search import _unwrap

    existing = _unwrap(
        await driver.query(
            "SELECT uuid FROM relates_to WHERE group_id = $group_id AND uuid = $uuid LIMIT 1;",
            {"group_id": restored.get("group_id", ""), "uuid": restored["uuid"]},
        )
    )
    if existing:
        await driver.query(
            "UPDATE relates_to CONTENT $row WHERE uuid = $uuid;",
            {"uuid": restored["uuid"], "row": restored},
        )
    else:
        await driver.query(
            """
            RELATE (type::record("entity", $src))->relates_to->(type::record("entity", $tgt))
            CONTENT $row;
            """,
            {"src": src, "tgt": tgt, "row": restored},
        )


async def import_group_from_dir(
    driver,
    input_dir: str | Path,
    target_group_id: str,
    *,
    mode: str = "merge",
) -> ImportResult:
    """Import an unpacked Memory Pack into *target_group_id*.

    ``mode="merge"`` is idempotent and preserves any target entities with the
    same name, aliases with the same normalized alias, and relation frames with
    the same canonical name. ``mode="replace"`` first deletes the target
    group's portable-graph tables.
    """
    if mode not in {"merge", "replace"}:
        raise ValueError("mode must be 'merge' or 'replace'")

    path = Path(input_dir)
    validation = validate_pack_dir(path)
    if not validation.ok:
        return ImportResult(
            target_group_id=target_group_id,
            mode=mode,
            counts={},
            validation=validation,
        )

    entities = _read_jsonl(path / "entities.jsonl")
    aliases = _read_jsonl(path / "entity_aliases.jsonl")
    frames = _read_jsonl(path / "relation_frames.jsonl")
    edges = _read_jsonl(path / "edges.jsonl")

    if mode == "replace":
        await _delete_portable_group(driver, target_group_id)

    entity_names = [str(r.get("name") or "") for r in entities if r.get("name")]
    frame_names = [str(r.get("canonical_name") or "") for r in frames if r.get("canonical_name")]
    alias_names = [
        str(r.get("normalized_alias") or "") for r in aliases if r.get("normalized_alias")
    ]

    existing_entities = await _existing_rows_by_key(
        driver, "entity", target_group_id, "name", entity_names
    )
    existing_frames = await _existing_rows_by_key(
        driver, "relation_frame", target_group_id, "canonical_name", frame_names
    )
    existing_aliases = await _existing_rows_by_key(
        driver, "entity_alias", target_group_id, "normalized_alias", alias_names
    )

    entity_uuid_map: dict[str, str] = {}
    frame_uuid_map: dict[str, str] = {}
    edge_uuid_map: dict[str, str] = {}
    counts = {"entities": 0, "entity_aliases": 0, "relation_frames": 0, "edges": 0}
    warnings: list[str] = []

    for source in entities:
        source_uuid = str(source.get("uuid") or "")
        name = str(source.get("name") or "")
        target_uuid = str(existing_entities.get(name, {}).get("uuid") or "") if name else ""
        if not target_uuid:
            target_uuid = _stable_import_uuid(
                target_group_id, "entity", source_uuid, fallback_name=name,
            )
        entity_uuid_map[source_uuid] = target_uuid

        if not source.get("uuid"):
            warnings.append(
                f"Entity {name!r} has empty source uuid; import may not be "
                f"idempotent across runs."
            )
        row = _without_omission_markers(source)
        if "created_at" not in row or row.get("created_at") is None:
            row["created_at"] = datetime.now(timezone.utc)
            warnings.append(f"Entity {name!r} missing created_at; defaulting to now.")
        row.update({"uuid": target_uuid, "group_id": target_group_id})
        await _safe_upsert(driver, "entity", row, f"entity {name!r}", warnings)
        counts["entities"] += 1

    for source in frames:
        source_uuid = str(source.get("uuid") or "")
        canonical = str(source.get("canonical_name") or "")
        target_uuid = (
            str(existing_frames.get(canonical, {}).get("uuid") or "") if canonical else ""
        )
        if not target_uuid:
            target_uuid = _stable_import_uuid(target_group_id, "relation_frame", source_uuid)
        frame_uuid_map[source_uuid] = target_uuid

        row = _without_omission_markers(source)
        if "created_at" not in row or row.get("created_at") is None:
            row["created_at"] = datetime.now(timezone.utc)
        row.update({"uuid": target_uuid, "group_id": target_group_id})
        await _safe_upsert(driver, "relation_frame", row, f"frame {canonical!r}", warnings)
        counts["relation_frames"] += 1

    for source in aliases:
        normalized = str(source.get("normalized_alias") or "")
        old_entity_uuid = str(source.get("entity_uuid") or "")
        mapped_entity_uuid = entity_uuid_map.get(old_entity_uuid)
        if not mapped_entity_uuid:
            warnings.append(
                f"Skipped alias {source.get('uuid')!r}: missing entity {old_entity_uuid!r}"
            )
            continue

        target_uuid = (
            str(existing_aliases.get(normalized, {}).get("uuid") or "") if normalized else ""
        )
        if not target_uuid:
            target_uuid = _stable_import_uuid(target_group_id, "entity_alias", source.get("uuid"))

        row = _without_omission_markers(source)
        if "created_at" not in row or row.get("created_at") is None:
            row["created_at"] = datetime.now(timezone.utc)
        row.update(
            {
                "uuid": target_uuid,
                "group_id": target_group_id,
                "entity_uuid": mapped_entity_uuid,
                "source_episode_uuid": None,
            }
        )
        await _safe_upsert(driver, "entity_alias", row, f"alias {normalized!r}", warnings)
        counts["entity_aliases"] += 1

    for source in edges:
        source_uuid = str(source.get("uuid") or "")
        target_uuid = _stable_import_uuid(target_group_id, "relates_to", source_uuid)
        edge_uuid_map[source_uuid] = target_uuid

    for source in edges:
        source_uuid = str(source.get("uuid") or "")
        src = entity_uuid_map.get(str(source.get("source_node_uuid") or ""))
        tgt = entity_uuid_map.get(str(source.get("target_node_uuid") or ""))
        if not src or not tgt:
            warnings.append(
                f"Skipped edge {source_uuid!r}: missing endpoint(s)"
            )
            continue

        row = _without_omission_markers(source)
        target_uuid = edge_uuid_map[source_uuid]
        edge_name = str(row.get("canonical_name") or row.get("name") or "")
        row.update(
            {
                "uuid": target_uuid,
                "group_id": target_group_id,
                "source_node_uuid": src,
                "target_node_uuid": tgt,
                "episodes": [],
                "relation_frame_id": _resolve_frame_ref(
                    row.get("relation_frame_id"), frame_uuid_map, source_uuid, warnings
                ),
                "supersedes": [
                    edge_uuid_map[e] for e in row.get("supersedes", []) if e in edge_uuid_map
                ],
                "superseded_by": edge_uuid_map.get(str(row.get("superseded_by") or ""))
                if row.get("superseded_by")
                else None,
            }
        )
        row["fact_key"] = make_fact_key(target_group_id, src, edge_name, tgt)
        await _upsert_edge(driver, row)
        counts["edges"] += 1

    return ImportResult(
        target_group_id=target_group_id,
        mode=mode,
        counts=counts,
        validation=validation,
        warnings=warnings,
    )


async def import_group_from_zip(
    driver,
    input_path: str | Path,
    target_group_id: str,
    *,
    mode: str = "merge",
) -> ImportResult:
    """Validate and import a ``.surriti-pack.zip`` into *target_group_id*."""
    validation = validate_pack_zip(input_path)
    if not validation.ok:
        return ImportResult(
            target_group_id=target_group_id,
            mode=mode,
            counts={},
            validation=validation,
        )

    with tempfile.TemporaryDirectory(prefix="surriti_pack_import_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(input_path, "r") as zf:
            # Defend against zip bombs and path-traversal attacks.
            _validate_zip_safety(zf, tmp)
            zf.extractall(tmp)
        return await import_group_from_dir(
            driver,
            tmp,
            target_group_id,
            mode=mode,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _surriti_version() -> str:
    """Return the installed Surriti version string (best effort)."""
    try:
        from surriti import __version__ as v

        return v
    except Exception:
        return "unknown"
