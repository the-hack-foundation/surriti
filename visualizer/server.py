"""D3 visualizer server for Surriti's SurrealDB graph.

All filter parameters compose AND-style at the SurrealDB query level via
parameterized bindings -- never f-string interpolation -- and every
``EntityEdge``/``EntityNode`` field already on disk is surfaced on the
wire so the client can render without fabricating data.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
log = logging.getLogger("surriti.visualizer")

load_dotenv()
SURRITI_API_KEY = (os.environ.get("SURRITI_API_KEY") or "").strip()
ALLOW_INSECURE_LOCAL = os.environ.get("SURRITI_ALLOW_INSECURE_LOCAL", "1") == "1"


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix):].strip()
    return ""


def _is_authorized(request: Request) -> bool:
    client_host = request.client.host if request.client else None
    if ALLOW_INSECURE_LOCAL and _is_loopback_host(client_host):
        return True
    if not SURRITI_API_KEY:
        return False
    x_api_key = (request.headers.get("x-api-key") or request.query_params.get("api_key") or "").strip()
    bearer = _extract_bearer_token(request.headers.get("authorization"))
    presented = x_api_key or bearer
    return bool(presented) and secrets.compare_digest(presented, SURRITI_API_KEY)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _compact_raw(row: dict[str, Any]) -> dict[str, Any]:
    """Strip embeddings from the wire payload but keep every other field."""
    return {
        key: _jsonable(value)
        for key, value in row.items()
        if not key.endswith("_embedding")
    }


def _record_id(value: Any) -> str:
    if value is None:
        return ""
    raw = value if isinstance(value, str) else str(value)
    if ":" in raw:
        return raw.rsplit(":", 1)[-1].strip("`'\"")
    return raw.strip("`'\"")


def _unwrap(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, dict):
        if "result" in rows:
            return list(rows["result"] or [])
        return [rows]
    if not isinstance(rows, list):
        log.warning("unexpected SurrealDB result shape: %s", type(rows).__name__)
        return list(rows)
    if not rows:
        return []
    if all(isinstance(row, dict) and "result" not in row for row in rows):
        return list(rows)
    last = rows[-1]
    if isinstance(last, dict) and "result" in last:
        return list(last["result"] or [])
    if isinstance(last, list):
        return list(last)
    return list(rows)


def _node(table: str, row: dict[str, Any], *, fallback_name: str) -> dict[str, Any]:
    uuid = str(row.get("uuid") or _record_id(row.get("id")) or fallback_name)
    name = str(row.get("name") or fallback_name)
    labels = list(row.get("labels") or [])
    if table == "entity" and "Entity" not in labels:
        labels.insert(0, "Entity")
    if table == "episode":
        labels = ["Episode"]
    if table == "community":
        labels = ["Community"]
    return {
        "id": uuid,
        "uuid": uuid,
        "table": table,
        "kind": table,
        "name": name,
        "label": labels[0] if labels else table.title(),
        "labels": labels,
        "group_id": row.get("group_id") or "",
        "summary": row.get("summary") or "",
        # ``content`` is omitted from the bulk payload to keep responses
        # small; the transcript modal hits ``/api/episode/{uuid}`` lazily.
        "source": row.get("source") or "",
        "source_description": row.get("source_description") or "",
        "reference_time": _jsonable(row.get("reference_time")),
        "created_at": _jsonable(row.get("created_at")),
        "attributes": _jsonable(row.get("attributes") or {}),
        "entity_edges": list(row.get("entity_edges") or []),
        "raw": _compact_raw(row),
    }


def _edge(
    table: str,
    row: dict[str, Any],
    *,
    source: str,
    target: str,
    name: str,
) -> dict[str, Any]:
    uuid = str(row.get("uuid") or f"{table}:{source}:{target}:{name}")
    canonical_name = row.get("canonical_name") or ""
    return {
        "id": uuid,
        "uuid": uuid,
        "table": table,
        "kind": table,
        "source": source,
        "target": target,
        "name": name,
        "canonical_name": canonical_name,
        # Display label preference: canonical_name > name > table.
        "label": canonical_name or name or table,
        "fact": row.get("fact") or "",
        "group_id": row.get("group_id") or "",
        "episodes": list(row.get("episodes") or []),
        # Temporal
        "valid_at": _jsonable(row.get("valid_at")),
        "invalid_at": _jsonable(row.get("invalid_at")),
        "expired_at": _jsonable(row.get("expired_at")),
        "created_at": _jsonable(row.get("created_at")),
        # State / provenance
        "status": row.get("status") or "",
        "polarity": row.get("polarity") or "",
        "source_type": row.get("source_type") or "",
        "confidence": row.get("confidence"),
        "temporal": bool(row.get("temporal") or False),
        "singleton": bool(row.get("singleton") or False),
        "domain": row.get("domain") or "",
        "fact_key": row.get("fact_key") or "",
        # Frame + structure
        "relation_frame_id": row.get("relation_frame_id") or "",
        "qualifiers": _jsonable(row.get("qualifiers") or {}),
        "roles": _jsonable(row.get("roles") or {}),
        # Lineage
        "supersedes": list(row.get("supersedes") or []),
        "superseded_by": row.get("superseded_by") or "",
        "conflict_group_id": row.get("conflict_group_id") or "",
        "derived": bool(row.get("derived") or False),
        "derived_from": row.get("derived_from") or "",
        "attributes": _jsonable(row.get("attributes") or {}),
        "raw": _compact_raw(row),
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class VisualizerState:
    def __init__(self) -> None:
        self.db: Any | None = None


state = VisualizerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from surrealdb import AsyncSurreal

        url = os.environ.get("SURRITI_SURREAL_URL", "ws://localhost:8000/rpc")
        namespace = os.environ.get("SURRITI_SURREAL_NS", "myapp")
        database = os.environ.get("SURRITI_SURREAL_DB", "myapp")
        username = os.environ.get("SURRITI_SURREAL_USER")
        password = os.environ.get("SURRITI_SURREAL_PASS")
        if not username or not password:
            raise RuntimeError(
                "Missing SURRITI_SURREAL_USER/SURRITI_SURREAL_PASS for visualizer DB connection."
            )

        db = AsyncSurreal(url)
        await db.connect()
        if username and password:
            await db.signin({"username": username, "password": password})
        await db.use(namespace, database)
        state.db = db
        log.info("Visualizer connected to %s ns=%s db=%s", url, namespace, database)
    except Exception as exc:
        state.db = None
        log.warning("Visualizer could not connect to SurrealDB: %s", exc)
    yield
    if state.db is not None:
        await state.db.close()


app = FastAPI(title="Surriti Visualizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api") and not _is_authorized(request):
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"connected": state.db is not None}


async def _query(surql: str, variables: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if state.db is None:
        raise HTTPException(status_code=503, detail="SurrealDB is not connected")
    try:
        return _unwrap(await state.db.query(surql, variables or {}))
    except Exception as exc:
        if "does not exist" in str(exc).lower():
            return []
        raise


# ---------------------------------------------------------------------------
# Filter compiler
# ---------------------------------------------------------------------------


def _parse_iso(name: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid {name}: {exc}") from exc


def _build_relates_filter(
    *,
    group_id: str | None,
    include_invalid: bool,
    as_of: datetime | None,
    status: list[str] | None,
    source_type: list[str] | None,
    canonical_name: list[str] | None,
    conflict_only: bool,
    derived_only: bool,
    edge_visibility: str,
    min_confidence: float | None,
    valid_after: datetime | None,
    valid_before: datetime | None,
) -> tuple[str, dict[str, Any]]:
    """Translate the request filters into a SurrealDB WHERE clause + bindings.

    Every value travels as a ``$placeholder`` -- no f-string SQL.
    """

    clauses: list[str] = []
    params: dict[str, Any] = {}
    if group_id:
        clauses.append("group_id = $group_id")
        params["group_id"] = group_id
    if not include_invalid:
        clauses.append("(invalid_at IS NONE AND expired_at IS NONE)")
    if as_of is not None:
        clauses.append("(valid_at IS NONE OR valid_at <= $as_of)")
        clauses.append("(invalid_at IS NONE OR invalid_at > $as_of)")
        params["as_of"] = as_of
    if status:
        clauses.append("status IN $status")
        params["status"] = status
    if source_type:
        clauses.append("source_type IN $source_type")
        params["source_type"] = source_type
    if canonical_name:
        clauses.append("(canonical_name IN $canonical OR name IN $canonical)")
        params["canonical"] = [c.lower() for c in canonical_name]
    if conflict_only or edge_visibility == "conflicts":
        clauses.append("conflict_group_id IS NOT NONE AND conflict_group_id != ''")
    if derived_only or edge_visibility == "derived":
        clauses.append("derived = true")
    if edge_visibility == "non_derived":
        clauses.append("(derived IS NONE OR derived = false)")
    if edge_visibility == "invalidated":
        clauses.append("(invalid_at IS NOT NONE OR expired_at IS NOT NONE OR status = 'superseded')")
    if edge_visibility == "active":
        clauses.append("(invalid_at IS NONE AND expired_at IS NONE)")
        clauses.append("(status IS NONE OR status = '' OR status = 'active')")
    if min_confidence is not None:
        clauses.append("(confidence IS NONE OR confidence >= $min_conf)")
        params["min_conf"] = float(min_confidence)
    if valid_after is not None:
        clauses.append("(valid_at IS NONE OR valid_at >= $valid_after)")
        params["valid_after"] = valid_after
    if valid_before is not None:
        clauses.append("(valid_at IS NONE OR valid_at <= $valid_before)")
        params["valid_before"] = valid_before
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/groups")
async def groups() -> dict[str, Any]:
    rows = await _query(
        "SELECT group_id, count() AS count FROM entity GROUP BY group_id;"
    )
    return {"groups": [_jsonable(row) for row in rows if row.get("group_id") is not None]}


@app.get("/api/graph")
async def graph(
    group_id: str | None = Query(default=None),
    include_invalid: bool = Query(default=True),
    limit: int = Query(default=1500, ge=50, le=10000),
    aggregate_mentions: bool = Query(default=True),
    as_of: str | None = Query(default=None),
    view: str = Query(default="truth", pattern="^(truth|raw|provenance|conflicts|timeline|frames|full|entities|episodes)$"),
    status: list[str] | None = Query(default=None),
    source_type: list[str] | None = Query(default=None),
    canonical_name: list[str] | None = Query(default=None),
    conflict_only: bool = Query(default=False),
    derived_only: bool = Query(default=False),
    edge_visibility: str = Query(default="all", pattern="^(all|active|invalidated|conflicts|derived|non_derived)$"),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    valid_after: str | None = Query(default=None),
    valid_before: str | None = Query(default=None),
    ego_uuid: str | None = Query(default=None),
    ego_hops: int = Query(default=1, ge=1, le=2),
) -> dict[str, Any]:
    """Build the graph payload.

    ``view`` controls which tables participate:

    Semantic lenses (``truth``, ``raw``, ``provenance``, ``conflicts``,
    ``timeline``, ``frames``) are normalized to the legacy table projections
    below before querying.

    - ``full``: episodes + entities + communities, with
      ``mentions``, ``relates_to``, ``has_member`` edges.
    - ``entities``: hide episode nodes and the ``mentions`` edge entirely;
      ``relates_to.episodes`` still travels on the wire so the inspector
      can click through to the source episodes.
    - ``episodes``: episodes + their mentioned entities + ``mentions`` edges,
      hiding entity-to-entity ``relates_to``.
    """

    parsed_as_of = _parse_iso("as_of", as_of)
    parsed_after = _parse_iso("valid_after", valid_after)
    parsed_before = _parse_iso("valid_before", valid_before)

    where_group = "WHERE group_id = $group_id" if group_id else ""
    base_params: dict[str, Any] = {"limit": limit}
    if group_id:
        base_params["group_id"] = group_id

    view_map = {
        "truth": "entities",
        "raw": "full",
        "provenance": "full",
        "conflicts": "entities",
        "timeline": "entities",
        "frames": "entities",
    }
    api_view = view_map.get(view, view)
    if view == "truth" and edge_visibility == "all":
        edge_visibility = "active"
        status = status or ["active"]
        include_invalid = False
    elif view == "conflicts":
        edge_visibility = "conflicts"
        status = status or ["needs_resolution"]
    elif view == "raw":
        include_invalid = True

    rel_where, rel_params = _build_relates_filter(
        group_id=group_id,
        include_invalid=include_invalid,
        as_of=parsed_as_of,
        status=status,
        source_type=source_type,
        canonical_name=canonical_name,
        conflict_only=conflict_only,
        derived_only=derived_only,
        edge_visibility=edge_visibility,
        min_confidence=min_confidence,
        valid_after=parsed_after,
        valid_before=parsed_before,
    )
    rel_params["limit"] = limit

    want_entities = api_view in ("full", "entities")
    want_episodes = api_view in ("full", "episodes")
    want_relates = api_view in ("full", "entities")
    want_mentions = api_view in ("full", "episodes")

    entities = await _query(
        f"SELECT * FROM entity {where_group} LIMIT $limit;", base_params
    ) if want_entities else []
    episodes = await _query(
        f"SELECT * FROM episode {where_group} LIMIT $limit;", base_params
    ) if want_episodes else []
    communities = await _query(
        f"SELECT * FROM community {where_group} LIMIT $limit;", base_params
    ) if want_entities else []
    relates = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM relates_to {rel_where} LIMIT $limit;
        """,
        rel_params,
    ) if want_relates else []
    mentions = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM mentions {where_group} LIMIT $limit;
        """,
        base_params,
    ) if want_mentions else []
    members = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM has_member {where_group} LIMIT $limit;
        """,
        base_params,
    ) if want_entities else []

    nodes: dict[str, dict[str, Any]] = {}
    for row in entities:
        node = _node("entity", row, fallback_name="Entity")
        nodes[node["id"]] = node
    for row in episodes:
        node = _node("episode", row, fallback_name="Episode")
        nodes[node["id"]] = node
    for row in communities:
        node = _node("community", row, fallback_name="Community")
        nodes[node["id"]] = node

    links: list[dict[str, Any]] = []
    for row in relates:
        source = str(row.get("source_node_uuid") or row.get("source_uuid") or _record_id(row.get("in")))
        target = str(row.get("target_node_uuid") or row.get("target_uuid") or _record_id(row.get("out")))
        if source in nodes and target in nodes:
            links.append(_edge("relates_to", row, source=source, target=target,
                               name=row.get("name") or "relates_to"))

    if aggregate_mentions and mentions:
        bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in mentions:
            source = str(row.get("source_uuid") or _record_id(row.get("in")))
            target = str(row.get("target_uuid") or _record_id(row.get("out")))
            if source in nodes and target in nodes:
                bucket.setdefault((source, target), []).append(row)
        for (source, target), rows in bucket.items():
            primary = rows[0]
            link = _edge("mentions", primary, source=source, target=target, name="mentions")
            link["count"] = len(rows)
            link["aggregated"] = True
            link["uuids"] = [str(r.get("uuid")) for r in rows if r.get("uuid")]
            links.append(link)
    else:
        for row in mentions:
            source = str(row.get("source_uuid") or _record_id(row.get("in")))
            target = str(row.get("target_uuid") or _record_id(row.get("out")))
            if source in nodes and target in nodes:
                links.append(_edge("mentions", row, source=source, target=target, name="mentions"))

    for row in members:
        source = str(row.get("source_uuid") or _record_id(row.get("in")))
        target = str(row.get("target_uuid") or _record_id(row.get("out")))
        if source in nodes and target in nodes:
            links.append(_edge("has_member", row, source=source, target=target, name="has_member"))

    # ------------------------------------------------------------------
    # Ego clamp -- LOD mode used by the "Ego" view in the visualizer.
    # We BFS from ``ego_uuid`` over the already-built link list (cheap,
    # all the filtering is done) and prune nodes/links that fall outside
    # the requested hop radius. Edges anchored at unknown nodes are
    # dropped so the canvas stays clean.
    # ------------------------------------------------------------------
    if ego_uuid and ego_uuid in nodes:
        adjacency: dict[str, set[str]] = {}
        for link in links:
            adjacency.setdefault(link["source"], set()).add(link["target"])
            adjacency.setdefault(link["target"], set()).add(link["source"])
        keep: set[str] = {ego_uuid}
        frontier: set[str] = {ego_uuid}
        for _ in range(ego_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for nb in adjacency.get(nid, ()):
                    if nb not in keep:
                        keep.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        nodes = {nid: n for nid, n in nodes.items() if nid in keep}
        links = [
            link for link in links
            if link["source"] in nodes and link["target"] in nodes
        ]

    degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for link in links:
        degree[link["source"]] = degree.get(link["source"], 0) + 1
        degree[link["target"]] = degree.get(link["target"], 0) + 1
    for node_id, count in degree.items():
        nodes[node_id]["degree"] = count

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "meta": {
            "view": view,
            "api_view": api_view,
            "group_id": group_id,
            "include_invalid": include_invalid,
            "aggregate_mentions": aggregate_mentions,
            "as_of": parsed_as_of.isoformat() if parsed_as_of else None,
            "filters": {
                "status": status or [],
                "source_type": source_type or [],
                "canonical_name": canonical_name or [],
                "conflict_only": conflict_only,
                "derived_only": derived_only,
                "edge_visibility": edge_visibility,
                "min_confidence": min_confidence,
                "valid_after": parsed_after.isoformat() if parsed_after else None,
                "valid_before": parsed_before.isoformat() if parsed_before else None,
            },
            "node_count": len(nodes),
            "link_count": len(links),
            "generated_at": _utcnow_iso(),
        },
    }


@app.get("/api/timeline_bounds")
async def timeline_bounds(
    group_id: str | None = Query(default=None),
) -> dict[str, Any]:
    where = "WHERE group_id = $group_id" if group_id else ""
    params: dict[str, Any] = {"group_id": group_id} if group_id else {}
    rows = await _query(
        f"""
        SELECT
            math::min(valid_at) AS min_valid,
            math::max(valid_at) AS max_valid,
            math::max(invalid_at) AS max_invalid,
            math::max(created_at) AS max_created
        FROM relates_to {where} GROUP ALL;
        """,
        params,
    )
    row = rows[0] if rows else {}
    upper_candidates = [row.get("max_valid"), row.get("max_invalid"), row.get("max_created")]
    upper = max((d for d in upper_candidates if d is not None), default=None)
    return {
        "min": _jsonable(row.get("min_valid")),
        "max": _jsonable(upper),
        "now": _utcnow_iso(),
        "group_id": group_id,
    }


@app.get("/api/conflicts")
async def conflicts(
    group_id: str | None = Query(default=None),
    conflict_group_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    """Group ``relates_to`` edges by ``conflict_group_id``.

    A group is returned when *any* member has ``status="needs_resolution"``.
    Subject + frame metadata are joined client-side so the inspector can
    badge each group with its directionality / cardinality / policy.
    """

    clauses = ["conflict_group_id IS NOT NONE", "conflict_group_id != ''"]
    params: dict[str, Any] = {"limit": limit}
    if group_id:
        clauses.append("group_id = $group_id")
        params["group_id"] = group_id
    if conflict_group_id:
        clauses.append("conflict_group_id = $cg")
        params["cg"] = conflict_group_id
    where = "WHERE " + " AND ".join(clauses)
    rows = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM relates_to {where} LIMIT $limit;
        """,
        params,
    )

    groups_acc: dict[str, dict[str, Any]] = {}
    subject_uuids: set[str] = set()
    for row in rows:
        cg = str(row.get("conflict_group_id") or "")
        if not cg:
            continue
        source = str(row.get("source_uuid") or _record_id(row.get("in")))
        target = str(row.get("target_uuid") or _record_id(row.get("out")))
        edge = _edge("relates_to", row, source=source, target=target,
                     name=row.get("name") or "relates_to")
        bucket = groups_acc.setdefault(cg, {
            "conflict_group_id": cg,
            "edges": [],
            "subject_uuid": source,
            "needs_resolution": False,
            "canonical_name": edge["canonical_name"] or edge["name"],
        })
        bucket["edges"].append(edge)
        if edge["status"] == "needs_resolution":
            bucket["needs_resolution"] = True
        subject_uuids.add(source)

    # Only return groups that actually contain a needs_resolution edge.
    out_groups = [g for g in groups_acc.values() if g["needs_resolution"]]

    subjects: dict[str, dict[str, Any]] = {}
    if subject_uuids:
        subj_rows = await _query(
            "SELECT * FROM entity WHERE uuid IN $uuids;",
            {"uuids": list(subject_uuids)},
        )
        for r in subj_rows:
            n = _node("entity", r, fallback_name="Entity")
            subjects[n["id"]] = {"uuid": n["id"], "name": n["name"], "group_id": n["group_id"]}

    for g in out_groups:
        g["subject"] = subjects.get(g["subject_uuid"]) or {"uuid": g["subject_uuid"]}

    return {
        "groups": out_groups,
        "count": len(out_groups),
        "generated_at": _utcnow_iso(),
    }


@app.get("/api/entity/{uuid}/timeline")
async def entity_timeline(
    uuid: str,
    group_id: str | None = Query(default=None),
    predicate: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    """Chronological view of every fact where ``uuid`` is the subject.

    Includes target stubs and the full edge so the client can render a
    ``supersedes`` / ``superseded_by`` chain.
    """

    clauses = ["record::id(in) = $uuid"]
    params: dict[str, Any] = {"uuid": uuid, "limit": limit}
    if group_id:
        clauses.append("group_id = $group_id")
        params["group_id"] = group_id
    if predicate:
        clauses.append("(name = $pred OR canonical_name = $pred)")
        params["pred"] = predicate.lower()
    where = "WHERE " + " AND ".join(clauses)
    rows = await _query(
        f"""
        SELECT *,
            record::id(in) AS source_uuid,
            record::id(out) AS target_uuid,
            time::unix(valid_at OR created_at) AS sort_ts
        FROM relates_to {where}
        ORDER BY sort_ts
        LIMIT $limit;
        """,
        params,
    )

    target_uuids = {str(r.get("target_uuid") or _record_id(r.get("out"))) for r in rows}
    target_uuids.discard("")
    targets: dict[str, dict[str, Any]] = {}
    if target_uuids:
        t_rows = await _query(
            "SELECT * FROM entity WHERE uuid IN $uuids;",
            {"uuids": list(target_uuids)},
        )
        for r in t_rows:
            n = _node("entity", r, fallback_name="Entity")
            targets[n["id"]] = {"uuid": n["id"], "name": n["name"]}

    subj_rows = await _query("SELECT * FROM entity WHERE uuid = $uuid;", {"uuid": uuid})
    subject = _node("entity", subj_rows[0], fallback_name="Entity") if subj_rows else None

    events: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_uuid") or _record_id(row.get("in")))
        target = str(row.get("target_uuid") or _record_id(row.get("out")))
        edge = _edge("relates_to", row, source=source, target=target,
                     name=row.get("name") or "relates_to")
        edge["target"] = targets.get(target) or {"uuid": target}
        events.append(edge)

    return {
        "subject": subject,
        "events": events,
        "count": len(events),
        "generated_at": _utcnow_iso(),
    }


@app.get("/api/entity/{uuid}/profile")
async def entity_profile(
    uuid: str,
    group_id: str | None = Query(default=None),
    fact_limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """Dossier payload for a single entity.

    Returns the canonical name, alias chips, dossier summary,
    salience / mention metadata, and the most recent valid facts where
    the entity is on either side. This is the data the visualizer's
    Dossier panel renders.
    """

    e_clauses = ["uuid = $uuid"]
    e_params: dict[str, Any] = {"uuid": uuid}
    if group_id:
        e_clauses.append("group_id = $group_id")
        e_params["group_id"] = group_id
    e_rows = await _query(
        f"SELECT * FROM entity WHERE {' AND '.join(e_clauses)} LIMIT 1;",
        e_params,
    )
    if not e_rows:
        raise HTTPException(status_code=404, detail="entity not found")
    row = e_rows[0]

    # Aliases that resolve TO this entity.
    a_clauses = ["entity_uuid = $uuid"]
    a_params: dict[str, Any] = {"uuid": uuid}
    if group_id:
        a_clauses.append("group_id = $group_id")
        a_params["group_id"] = group_id
    aliases = await _query(
        f"SELECT alias, normalized_alias, confidence, source_episode_uuid "
        f"FROM entity_alias WHERE {' AND '.join(a_clauses)};",
        a_params,
    )

    # Top facts (recency-ordered, both directions).
    f_clauses = ["(record::id(in) = $uuid OR record::id(out) = $uuid)"]
    f_params: dict[str, Any] = {"uuid": uuid, "limit": fact_limit}
    if group_id:
        f_clauses.append("group_id = $group_id")
        f_params["group_id"] = group_id
    f_clauses.append("(invalid_at IS NONE OR invalid_at = NONE)")
    facts = await _query(
        f"""
        SELECT *,
            record::id(in) AS source_uuid,
            record::id(out) AS target_uuid,
            time::unix(valid_at OR created_at) AS sort_ts
        FROM relates_to WHERE {' AND '.join(f_clauses)}
        ORDER BY sort_ts DESC
        LIMIT $limit;
        """,
        f_params,
    )

    return {
        "uuid": str(row.get("uuid") or uuid),
        "group_id": row.get("group_id"),
        "name": row.get("name"),
        "canonical_name": row.get("canonical_name") or row.get("name"),
        "aliases": list(row.get("aliases") or []),
        "alias_records": [_jsonable(a) for a in aliases],
        "labels": list(row.get("labels") or []),
        "summary": row.get("summary") or "",
        "profile_summary": row.get("profile_summary") or "",
        "salience": row.get("salience") or 0,
        "mention_count": row.get("mention_count") or 0,
        "last_seen_at": _jsonable(row.get("last_seen_at")),
        "merged_into": row.get("merged_into"),
        "attributes": _jsonable(row.get("attributes") or {}),
        "facts": [_jsonable(f) for f in facts],
        "generated_at": _utcnow_iso(),
    }


@app.get("/api/episode/{uuid}")
async def episode_detail(
    uuid: str,
    group_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Full episode payload (including ``content``) for the transcript modal."""
    clauses = ["uuid = $uuid"]
    params: dict[str, Any] = {"uuid": uuid}
    if group_id:
        clauses.append("group_id = $group_id")
        params["group_id"] = group_id
    rows = await _query(
        f"SELECT * FROM episode WHERE {' AND '.join(clauses)} LIMIT 1;",
        params,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="episode not found")
    row = rows[0]
    node = _node("episode", row, fallback_name="Episode")
    node["content"] = row.get("content") or ""
    # Mention count: cheap separate query bound by uuid.
    m_rows = await _query(
        "SELECT count() AS c FROM mentions WHERE record::id(in) = $uuid GROUP ALL;",
        {"uuid": uuid},
    )
    node["mention_count"] = int(m_rows[0].get("c") or 0) if m_rows else 0
    return node


@app.get("/api/frames")
async def frames(
    group_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Frame catalogue for the filter dropdown.

    Combines the in-process ``DEFAULT_FRAMES`` (always available) with
    distinct ``canonical_name``/``name`` values observed in the live
    database. Deduped by canonical name. Each frame surfaces the
    metadata the inspector + filter UI need to render badges without
    fabricating values.
    """
    try:
        from surriti.relation_frames import DEFAULT_FRAMES
    except Exception:
        DEFAULT_FRAMES = ()  # type: ignore[assignment]

    out: dict[str, dict[str, Any]] = {}
    for frame in DEFAULT_FRAMES:
        out[frame.canonical_name.lower()] = {
            "canonical_name": frame.canonical_name,
            "aliases": list(frame.aliases),
            "directionality": frame.directionality,
            "temporal_kind": frame.temporal_kind,
            "cardinality": frame.cardinality,
            "contradiction_policy": frame.contradiction_policy,
            "inverse_name": frame.inverse_name,
            "confidence": float(frame.confidence),
            "source": "seed",
        }

    where = "WHERE group_id = $group_id" if group_id else ""
    params: dict[str, Any] = {"group_id": group_id} if group_id else {}
    try:
        rows = await _query(
            f"SELECT canonical_name, name FROM relates_to {where} GROUP BY canonical_name, name;",
            params,
        )
    except Exception as exc:
        log.warning("frames discovery query failed: %s", exc)
        rows = []
    for r in rows:
        for key in (r.get("canonical_name"), r.get("name")):
            if not key:
                continue
            kk = str(key).lower()
            if kk in out:
                continue
            out[kk] = {
                "canonical_name": str(key),
                "aliases": [],
                "directionality": "unknown",
                "temporal_kind": "unknown",
                "cardinality": "unknown",
                "contradiction_policy": "coexist",
                "inverse_name": None,
                "confidence": 0.5,
                "source": "discovered",
            }

    return {
        "frames": sorted(out.values(), key=lambda f: f["canonical_name"]),
        "count": len(out),
        "generated_at": _utcnow_iso(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "server:app",
        host=os.environ.get("VISUALIZER_HOST", "0.0.0.0"),
        port=int(os.environ.get("VISUALIZER_PORT", "1337")),
        reload=os.environ.get("VISUALIZER_RELOAD", "0") == "1",
        app_dir=str(ROOT),
    )
