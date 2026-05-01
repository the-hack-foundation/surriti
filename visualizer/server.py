"""D3 visualizer server for Surriti's SurrealDB graph."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

load_dotenv()


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
    return {
        key: _jsonable(value)
        for key, value in row.items()
        if not key.endswith("_embedding")
    }


def _record_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value
    else:
        raw = str(value)
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


def _node(
    table: str,
    row: dict[str, Any],
    *,
    fallback_name: str,
) -> dict[str, Any]:
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
        "content": row.get("content") or "",
        "source": row.get("source") or "",
        "reference_time": _jsonable(row.get("reference_time")),
        "created_at": _jsonable(row.get("created_at")),
        "attributes": _jsonable(row.get("attributes") or {}),
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
    return {
        "id": uuid,
        "uuid": uuid,
        "table": table,
        "kind": table,
        "source": source,
        "target": target,
        "name": name,
        "fact": row.get("fact") or "",
        "group_id": row.get("group_id") or "",
        "episodes": list(row.get("episodes") or []),
        "valid_at": _jsonable(row.get("valid_at")),
        "invalid_at": _jsonable(row.get("invalid_at")),
        "expired_at": _jsonable(row.get("expired_at")),
        "created_at": _jsonable(row.get("created_at")),
        "attributes": _jsonable(row.get("attributes") or {}),
        "raw": _compact_raw(row),
    }


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
        username = os.environ.get("SURRITI_SURREAL_USER", "root")
        password = os.environ.get("SURRITI_SURREAL_PASS", "root")

        db = AsyncSurreal(url)
        await db.connect()
        if username and password:
            await db.signin({"username": username, "password": password})
        await db.use(namespace, database)
        state.db = db
        print(f"Visualizer connected to {url} ns={namespace} db={database}")
    except Exception as exc:
        state.db = None
        print(f"Visualizer could not connect to SurrealDB: {exc}")
    yield
    if state.db is not None:
        await state.db.close()


app = FastAPI(title="Surriti Visualizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


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


@app.get("/api/groups")
async def groups() -> dict[str, Any]:
    rows = await _query(
        """
        SELECT group_id, count() AS count FROM entity GROUP BY group_id;
        """
    )
    return {"groups": [_jsonable(row) for row in rows if row.get("group_id") is not None]}


@app.get("/api/graph")
async def graph(
    group_id: str | None = Query(default=None),
    include_invalid: bool = Query(default=True),
    limit: int = Query(default=1500, ge=50, le=10000),
) -> dict[str, Any]:
    where_group = "WHERE group_id = $group_id" if group_id else ""
    rel_where = []
    if group_id:
        rel_where.append("group_id = $group_id")
    if not include_invalid:
        rel_where.append("(invalid_at IS NONE AND expired_at IS NONE)")
    rel_clause = ("WHERE " + " AND ".join(rel_where)) if rel_where else ""
    params = {"group_id": group_id, "limit": limit}

    entities = await _query(f"SELECT * FROM entity {where_group} LIMIT $limit;", params)
    episodes = await _query(f"SELECT * FROM episode {where_group} LIMIT $limit;", params)
    communities = await _query(f"SELECT * FROM community {where_group} LIMIT $limit;", params)
    relates = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM relates_to {rel_clause} LIMIT $limit;
        """,
        params,
    )
    mentions = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM mentions {where_group} LIMIT $limit;
        """,
        params,
    )
    members = await _query(
        f"""
        SELECT *, record::id(in) AS source_uuid, record::id(out) AS target_uuid
        FROM has_member {where_group} LIMIT $limit;
        """,
        params,
    )

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
        source = str(row.get("source_node_uuid") or _record_id(row.get("in")))
        target = str(row.get("target_node_uuid") or _record_id(row.get("out")))
        if source in nodes and target in nodes:
            links.append(_edge("relates_to", row, source=source, target=target, name=row.get("name") or "relates_to"))
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
            "group_id": group_id,
            "include_invalid": include_invalid,
            "node_count": len(nodes),
            "link_count": len(links),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.environ.get("VISUALIZER_HOST", "0.0.0.0"),
        port=int(os.environ.get("VISUALIZER_PORT", "1337")),
        reload=os.environ.get("VISUALIZER_RELOAD", "0") == "1",
        app_dir=str(ROOT),
    )
