"""Testing utilities — including an in-memory SurrealDB stand-in.

Importable as ``from surriti.testing import InMemoryDriver``. Useful for
unit tests, demos, and CI environments where running a real SurrealDB
instance is undesirable. Implements just enough of the SurrealDB query
surface used by Surriti's ingest and search pipelines to drive
end-to-end runs offline.

Not for production use.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from surriti.embedder import cosine_similarity


class InMemoryDriver:
    """In-process stand-in for :class:`surriti.SurrealDriver`."""

    embedding_dim: int = 64

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim
        self.records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    async def connect(self) -> None:  # pragma: no cover - no-op
        return None

    async def close(self) -> None:  # pragma: no cover - no-op
        return None

    async def init_schema(self) -> None:  # pragma: no cover - no-op
        return None

    async def clear(self) -> None:
        self.records.clear()

    async def query(self, surql: str, variables: dict[str, Any] | None = None):
        s = surql.strip()
        v = variables or {}
        if s.startswith('CREATE type::thing("episode"'):
            self._insert("episode", v)
            return [[{"ok": True}]]
        if s.startswith('CREATE type::thing("entity"'):
            self._insert("entity", v)
            return [[{"ok": True}]]
        if "RELATE" in s and "->relates_to->" in s:
            self._insert(
                "relates_to",
                {
                    **v,
                    "in": v["src"],
                    "out": v["tgt"],
                    "source_node_uuid": v["src"],
                    "target_node_uuid": v["tgt"],
                },
            )
            return [[{"ok": True}]]
        if "RELATE" in s and "->mentions->" in s:
            self._insert("mentions", {**v, "in": v["ep"], "out": v["en"]})
            return [[{"ok": True}]]
        if s.startswith("SELECT * FROM entity WHERE group_id"):
            names = set(v.get("names") or [])
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g") and r.get("name") in names
            ]
            return [rows]
        if "FROM relates_to" in s and "fact_embedding <|" in s:
            vec = v.get("vec")
            scored = []
            for r in self.records["relates_to"]:
                if v.get("group_id") not in (None, r.get("group_id")):
                    continue
                if not r.get("fact_embedding"):
                    continue
                scored.append((cosine_similarity(vec, r["fact_embedding"]), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [[r for _, r in scored[:10]]]
        if "FROM relates_to" in s and "fact @1@" in s:
            q = (v.get("q") or "").lower()
            tokens = [t for t in q.split() if t]
            rows = [
                r for r in self.records["relates_to"]
                if any(t in r.get("fact", "").lower() for t in tokens)
                and v.get("group_id") in (None, r.get("group_id"))
            ]
            return [rows]
        if s.startswith("SELECT * FROM episode"):
            rows = [r for r in self.records["episode"] if r.get("group_id") == v.get("g")]
            rows.sort(key=lambda r: r.get("reference_time"), reverse=True)
            return [rows[: v.get("n", 10)]]
        if s.startswith("SELECT name, content FROM episode WHERE uuid IN"):
            uuids = set(v.get("u") or [])
            rows = [r for r in self.records["episode"] if r.get("uuid") in uuids]
            return [rows]
        if s.startswith("UPDATE relates_to"):
            uuids = set(v.get("uuids") or [])
            for r in self.records["relates_to"]:
                if r.get("uuid") in uuids:
                    r["invalid_at"] = v.get("invalid_at")
                    r["expired_at"] = v.get("expired_at")
            return [[{"ok": True}]]
        return [[]]

    def _insert(self, table: str, payload: dict[str, Any]) -> None:
        record = {k: val for k, val in payload.items() if k not in {"src", "tgt", "ep", "en"}}
        self.records[table].append(record)


__all__ = ["InMemoryDriver"]
