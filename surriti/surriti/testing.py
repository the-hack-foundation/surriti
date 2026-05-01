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

    def __init__(
        self,
        embedding_dim: int = 64,
        *,
        enforce_entity_name_uniq: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.enforce_entity_name_uniq = enforce_entity_name_uniq

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
        if s.startswith('CREATE type::record("episode"'):
            self._insert("episode", v)
            return [[{"ok": True}]]
        if s.startswith('CREATE type::record("entity"'):
            if self.enforce_entity_name_uniq:
                for r in self.records["entity"]:
                    if (
                        r.get("group_id") == v.get("group_id")
                        and r.get("name") == v.get("name")
                    ):
                        raise RuntimeError(
                            "Database index entity_name_uniq already contains "
                            f"[{v.get('group_id')!r}, {v.get('name')!r}]"
                        )
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
        if s.startswith("SELECT * FROM entity WHERE group_id") and "name = $n" in s:
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g") and r.get("name") == v.get("n")
            ]
            return [rows[:1]]
        if s.startswith("SELECT * FROM entity WHERE group_id"):
            names = set(v.get("names") or [])
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g") and r.get("name") in names
            ]
            return [rows]
        if "FROM relates_to" in s and "AND in = type::record" in s and "AND out = type::record" in s:
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id")
                and r.get("source_node_uuid") == v.get("src")
                and r.get("target_node_uuid") == v.get("tgt")
                and r.get("name") == v.get("name")
                and r.get("invalid_at") is None
                and (r.get("status") in (None, "active"))
            ]
            return [rows[:10]]
        if "FROM relates_to" in s and "fact_key = $key" in s:
            # _find_equivalent_edge primary lookup.
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id")
                and r.get("fact_key") == v.get("key")
                and r.get("invalid_at") is None
            ]
            return [rows[:10]]
        if "FROM relates_to" in s and "valid_at <= $as_of" in s:
            # get_facts_as_of: edges valid at the given timestamp.
            rows = []
            for r in self.records["relates_to"]:
                if r.get("group_id") != v.get("group_id"):
                    continue
                if r.get("source_node_uuid") != v.get("src"):
                    continue
                va = r.get("valid_at")
                if va is not None and va > v.get("as_of"):
                    continue
                ia = r.get("invalid_at")
                if ia is not None and ia <= v.get("as_of"):
                    continue
                if "name" in v and r.get("name") != v.get("name"):
                    continue
                if "domain" in v and r.get("domain") != v.get("domain"):
                    continue
                rows.append(r)
            rows.sort(
                key=lambda r: (r.get("valid_at") is not None, r.get("valid_at")),
                reverse=True,
            )
            return [rows[: v.get("limit", 200)]]
        if "FROM relates_to" in s and "AND in = type::record" in s and 'status = "active"' in s:
            # Singleton-slot closer / get_current_facts: filter by
            # (group_id, subject, name, active) without constraining
            # the target node (the closer keeps the matching object).
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id")
                and r.get("source_node_uuid") == v.get("src")
                and (("name" not in v) or r.get("name") == v.get("name"))
                and (r.get("status") in (None, "active"))
                and r.get("invalid_at") is None
            ]
            if "domain" in v:
                rows = [r for r in rows if r.get("domain") == v.get("domain")]
            limit = v.get("limit", 50)
            return [rows[:limit]]
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
        if "FROM entity" in s and "name_embedding <|" in s:
            vec = v.get("vec")
            scored = []
            for r in self.records["entity"]:
                if v.get("group_id") not in (None, r.get("group_id")):
                    continue
                if not r.get("name_embedding"):
                    continue
                scored.append((cosine_similarity(vec, r["name_embedding"]), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [[r for _, r in scored[:10]]]
        if "FROM entity" in s and "name @1@" in s:
            q = (v.get("q") or "").lower()
            tokens = [t for t in q.split() if t]
            rows = [
                r for r in self.records["entity"]
                if any(t in r.get("name", "").lower() for t in tokens)
                and v.get("group_id") in (None, r.get("group_id"))
            ]
            return [rows]
        if s.startswith("SELECT * FROM episode"):
            groups = set(v.get("groups") or [])
            rows = [
                r for r in self.records["episode"]
                if (
                    (groups and r.get("group_id") in groups)
                    or (not groups and r.get("group_id") == v.get("g"))
                )
            ]
            if v.get("source") is not None:
                rows = [r for r in rows if r.get("source") == v.get("source")]
            if v.get("ref") is not None:
                rows = [r for r in rows if r.get("reference_time") <= v.get("ref")]
            rows.sort(key=lambda r: r.get("reference_time"), reverse=True)
            return [rows[: v.get("n", 10)]]
        if s.startswith("SELECT content FROM episode WHERE uuid IN") or \
                s.startswith("SELECT name, content FROM episode WHERE uuid IN"):
            uuids = set(v.get("u") or [])
            rows = [r for r in self.records["episode"] if r.get("uuid") in uuids]
            return [rows]
        if s.startswith("UPDATE relates_to") and "array::distinct" in s:
            extra = list(v.get("episodes") or [])
            for r in self.records["relates_to"]:
                if r.get("uuid") == v.get("uuid"):
                    merged = list(r.get("episodes") or [])
                    for episode_uuid in extra:
                        if episode_uuid not in merged:
                            merged.append(episode_uuid)
                    r["episodes"] = merged
            return [[{"ok": True}]]
        if s.startswith("UPDATE relates_to"):
            uuids = set(v.get("uuids") or [])
            for r in self.records["relates_to"]:
                if r.get("uuid") in uuids:
                    r["invalid_at"] = v.get("invalid_at")
                    r["expired_at"] = v.get("expired_at")
                    if 'status' in s or 'superseded' in s:
                        r["status"] = "superseded"
                        r["superseded_by"] = v.get("superseded_by")
            return [[{"ok": True}]]
        if s.startswith('UPDATE type::record("entity"'):
            for r in self.records["entity"]:
                if r.get("uuid") == v.get("uuid"):
                    if "summary" in v:
                        r["summary"] = v["summary"]
                    if "labels" in v:
                        r["labels"] = v["labels"]
                    if "attributes" in v:
                        r["attributes"] = v["attributes"]
            return [[{"ok": True}]]
        return [[]]

    def _insert(self, table: str, payload: dict[str, Any]) -> None:
        record = {k: val for k, val in payload.items() if k not in {"src", "tgt", "ep", "en"}}
        self.records[table].append(record)


__all__ = ["InMemoryDriver"]
