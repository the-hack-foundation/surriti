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
        if " IN $values" in s and "WHERE group_id = $g" in s:
            import re as _re

            _m = _re.search(r"FROM\s+(\w+)\s+WHERE.*?AND\s+(\w+)\s+IN\s+\$values", s)
            if _m:
                tbl, field = _m.group(1), _m.group(2)
                wanted = set(v.get("values") or [])
                return [[
                    r for r in self.records[tbl]
                    if r.get("group_id") == v.get("g") and r.get(field) in wanted
                ]]
        if s.startswith('CREATE type::record("episode"'):
            self._insert("episode", v)
            return [[{"ok": True}]]
        if s.startswith('CREATE type::record("entity_alias"'):
            # Enforce the (group_id, normalized_alias) unique index so
            # alias writes converge in tests just like in production.
            for r in self.records["entity_alias"]:
                if (
                    r.get("group_id") == v.get("group_id")
                    and r.get("normalized_alias") == v.get("normalized_alias")
                ):
                    raise RuntimeError(
                        "Database index entity_alias_unique already contains "
                        f"[{v.get('group_id')!r}, {v.get('normalized_alias')!r}]"
                    )
            self._insert("entity_alias", v)
            return [[{"ok": True}]]
        if s.startswith("SELECT * FROM entity_alias WHERE group_id"):
            wanted = set(v.get("aliases") or [])
            rows = [
                r for r in self.records["entity_alias"]
                if r.get("group_id") == v.get("g")
                and r.get("normalized_alias") in wanted
            ]
            return [rows]
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
            row = dict(v.get("row") or v)
            row.setdefault("uuid", v.get("uuid"))
            row["in"] = v["src"]
            row["out"] = v["tgt"]
            row["source_node_uuid"] = v["src"]
            row["target_node_uuid"] = v["tgt"]
            self._insert("relates_to", row)
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
        if s.startswith("SELECT * FROM entity") and "name = $name" in s:
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("group_id") and r.get("name") == v.get("name")
            ]
            return [rows[:1]]
        if s.startswith("SELECT * FROM entity WHERE group_id") and "uuid = $u" in s:
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g") and r.get("uuid") == v.get("u")
            ]
            return [rows[:1]]
        if s.startswith("SELECT * FROM entity") and "uuid = $uuid" in s:
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("group_id") and r.get("uuid") == v.get("uuid")
            ]
            return [rows[:1]]
        if s.startswith("SELECT * FROM entity WHERE group_id") and "uuid IN" in s:
            wanted = set(v.get("u") or [])
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g") and r.get("uuid") in wanted
            ]
            return [rows]
        if s.startswith("SELECT uuid FROM entity WHERE group_id"):
            rows = [
                {"uuid": r.get("uuid")}
                for r in self.records["entity"]
                if r.get("group_id") == v.get("g")
            ]
            return [rows]
        if s.startswith("SELECT * FROM entity WHERE group_id"):
            names = set(v.get("names") or [])
            rows = [
                r for r in self.records["entity"]
                if r.get("group_id") == v.get("g")
                and (not names or r.get("name") in names)
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
        if "FROM relates_to" in s and "uuid = $uuid" in s:
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id") and r.get("uuid") == v.get("uuid")
            ]
            return [rows[:1]]
        if "FROM relates_to" in s and "uuid IN $uuids" in s:
            wanted = set(v.get("uuids") or [])
            rows = [
                {"uuid": r.get("uuid"), "recall_count": r.get("recall_count")}
                for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id") and r.get("uuid") in wanted
            ]
            return [rows]
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
        if "FROM relates_to" in s and 'status = "needs_resolution"' in s:
            # get_conflicts: surface unresolved-conflict edges.
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("group_id")
                and r.get("status") == "needs_resolution"
            ]
            return [rows[: v.get("limit", 100)]]
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
            if 'AND name = "has_trait"' in s:
                rows = [r for r in rows if r.get("name") == "has_trait"]
            if 'AND name = "has_belief"' in s:
                rows = [r for r in rows if r.get("name") == "has_belief"]
            if 'AND name = "has_pattern"' in s:
                rows = [r for r in rows if r.get("name") == "has_pattern"]
            if "is_belief = true" in s:
                rows = [r for r in rows if r.get("is_belief") is True]
            limit = v.get("limit", 50)
            return [rows[:limit]]
        if "FROM relates_to" in s and "(in = $rec OR out = $rec)" in s:
            rec = v.get("rec", "")
            uuid_part = rec.split(":", 1)[1] if ":" in rec else rec
            rows = [
                r for r in self.records["relates_to"]
                if r.get("group_id") == v.get("g")
                and (
                    r.get("source_node_uuid") == uuid_part
                    or r.get("target_node_uuid") == uuid_part
                )
                and r.get("invalid_at") is None
            ]
            limit = int(v.get("lim") or 30)
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
        if "FROM relates_to" in s and ("fact @0@" in s or "fact @1@" in s):
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
        if "FROM entity" in s and ("name @0@" in s or "name @1@" in s):
            q = (v.get("q") or "").lower()
            tokens = [t for t in q.split() if t]
            rows = [
                r for r in self.records["entity"]
                if any(t in r.get("name", "").lower() for t in tokens)
                and v.get("group_id") in (None, r.get("group_id"))
            ]
            return [rows]
        if "SELECT count() as cnt FROM episode" in s:
            rows = [
                r for r in self.records["episode"]
                if r.get("group_id") == v.get("group_id")
                and "self_" in str(r.get("source") or "")
            ]
            return [[{"cnt": len(rows)}]]
        if "FROM episode" in s and "uuid IN $episode_uuids" in s:
            wanted = set(v.get("episode_uuids") or [])
            rows = [
                r for r in self.records["episode"]
                if r.get("group_id") == v.get("group_id")
                and r.get("uuid") in wanted
                and "self_" in str(r.get("source") or "")
            ]
            rows.sort(key=lambda r: r.get("created_at"), reverse=True)
            return [rows[:100]]
        if "FROM episode" in s and "cognition_processed_at IS NONE" in s:
            rows = [
                {"group_id": r.get("group_id", ""), "uuid": r.get("uuid")}
                for r in self.records["episode"]
                if r.get("cognition_processed_at") is None
            ]
            rows.reverse()
            return [rows[: int(v.get("limit") or 100)]]
        if (s.startswith("SELECT * FROM episode") or "FROM episode" in s) and "uuid IN $u" not in s:
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
            if "source CONTAINS 'self_'" in s:
                rows = [r for r in rows if "self_" in str(r.get("source") or "")]
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
        if s.startswith("UPDATE relates_to") and "last_recalled_at" in s:
            for r in self.records["relates_to"]:
                if r.get("group_id") == v.get("group_id") and r.get("uuid") == v.get("uuid"):
                    r["recall_count"] = int(v.get("recall_count") or 0)
                    r["last_recalled_at"] = v.get("last_recalled_at")
            return [[{"ok": True}]]
        if s.startswith("UPDATE relates_to") and "fact = $fact" in s:
            for r in self.records["relates_to"]:
                if r.get("group_id") == v.get("group_id") and r.get("uuid") == v.get("uuid"):
                    for key in ("fact", "confidence", "is_belief", "attributes"):
                        if key in v:
                            r[key] = v[key]
                    r["status"] = "active"
                    r["invalid_at"] = None
            return [[{"ok": True}]]
        if s.startswith("UPDATE episode SET") and "cognition_processed_at" in s:
            wanted = set(v.get("episode_uuids") or [])
            for r in self.records["episode"]:
                if r.get("group_id") == v.get("group_id") and r.get("uuid") in wanted:
                    r["cognition_processed_at"] = v.get("processed_at")
                    r["cognition_version"] = v.get("version")
            return [[{"ok": True}]]
        if s.startswith("UPDATE relates_to") and "record::id(in) IN $aliases" in s:
            aliases = set(v.get("aliases") or [])
            for r in self.records["relates_to"]:
                if r.get("group_id") != v.get("group_id"):
                    continue
                if r.get("source_node_uuid") in aliases or r.get("in") in aliases:
                    r["in"] = v.get("canonical")
                    r["source_node_uuid"] = v.get("canonical")
                if r.get("target_node_uuid") in aliases or r.get("out") in aliases:
                    r["out"] = v.get("canonical")
                    r["target_node_uuid"] = v.get("canonical")
            for r in self.records["mentions"]:
                if r.get("group_id") == v.get("group_id") and r.get("out") in aliases:
                    r["out"] = v.get("canonical")
            self.records["entity"] = [
                r for r in self.records["entity"]
                if not (r.get("group_id") == v.get("group_id") and r.get("uuid") in aliases)
            ]
            return [[{"ok": True}]]
        if s.startswith("UPDATE relates_to") and "conflict_group_id = $cg" in s:
            uuids = set(v.get("uuids") or [])
            for r in self.records["relates_to"]:
                if r.get("uuid") in uuids:
                    r["conflict_group_id"] = v.get("cg")
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
                    # Profile / dossier fields written by
                    # surriti.profiles.refresh_entity_profiles.
                    for key in (
                        "profile_summary",
                        "profile_embedding",
                        "mention_count",
                        "salience",
                        "last_seen_at",
                        "canonical_name",
                        "aliases",
                        "merged_into",
                    ):
                        if key in v:
                            r[key] = v[key]
            return [[{"ok": True}]]

        # ── Generic CREATE type::record(...) fallback ───────────────────
        import re as _re
        _m = _re.match(r"CREATE type::record\(\"(\w+)\"", s)
        if _m:
            tbl = _m.group(1)
            if tbl not in self.records:
                self.records[tbl] = []
            self._insert(tbl, v.get("row") or v)
            return [[{"ok": True}]]

        # ── Generic UPSERT type::record(...) fallback ───────────────────
        _um = _re.match(r"UPSERT type::record\(\"(\w+)\"", s)
        if _um:
            tbl = _um.group(1)
            if tbl not in self.records:
                self.records[tbl] = []
            payload = dict(v.get("row") or v)
            payload.setdefault("uuid", v.get("uuid"))
            # Upsert: replace existing with same uuid, or append.
            existing = None
            for i, r in enumerate(self.records[tbl]):
                if r.get("uuid") == payload.get("uuid"):
                    existing = i
                    break
            if existing is not None:
                merged = {**self.records[tbl][existing], **payload}
                self.records[tbl][existing] = merged
            else:
                self._insert(tbl, payload)
            return [[{"ok": True}]]

        # ── Memory Pack export queries ─────────────────────────────────
        _exp_tables = {
            "entity": "entity",
            "entity_alias": "entity_alias",
            "relation_frame": "relation_frame",
            "relates_to": "relates_to",
        }
        for _tbl, _rec in _exp_tables.items():
            # Paginated export: SELECT ... FROM <table> WHERE group_id = $g
            #   ORDER BY created_at, uuid LIMIT $limit START $offset
            if (
                _re.search(rf"\bFROM\s+{_tbl}\b", s)
                and "ORDER BY created_at, uuid" in s
                and "$limit" in s
            ):
                offset = int(v.get("offset") or 0)
                limit = int(v.get("limit") or 1000)
                rows = [
                    r for r in self.records[_rec]
                    if r.get("group_id") == v.get("g")
                ]
                # Sort by created_at then uuid for deterministic pagination.
                rows.sort(
                    key=lambda r: (
                        str(r.get("created_at") or ""),
                        str(r.get("uuid") or ""),
                    )
                )
                page = rows[offset : offset + limit]
                # Inject `in` / `out` aliases for relates_to if the
                # query requested them (record::id(in) AS `in`).
                if _tbl == "relates_to" and "record::id(in)" in s:
                    for r in page:
                        r.setdefault("in", r.get("source_node_uuid", r.get("uuid")))
                        r.setdefault("out", r.get("target_node_uuid", r.get("uuid")))
                return [page]

        # ── DELETE queries (used by import replace) ─────────────────────
        for _tbl in ("relates_to", "entity_alias", "relation_frame", "entity"):
            if s.startswith(f"DELETE {_tbl} WHERE uuid"):
                before = len(self.records[_tbl])
                self.records[_tbl] = [
                    r for r in self.records[_tbl]
                    if r.get("uuid") != v.get("uuid")
                ]
                after = len(self.records[_tbl])
                return [[{"deleted": before - after}]]
            if s.startswith(f"DELETE {_tbl} WHERE group_id") or \
                    f"DELETE {_tbl} WHERE group_id" in s:
                before = len(self.records[_tbl])
                self.records[_tbl] = [
                    r for r in self.records[_tbl]
                    if r.get("group_id") != v.get("g")
                ]
                after = len(self.records[_tbl])
                return [[{"deleted": before - after}]]

        return [[]]

    def _insert(self, table: str, payload: dict[str, Any]) -> None:
        record = {k: val for k, val in payload.items() if k not in {"src", "tgt", "ep", "en"}}
        if table in {"entity", "community"} and "emb" in record:
            record["name_embedding"] = record.pop("emb")
        if table == "relates_to" and "emb" in record:
            record["fact_embedding"] = record.pop("emb")
        self.records[table].append(record)


__all__ = ["InMemoryDriver"]
