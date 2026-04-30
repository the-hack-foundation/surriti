"""Search filters - subset of Graphiti's SearchFilters tailored for Surriti.

Filters are applied in two places:
1. As SurrealQL WHERE clauses appended to vector / full-text candidate queries.
2. As post-fetch Python filters when expressing the predicate in SurrealQL
   would be awkward (multi-window date ranges, label/edge type subsets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ComparisonOperator(str, Enum):
    eq = "="
    ne = "<>"
    gt = ">"
    lt = "<"
    gte = ">="
    lte = "<="
    is_null = "IS NULL"
    is_not_null = "IS NOT NULL"


@dataclass
class DateFilter:
    date: datetime | None
    op: ComparisonOperator


@dataclass
class PropertyFilter:
    name: str
    value: Any
    op: ComparisonOperator = ComparisonOperator.eq


@dataclass
class SearchFilters:
    """All fields are optional. ``None`` = no constraint."""

    node_labels: list[str] | None = None
    edge_types: list[str] | None = None
    edge_uuids: list[str] | None = None
    # Each outer list element is a "window" combined via OR; inner list via AND.
    valid_at: list[list[DateFilter]] = field(default_factory=list)
    invalid_at: list[list[DateFilter]] = field(default_factory=list)
    created_at: list[list[DateFilter]] = field(default_factory=list)
    expired_at: list[list[DateFilter]] = field(default_factory=list)
    property_filters: list[PropertyFilter] = field(default_factory=list)


def _date_clause_matches(value: datetime | None, clauses: list[list[DateFilter]]) -> bool:
    if not clauses:
        return True
    for window in clauses:  # OR across windows
        ok = True
        for clause in window:  # AND within window
            if not _eval_date(value, clause):
                ok = False
                break
        if ok:
            return True
    return False


def _eval_date(value: datetime | None, clause: DateFilter) -> bool:
    if clause.op is ComparisonOperator.is_null:
        return value is None
    if clause.op is ComparisonOperator.is_not_null:
        return value is not None
    if value is None or clause.date is None:
        return False
    table = {
        ComparisonOperator.eq: lambda a, b: a == b,
        ComparisonOperator.ne: lambda a, b: a != b,
        ComparisonOperator.gt: lambda a, b: a > b,
        ComparisonOperator.lt: lambda a, b: a < b,
        ComparisonOperator.gte: lambda a, b: a >= b,
        ComparisonOperator.lte: lambda a, b: a <= b,
    }
    fn = table.get(clause.op)
    if fn is None:
        return True
    return fn(value, clause.date)


def edge_passes_filters(edge_row: dict[str, Any], filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.edge_types is not None and edge_row.get("name") not in filters.edge_types:
        return False
    if filters.edge_uuids is not None and edge_row.get("uuid") not in filters.edge_uuids:
        return False
    for date_field in ("valid_at", "invalid_at", "created_at", "expired_at"):
        clauses = getattr(filters, date_field)
        if not _date_clause_matches(edge_row.get(date_field), clauses):
            return False
    for pf in filters.property_filters:
        if not _eval_property(edge_row, pf):
            return False
    return True


def _eval_property(row: dict[str, Any], pf: PropertyFilter) -> bool:
    actual = row.get(pf.name)
    if actual is None:
        attrs = row.get("attributes") or {}
        if isinstance(attrs, dict):
            actual = attrs.get(pf.name)
    if pf.op is ComparisonOperator.is_null:
        return actual is None
    if pf.op is ComparisonOperator.is_not_null:
        return actual is not None
    if actual is None:
        return False
    table = {
        ComparisonOperator.eq: lambda a, b: a == b,
        ComparisonOperator.ne: lambda a, b: a != b,
        ComparisonOperator.gt: lambda a, b: a > b,
        ComparisonOperator.lt: lambda a, b: a < b,
        ComparisonOperator.gte: lambda a, b: a >= b,
        ComparisonOperator.lte: lambda a, b: a <= b,
    }
    fn = table.get(pf.op)
    if fn is None:
        return True
    try:
        return fn(actual, pf.value)
    except TypeError:
        return False


def node_passes_filters(node_row: dict[str, Any], filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.node_labels is not None:
        labels = set(node_row.get("labels") or [])
        if not labels.intersection(filters.node_labels):
            return False
    for pf in filters.property_filters:
        if not _eval_property(node_row, pf):
            return False
    return True
