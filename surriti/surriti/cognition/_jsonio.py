"""Shared helpers for cognition modules."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_LEADING_TEXT_RE = re.compile(r"^[^{\[]*", re.DOTALL)


def parse_json_loose(raw: str | None) -> Any:
    """Parse JSON returned by a model, tolerating code fences and prose
    around the payload. Returns ``None`` on any parse failure."""

    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first JSON object/array.
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not match:
            logger.debug("parse_json_loose: no JSON found in %r", text[:200])
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.debug("parse_json_loose: failed to parse %r", text[:200])
            return None


def snake_case(name: str) -> str:
    """Coerce a free-form string into a snake_case label."""

    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").strip()).strip("_").lower()
    return s or "unknown"
