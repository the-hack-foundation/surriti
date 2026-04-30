"""Public exception hierarchy for Surriti.

All errors raised by Surriti inherit from :class:`SurritiError` so callers
can catch them as a group::

    try:
        await memory.add_episode(...)
    except SurritiError as e:
        log.exception("memory write failed", exc_info=e)
"""

from __future__ import annotations


class SurritiError(Exception):
    """Base class for all Surriti errors."""


class SurritiConfigError(SurritiError):
    """Configuration is missing or invalid (e.g. no API key, bad URL)."""


class SurritiConnectionError(SurritiError):
    """Failure to connect to or communicate with SurrealDB."""


class SurritiSchemaError(SurritiError):
    """Schema initialisation or migration failed."""


class SurritiLLMError(SurritiError):
    """The configured LLM client returned an invalid or unparseable response."""


class SurritiNotFoundError(SurritiError):
    """A requested record (node, edge, episode) does not exist."""


__all__ = [
    "SurritiConfigError",
    "SurritiConnectionError",
    "SurritiError",
    "SurritiLLMError",
    "SurritiNotFoundError",
    "SurritiSchemaError",
]
