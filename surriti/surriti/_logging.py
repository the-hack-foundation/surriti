"""Optional structured logging helper.

Surriti uses Python's stdlib :mod:`logging`. By default it emits no output
beyond what your application configures. Call :func:`setup_logging` from a
``__main__`` block, a notebook, or your test bootstrap to get a sensible
default config.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(
    level: LogLevel | int = "INFO",
    *,
    fmt: str = _DEFAULT_FMT,
    propagate: bool = False,
) -> logging.Logger:
    """Configure a console handler for the ``surriti`` logger tree.

    Idempotent: re-invoking it just updates the level. Respects
    ``SURRITI_LOG_LEVEL`` env var when called with the default level.
    """

    if isinstance(level, str) and level == "INFO":
        level = os.environ.get("SURRITI_LOG_LEVEL", "INFO")

    logger = logging.getLogger("surriti")
    logger.setLevel(level)
    logger.propagate = propagate

    if not any(getattr(h, "_surriti_default", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        handler._surriti_default = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


__all__ = ["setup_logging"]
