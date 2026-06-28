"""Thin async wrapper around the SurrealDB Python SDK.

Surriti talks to SurrealDB exclusively through this driver. It exposes a
small surface (connect/close, query, create/select/relate) so that tests can
swap in a fake driver and so that callers never need to import the Surreal
SDK directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from surriti.errors import SurritiConnectionError, SurritiSchemaError
from surriti.schema import ALL_TABLES, schema_ddl

logger = logging.getLogger(__name__)


class SurrealDriver:
    """Async SurrealDB driver wrapper.

    Parameters
    ----------
    url:
        WebSocket or HTTP endpoint of the SurrealDB server, e.g.
        ``ws://localhost:8000/rpc``. Use ``mem://`` for an in-process
        instance (only the surrealdb SDK supports this).
    namespace, database:
        Surreal namespace and database to use.
    username, password:
        Optional credentials for ``signin``.
    embedding_dim:
        Dimensionality used by ``init_schema``.
    """

    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "surriti",
        database: str = "surriti",
        username: str | None = None,
        password: str | None = None,
        embedding_dim: int = 768,
    ) -> None:
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}"
            )
        self.embedding_dim = embedding_dim
        self._db: Any | None = None

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_env(cls) -> "SurrealDriver":
        """Build a driver from ``SURRITI_SURREAL_*`` environment variables."""

        return cls(
            url=os.environ.get("SURRITI_SURREAL_URL", "ws://localhost:8000/rpc"),
            namespace=os.environ.get("SURRITI_SURREAL_NS", "surriti"),
            database=os.environ.get("SURRITI_SURREAL_DB", "surriti"),
            username=os.environ.get("SURRITI_SURREAL_USER"),
            password=os.environ.get("SURRITI_SURREAL_PASS"),
            embedding_dim=int(os.environ.get("SURRITI_EMBEDDING_DIM", "768")),
        )

    # ---------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        # Imported lazily so the rest of the package is testable without the
        # surrealdb extra installed.
        try:
            from surrealdb import AsyncSurreal  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise SurritiConnectionError(
                "The 'surrealdb' package is required. Install it with `pip install surrealdb`."
            ) from exc

        try:
            self._db = AsyncSurreal(self.url)
            await self._db.connect()
            if self.username and self.password:
                await self._db.signin({"username": self.username, "password": self.password})
            await self._db.use(self.namespace, self.database)
        except SurritiConnectionError:
            raise
        except Exception as exc:
            raise SurritiConnectionError(
                f"Could not connect to SurrealDB at {self.url!r}: {exc}"
            ) from exc
        logger.debug("Connected to SurrealDB at %s", self.url)

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("Error closing SurrealDB connection")
            self._db = None

    async def __aenter__(self) -> "SurrealDriver":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------------------------------------------------------------- helpers
    @property
    def db(self) -> Any:
        if self._db is None:
            raise SurritiConnectionError(
                "SurrealDriver is not connected. Call `await driver.connect()` "
                "or use it as an async context manager."
            )
        return self._db

    async def query(
        self, surql: str, variables: dict[str, Any] | None = None
    ) -> list[Any]:
        """Run a SurrealQL query and return the (last statement's) result.

        Transparently reconnects ONCE if the underlying websocket has been
        torn down (e.g. SurrealDB was restarted while myapp kept running).
        Recognised stale-connection signatures include "no close frame",
        "ConnectionClosed", "WebSocket", and "not connected".

        Retries ONCE on transaction-conflict errors (SurrealDB's optimistic
        concurrency control).
        """

        import asyncio

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return await self.db.query(surql, variables or {})
            except SurritiConnectionError:
                raise
            except Exception as exc:
                msg = str(exc).lower()
                # Retry on transaction conflicts (optimistic concurrency)
                if "transaction conflict" in msg or "resource busy" in msg:
                    if attempt < max_retries:
                        backoff = 0.1 * (2**attempt)
                        logger.debug(
                            "Transaction conflict on query; retrying in %.2fs (attempt %d/%d).",
                            backoff,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    # Exhausted retries — re-raise
                    raise
                stale = any(
                    tok in msg
                    for tok in (
                        "no close frame",
                        "connectionclosed",
                        "connection closed",
                        "websocket",
                        "not connected",
                        "broken pipe",
                        "connection reset",
                    )
                )
                if not stale:
                    raise
                logger.warning(
                    "SurrealDB connection appears stale (%s); reconnecting and retrying.",
                    exc,
                )
                try:
                    await self.close()
                except Exception:  # pragma: no cover
                    logger.debug("close() raised during reconnect; ignoring", exc_info=True)
                await self.connect()
                return await self.db.query(surql, variables or {})

    async def init_schema(self) -> None:
        """Apply the Surriti schema. Idempotent - safe to call repeatedly."""

        try:
            await self.query(schema_ddl(self.embedding_dim))
            # Backfill cognitive-layer fields added after initial deployment.
            # Records inserted before these fields existed have NONE, which
            # fails SurrealDB's TYPE validation on any subsequent UPDATE.
            await self.query(
                "UPDATE relates_to SET recall_count = 0"
                " WHERE recall_count IS NONE;"
            )
            await self.query(
                "UPDATE relates_to SET reinforcement_count = 1"
                " WHERE reinforcement_count IS NONE;"
            )
            await self.query(
                "UPDATE relates_to SET weight = 1.0"
                " WHERE weight IS NONE;"
            )
            await self.query(
                "UPDATE relates_to SET decay_score = 1.0"
                " WHERE decay_score IS NONE;"
            )
            await self.query(
                "UPDATE entity SET salience = 0.0"
                " WHERE salience IS NONE;"
            )
            await self.query(
                "UPDATE entity SET mention_count = 0"
                " WHERE mention_count IS NONE;"
            )
        except Exception as exc:
            raise SurritiSchemaError(f"Failed to initialise schema: {exc}") from exc

    async def clear(self) -> None:
        """Delete every record in tables managed by Surriti. Useful for tests.

        Retries each DELETE individually to avoid transaction conflicts when
        cleaning up large datasets.
        """

        for table in ALL_TABLES:
            for _ in range(3):
                try:
                    await self.query(f"DELETE {table};")
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    if ("transaction conflict" in msg or "resource busy" in msg):
                        import asyncio; await asyncio.sleep(0.2)
                        continue
                    raise

