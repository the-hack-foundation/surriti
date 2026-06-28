"""Hermes Agent memory provider backed by Surriti.

Talks to a running Surriti service (myapp) over HTTP. The service does the
heavy lifting: LLM-driven entity/fact extraction, contradiction resolution,
temporal validity tracking, and hybrid vector + BM25 recall over a
SurrealDB knowledge graph.

Two pure endpoints are used:

    POST /recall  {query, user_id, limit}    -> {facts, entities}
    POST /store   {content, user_id, ...}    -> {episode_uuid, ...}

`prefetch` is synchronous (returns recall context for the next turn).
`sync_turn` MUST be non-blocking, so it dispatches to a daemon thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 15.0
CONFIG_FILENAME = "surriti.json"

# Recall results are cached briefly to absorb retries and to let queue_prefetch
# hand off to prefetch with zero latency.
_CACHE_TTL_SECONDS = 30.0
_CACHE_MAX_ENTRIES = 32


class SurritiMemoryProvider(MemoryProvider):
    """Memory provider that proxies recall/store to the Surriti HTTP service."""

    def __init__(self) -> None:
        self._url: str = ""
        self._timeout: float = DEFAULT_TIMEOUT
        self._user_id: str = "default"
        self._session_id: str = ""
        self._hermes_home: Path | None = None
        self._sync_thread: threading.Thread | None = None
        self._available: bool | None = None
        self._client: httpx.Client | None = None
        # Cache: query -> (expires_at, formatted_block)
        self._cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        # Prefetch tracking: query -> Event signaling completion
        self._prefetch_inflight: dict[str, threading.Event] = {}
        self._prefetch_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "surriti"

    # ------------------------------------------------------------------
    # Availability — NO network calls (per provider contract)
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        # Available iff a URL is configured (env var or config file).
        if os.environ.get("SURRITI_URL"):
            self._available = True
            return True
        cfg = self._read_config_file()
        self._available = bool(cfg.get("url"))
        return self._available

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home")
        self._hermes_home = Path(hermes_home) if hermes_home else None

        cfg = self._read_config_file()
        self._url = (
            os.environ.get("SURRITI_URL")
            or cfg.get("url")
            or DEFAULT_URL
        ).rstrip("/")
        self._timeout = float(cfg.get("timeout") or DEFAULT_TIMEOUT)
        self._user_id = cfg.get("user_id") or os.environ.get("SURRITI_USER_ID", "default")
        # Persistent client with HTTP keep-alive — cuts ~10-30ms off each call
        # by reusing the TCP connection across recall/store requests.
        self._client = httpx.Client(
            base_url=self._url,
            timeout=self._timeout,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        logger.info(
            "Surriti memory provider initialized: url=%s user_id=%s session=%s",
            self._url, self._user_id, self._session_id,
        )

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Config schema (consumed by `hermes memory setup`)
    # ------------------------------------------------------------------
    def get_config_schema(self) -> list[dict]:
        return [
            {
                "key": "url",
                "description": "Surriti service base URL",
                "default": DEFAULT_URL,
                "env_var": "SURRITI_URL",
            },
            {
                "key": "user_id",
                "description": "Default tenant/group_id for memory isolation",
                "default": "default",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Persist non-secret config to $HERMES_HOME/surriti.json."""
        path = Path(hermes_home) / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop empty values so defaults still apply on next load.
        clean = {k: v for k, v in values.items() if v not in ("", None)}
        path.write_text(json.dumps(clean, indent=2))

    # ------------------------------------------------------------------
    # Tools — none. Recall is automatic via prefetch().
    # ------------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs: Any) -> Any:
        return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    # System prompt block — describes the provider
    # ------------------------------------------------------------------
    def system_prompt_block(self) -> str | None:
        return (
            "You have persistent memory powered by Surriti, a temporal "
            "knowledge graph. Relevant facts from prior conversations are "
            "injected before each turn under MEMORY. Treat MEMORY as "
            "authoritative — do not deny knowledge that is present there."
        )

    # ------------------------------------------------------------------
    # Recall — synchronous, returns context for the next API call.
    # Hits the cache first (populated by queue_prefetch when available).
    # ------------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        key = self._cache_key(query)

        # Fast path: queue_prefetch already finished.
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # If queue_prefetch is in flight for this exact query, wait for it
        # rather than firing a duplicate request.
        with self._prefetch_lock:
            event = self._prefetch_inflight.get(key)
        if event is not None:
            # Bounded wait — fall through to a sync fetch on timeout.
            if event.wait(timeout=self._timeout):
                cached = self._cache_get(key)
                if cached is not None:
                    return cached

        return self._do_recall(query, key)

    def queue_prefetch(self, query: str) -> None:
        """Fire recall in the background while the user/agent is still busy.

        Hermes calls this as soon as a query is known. By the time `prefetch`
        is called for the same query, the result is already cached and
        returns instantly.
        """
        if not query or not query.strip():
            return
        key = self._cache_key(query)

        if self._cache_get(key) is not None:
            return  # already cached

        with self._prefetch_lock:
            if key in self._prefetch_inflight:
                return  # already in flight
            event = threading.Event()
            self._prefetch_inflight[key] = event

        def _worker() -> None:
            try:
                self._do_recall(query, key)
            finally:
                event.set()
                with self._prefetch_lock:
                    self._prefetch_inflight.pop(key, None)

        threading.Thread(target=_worker, daemon=True).start()

    def _do_recall(self, query: str, cache_key: str) -> str:
        """Perform the HTTP recall, format the block, and cache it."""
        client = self._client
        if client is None:
            # Fallback if shutdown raced — create an ephemeral client.
            client = httpx.Client(timeout=self._timeout)
            owns_client = True
        else:
            owns_client = False
        try:
            resp = client.post(
                "/recall" if not owns_client else f"{self._url}/recall",
                json={"query": query, "user_id": self._user_id, "limit": 10},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Surriti recall failed: %s", exc)
            return ""
        finally:
            if owns_client:
                try:
                    client.close()
                except Exception:
                    pass

        block = self._format_recall_block(
            data.get("facts") or [],
            data.get("entities") or [],
        )
        self._cache_put(cache_key, block)
        return block

    @staticmethod
    def _format_recall_block(facts: list[dict], entities: list[dict]) -> str:
        if not facts and not entities:
            return ""
        parts: list[str] = ["MEMORY (from Surriti knowledge graph):"]
        if entities:
            ent_lines = []
            for e in entities[:8]:
                name = e.get("name") or "?"
                labels = e.get("labels") or []
                tag = f" ({', '.join(labels)})" if labels else ""
                ent_lines.append(f"- {name}{tag}")
            parts.append("Entities:\n" + "\n".join(ent_lines))
        if facts:
            fact_lines = []
            for f in facts[:12]:
                text = (f.get("fact") or "").strip()
                if text:
                    fact_lines.append(f"- {text}")
            if fact_lines:
                parts.append("Facts:\n" + "\n".join(fact_lines))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Ingest — non-blocking (daemon thread)
    # ------------------------------------------------------------------
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list | None = None,
    ) -> None:
        if not user_content and not assistant_content:
            return

        # Prefer storing the user message — that's where new facts live.
        # Skip the assistant content unless the user said nothing (rare).
        content = user_content or assistant_content
        source_type = "user" if user_content else "assistant"

        def _send() -> None:
            client = self._client
            owns_client = False
            if client is None:
                client = httpx.Client(timeout=self._timeout)
                owns_client = True
            try:
                resp = client.post(
                    "/store" if not owns_client else f"{self._url}/store",
                    json={
                        "content": content,
                        "user_id": self._user_id,
                        "name": "chat",
                        "source_type": source_type,
                        "source_description": f"hermes:{session_id or self._session_id}",
                    },
                )
                resp.raise_for_status()
                # New facts may have landed — invalidate stale recall cache.
                self._cache_clear()
            except Exception as exc:
                logger.warning("Surriti sync_turn failed: %s", exc)
            finally:
                if owns_client:
                    try:
                        client.close()
                    except Exception:
                        pass

        # Wait for any in-flight sync to finish first, then dispatch fresh.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_send, daemon=True)
        self._sync_thread.start()

    def on_session_end(self, messages: list) -> None:
        # Make sure any pending sync completes before process exit.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(query: str) -> str:
        return query.strip().lower()

    def _cache_get(self, key: str) -> str | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, block = entry
            if expires_at < time.monotonic():
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return block

    def _cache_put(self, key: str, block: str) -> None:
        with self._cache_lock:
            self._cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, block)
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)

    def _cache_clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _read_config_file(self) -> dict:
        if self._hermes_home is None:
            home = os.environ.get("HERMES_HOME")
            if not home:
                return {}
            self._hermes_home = Path(home)
        path = self._hermes_home / CONFIG_FILENAME
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text()) or {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return {}


def register(ctx) -> None:
    """Plugin entry point — called by the memory plugin discovery system."""
    ctx.register_memory_provider(SurritiMemoryProvider())
