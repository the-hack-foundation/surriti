"""Surriti client for Squire — lightweight wrapper around the Surriti HTTP service.

Usage
-----
    from surriti_client import SurritiClient

    client = SurritiClient()

    # Store a memory (episode) — also triggers LLM response
    result = client.query("michael", "Michael said his book chapter 1 is about work as a physics framework")
    # → {"response": "...", "recalled_facts": [...]}

    # Store without LLM response (just save to memory)
    result = client.store("michael", "Michael prefers concise responses")

    # Recall relevant context for a query
    context = client.recall("michael", "What is Michael's book about?")
    # → MemoryContext(profiles, facts, episodes, traits, goals)

    # Direct fact storage (no LLM)
    client.add_fact("michael", "Michael", "prefers", "concise responses", source_type="user")

    # Self-awareness episodes
    client.store_self("michael", "self_observation", "Squire was too verbose in the last response")

    # List all stored nodes
    nodes, edges = client.list_memory("michael")

    # Clear all memory for a user
    client.clear("michael")

Architecture
------------
Wraps the Surriti FastAPI service (port 3000) via HTTP. Two modes:

1. **Query mode** (``query()``) — calls ``/prompt`` endpoint:
   recall → build system prompt with memory → LLM response → store episode
   Blocking, single-call. Best for turn-by-turn conversation.

2. **Direct mode** (``store()``, ``recall()``, ``add_fact()``) — direct
   Surriti memory operations. Best for structured memory where Squire
   controls the LLM call itself.

The ``query()`` method is the simplest path. For full control, use the
direct methods and manage the LLM call yourself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SURRITI_URL = os.environ.get("SURRITI_URL", "http://localhost:3000")
DEFAULT_GROUP_ID = "squire"
CACHE_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class MemoryContext:
    """Result of a recall operation — structured memory bundle."""
    query: str = ""
    profiles: list[dict] = field(default_factory=list)
    """Entity profiles (dossiers) for mentioned entities."""
    facts: list[dict] = field(default_factory=list)
    """Relevant fact edges (subject → predicate → object)."""
    episodes: list[dict] = field(default_factory=list)
    """Related episodic memories."""
    communities: list[dict] = field(default_factory=list)
    """Community clusters of related entities."""
    traits: list[dict] = field(default_factory=list)
    """Synthesized trait entities."""
    goals: list[dict] = field(default_factory=list)
    """Active goal entities."""
    prediction: dict | None = None
    """Per-group prediction bundle (topics, preferences)."""
    resolved_entities: list[dict] = field(default_factory=list)
    """Entity resolution details (mention → canonical)."""

    @property
    def has_memory(self) -> bool:
        return bool(self.facts or self.profiles or self.episodes)

    def summary(self) -> str:
        """Human-readable summary of the context."""
        lines = []
        if self.profiles:
            lines.append(f"Entities: {', '.join(p.get('name', '?') for p in self.profiles)}")
        if self.facts:
            lines.append(f"Facts: {len(self.facts)} relevant facts")
        if self.traits:
            lines.append(f"Traits: {', '.join(t.get('name', '?') for t in self.traits)}")
        if self.goals:
            lines.append(f"Goals: {', '.join(g.get('name', '?') for g in self.goals)}")
        return "\n".join(lines) if lines else "(no memory found)"


@dataclass
class StoreResult:
    """Result of storing an episode."""
    episode_uuid: str = ""
    entities_added: int = 0
    edges_added: int = 0
    invalidated: int = 0
    new_facts: list[dict] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of a full query (recall → LLM → store)."""
    response: str = ""
    recalled_facts: list[str] = field(default_factory=list)
    store_result: StoreResult | None = None


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------
class LocalCache:
    """Lightweight JSON cache for frequently-accessed facts.

    Reduces Surriti API calls for hot facts (user preferences, project state)
    by caching them locally for CACHE_TTL seconds.
    """

    def __init__(self, group_id: str = DEFAULT_GROUP_ID):
        cache_dir = Path.home() / ".hermes" / "surriti-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / f"{group_id}.json"
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    self._data = data.get("facts", {})
                    self._timestamps = {k: v for k, v in data.get("timestamps", {}).items()}
            except (json.JSONDecodeError, IOError):
                self._data = {}
                self._timestamps = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({
                "facts": self._data,
                "timestamps": self._timestamps,
            }, f, indent=2)

    def get(self, key: str) -> dict | None:
        """Get cached fact by key. Returns None if expired or missing."""
        ts = self._timestamps.get(key, 0)
        if time.time() - ts > CACHE_TTL:
            return None
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        """Cache a fact by key."""
        self._data[key] = value
        self._timestamps[key] = time.time()
        self._save()

    def delete(self, key: str) -> None:
        """Remove a cached fact."""
        self._data.pop(key, None)
        self._timestamps.pop(key, None)
        self._save()

    def clear(self) -> None:
        """Clear all cached facts."""
        self._data.clear()
        self._timestamps.clear()
        self._save()


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class SurritiClient:
    """Client for the Surriti memory service.

    Parameters
    ----------
    base_url : str
        Surriti service URL (default: http://localhost:3000).
    group_id : str
        Default group/tenant ID for memory isolation.
    """

    def __init__(self, base_url: str | None = None, group_id: str = DEFAULT_GROUP_ID):
        self.base_url = (base_url or SURRITI_URL).rstrip("/")
        self.group_id = group_id
        self._cache = LocalCache(group_id)

    # ------------------------------------------------------------------
    # Query mode — blocking recall → LLM → store
    # ------------------------------------------------------------------
    def query(self, user_id: str, content: str) -> QueryResult:
        """One-shot query: recall memory → LLM response → store episode.

        This is the simplest interface for turn-by-turn conversation.
        It calls the ``/prompt`` endpoint which handles the full pipeline:
        1. Search memory for relevant facts
        2. Build system prompt with memory context
        3. Generate LLM response
        4. Store the user's message as an episode

        Parameters
        ----------
        user_id : str
            The user's identifier (maps to Surriti group_id).
        content : str
            The user's message or query.

        Returns
        -------
        QueryResult with response text, recalled facts, and store result.
        """
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/prompt",
                json={"text": content, "user_id": user_id},
            )
            resp.raise_for_status()
            data = resp.json()

        return QueryResult(
            response=data.get("response", ""),
            recalled_facts=data.get("recalled_facts", []),
        )

    # ------------------------------------------------------------------
    # Direct mode — structured memory operations
    # ------------------------------------------------------------------
    def store(self, user_id: str, content: str, *, name: str = "chat",
              source: str = "message", source_description: str = "") -> StoreResult:
        """Store an episode in Surriti memory.

        Sends the content through the ``/send`` + WebSocket pipeline,
        which extracts entities/facts via LLM and stores them.

        Parameters
        ----------
        user_id : str
            The user's identifier (group_id).
        content : str
            The episode content to store.
        name : str
            Episode name/type (default: "chat").
        source : str
            Episode source type (message, json, text, fact_triple, etc.).
        source_description : str
            Optional description of the source.

        Returns
        -------
        StoreResult with episode UUID, entity/edge counts, and new facts.
        """
        # Use /send + WebSocket for streaming store
        import uuid
        session_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        try:
            import websockets
        except ImportError:
            # Fallback: just use /prompt which also stores
            result = self.query(user_id, content)
            return StoreResult(
                entities_added=0,
                edges_added=0,
            )

        facts = []
        entities_added = 0
        edges_added = 0
        invalidated = 0
        episode_uuid = ""

        async def _store():
            nonlocal facts, entities_added, edges_added, invalidated, episode_uuid
            async with websockets.connect(f"{self.base_url.replace('http', 'ws')}/ws/{session_id}") as ws:
                # Queue the turn
                await ws.send(json.dumps({
                    "session_id": session_id,
                    "user_id": user_id,
                    "content": content,
                    "mode": "direct",
                }))

                # Drain events until done
                async for msg in ws:
                    event = json.loads(msg)
                    if event.get("type") == "memory_store":
                        episode_uuid = event.get("episode_uuid", "")
                        entities_added = event.get("entities_added", 0)
                        edges_added = event.get("edges_added", 0)
                        invalidated = event.get("invalidated", 0)
                        facts = event.get("new_facts", [])
                    elif event.get("type") == "error":
                        log.warning("Store error: %s", event.get("message"))
                        break
                    elif event.get("type") == "done":
                        break

        try:
            asyncio.run(_store())
        except Exception as e:
            log.warning("Store failed: %s", e)

        return StoreResult(
            episode_uuid=episode_uuid,
            entities_added=entities_added,
            edges_added=edges_added,
            invalidated=invalidated,
            new_facts=facts,
        )

    def recall(self, user_id: str, query: str, *, depth: str = "normal",
               limit: int = 20) -> MemoryContext:
        """Recall relevant memory for a query.

        Uses the ``/prompt`` endpoint to search memory and return
        recalled facts. The depth parameter controls how much context
        is retrieved (passed through to the underlying Surriti engine).

        Parameters
        ----------
        user_id : str
            The user's identifier (group_id).
        query : str
            The search query.
        depth : str
            Recall depth (fast/normal/deep).
        limit : int
            Maximum results to return.

        Returns
        -------
        MemoryContext with profiles, facts, episodes, traits, goals, etc.
        """
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/prompt",
                json={"text": query, "user_id": user_id},
            )
            resp.raise_for_status()
            data = resp.json()

        recalled_facts = data.get("recalled_facts", [])

        return MemoryContext(
            query=query,
            facts=[{"fact": f} for f in recalled_facts],
        )

    def search(self, user_id: str, query: str, *, limit: int = 10) -> list[dict]:
        """Search memory for relevant facts.

        Lightweight search that returns matching facts without triggering
        a full recall/store cycle.

        Parameters
        ----------
        user_id : str
            The user's identifier (group_id).
        query : str
            Search query.
        limit : int
            Maximum results.

        Returns
        -------
        List of matching fact dicts.
        """
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/prompt",
                json={"text": query, "user_id": user_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("recalled_facts", [])

    # ------------------------------------------------------------------
    # Self-awareness episodes
    # ------------------------------------------------------------------
    def store_self(self, user_id: str, episode_type: str, content: str,
                   *, name: str | None = None) -> StoreResult:
        """Store a self-referential episode for operational self-awareness.

        Self-episodes track Squire's own behavior patterns, corrections,
        successes, and observations. These feed into the self-model
        extraction pipeline.

        Supported types:
        - ``self_observation`` — explicit reflection ("I was too verbose")
        - ``self_correction`` — noticing a mistake
        - ``self_success`` — noticing a win
        - ``self_pattern`` — recurring behavioral trend

        Parameters
        ----------
        user_id : str
            The user's identifier (group_id).
        episode_type : str
            One of the self-episode types above.
        content : str
            The self-observation content.
        name : str | None
            Optional episode name. Defaults to the type.

        Returns
        -------
        StoreResult with the stored episode details.
        """
        # Validate episode_type against known self-episode types
        valid_types = ("self_observation", "self_correction", "self_success", "self_pattern")
        if episode_type not in valid_types:
            raise ValueError(
                f"Invalid episode_type {episode_type!r}. "
                f"Must be one of: {', '.join(valid_types)}"
            )

        episode_name = name or f"self_{episode_type}"
        return self.store(
            user_id,
            content,
            name=episode_name,
            source="message",
            source_description=f"self_{episode_type}",
        )

    # ------------------------------------------------------------------
    # Direct fact storage (no LLM)
    # ------------------------------------------------------------------
    def add_fact(self, user_id: str, subject: str, predicate: str,
                 object_: str, *, fact: str | None = None,
                 source_type: str = "user") -> StoreResult:
        """Add a structured fact without LLM extraction.

        Bypasses the LLM extractor and directly stores a (subject, predicate, object)
        triple. Useful for storing known facts, preferences, or constraints
        that don't need extraction.

        Parameters
        ----------
        user_id : str
            The user's identifier (group_id).
        subject : str
            The subject entity name.
        predicate : str
            The relation/predicate (e.g., "prefers", "works_at", "lives_in").
        object_ : str
            The object entity name or value.
        fact : str | None
            Human-readable fact text. Defaults to "subject predicate object".
        source_type : str
            Provenance: "user" (authoritative), "assistant", "tool", "system".

        Returns
        -------
        StoreResult with the stored fact details.
        """
        fact_text = fact or f"{subject} {predicate} {object_}."
        content = f"FACT: {subject} {predicate} {object_}."
        return self.store(
            user_id,
            content,
            name=f"fact_{subject}_{predicate}",
            source="fact_triple",
            source_description=f"{subject} -> {predicate} -> {object_}",
        )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def list_memory(self, user_id: str) -> tuple[list[dict], list[dict]]:
        """List all entities and edges for a user.

        Returns
        -------
        Tuple of (nodes, edges) as dicts.
        """
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self.base_url}/memory/{user_id}")
            resp.raise_for_status()
            data = resp.json()
            return data.get("nodes", []), data.get("edges", [])

    def clear(self, user_id: str) -> dict:
        """Clear all memory for a user.

        Returns
        -------
        Dict with status and tables cleared count.
        """
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(f"{self.base_url}/memory/{user_id}")
            resp.raise_for_status()
            self._cache.clear()
            return resp.json()

    def health(self) -> dict:
        """Check Surriti service health."""
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Convenience: inject memory into a system prompt
    # ------------------------------------------------------------------
    def build_system_prompt(self, context: MemoryContext,
                            base_system: str = "") -> str:
        """Build a system prompt enriched with Surriti memory context.

        Parameters
        ----------
        context : MemoryContext
            The recall result to inject.
        base_system : str
            Base system prompt (Squire's default instructions).

        Returns
        -------
        Complete system prompt with memory section.
        """
        parts = []

        if base_system:
            parts.append(base_system)

        if context.profiles:
            ent_lines = []
            for p in context.profiles:
                name = p.get("name", "?")
                summary = p.get("summary", "")
                labels = p.get("labels", [])
                line = f"- **{name}**"
                if labels:
                    line += f" ({', '.join(labels)})"
                if summary:
                    line += f"\n  Summary: {summary}"
                ent_lines.append(line)
            parts.append(f"\nMEMORY ENTITIES:\n" + "\n".join(ent_lines))

        if context.facts:
            fact_lines = []
            for f in context.facts:
                fact_text = f.get("fact", "") or f.get("name", "")
                if fact_text:
                    fact_lines.append(f"- {fact_text}")
            if fact_lines:
                parts.append("\nMEMORY FACTS:\n" + "\n".join(fact_lines))

        if context.traits:
            trait_lines = [f"- {t.get('name', '?')}" for t in context.traits]
            parts.append("\nMEMORY TRAITS:\n" + "\n".join(trait_lines))

        if context.goals:
            goal_lines = [f"- {g.get('name', '?')}" for g in context.goals]
            parts.append("\nMEMORY GOALS:\n" + "\n".join(goal_lines))

        if not context.has_memory:
            parts.append("\nMEMORY: (empty — no prior context)")

        return "\n".join(parts) if parts else base_system or "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick health check
        client = SurritiClient()
        health = client.health()
        print(f"Health: {health}")

        # Quick store + recall test
        result = client.store("michael", "Michael's book is about work as a physics framework")
        print(f"Stored: {result.entities_added} entities, {result.edges_added} edges")

        context = client.recall("michael", "work as physics")
        print(f"Recalled: {context.summary()}")
    else:
        print("Usage: python surriti_client.py test")
