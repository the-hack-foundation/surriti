"""Tiny LLM HTTP/WS service backed by an Ollama runtime + surriti memory.

Endpoints
---------
GET  /health                   liveness probe
POST /prompt                   one-shot blocking inference (kept for curl)
GET  /memory/{user_id}         dump recent facts (debugging)

POST /send                     queue a streaming turn (returns immediately)
WS   /ws/{session_id}          stream events for that session

Streaming event types (over the WebSocket)
-----------------------------------------
{type:"step_start",  label:"surriti_recall"}
{type:"memory_recall", query, edges:[{fact, name, source, target, valid_at, score}],
                       nodes:[{uuid, name, labels, summary}]}
{type:"step_done",   label:"surriti_recall"}

{type:"chunk",       text}                  # streamed LLM token(s)

{type:"step_start",  label:"surriti_store"}
{type:"memory_store",episode_uuid, entities_added, edges_added, invalidated,
                     new_facts:[{fact, name, source, target}]}
{type:"step_done",   label:"surriti_store"}

{type:"done"}
{type:"error",       message}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_EXTRACT_MODEL = os.environ.get("OLLAMA_EXTRACT_MODEL", "qwen2.5:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM    = int(os.environ.get("EMBED_DIM", "768"))
PORT = int(os.environ.get("PORT", "3000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

SURREAL_URL  = os.environ.get("SURRITI_SURREAL_URL",  "ws://surrealdb:8000/rpc")
SURREAL_NS   = os.environ.get("SURRITI_SURREAL_NS",   "myapp")
SURREAL_DB   = os.environ.get("SURRITI_SURREAL_DB",   "myapp")
SURREAL_USER = os.environ.get("SURRITI_SURREAL_USER", "root")
SURREAL_PASS = os.environ.get("SURRITI_SURREAL_PASS", "root")

# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------
_memory: Any = None     # surriti.Surriti | None
_http: httpx.AsyncClient | None = None

# Per-session event queues, written by /send, drained by /ws/{session_id}
_sessions: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# Ollama-backed embedder for surriti
# ---------------------------------------------------------------------------
class OllamaEmbedder:
    """Calls Ollama's /api/embeddings to produce vectors for surriti."""

    def __init__(self, http: httpx.AsyncClient, base_url: str,
                 model: str, embedding_dim: int) -> None:
        self._http = http
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self.model = model
        self.embedding_dim = embedding_dim

    async def create(self, input_data: str) -> list[float]:
        text = (input_data or "").strip() or " "
        r = await self._http.post(
            self._url, json={"model": self.model, "prompt": text}, timeout=60,
        )
        r.raise_for_status()
        vec = r.json().get("embedding") or []
        # Pad/truncate to declared dim so SurrealDB HNSW stays happy.
        if len(vec) < self.embedding_dim:
            vec = list(vec) + [0.0] * (self.embedding_dim - len(vec))
        elif len(vec) > self.embedding_dim:
            vec = list(vec)[: self.embedding_dim]
        return [float(x) for x in vec]

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        return [await self.create(t) for t in input_data]


async def _ensure_ollama_model(model: str) -> None:
    """Make sure ``model`` is available locally in ollama. Pulls if missing."""
    assert _http is not None
    # Wait for ollama to respond
    for _ in range(60):
        try:
            r = await _http.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(1)
    else:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}")

    tags = r.json().get("models", [])
    have = any(m.get("name", "").startswith(model) for m in tags)
    if have:
        log.info("Ollama model already present: %s", model)
        return

    log.info("Pulling model %s from ollama (this may take a while)...", model)
    async with _http.stream(
        "POST", f"{OLLAMA_URL}/api/pull",
        json={"model": model, "stream": True},
        timeout=None,
    ) as resp:
        last_status = ""
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            status = evt.get("status", "")
            if status and status != last_status:
                log.info("ollama pull: %s", status)
                last_status = status
            if evt.get("error"):
                raise RuntimeError(f"ollama pull failed: {evt['error']}")
    log.info("Ollama model ready: %s", model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory, _http

    _http = httpx.AsyncClient()

    try:
        await _ensure_ollama_model(OLLAMA_MODEL)
    except Exception as exc:
        log.warning("Ollama chat model preparation failed: %s", exc)
    try:
        await _ensure_ollama_model(OLLAMA_EXTRACT_MODEL)
    except Exception as exc:
        log.warning("Ollama extraction model preparation failed: %s", exc)
    try:
        await _ensure_ollama_model(OLLAMA_EMBED_MODEL)
    except Exception as exc:
        log.warning("Ollama embedding model preparation failed: %s", exc)

    try:
        from openai import AsyncOpenAI
        from surriti import Surriti
        from surriti.driver import SurrealDriver
        from surriti.llm_clients import OpenAILLMClient

        # Point the openai sdk at ollama's OpenAI-compatible endpoint.
        oa_client = AsyncOpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
        llm_client = OpenAILLMClient(model=OLLAMA_EXTRACT_MODEL, client=oa_client)
        embedder = OllamaEmbedder(_http, OLLAMA_URL, OLLAMA_EMBED_MODEL, EMBED_DIM)

        driver = SurrealDriver(
            url=SURREAL_URL, namespace=SURREAL_NS, database=SURREAL_DB,
            username=SURREAL_USER, password=SURREAL_PASS, embedding_dim=EMBED_DIM,
        )
        mem = Surriti(driver, llm_client=llm_client, embedder=embedder)
        await mem.connect()
        _memory = mem
        log.info("Surriti memory connected to %s (chat=%s extract=%s embed=%s dim=%d)",
                 SURREAL_URL, OLLAMA_MODEL, OLLAMA_EXTRACT_MODEL,
                 OLLAMA_EMBED_MODEL, EMBED_DIM)
    except Exception as exc:  # pragma: no cover - integration path
        log.warning("Surriti memory unavailable (%s) -- running without memory.", exc)
        _memory = None

    yield

    if _memory is not None:
        await _memory.close()
    if _http is not None:
        await _http.aclose()


app = FastAPI(title="TinyToolLLM + ollama + surriti", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PromptRequest(BaseModel):
    text: str
    user_id: str = "default"


class PromptResponse(BaseModel):
    response: str
    recalled_facts: list[str] = []


class SendRequest(BaseModel):
    session_id: str
    user_id: str = "default"
    content: str
    # `mode` and `files` are accepted but ignored (CLI compatibility)
    mode: str | None = None
    files: list[dict] | None = None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def _is_connection_error(exc: BaseException) -> bool:
    """Is this exception a sign that surriti's WS to SurrealDB died?"""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in {"ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"}:
        return True
    return any(s in msg for s in (
        "no close frame", "connection closed", "connection reset",
        "broken pipe", "websocket", "not connected", "is not connected",
    ))


async def _reconnect_memory() -> bool:
    """Best-effort reconnect of the global surriti memory instance."""
    global _memory
    if _memory is None:
        return False
    log.warning("Reconnecting surriti memory to %s ...", SURREAL_URL)
    try:
        try:
            await _memory.close()
        except Exception:
            pass
        await _memory.connect()
        log.info("Surriti memory reconnected.")
        return True
    except Exception as exc:
        log.error("Surriti reconnect failed: %s", exc)
        return False


async def _memory_call(coro_factory):
    """Run a memory coroutine, reconnecting and retrying once on conn errors."""
    try:
        return await coro_factory()
    except Exception as exc:
        if not _is_connection_error(exc):
            raise
        log.warning("Memory op hit a connection error (%s); reconnecting.", exc)
        if not await _reconnect_memory():
            raise
        return await coro_factory()


EXTRACTION_INSTRUCTIONS = (
    "Extraction rules:\n"
    "1. Extract entities (people, places, organisations, foods, hobbies, "
    "topics) that are mentioned in the input. Things like 'pizza', "
    "'pasta', 'developer', 'Eds Pizza', 'Mark' are all valid entities.\n"
    "2. Pronouns are NOT entities. Never emit 'I', 'me', 'my', 'you', "
    "'he', 'she', 'they', 'user', 'assistant', or generic 'person' as "
    "entity names. When the speaker says 'I'/'my'/'me', resolve it to "
    "the user's known name from earlier context (e.g. 'Michael') and "
    "use that name as the entity.\n"
    "3. Predicate names must describe the actual relation. Good "
    "predicates: works_at, works_with, worked_with, lives_in, likes, "
    "dislikes, prefers, used_to_like, favorite_food, "
    "favorite_restaurant, knows, owns, is_a. Avoid 'related_to' unless "
    "absolutely nothing else fits.\n"
    "4. Do not invent example entities such as Alice, Bob, or Acme "
    "Corp. Only use names that appear in the input or earlier context.\n"
    "5. If a single statement involves multiple objects (e.g. 'Peter "
    "and Matt'), emit one separate fact per object.\n"
    "6. If the message is a greeting, question, or acknowledgement "
    "with no factual claim, return zero facts (entities are still ok)."
)


def _node_index(nodes: list) -> dict[str, str]:
    return {n.uuid: n.name for n in nodes}


def _edge_view(edge, name_by_uuid: dict[str, str], score: float | None = None) -> dict:
    return {
        "fact":   edge.fact or "",
        "name":   edge.name or "",
        "source": name_by_uuid.get(edge.source_node_uuid, edge.source_node_uuid),
        "target": name_by_uuid.get(edge.target_node_uuid, edge.target_node_uuid),
        "valid_at":   str(edge.valid_at) if edge.valid_at else None,
        "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
        "score": score,
    }


def _node_view(node) -> dict:
    return {
        "uuid":    node.uuid,
        "name":    node.name,
        "labels":  list(node.labels) if node.labels else [],
        "summary": node.summary or "",
    }


# ---------------------------------------------------------------------------
# Inference (Ollama)
# ---------------------------------------------------------------------------
async def _generate_blocking(messages: list[dict]) -> str:
    """One-shot generation used by the legacy /prompt endpoint."""
    assert _http is not None
    r = await _http.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": MAX_NEW_TOKENS},
        },
        timeout=None,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


async def _stream_generate(messages: list[dict], emit) -> str:
    """Stream tokens through ``emit`` and return the assembled response."""
    assert _http is not None
    pieces: list[str] = []
    async with _http.stream(
        "POST", f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": MAX_NEW_TOKENS},
        },
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            chunk = (evt.get("message") or {}).get("content", "")
            if chunk:
                pieces.append(chunk)
                await emit({"type": "chunk", "text": chunk})
            if evt.get("done"):
                break
    return "".join(pieces)


# ---------------------------------------------------------------------------
# The streaming turn -- writes events to a per-session queue
# ---------------------------------------------------------------------------
async def _run_turn(req: SendRequest) -> None:
    queue = _sessions.get(req.session_id)
    if queue is None:
        log.warning("No live websocket for session %s; dropping turn.", req.session_id)
        return

    async def emit(event: dict) -> None:
        await queue.put(event)

    try:
        recalled_facts: list[str] = []

        # ---- 1. Recall ------------------------------------------------------
        if _memory is not None:
            await emit({"type": "step_start", "label": "surriti_recall"})
            try:
                results = await _memory_call(
                    lambda: _memory.search(req.content, group_id=req.user_id)
                )
                names = _node_index(results.nodes)
                edges_view = [_edge_view(e, names, results.scores.get(e.uuid))
                              for e in results.edges[:8]]
                nodes_view = [_node_view(n) for n in results.nodes[:8]]
                recalled_facts = [e["fact"] for e in edges_view if e["fact"]]
                await emit({
                    "type":  "memory_recall",
                    "query": req.content,
                    "edges": edges_view,
                    "nodes": nodes_view,
                })
            except Exception as exc:
                log.warning("Memory recall failed: %s", exc)
                await emit({"type": "step_error", "label": "surriti_recall",
                            "error": str(exc)})
            finally:
                await emit({"type": "step_done", "label": "surriti_recall"})

        # ---- 2. Build prompt ------------------------------------------------
        messages: list[dict] = []
        if recalled_facts:
            ctx = "Relevant context from memory:\n" + "\n".join(
                f"- {f}" for f in recalled_facts
            )
            messages.append({"role": "system", "content": ctx})
        messages.append({"role": "user", "content": req.content})

        # ---- 3. Stream LLM tokens ------------------------------------------
        response_text = await _stream_generate(messages, emit)

        # ---- 4. Store -------------------------------------------------------
        if _memory is not None:
            await emit({"type": "step_start", "label": "surriti_store"})
            try:
                # Pull a few prior episodes for this user to give the
                # extractor turn-to-turn context (so "I like pizza" is
                # anchored to the speaker's earlier-introduced name, etc.).
                prev_uuids: list[str] = []
                try:
                    prev = await _memory_call(
                        lambda: _memory.retrieve_episodes(
                            group_ids=[req.user_id], last_n=4,
                        )
                    )
                    prev_uuids = [p.uuid for p in prev]
                except Exception as exc:
                    log.warning("retrieve_episodes failed: %s", exc)

                stored = await _memory_call(
                    lambda: _memory.add_episode(
                        name=f"turn-{req.user_id}",
                        episode_body=req.content,
                        group_id=req.user_id,
                        previous_episode_uuids=prev_uuids or None,
                        custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
                    )
                )
                names = _node_index(stored.nodes)
                new_facts = [_edge_view(e, names) for e in stored.edges]
                await emit({
                    "type":            "memory_store",
                    "episode_uuid":    stored.episode.uuid,
                    "entities_added":  len(stored.nodes),
                    "edges_added":     len(stored.edges),
                    "invalidated":     len(stored.invalidated_edges),
                    "new_facts":       new_facts,
                })
            except Exception as exc:
                log.warning("Memory store failed: %s", exc)
                await emit({"type": "step_error", "label": "surriti_store",
                            "error": str(exc)})
            finally:
                await emit({"type": "step_done", "label": "surriti_store"})

        await emit({"type": "done"})

    except Exception as exc:  # pragma: no cover
        log.exception("Turn failed")
        await emit({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints -- streaming
# ---------------------------------------------------------------------------
@app.post("/send")
async def send(req: SendRequest) -> dict:
    """Queue a streaming turn. The events are pushed to ``/ws/{session_id}``."""
    if req.session_id not in _sessions:
        return {"status": "no_websocket",
                "detail": f"No active websocket for session {req.session_id!r}"}
    asyncio.create_task(_run_turn(req))
    return {"status": "queued"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue
    log.info("WebSocket connected: session=%s", session_id)
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected: session=%s", session_id)
    finally:
        _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Endpoints -- legacy / diagnostic
# ---------------------------------------------------------------------------
@app.post("/prompt", response_model=PromptResponse)
async def prompt(req: PromptRequest) -> PromptResponse:
    recalled: list[str] = []
    if _memory is not None:
        try:
            results = await _memory_call(
                lambda: _memory.search(req.text, group_id=req.user_id)
            )
            recalled = [e.fact for e in results.edges[:5] if e.fact]
        except Exception as exc:
            log.warning("Memory search failed: %s", exc)

    messages: list[dict] = []
    if recalled:
        messages.append({"role": "system",
                         "content": "Relevant context from memory:\n"
                                    + "\n".join(f"- {f}" for f in recalled)})
    messages.append({"role": "user", "content": req.text})

    response_text = await _generate_blocking(messages)

    if _memory is not None:
        try:
            prev_uuids: list[str] = []
            try:
                prev = await _memory_call(
                    lambda: _memory.retrieve_episodes(
                        group_ids=[req.user_id], last_n=4,
                    )
                )
                prev_uuids = [p.uuid for p in prev]
            except Exception as exc:
                log.warning("retrieve_episodes failed: %s", exc)

            await _memory_call(
                lambda: _memory.add_episode(
                    name=f"turn-{req.user_id}",
                    episode_body=req.text,
                    group_id=req.user_id,
                    previous_episode_uuids=prev_uuids or None,
                    custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
                )
            )
        except Exception as exc:
            log.warning("Memory store failed: %s", exc)

    return PromptResponse(response=response_text, recalled_facts=recalled)


@app.get("/memory/{user_id}")
async def get_memory(user_id: str) -> dict:
    """List EVERY entity and edge stored under ``group_id == user_id``.

    Implemented as raw SELECTs (rather than a semantic ``search``) so the
    caller can audit the whole knowledge graph for that user.
    """
    if _memory is None:
        return {"status": "memory_unavailable", "nodes": [], "edges": []}

    from surriti.search import _unwrap
    from surriti.utils import parse_entity, parse_edge

    async def _dump():
        node_rows = _unwrap(await _memory.driver.query(
            "SELECT * FROM entity WHERE group_id = $g ORDER BY created_at;",
            {"g": user_id},
        ))
        edge_rows = _unwrap(await _memory.driver.query(
            "SELECT * FROM relates_to WHERE group_id = $g ORDER BY created_at;",
            {"g": user_id},
        ))
        return [parse_entity(r) for r in node_rows], \
               [parse_edge(r) for r in edge_rows]

    try:
        nodes, edges = await _memory_call(_dump)
    except Exception as exc:
        log.warning("Memory dump failed: %s", exc)
        return {"status": "error", "detail": str(exc),
                "nodes": [], "edges": []}

    names = _node_index(nodes)
    return {
        "status": "ok",
        "nodes":  [_node_view(n) for n in nodes],
        "edges":  [_edge_view(e, names) for e in edges],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "memory": _memory is not None,
        "model":  OLLAMA_MODEL,
        "extract_model": OLLAMA_EXTRACT_MODEL,
        "embed_model":   OLLAMA_EMBED_MODEL,
        "ollama": OLLAMA_URL,
        "active_sessions": list(_sessions.keys()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
