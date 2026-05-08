"""Tiny FastAPI service backed by vLLM (OpenAI-compatible) + surriti memory.

This module models the *pip-import* developer experience for ``surriti``:
memory is built with ``surriti.Surriti.from_env(...)``, an in-process
``fastembed`` model is supplied as the embedder, and the LLM is reached via
the OpenAI Async client pointed at vLLM's ``/v1`` endpoint.

Endpoints
---------
GET  /health                   liveness probe
POST /prompt                   one-shot blocking inference
GET  /memory/{user_id}         dump every entity/edge for a user

POST /send                     queue a streaming turn (returns immediately)
WS   /ws/{session_id}          stream events for that session

Streaming event types (over the WebSocket)
------------------------------------------
{type:"step_start",  label:"surriti_recall"}
{type:"memory_recall", query, edges:[...], nodes:[...]}
{type:"step_done",   label:"surriti_recall"}

{type:"chunk",       text}                  # streamed LLM token(s)

{type:"step_start",  label:"surriti_store"}
{type:"memory_store",episode_uuid, entities_added, edges_added, invalidated,
                     new_facts:[...]}
{type:"step_done",   label:"surriti_store"}

{type:"done"}
{type:"error",       message}
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from pydantic import BaseModel

from surriti import Surriti
from surriti.driver import SurrealDriver
from surriti.llm_clients import OpenAILLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1").rstrip("/")
VLLM_MODEL    = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-4B")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBED_DIM     = int(os.environ.get("EMBED_DIM", "768"))
PORT          = int(os.environ.get("PORT", "3000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1536"))
CHAT_TEMPERATURE = float(os.environ.get("CHAT_TEMPERATURE", "0.3"))
# Qwen3-family thinking models leak `<think>` traces and burn the token
# budget reasoning before they ever emit user-visible text. We disable
# thinking by default for chat *and* extraction. Set to "1" to re-enable.
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
_CHAT_EXTRA_BODY: dict = {
    "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
}
# Short by default — docker-compose `depends_on: condition: service_healthy`
# already gates startup on vLLM. Override via env if running standalone.
VLLM_WAIT_S   = float(os.environ.get("VLLM_WAIT_S", "30"))

SURREAL_URL  = os.environ.get("SURRITI_SURREAL_URL",  "ws://surrealdb:8000/rpc")
SURREAL_NS   = os.environ.get("SURRITI_SURREAL_NS",   "myapp")
SURREAL_DB   = os.environ.get("SURRITI_SURREAL_DB",   "myapp")
# WARNING: "root"/"root" are dev/local defaults only. Set these env vars to
# strong credentials before any non-local deployment.
SURREAL_USER = os.environ.get("SURRITI_SURREAL_USER", "root")
SURREAL_PASS = os.environ.get("SURRITI_SURREAL_PASS", "root")


# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------
_memory: Surriti | None = None
_oa: AsyncOpenAI | None = None

# Per-session event queues, written by /send, drained by /ws/{session_id}
_sessions: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# In-process fastembed embedder for surriti
# ---------------------------------------------------------------------------
class FastEmbedEmbedder:
    """Implements ``surriti.embedder.EmbedderClient`` using fastembed."""

    def __init__(self, model_name: str, embedding_dim: int) -> None:
        from fastembed import TextEmbedding  # heavy import; do it lazily

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = TextEmbedding(model_name=model_name)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        # fastembed yields numpy arrays — convert to plain floats.
        out: list[list[float]] = []
        for vec in self._model.embed(texts):
            out.append([float(x) for x in vec])
        return out

    async def create(self, input_data: str) -> list[float]:
        text = (input_data or "").strip() or " "
        vecs = await asyncio.to_thread(self._embed_sync, [text])
        return vecs[0]

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        texts = [(t or "").strip() or " " for t in input_data]
        return await asyncio.to_thread(self._embed_sync, texts)


# ---------------------------------------------------------------------------
# vLLM readiness wait (build-once at lifespan)
# ---------------------------------------------------------------------------
async def _wait_for_vllm(url: str, timeout_s: float = 30.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as h:
        while True:
            try:
                r = await h.get(f"{url}/models")
                if r.status_code == 200:
                    log.info("vLLM ready at %s", url)
                    return
            except Exception:
                pass
            if asyncio.get_event_loop().time() >= deadline:
                raise RuntimeError(f"vLLM not reachable at {url} after {timeout_s}s")
            await asyncio.sleep(2.0)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory, _oa

    # OpenAI client pointed at vLLM. ``api_key`` is required by the SDK but
    # vLLM ignores it.
    _oa = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    try:
        await _wait_for_vllm(VLLM_BASE_URL, timeout_s=VLLM_WAIT_S)
    except Exception as exc:
        log.warning("vLLM readiness wait failed: %s -- continuing anyway.", exc)

    try:
        embedder = FastEmbedEmbedder(EMBED_MODEL, EMBED_DIM)
        llm_client = OpenAILLMClient(
            model=VLLM_MODEL, client=_oa, extra_body=_CHAT_EXTRA_BODY,
        )
        driver = SurrealDriver(
            url=SURREAL_URL, namespace=SURREAL_NS, database=SURREAL_DB,
            username=SURREAL_USER, password=SURREAL_PASS,
            embedding_dim=EMBED_DIM,
        )
        mem = Surriti(driver, llm_client=llm_client, embedder=embedder)
        await mem.connect()
        _memory = mem
        log.info(
            "Surriti memory connected to %s (model=%s embed=%s dim=%d)",
            SURREAL_URL, VLLM_MODEL, EMBED_MODEL, EMBED_DIM,
        )
    except Exception as exc:  # pragma: no cover - integration path
        log.warning("Surriti memory unavailable (%s) -- running without memory.", exc)
        _memory = None

    yield

    if _memory is not None:
        await _memory.close()
    if _oa is not None:
        await _oa.close()


app = FastAPI(title="surriti + vllm demo", lifespan=lifespan)


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
    # Accepted but ignored; kept for CLI compatibility.
    mode: str | None = None
    files: list[dict] | None = None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
EXTRACTION_INSTRUCTIONS = (
    "Extraction rules:\n"
    "1. Extract entities (people, places, organisations, foods, hobbies, "
    "topics) that are mentioned in the input.\n"
    "2. Pronouns are NOT entities. Never emit 'I', 'me', 'my', 'you', "
    "'he', 'she', 'they', 'user', 'assistant', or generic 'person' as "
    "entity names. When the speaker says 'I'/'my'/'me', resolve it to "
    "the user's known name from earlier context (e.g. 'Michael') and "
    "use that name as the entity.\n"
    "3. Predicate names must describe the actual relation as a "
    "snake_case verb phrase. Pick whatever verb fits the sentence; "
    "do NOT translate to a fixed vocabulary.\n"
    "4. Do not invent example entities (Alice, Bob, Acme Corp, etc.). "
    "Only use names that appear in the input or earlier context.\n"
    "5. If a single statement involves multiple objects (e.g. 'Peter "
    "and Matt'), emit one separate fact per object.\n"
    "6. If the message is a greeting, question, or acknowledgement "
    "with no factual claim, return zero facts (entities are still ok)."
)


async def _bootstrap_display_name(user_id: str, stored, current: str | None) -> None:
    """If the LLM extracted an ``is_named`` fact from the speaker to a real
    name (e.g. ``default -[is_named]-> Auley``), persist that name as the
    User node's ``display_name`` so subsequent turns get a richer speaker
    hint. No-op if a display_name is already set or no such edge exists.
    """
    if current or _memory is None:
        return
    nodes_by_uuid = {n.uuid: n for n in stored.nodes}
    for edge in stored.edges:
        if (edge.name or "").lower() != "is_named":
            continue
        src_node = nodes_by_uuid.get(edge.source_node_uuid)
        tgt_node = nodes_by_uuid.get(edge.target_node_uuid)
        if not src_node or not tgt_node:
            continue
        if src_node.name != user_id or not tgt_node.name:
            continue
        if tgt_node.name == user_id:
            continue
        try:
            await _memory.upsert_user(group_id=user_id, display_name=tgt_node.name)
            log.info("Bootstrapped display_name for %r: %r", user_id, tgt_node.name)
        except Exception as exc:
            log.warning("display_name bootstrap failed: %s", exc)
        return


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
# Inference (vLLM via OpenAI client)
# ---------------------------------------------------------------------------
async def _generate_blocking(messages: list[dict]) -> str:
    """One-shot generation used by the legacy /prompt endpoint."""
    assert _oa is not None
    resp = await _oa.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=CHAT_TEMPERATURE,
        stream=False,
        extra_body=_CHAT_EXTRA_BODY,
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


async def _stream_generate(messages: list[dict], emit) -> str:
    """Stream tokens through ``emit`` and return the assembled response."""
    assert _oa is not None
    pieces: list[str] = []
    stream = await _oa.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=CHAT_TEMPERATURE,
        stream=True,
        extra_body=_CHAT_EXTRA_BODY,
    )
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        chunk = (delta.content if delta else "") or ""
        if chunk:
            pieces.append(chunk)
            await emit({"type": "chunk", "text": chunk})
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
        recalled_nodes: list[dict] = []

        # ---- 1. Recall ------------------------------------------------------
        if _memory is not None:
            await emit({"type": "step_start", "label": "surriti_recall"})
            try:
                results = await _memory.search(req.content, group_id=req.user_id)
                names = _node_index(results.nodes)
                edges_view = [_edge_view(e, names, results.scores.get(e.uuid))
                              for e in results.edges[:8]]
                nodes_view = [_node_view(n) for n in results.nodes[:8]]
                recalled_facts = [e["fact"] for e in edges_view
                                  if e["fact"] and not e.get("invalid_at")]
                recalled_nodes = nodes_view
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
        sys_parts: list[str] = [
            "You are a helpful assistant with persistent memory about the "
            f"user (user_id={req.user_id}). The MEMORY section below was "
            "retrieved from your own knowledge graph for this user and is "
            "authoritative. Treat it as facts you already know. NEVER say "
            "things like 'I don't have access to personal information' or "
            "'I'm just an AI' when the answer is present in MEMORY -- "
            "instead, answer directly using the memory. If MEMORY is empty "
            "or doesn't cover the question, say so plainly and ask a brief "
            "follow-up."
        ]
        if recalled_nodes:
            ent_lines = [
                f"- {n['name']}" + (f" ({', '.join(n['labels'])})"
                                    if n.get('labels') else "")
                for n in recalled_nodes
            ]
            sys_parts.append("MEMORY entities:\n" + "\n".join(ent_lines))
        if recalled_facts:
            sys_parts.append(
                "MEMORY facts:\n" + "\n".join(f"- {f}" for f in recalled_facts)
            )
        if not recalled_facts and not recalled_nodes:
            sys_parts.append("MEMORY: (empty)")
        messages: list[dict] = [
            {"role": "system", "content": "\n\n".join(sys_parts)},
            {"role": "user", "content": req.content},
        ]

        # ---- 3. Stream LLM tokens ------------------------------------------
        await _stream_generate(messages, emit)

        # ---- 4. Store -------------------------------------------------------
        if _memory is not None:
            await emit({"type": "step_start", "label": "surriti_store"})
            try:
                prev_uuids: list[str] = []
                try:
                    prev = await _memory.retrieve_episodes(
                        group_ids=[req.user_id], last_n=4,
                    )
                    prev_uuids = [p.uuid for p in prev]
                except Exception as exc:
                    log.warning("retrieve_episodes failed: %s", exc)

                # Ensure the canonical User entity exists for this tenant
                # and look up the friendly display name (if the LLM has
                # already learned one from a prior turn). Both are passed
                # as speaker context so the extractor anchors first-person
                # pronouns to the right entity.
                speaker_name: str | None = None
                try:
                    user_node = await _memory.upsert_user(group_id=req.user_id)
                    speaker_name = (user_node.attributes or {}).get("display_name")
                except Exception as exc:
                    log.warning("upsert_user failed: %s", exc)

                stored = await _memory.add_episode(
                    name="chat",
                    episode_body=req.content,
                    group_id=req.user_id,
                    previous_episode_uuids=prev_uuids or None,
                    custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
                    speaker_id=req.user_id,
                    speaker_name=speaker_name,
                    source_type="user",
                )
                # Bootstrap display_name from any is_named edge that points
                # from the user_id to a real name. Self-healing: future
                # turns will see the rich speaker hint with the actual name.
                await _bootstrap_display_name(req.user_id, stored, current=speaker_name)
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
            results = await _memory.search(req.text, group_id=req.user_id)
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
                prev = await _memory.retrieve_episodes(
                    group_ids=[req.user_id], last_n=4,
                )
                prev_uuids = [p.uuid for p in prev]
            except Exception as exc:
                log.warning("retrieve_episodes failed: %s", exc)

            speaker_name: str | None = None
            try:
                un = await _memory.upsert_user(group_id=req.user_id)
                speaker_name = (un.attributes or {}).get("display_name")
            except Exception as exc:
                log.warning("upsert_user failed: %s", exc)

            stored = await _memory.add_episode(
                name="chat",
                episode_body=req.text,
                group_id=req.user_id,
                previous_episode_uuids=prev_uuids or None,
                custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
                speaker_id=req.user_id,
                speaker_name=speaker_name,
                source_type="user",
            )
            await _bootstrap_display_name(req.user_id, stored, current=speaker_name)
        except Exception as exc:
            log.warning("Memory store failed: %s", exc)

    return PromptResponse(response=response_text, recalled_facts=recalled)


@app.get("/memory/{user_id}")
async def get_memory(user_id: str) -> dict:
    """List EVERY entity and edge stored under ``group_id == user_id``."""
    if _memory is None:
        return {"status": "memory_unavailable", "nodes": [], "edges": []}

    from surriti.search import _unwrap
    from surriti.utils import parse_edge, parse_entity

    try:
        node_rows = _unwrap(await _memory.driver.query(
            "SELECT * FROM entity WHERE group_id = $g ORDER BY created_at;",
            {"g": user_id},
        ))
        edge_rows = _unwrap(await _memory.driver.query(
            "SELECT * FROM relates_to WHERE group_id = $g ORDER BY created_at;",
            {"g": user_id},
        ))
    except Exception as exc:
        log.warning("Memory dump failed: %s", exc)
        return {"status": "error", "detail": str(exc), "nodes": [], "edges": []}

    nodes = [parse_entity(r) for r in node_rows]
    edges = [parse_edge(r) for r in edge_rows]
    names = _node_index(nodes)
    return {
        "status": "ok",
        "nodes":  [_node_view(n) for n in nodes],
        "edges":  [_edge_view(e, names) for e in edges],
    }


@app.delete("/memory/{user_id}")
async def delete_memory(user_id: str) -> dict:
    """Wipe every entity, edge, episode, and mention for ``group_id == user_id``.

    Used by the CLI's ``/clear`` command to reset a tenant's graph between
    experiments. Idempotent: deleting an empty tenant returns ``deleted=0``.
    """
    if _memory is None:
        return {"status": "memory_unavailable", "deleted": 0}

    deleted = 0
    for table in ("relates_to", "mentions", "entity", "episode", "community",
                  "has_member"):
        try:
            await _memory.driver.query(
                f"DELETE {table} WHERE group_id = $g;",
                {"g": user_id},
            )
            deleted += 1
        except Exception as exc:
            log.warning("Delete from %s failed for %r: %s", table, user_id, exc)

    return {"status": "ok", "user_id": user_id, "tables_cleared": deleted}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "memory": _memory is not None,
        "model":  VLLM_MODEL,
        "embed_model": EMBED_MODEL,
        "embed_dim":   EMBED_DIM,
        "vllm":   VLLM_BASE_URL,
        "active_sessions": list(_sessions.keys()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
