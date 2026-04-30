"""Terminal frontend for the surriti-backed myapp service.

Usage
-----
    python cli.py --http http://localhost:3000 --ws ws://localhost:3000

The CLI POSTs each line to ``/send`` and renders the streaming events that the
service pushes back over ``/ws/{session_id}``:

  * a "graph recall" panel showing the facts + entities surriti retrieved
  * the assistant's streamed answer
  * a "graph store" panel showing what was added/invalidated this turn
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from typing import Optional

import httpx
import websockets
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

PROMPT = "\033[1;92m> \033[0m"
SPINNER = ["|", "/", "-", "\\"]
RECV_TIMEOUT = 180.0


async def ainput(prompt: str = "") -> str:
    return await asyncio.to_thread(lambda: input(prompt))


# ---------------------------------------------------------------------------
# Per-turn state
# ---------------------------------------------------------------------------

class TurnState:
    __slots__ = (
        "recall_active", "recall_query", "recall_edges", "recall_nodes",
        "store_active", "store_summary",
        "response", "responding",
        "activity", "error",
        "spinner_t0", "spinner_frame",
    )

    def __init__(self) -> None:
        self.recall_active: bool = False
        self.recall_query: Optional[str] = None
        self.recall_edges: list[dict] = []
        self.recall_nodes: list[dict] = []
        self.store_active: bool = False
        self.store_summary: Optional[dict] = None
        self.response: str = ""
        self.responding: bool = False
        self.activity: list[tuple[str, str]] = []
        self.error: Optional[str] = None
        self.spinner_t0: float = time.time()
        self.spinner_frame: int = 0

    def add_activity(self, style: str, text: str) -> None:
        self.activity.append((style, text))
        if len(self.activity) > 8:
            self.activity = self.activity[-8:]

    def append_response(self, text: str) -> None:
        if not text:
            return
        self.responding = True
        self.response += text

    def tick(self) -> None:
        self.spinner_frame = (self.spinner_frame + 1) % len(SPINNER)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_recall(s: TurnState) -> Optional[RenderableType]:
    if not (s.recall_active or s.recall_edges or s.recall_nodes):
        return None

    parts: list[RenderableType] = []

    if s.recall_query:
        parts.append(Text.assemble(("query  ", "dim"), (s.recall_query, "white")))

    if s.recall_edges:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
                    expand=True, padding=(0, 1))
        tbl.add_column("source", style="bright_cyan", overflow="fold")
        tbl.add_column("rel",    style="magenta",     overflow="fold")
        tbl.add_column("target", style="bright_cyan", overflow="fold")
        tbl.add_column("fact",   style="white",       overflow="fold")
        tbl.add_column("score",  style="dim",         justify="right", width=6)
        for e in s.recall_edges:
            score = f"{e['score']:.2f}" if isinstance(e.get("score"), (int, float)) else "-"
            tbl.add_row(e.get("source") or "-", e.get("name") or "relates_to",
                        e.get("target") or "-", e.get("fact") or "", score)
        parts.append(tbl)
    elif s.recall_active:
        parts.append(Text(f"  {SPINNER[s.spinner_frame]} searching...", style="dim magenta"))
    else:
        parts.append(Text("  (no facts recalled)", style="dim"))

    if s.recall_nodes:
        node_lines = []
        for n in s.recall_nodes:
            label = ", ".join(n.get("labels") or []) or "Entity"
            node_lines.append(
                Text.assemble(
                    ("- ", "dim"),
                    (n.get("name") or "?", "bright_cyan"),
                    ("  [", "dim"), (label, "dim italic"), ("]", "dim"),
                )
            )
        parts.append(Panel(
            Group(*node_lines),
            title="[dim]entities[/]",
            border_style="dim magenta",
            box=box.MINIMAL,
            padding=(0, 1),
        ))

    return Panel(
        Group(*parts),
        title="[bold magenta]graph recall[/]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _render_store(s: TurnState) -> Optional[RenderableType]:
    if not (s.store_active or s.store_summary):
        return None

    if s.store_active and not s.store_summary:
        body: RenderableType = Text(
            f"  {SPINNER[s.spinner_frame]} storing episode...", style="dim green",
        )
    else:
        sm = s.store_summary or {}
        head = Text.assemble(
            ("entities ", "dim"), (str(sm.get("entities_added", 0)), "bright_green"),
            ("   facts ",  "dim"), (str(sm.get("edges_added", 0)),    "bright_green"),
            ("   invalidated ", "dim"),
            (str(sm.get("invalidated", 0)),
             "yellow" if sm.get("invalidated") else "dim"),
        )
        rows: list[RenderableType] = [head]
        new_facts = sm.get("new_facts") or []
        if new_facts:
            tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
            tbl.add_column(style="bright_green", overflow="fold")
            tbl.add_column(style="magenta",      overflow="fold")
            tbl.add_column(style="bright_green", overflow="fold")
            tbl.add_column(style="white",        overflow="fold")
            for f in new_facts:
                tbl.add_row(f.get("source") or "-", f.get("name") or "relates_to",
                            f.get("target") or "-", f.get("fact") or "")
            rows.append(tbl)
        body = Group(*rows)

    return Panel(
        body,
        title="[bold green]graph store[/]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_turn(s: TurnState) -> RenderableType:
    panels: list[RenderableType] = []

    recall = _render_recall(s)
    if recall is not None:
        panels.append(recall)

    if s.activity:
        rows = [Text.from_markup(f"[{st}]{tx}[/{st}]") for st, tx in s.activity]
        panels.append(Panel(
            Group(*rows),
            title="[dim]activity[/]",
            border_style="bright_black",
            box=box.MINIMAL,
            padding=(0, 1),
        ))

    if s.response.strip() or s.responding:
        body: RenderableType = Markdown(s.response if s.response.strip() else " ")
        panels.append(Panel(
            body,
            title="[bold bright_white]assistant[/]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    store = _render_store(s)
    if store is not None:
        panels.append(store)

    if s.error:
        panels.append(Panel(
            Text(s.error, style="red"),
            title="[red]error[/]",
            border_style="red",
            box=box.ROUNDED,
        ))

    return Group(*panels) if panels else Text("")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def print_header(session_id: str, user_id: str) -> None:
    console.print(Panel.fit(
        "[bold bright_cyan]surriti chat[/]\n"
        "[dim]temporal knowledge graph + tiny LLM[/]",
        border_style="bright_cyan",
        box=box.ROUNDED,
    ))
    console.print(Text.assemble(
        ("session  ", "dim"), (session_id, "bright_cyan"), "\n",
        ("user     ", "dim"), (user_id, "bright_cyan"), "\n",
        ("/quit  /memory  /clear  /screen", "dim"),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def _show_memory(client: httpx.AsyncClient, http_base: str, user_id: str) -> None:
    try:
        resp = await client.get(f"{http_base}/memory/{user_id}")
        resp.raise_for_status()
    except httpx.RequestError as exc:
        console.print(f"[red]connection error:[/] {exc}")
        return
    data = resp.json()
    edges = data.get("edges") or []
    nodes = data.get("nodes") or []

    if not edges and not nodes:
        console.print(f"[dim]no memory yet for user {user_id!r}[/]")
        return

    if edges:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
                    title=f"[bold]all facts for {user_id}[/]", title_justify="left")
        tbl.add_column("source", style="bright_cyan")
        tbl.add_column("rel",    style="magenta")
        tbl.add_column("target", style="bright_cyan")
        tbl.add_column("fact",   style="white", overflow="fold")
        tbl.add_column("valid_at", style="dim")
        for e in edges:
            tbl.add_row(e.get("source") or "-", e.get("name") or "-",
                        e.get("target") or "-", e.get("fact") or "",
                        (e.get("valid_at") or "").split(".")[0])
        console.print(tbl)

    if nodes:
        console.print(Text.assemble(("entities: ", "dim"),
                                    *(Text.assemble((n.get("name") or "?", "bright_cyan"),
                                                    "  ") for n in nodes)))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(http_base: str, ws_base: str, session_id: str, user_id: str) -> None:
    http_base = http_base.rstrip("/")
    ws_base   = ws_base.rstrip("/")
    ws_url   = f"{ws_base}/ws/{session_id}"
    send_url = f"{http_base}/send"

    print_header(session_id, user_id)

    async with (
        websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws,
        httpx.AsyncClient(timeout=30.0) as client,
    ):
        while True:
            try:
                line = (await ainput(PROMPT)).strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"/quit", "/exit", "exit", "quit"}:
                break
            if line == "/memory":
                await _show_memory(client, http_base, user_id)
                continue
            if line == "/clear":
                # Destructive: wipe the tenant's graph on the server, then
                # clear the terminal. Use /screen for screen-only clear.
                try:
                    resp = await client.delete(f"{http_base}/memory/{user_id}")
                    if resp.status_code >= 400:
                        console.print(
                            f"[red]server refused delete: http {resp.status_code} "
                            f"{resp.text.strip()}[/]"
                        )
                    else:
                        console.print(
                            f"[dim]wiped memory for user {user_id!r}[/]"
                        )
                except httpx.RequestError as exc:
                    console.print(f"[red]connection error:[/] {exc}")
                console.clear()
                print_header(session_id, user_id)
                continue
            if line == "/screen":
                console.clear()
                print_header(session_id, user_id)
                continue

            # POST the message to /send
            body = {"session_id": session_id, "user_id": user_id, "content": line}
            try:
                resp = await client.post(send_url, json=body)
            except httpx.RequestError as exc:
                console.print(f"[red]connection error:[/] {exc}")
                continue
            if resp.status_code >= 400:
                console.print(f"[red]http {resp.status_code}:[/] {resp.text.strip()}")
                continue
            data = resp.json()
            if data.get("status") == "no_websocket":
                console.print(f"[red]server says: {data.get('detail')}[/]")
                continue

            # Stream the turn
            turn = TurnState()
            done = False

            async def _tick() -> None:
                try:
                    while True:
                        await asyncio.sleep(0.15)
                        turn.tick()
                        live.update(render_turn(turn))
                except asyncio.CancelledError:
                    pass

            with Live(render_turn(turn), console=console, refresh_per_second=10,
                      transient=False) as live:
                tick_task = asyncio.create_task(_tick())
                try:
                    while not done:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                        except asyncio.TimeoutError:
                            turn.error = "recv timeout - server unresponsive"
                            done = True
                            break
                        except websockets.ConnectionClosed:
                            turn.error = "websocket closed unexpectedly"
                            done = True
                            break

                        msg = json.loads(raw)
                        t = msg.get("type")

                        if t == "step_start":
                            label = msg.get("label", "?")
                            if label == "surriti_recall":
                                turn.recall_active = True
                            elif label == "surriti_store":
                                turn.store_active = True
                            turn.add_activity("dim cyan", f"> {label}")

                        elif t == "step_done":
                            label = msg.get("label", "?")
                            if label == "surriti_recall":
                                turn.recall_active = False
                            elif label == "surriti_store":
                                turn.store_active = False
                            turn.add_activity("green", f"  done: {label}")

                        elif t == "step_error":
                            label = msg.get("label", "?")
                            turn.add_activity("red", f"  error: {label}: "
                                                     f"{msg.get('error', '')}")
                            if label == "surriti_recall":
                                turn.recall_active = False
                            elif label == "surriti_store":
                                turn.store_active = False

                        elif t == "memory_recall":
                            turn.recall_query = msg.get("query")
                            turn.recall_edges = msg.get("edges") or []
                            turn.recall_nodes = msg.get("nodes") or []

                        elif t == "memory_store":
                            turn.store_summary = msg

                        elif t == "chunk":
                            turn.append_response(msg.get("text", ""))

                        elif t == "done":
                            turn.responding = False
                            done = True

                        elif t == "error":
                            turn.error = msg.get("message", "?")
                            done = True

                        else:
                            turn.add_activity("dim", f"[ws] {json.dumps(msg)}")

                        live.update(render_turn(turn))
                finally:
                    tick_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await tick_task

            console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat for surriti+myapp")
    parser.add_argument("--http", default="http://localhost:3000",
                        help="myapp HTTP base URL")
    parser.add_argument("--ws", default="ws://localhost:3000",
                        help="myapp WebSocket base URL")
    parser.add_argument("--session", default="dev", help="Session ID")
    parser.add_argument("--user", default="default", help="User ID")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.http, args.ws, args.session, args.user))
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted.[/]")


if __name__ == "__main__":
    main()
