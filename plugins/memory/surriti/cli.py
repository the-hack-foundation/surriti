"""Hermes CLI subcommands for the Surriti memory provider.

Discovered automatically by Hermes when this provider is the active
``memory.provider`` in config. Adds:

    hermes surriti status        — show service health and config
    hermes surriti dump          — dump all entities and edges for the user
    hermes surriti recall QUERY  — preview what a query would recall
    hermes surriti clear         — wipe all memory for the user (with --yes)
    hermes surriti config        — show the active config file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

CONFIG_FILENAME = "surriti.json"
DEFAULT_URL = "http://localhost:3000"


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    return Path(home) if home else Path.home() / ".hermes"


def _load_config() -> dict:
    path = _hermes_home() / CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _url() -> str:
    return (os.environ.get("SURRITI_URL") or _load_config().get("url") or DEFAULT_URL).rstrip("/")


def _user_id() -> str:
    return (
        os.environ.get("SURRITI_USER_ID")
        or _load_config().get("user_id")
        or "default"
    )


def _cmd_status() -> int:
    url = _url()
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{url}/health")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        print(f"ERROR: cannot reach {url}: {exc}", file=sys.stderr)
        return 1
    print(f"Surriti service: {url}")
    print(f"  status:      {data.get('status')}")
    print(f"  memory:      {data.get('memory')}")
    print(f"  model:       {data.get('model')}")
    print(f"  embed_model: {data.get('embed_model')}")
    print(f"  user_id:     {_user_id()}")
    return 0


def _cmd_dump() -> int:
    url = _url()
    uid = _user_id()
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{url}/memory/{uid}")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        print(f"ERROR: dump failed: {exc}", file=sys.stderr)
        return 1
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    print(f"User: {uid}    Entities: {len(nodes)}    Edges: {len(edges)}\n")
    if nodes:
        print("=== Entities ===")
        for n in sorted(nodes, key=lambda x: x.get("name", "")):
            labels = ", ".join(n.get("labels") or [])
            tag = f" [{labels}]" if labels else ""
            print(f"  - {n.get('name')}{tag}")
        print()
    if edges:
        print("=== Facts ===")
        for e in edges:
            fact = (e.get("fact") or "").strip()
            inv = " [INVALID]" if e.get("invalid_at") else ""
            print(f"  - {e.get('source')} -[{e.get('name')}]-> {e.get('target')}{inv}")
            if fact:
                print(f"      \"{fact}\"")
    return 0


def _cmd_recall(query: str) -> int:
    url = _url()
    uid = _user_id()
    if not query:
        print("ERROR: query required", file=sys.stderr)
        return 2
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{url}/recall",
                json={"query": query, "user_id": uid, "limit": 10},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        print(f"ERROR: recall failed: {exc}", file=sys.stderr)
        return 1
    facts = data.get("facts", [])
    entities = data.get("entities", [])
    print(f"Query: {query!r}    user_id={uid}    facts={len(facts)} entities={len(entities)}\n")
    if entities:
        print("=== Entities ===")
        for e in entities:
            print(f"  - {e.get('name')}")
        print()
    if facts:
        print("=== Facts ===")
        for f in facts:
            text = (f.get("fact") or "").strip()
            score = f.get("score")
            score_str = f"  (score={score:.3f})" if isinstance(score, (int, float)) else ""
            print(f"  - {text}{score_str}")
    return 0


def _cmd_clear(yes: bool) -> int:
    uid = _user_id()
    if not yes:
        print(f"This will wipe ALL memory for user_id={uid!r}.")
        print("Re-run with --yes to confirm.")
        return 2
    url = _url()
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.delete(f"{url}/memory/{uid}")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        print(f"ERROR: clear failed: {exc}", file=sys.stderr)
        return 1
    print(f"Cleared user_id={uid!r}: {data}")
    return 0


def _cmd_config() -> int:
    cfg_path = _hermes_home() / CONFIG_FILENAME
    cfg = _load_config()
    print(f"Config file: {cfg_path} ({'exists' if cfg_path.exists() else 'missing'})")
    print(f"Active URL:  {_url()}")
    print(f"Active user_id: {_user_id()}")
    if cfg:
        print("Contents:")
        print(json.dumps(cfg, indent=2))
    return 0


def surriti_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "surriti_command", None)
    if sub == "status":
        return _cmd_status()
    if sub == "dump":
        return _cmd_dump()
    if sub == "recall":
        return _cmd_recall(getattr(args, "query", ""))
    if sub == "clear":
        return _cmd_clear(getattr(args, "yes", False))
    if sub == "config":
        return _cmd_config()
    print("Usage: hermes surriti <status|dump|recall|clear|config>")
    return 2


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the `hermes surriti` argparse tree."""
    subs = subparser.add_subparsers(dest="surriti_command")
    subs.add_parser("status", help="Show Surriti service health and active config")
    subs.add_parser("dump", help="Dump all entities and edges for the active user")
    p_recall = subs.add_parser("recall", help="Preview what a query would recall")
    p_recall.add_argument("query", help="Search query")
    p_clear = subs.add_parser("clear", help="Wipe all memory for the active user")
    p_clear.add_argument("--yes", action="store_true", help="Skip confirmation")
    subs.add_parser("config", help="Show the active Surriti config")
    subparser.set_defaults(func=surriti_command)
