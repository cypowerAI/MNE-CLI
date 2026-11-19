"""Minimal stdio MCP server.

Every CLI command becomes an MCP tool, so Claude Desktop, Cursor, Continue and
similar clients can call MNE analyses directly. No SDK dependency: the protocol
is newline-delimited JSON-RPC 2.0, which is what current MCP clients speak.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .context import default_store_dir
from .errors import classify
from .registry import mcp_tools
from .runner import execute
from .store import Store

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


def manifest() -> dict[str, Any]:
    tools = mcp_tools()
    for tool in tools:
        tool.pop("_command", None)
    return {
        "server": {"name": "mne-cli", "version": __version__, "transport": "stdio"},
        "protocolVersion": PROTOCOL_VERSION,
        "tools": tools,
        "count": len(tools),
    }


def serve(store_dir: str | None = None, log_level: str = "ERROR") -> None:
    store = Store(default_store_dir(store_dir))
    tools = {t["_command"]: t for t in mcp_tools()}
    out = sys.stdout

    def send(payload: dict[str, Any]) -> None:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        out.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue

        if isinstance(msg, list):  # batch
            requests = msg
        else:
            requests = [msg]

        for request in requests:
            response = _handle(request, tools, store, log_level)
            if response is not None:
                send(response)


def _handle(request: dict[str, Any], tools: dict[str, Any], store: Store, log_level: str) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    notify = request_id is None

    if method is None:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "invalid request"}}

    if method in {"notifications/initialized", "initialized", "notifications/cancelled"}:
        return None

    if method == "initialize":
        version = (request.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version if version in SUPPORTED_VERSIONS else PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mne-cli", "version": __version__},
                "instructions": (
                    "mne-cli wraps MNE-Python. Call mne_cli_commands first to see every tool, "
                    "then mne_cli_describe for exact parameters. Objects are passed between tools "
                    "as handles such as raw_0001."
                ),
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    if method == "tools/list":
        payload = [{k: v for k, v in tool.items() if k != "_command"} for tool in tools.values()]
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": payload}}

    if method == "tools/call":
        params = request.get("params") or {}
        tool_name = params.get("name")
        args = params.get("arguments") or {}
        command = tool_name
        if tool_name not in tools:
            command = next((c for t, c in ((v["name"], k) for k, v in tools.items()) if t == tool_name), None)
        if not command:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": tool_name}}, ensure_ascii=False)}],
                    "isError": True,
                },
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": _call(command, args, store, log_level)}

    if method in {"resources/list", "prompts/list", "resources/templates/list"}:
        key = {"resources/list": "resources", "prompts/list": "prompts", "resources/templates/list": "resourceTemplates"}[method]
        return {"jsonrpc": "2.0", "id": request_id, "result": {key: []}}

    if notify:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


def _call(command: str, args: dict[str, Any], store: Store, log_level: str) -> dict[str, Any]:
    original = sys.stdout
    sys.stdout = sys.stderr  # keep the protocol channel clean
    try:
        body, _ctx = execute(command, args, store=store, log_level=log_level)
        payload = {"ok": True, "command": command, **body}
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str, indent=1)}], "isError": False}
    except BaseException as exc:  # noqa: BLE001
        err = classify(exc)
        payload = {"ok": False, "command": command, "error": err.to_dict()}
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}], "isError": True}
    finally:
        sys.stdout = original
