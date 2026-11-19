"""MCP bridge commands."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..registry import Param, command


@command(
    "mcp.manifest",
    "Print the MCP tool manifest (name, description, JSON input schema) without starting a server.",
    params=[Param("out", "path", "optionally write the manifest to a .json file")],
    returns="{server, protocolVersion, count, tools[]}",
    examples=[{"args": "mcp manifest --out manifest.json"}],
    notes="Use this manifest when wiring mne-cli into an HTTP/JSON based agent runtime.",
    tags=("mcp", "meta"),
)
def cmd_mcp_manifest(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import json
    import sys

    from ..mcp import manifest

    payload = manifest()
    if p.get("out"):
        from pathlib import Path

        path = Path(p["out"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        ctx.add_artifact("json", path, "mcp manifest")
        payload["path"] = str(path.resolve())
    sys.stderr.write(f"mne-cli: {payload['count']} MCP tools ready\n")
    return payload


@command(
    "mcp.serve",
    "Run a stdio MCP server so an MCP client (Claude Desktop, Cursor...) can call every command.",
    params=[Param("log-level-mcp", "str", "accepted for symmetry; use the global --log-level", default="ERROR")],
    returns="{stopped, reason}",
    examples=[{"args": "mcp serve", "note": "configure this as the server command in your MCP client"}],
    notes="Protocol messages go to stdout; all MNE logs are redirected to stderr.",
    tags=("mcp",),
)
def cmd_mcp_serve(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    from ..mcp import serve

    try:
        serve(str(ctx.store.root), ctx.log_level)
    except KeyboardInterrupt:
        return {"stopped": True, "reason": "interrupted"}
    return {"stopped": True, "reason": "stdin closed"}
