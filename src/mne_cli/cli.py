"""Entry point: parse arguments, run one command, print one JSON envelope."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Headless plotting must be configured before matplotlib/mne get imported.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_BROWSER_BACKEND", "matplotlib")

from . import __version__
from .commands import load_all
from .errors import UsageError, classify
from .registry import build_parser, coerce, dispatch_name, get
from .runner import execute


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _collect_params(ns, cmd) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for param in cmd.params:
        raw = getattr(ns, param.dest, None)
        if raw is None:
            values[param.dest] = param.default
        else:
            values[param.dest] = coerce(param, raw)
    return values


def _render_text(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        err = payload["error"]
        lines = [f"error [{err['code']}] {err['message']}"]
        if err.get("hint"):
            lines.append(f"hint: {err['hint']}")
        return "\n".join(lines)

    lines = [f"{payload['command']} ok in {payload.get('elapsed_ms', 0)} ms"]
    data = payload.get("data") or {}
    for key, value in data.items():
        if value is None or value == []:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"{key}: {text}")
    if payload.get("handle"):
        lines.append(f"handle: {payload['handle']}")
    for art in payload.get("artifacts") or []:
        lines.append(f"file: {art['path']}")
    for note in payload.get("warnings") or []:
        lines.append(f"warning: {note}")
    for step in payload.get("next") or []:
        lines.append(f"next: {step}")
    return "\n".join(lines)


def _emit(payload: dict[str, Any], fmt: str, code: int, quiet: bool = False) -> int:
    if quiet:  # stdio MCP server: stdout belongs to JSON-RPC, not to the envelope
        return code
    if fmt == "text":
        print(_render_text(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    return code


def main(argv: list[str] | None = None) -> int:
    _console()
    load_all()
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--version" in argv and len(argv) == 1:
        print(f"mne-cli {__version__}")
        return 0

    try:
        # `mne-cli epochs.build` and `mne-cli epochs build` are both accepted.
        if argv and "." in argv[0]:
            from .registry import REGISTRY

            if argv[0] in REGISTRY:
                argv[0:1] = argv[0].split(".", 1)
        ns = parser.parse_args(argv)
    except SystemExit as exc:  # --help and --version exit cleanly
        return int(exc.code or 2)
    except UsageError as exc:
        payload = {
            "ok": False,
            "command": None,
            "error": exc.to_dict(),
            "elapsed_ms": 0,
        }
        return _emit(payload, "json", exc.exit_code)

    name = dispatch_name(ns)
    if name is None:
        # Bare invocation is the discovery entry point.
        name = "commands"

    fmt = getattr(ns, "format", "json") or "json"
    store_dir = getattr(ns, "store", None)
    log_level = getattr(ns, "log_level", "WARNING") or "WARNING"

    started = time.perf_counter()
    try:
        cmd = get(name)
        params = _collect_params(ns, cmd)
        body, _ctx = execute(name, params, store_dir=store_dir, log_level=log_level)
        payload: dict[str, Any] = {"ok": True, "command": name, **body}
        payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return _emit(payload, fmt, 0, quiet=(name == "mcp.serve"))

    except BaseException as exc:  # noqa: BLE001 - every failure must be an envelope
        if isinstance(exc, KeyboardInterrupt):
            payload = {"ok": False, "command": name, "error": {"code": "INTERRUPTED", "message": "interrupted"}}
            return _emit(payload, fmt, 130, quiet=(name == "mcp.serve"))
        err = classify(exc) if not isinstance(exc, UsageError) else exc
        payload = {
            "ok": False,
            "command": name,
            "error": err.to_dict(),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        return _emit(payload, fmt, err.exit_code, quiet=(name == "mcp.serve"))


if __name__ == "__main__":
    raise SystemExit(main())
