"""Multi-step pipelines: an agent writes one JSON plan, the CLI runs it.

Step arguments may reference an earlier handle with ``{{name}}`` where ``name``
is the ``as`` label of a previous step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .context import Ctx
from .errors import MneCliError, UsageError
from .registry import get
from .runner import execute

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def load_spec(spec: str | None, spec_json: Any) -> dict[str, Any]:
    if spec_json is not None:
        if isinstance(spec_json, dict):
            return spec_json
        return json.loads(str(spec_json))
    if not spec:
        raise UsageError("no pipeline spec given", hint="Pass --spec plan.json or --spec-json '{...}'.")
    path = Path(spec).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if spec.strip().startswith("{"):
        return json.loads(spec)
    raise UsageError(f"pipeline spec not found: {spec}")


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in (args or {}).items():
        clean = str(key).lstrip("-").replace("-", "_")
        out[clean] = value
    return out


def _substitute(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        matches = PLACEHOLDER.findall(value)
        if not matches:
            return value
        if len(matches) == 1 and value.strip() == "{{%s}}" % matches[0]:
            if matches[0] not in bindings:
                raise UsageError(f"pipeline references unknown step '{matches[0]}'")
            return bindings[matches[0]]
        for name in matches:
            if name not in bindings:
                raise UsageError(f"pipeline references unknown step '{name}'")
            value = value.replace("{{%s}}" % name, str(bindings[name]))
        return value
    if isinstance(value, list):
        return [_substitute(v, bindings) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, bindings) for k, v in value.items()}
    return value


def run_pipeline(
    ctx: Ctx,
    spec: dict[str, Any],
    *,
    dry_run: bool = False,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    steps = spec.get("steps") or []
    if not steps:
        raise UsageError("pipeline has no steps", hint='Add {"steps": [{"command": "raw.load", "args": {...}}]}.')

    bindings: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    for index, step in enumerate(steps):
        name = step.get("command") or step.get("name")
        if not name:
            raise UsageError(f"step {index} has no command")
        args = _substitute(_normalize_args(step.get("args") or {}), bindings)

        try:
            cmd = get(name)
            unknown = sorted(set(args) - {p.dest for p in cmd.params})
            if unknown:
                raise UsageError(
                    f"step {index} ({name}): unknown parameter(s): {', '.join(unknown)}",
                    hint="Allowed: " + ", ".join(sorted(p.dest for p in cmd.params)),
                )

            if dry_run:
                trace.append({"index": index, "command": name, "args": args, "ok": True, "dry_run": True})
                if step.get("as"):
                    bindings[step["as"]] = f"<{step['as']}>"
                continue

            body, sub_ctx = execute(name, args, ctx=ctx)
            step_warnings: list[str] = []
            if step.get("as"):
                if not sub_ctx.primary_handle:
                    # e.g. `psd` returns numbers but no object: keep going, the
                    # binding is simply left unbound and later steps fail loudly.
                    step_warnings.append(
                        f"step {index} ({name}) returned no handle; '{step['as']}' is unbound"
                    )
                else:
                    bindings[step["as"]] = sub_ctx.primary_handle

            trace.append(
                {
                    "index": index,
                    "command": name,
                    "ok": True,
                    "handle": sub_ctx.primary_handle,
                    "artifacts": sub_ctx.artifacts,
                    "summary": _brief(body),
                    **({"warnings": step_warnings} if step_warnings else {}),
                }
            )
            for note in sub_ctx.artifacts:
                ctx.artifacts.append(note)
        except BaseException as exc:  # noqa: BLE001 - record and optionally stop
            message = str(exc)
            trace.append({"index": index, "command": name, "ok": False, "args": args, "error": message})
            if stop_on_error:
                details = {"failed_step": index, "failed_command": name, "steps": trace, "bindings": bindings}
                if isinstance(exc, MneCliError):
                    exc.details = {**(exc.details or {}), **details}
                    raise
                raise MneCliError(f"step {index} ({name}) failed: {message}", details=details) from None

    if bindings:
        ctx.primary_handle = next(reversed(bindings.values())) if bindings else None

    return {
        "n_steps": len(steps),
        "n_failed": sum(1 for t in trace if not t["ok"]),
        "bindings": bindings,
        "steps": trace,
        "dry_run": dry_run,
    }


def _brief(body: dict[str, Any], keys: int = 6) -> dict[str, Any]:
    """Keep the pipeline trace small: a few scalar highlights per step."""
    data = body.get("data") or {}
    keep = {}
    for key, value in data.items():
        if len(keep) >= keys:
            break
        if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and len(value) <= 6):
            keep[key] = value
    if body.get("handle"):
        keep["handle"] = body["handle"]
    return keep
