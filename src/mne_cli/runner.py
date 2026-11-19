"""Single execution path shared by the CLI, the MCP server and pipelines."""

from __future__ import annotations

import warnings
from typing import Any

from .context import Ctx, default_store_dir
from .errors import UsageError
from .registry import get
from .result import to_jsonable
from .store import Store

_MNE_CONFIGURED = False

# Commands that never touch scientific computing: skip the expensive mne import.
_NO_MNE = {
    "commands",
    "describe",
    "version",
    "store.list",
    "store.show",
    "store.drop",
    "store.clear",
    "mcp.manifest",
    "dataset.list",
    "system.doctor",
}


def configure_mne(log_level: str = "WARNING") -> None:
    global _MNE_CONFIGURED
    import matplotlib

    matplotlib.use("Agg", force=True)
    import mne

    mne.set_log_level(log_level)
    _MNE_CONFIGURED = True


def execute(
    name: str,
    params: dict[str, Any],
    *,
    ctx: Ctx | None = None,
    store: Store | None = None,
    store_dir: str | None = None,
    log_level: str = "WARNING",
) -> tuple[dict[str, Any], Ctx]:
    """Run one command, returning the result body and the context it used."""
    cmd = get(name)
    allowed = {p.dest for p in cmd.params}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise UsageError(
            f"unknown parameter(s) for {name}: {', '.join(unknown)}",
            hint="Allowed: " + (", ".join(sorted(allowed)) or "(none)"),
        )

    resolved = {}
    for param in cmd.params:
        if param.dest in params and params[param.dest] is not None:
            resolved[param.dest] = params[param.dest]
        else:
            resolved[param.dest] = param.default

    if ctx is None:
        store_obj = store or Store(default_store_dir(store_dir))
        ctx = Ctx(store=store_obj, command=name, params=resolved, log_level=log_level)
    else:
        ctx = Ctx(store=ctx.store, command=name, params=resolved, fmt=ctx.fmt, log_level=log_level)

    configure_mne(log_level) if name not in _NO_MNE else None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = cmd.handler(ctx, resolved) or {}

    for item in caught:
        text = f"{item.category.__name__}: {item.message}"
        if len(text) > 500:
            text = text[:500] + "…"
        ctx.warnings.append(text)

    body: dict[str, Any] = {"data": to_jsonable(data if isinstance(data, dict) else {"result": data})}
    if ctx.primary_handle:
        body["handle"] = ctx.primary_handle
    if ctx.artifacts:
        body["artifacts"] = ctx.artifacts
    if ctx.warnings:
        body["warnings"] = ctx.warnings[:20]
    if ctx.next_steps:
        body["next"] = ctx.next_steps
    body["store"] = str(ctx.store.root)
    return body, ctx
