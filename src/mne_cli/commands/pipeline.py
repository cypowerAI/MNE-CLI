"""One-shot multi-step execution, so an agent needs a single call per analysis."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..pipeline import load_spec, run_pipeline
from ..registry import Param, command


@command(
    "run",
    "Execute a multi-step JSON pipeline (load -> filter -> epochs -> average ...) in one call.",
    params=[
        Param("spec", "str", "path to a pipeline .json file, or inline JSON"),
        Param("spec-json", "json", "inline pipeline object"),
        Param("dry-run", "bool", "validate the plan without executing it", default=False),
        Param("stop-on-error", "bool", "abort at the first failing step", default=True),
    ],
    returns="{n_steps, n_failed, bindings, steps: [{index, command, ok, handle}]}",
    examples=[
        {"args": "run --spec pipeline.json"},
        {
            "args": "run --spec-json '{\"steps\":[{\"command\":\"raw.load\",\"args\":{\"path\":\"x.fif\"},\"as\":\"r\"},"
            "{\"command\":\"raw.filter\",\"args\":{\"raw\":\"{{r}}\",\"l-freq\":1,\"h-freq\":40},\"as\":\"f\"}]}'",
            "note": "{{r}} is replaced by the handle of the step labelled r",
        },
        {"args": "run --spec pipeline.json --dry-run"},
    ],
    notes="Pipeline steps share one store, so later steps can address earlier handles.",
    tags=("workflow",),
)
def cmd_run(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    spec = load_spec(p.get("spec"), p.get("spec_json"))
    return run_pipeline(
        ctx,
        spec,
        dry_run=bool(p.get("dry_run", False)),
        stop_on_error=bool(p.get("stop_on_error", True)),
    )
