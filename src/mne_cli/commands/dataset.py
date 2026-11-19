"""Sample datasets, so an agent can bootstrap without hunting for recordings."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import UsageError
from ..registry import Param, command


def _dataset_modules():
    import mne.datasets as datasets

    found = {}
    for name in dir(datasets):
        if name.startswith("_"):
            continue
        obj = getattr(datasets, name)
        if inspect.ismodule(obj) and hasattr(obj, "data_path"):
            found[name] = obj
    return found


@command(
    "dataset.list",
    "List the built-in MNE sample datasets this CLI can download.",
    returns="{datasets: [{name, summary}]}",
    examples=[{"args": "dataset list"}],
    tags=("data",),
)
def cmd_dataset_list(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    items = []
    for name, module in sorted(_dataset_modules().items()):
        doc = (inspect.getdoc(module) or "").splitlines()
        summary = next((line.strip() for line in doc if line.strip()), "")
        items.append({"name": name, "summary": summary[:200]})
    return {"datasets": items, "n": len(items)}


@command(
    "dataset.demo",
    "Generate a small synthetic EEG recording (alpha rhythm, blinks, 2 event types) to try the CLI.",
    params=[
        Param("out", "path", "output .fif path", default="demo_raw.fif"),
        Param("duration", "float", "length in seconds", default=60.0),
        Param("sfreq", "float", "sampling rate in Hz", default=250.0),
        Param("n-channels", "int", "number of EEG channels (max 16, 10-20 names)", default=16),
        Param("n-events", "int", "number of trials, alternating code 1 (audio) and 2 (visual)", default=40),
        Param("seed", "int", "random seed", default=0),
    ],
    returns="{path, n_channels, duration_s, sfreq, n_events, event_id}",
    examples=[
        {"args": "dataset demo --out demo_raw.fif"},
        {"args": "dataset demo --out demo_raw.fif --duration 30 --n-events 20"},
    ],
    notes="Synthetic data: useful for wiring up a pipeline, never for scientific claims.",
    tags=("data", "demo"),
)
def cmd_dataset_demo(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    from ..demo import make_demo_raw

    raw, events = make_demo_raw(
        n_channels=int(p.get("n_channels") or 16),
        sfreq=float(p.get("sfreq") or 250.0),
        duration=float(p.get("duration") or 60.0),
        n_events=int(p.get("n_events") or 40),
        seed=int(p.get("seed") or 0),
    )
    out = Path(p.get("out") or "demo_raw.fif").expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(out), overwrite=True, verbose="ERROR")
    ctx.add_artifact("fif", out, "demo raw")
    return {
        "path": str(out),
        "n_channels": len(raw.ch_names),
        "duration_s": round(float(raw.times[-1]), 2),
        "sfreq": float(raw.info["sfreq"]),
        "n_events": int(len(events)),
        "event_id": {"audio": 1, "visual": 2},
        "hint": f"mne-cli raw load --path {out}",
    }


@command(
    "dataset.download",
    "Download (or locate) an MNE sample dataset and return its path.",
    params=[
        Param("name", "str", "dataset name from `dataset list`, e.g. sample, eegbci, testing", required=True, example="sample"),
        Param("path", "path", "base directory to download into (default: MNE data dir)"),
        Param("force-update", "bool", "re-download even if present", default=False),
        Param("subjects", "csv", "for datasets with subjects (e.g. eegbci): 1,2,3", example="1,2"),
        Param("runs", "csv", "for run-based datasets (e.g. eegbci): 3,4", example="3,4"),
    ],
    returns="{name, paths: [...], n_paths, existed_before}",
    examples=[
        {"args": "dataset download --name eegbci --subjects 1 --runs 3"},
        {"args": "dataset download --name sample", "note": "about 1.5 GB"},
    ],
    notes="Downloads can be large; `sample` is ~1.5 GB, `eegbci` a few MB.",
    tags=("data",),
)
def cmd_dataset_download(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    modules = _dataset_modules()
    name = p["name"]
    if name not in modules:
        raise UsageError(
            f"unknown dataset: {name}",
            hint="Run `mne-cli dataset list` to see the available names.",
        )
    module = modules[name]

    kwargs: dict[str, Any] = {
        "download": True,
        "force_update": bool(p.get("force_update")),
        "verbose": "ERROR",
    }
    if p.get("path"):
        kwargs["path"] = str(Path(p["path"]).expanduser())
    subjects = p.get("subjects")
    runs = p.get("runs")
    sig_params = inspect.signature(module.data_path).parameters
    if subjects and "subjects" in sig_params:
        kwargs["subjects"] = [int(s) for s in subjects] if str(subjects[0]).isdigit() else subjects
    if runs and "runs" in sig_params:
        kwargs["runs"] = [int(r) for r in runs] if str(runs[0]).isdigit() else runs

    result = module.data_path(**kwargs)
    paths = result if isinstance(result, (list, tuple)) else [result]
    paths = [str(Path(str(item)).resolve()) for item in paths]

    for item in paths:
        ctx.add_artifact("directory" if Path(item).is_dir() else "file", item, f"dataset:{name}")

    return {
        "name": name,
        "paths": paths,
        "n_paths": len(paths),
        "hint": "Pass one of these paths to `mne-cli raw load --path <file>`.",
    }
