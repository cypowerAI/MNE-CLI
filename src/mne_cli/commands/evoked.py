"""Evoked responses: inspect, crop, export, plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import DataError
from ..registry import Param, command
from ..utils import evoked_digest, resolve_picks


def evoked_opt(required: bool = True) -> Param:
    return Param("evoked", "str", "evoked handle (or a .fif path)", required=required, example="evoked_0001")


@command(
    "evoked.info",
    "Describe an evoked response: condition, number of averages, time window, amplitude range.",
    params=[evoked_opt()],
    returns="evoked digest",
    examples=[{"args": "evoked info --evoked evoked_0001"}],
    tags=("evoked", "info"),
)
def cmd_evoked_info(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    evoked = ctx.object("evoked", p["evoked"])
    return {**evoked_digest(evoked), "handle": p["evoked"]}


@command(
    "evoked.values",
    "Sample an evoked response: channel-wise peak amplitude and latency, or full matrix.",
    params=[
        evoked_opt(),
        Param("picks", "csv", "channels to report"),
        Param("mode", "choice", "peak summary (one row per channel) or the full matrix", default="peaks", choices=["peaks", "matrix"]),
    ],
    returns="peaks: [{channel, peak_amplitude, peak_latency_s, mean}]; matrix: {times_s, channels, data}",
    examples=[{"args": "evoked values --evoked evoked_0001 --picks Fz,Cz,Pz"}],
    tags=("evoked", "info"),
)
def cmd_evoked_values(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    evoked = ctx.object("evoked", p["evoked"])
    names = resolve_picks(evoked.info, p.get("picks")) or list(evoked.ch_names)
    data = evoked.copy().pick(names).data
    if len(names) == 0:
        raise DataError("no channels selected")

    if p.get("mode") == "matrix":
        times = evoked.times[: data.shape[1]]
        step = max(1, data.shape[1] // 50)
        return {
            "times_s": [round(float(t), 4) for t in times[::step]],
            "channels": names[:20],
            "data": [[round(float(v), 6) for v in row[::step]] for row in data[:20]],
            "note": "downsampled for context; use `evoked array` for the full matrix",
        }

    rows = []
    times = evoked.times
    for name, row in zip(names, data):
        idx = int(np.argmax(np.abs(row)))
        rows.append(
            {
                "channel": name,
                "peak_amplitude": round(float(row[idx]), 8),
                "peak_latency_s": round(float(times[idx]), 4),
                "mean": round(float(row.mean()), 8),
            }
        )
    return {"n_channels": len(rows), "peaks": rows}


@command(
    "evoked.crop",
    "Trim the time window of an evoked response.",
    params=[
        evoked_opt(),
        Param("tmin", "float", "new start in seconds"),
        Param("tmax", "float", "new end in seconds"),
    ],
    returns="evoked digest + handle",
    examples=[{"args": "evoked crop --evoked evoked_0001 --tmin 0 --tmax 0.6"}],
    tags=("evoked",),
)
def cmd_evoked_crop(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    evoked = ctx.object("evoked", p["evoked"]).copy()
    evoked.crop(tmin=p.get("tmin"), tmax=p.get("tmax"), verbose="ERROR")
    summary = evoked_digest(evoked)
    handle = ctx.keep("evoked", evoked, summary=summary)
    return {**summary, "handle": handle}


@command(
    "evoked.save",
    "Write an evoked response to a FIF file.",
    params=[evoked_opt(), Param("out", "path", "output .fif path", required=True)],
    returns="{path, size_mb}",
    examples=[{"args": "evoked save --evoked evoked_0001 --out avg.fif"}],
    tags=("evoked", "io"),
)
def cmd_evoked_save(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    evoked = ctx.object("evoked", p["evoked"])
    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    evoked.save(str(out), overwrite=True, verbose="ERROR")
    ctx.add_artifact("fif", out, "evoked")
    return {"path": str(out.resolve()), "size_mb": round(out.stat().st_size / 1e6, 2)}


@command(
    "evoked.array",
    "Export the evoked matrix (n_channels x n_times) to .npy.",
    params=[
        evoked_opt(),
        Param("out", "path", "output .npy path", required=True),
        Param("picks", "csv", "channels to export"),
        Param("scale", "float", "multiply values, e.g. 1e6 for microvolts", default=1.0),
    ],
    returns="{path, shape, channels}",
    examples=[{"args": "evoked array --evoked evoked_0001 --out avg.npy --scale 1e6"}],
    tags=("evoked", "io"),
)
def cmd_evoked_array(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    evoked = ctx.object("evoked", p["evoked"])
    names = resolve_picks(evoked.info, p.get("picks")) or list(evoked.ch_names)
    data = evoked.copy().pick(names).data * float(p.get("scale") or 1.0)
    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), data)
    ctx.add_artifact("npy", out, "evoked")
    return {"path": str(out.resolve()), "shape": list(data.shape), "channels": names}


@command(
    "evoked.plot",
    "Render an evoked response: butterfly, per-channel traces or a topomap.",
    params=[
        evoked_opt(),
        Param("out", "path", "PNG path (default: store/artifacts)"),
        Param("kind", "choice", "butterfly, traces or topomap", default="butterfly", choices=["butterfly", "traces", "topomap"]),
        Param("picks", "csv", "channels to draw"),
        Param("time-unit", "choice", "x axis unit", default="ms", choices=["ms", "s"]),
        Param("dpi", "int", "image resolution", default=120),
    ],
    returns="{path, kind, n_channels}",
    examples=[
        {"args": "evoked plot --evoked evoked_0001 --kind butterfly"},
        {"args": "evoked plot --evoked evoked_0001 --kind topomap --picks Fz,Cz,Pz"},
    ],
    tags=("evoked", "viz"),
)
def cmd_evoked_plot(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    evoked = ctx.object("evoked", p["evoked"])
    names = resolve_picks(evoked.info, p.get("picks"))
    kind = p.get("kind") or "butterfly"
    if kind == "butterfly":
        fig = evoked.plot(picks=names, show=False, time_unit=p.get("time_unit") or "ms", verbose="ERROR")
    elif kind == "traces":
        fig = evoked.plot(
            picks=names or list(evoked.ch_names)[:12],
            spatial_colors=False,
            show=False,
            time_unit=p.get("time_unit") or "ms",
            verbose="ERROR",
        )
    else:
        fig = evoked.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type="eeg" if _has_eeg(evoked) else None, show=False)
    if isinstance(fig, list):
        fig = fig[0]
    out = ctx.out_path(f"{p['evoked']}_{kind}", "png", p.get("out"))
    fig.savefig(str(out), dpi=int(p.get("dpi") or 120), bbox_inches="tight")
    plt.close(fig)
    ctx.add_artifact("png", out, f"evoked {kind}")
    return {"path": str(out.resolve()), "kind": kind, "n_channels": len(names) if names else len(evoked.ch_names)}


def _has_eeg(evoked) -> bool:
    return any(ch["kind"] == 2 for ch in evoked.info["chs"])
