"""Epoching: cut trials around events."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..errors import DataError
from ..registry import Param, command
from ..utils import epochs_digest, resolve_picks
from .io_ import raw_opt


@command(
    "epochs.build",
    "Cut a raw object into trials time-locked to events.",
    params=[
        raw_opt(),
        Param("events", "str", "events handle from `raw events`, or a .npy path"),
        Param("events-json", "json", "events inline, e.g. [[100,0,1],[900,0,2]] (samples, 0, code)"),
        Param("event-id", "json", "conditions to keep, e.g. '{\"audio\":1,\"visual\":2}'; default keeps all codes"),
        Param("tmin", "float", "trial start relative to the event in seconds", default=-0.2),
        Param("tmax", "float", "trial end relative to the event in seconds", default=0.8),
        Param("baseline", "json", "baseline window, e.g. [null,0] or [-0.2,0]; null disables correction", default=[None, 0]),
        Param("reject", "json", "peak-to-peak rejection in volts, e.g. '{\"eeg\":150e-6}'"),
        Param("flat", "json", "flat rejection in volts, e.g. '{\"eeg\":1e-6}'"),
        Param("picks", "csv", "channels to keep in the epochs"),
        Param("ch-types", "csv", "channel types to keep, e.g. eeg"),
        Param("decim", "int", "decimate in time by this factor"),
        Param("preload", "bool", "load samples now (needed by most downstream steps)", default=True),
        Param("from-annotations", "bool", "build from raw annotations instead of an events array", default=False),
    ],
    returns="{n_epochs, event_id, tmin, tmax, n_events_by_condition, handle}",
    examples=[
        {"args": "epochs.build --raw raw_0001 --events events_0001 --tmin -0.2 --tmax 0.8"},
        {"args": "epochs.build --raw raw_0001 --from-annotations --event-id '{\"audio\":1}'"},
    ],
    notes="Volt-based thresholds are tiny numbers: 100 uV is 1e-4.",
    tags=("epochs",),
)
def cmd_epochs_build(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne
    import numpy as np

    raw = ctx.object("raw", p["raw"])
    picks = resolve_picks(raw.info, p.get("picks"), p.get("ch_types"))
    baseline = p.get("baseline")
    if isinstance(baseline, list):
        baseline = tuple(baseline)

    if p.get("from_annotations"):
        if len(raw.annotations) == 0:
            raise DataError("raw has no annotations", hint="Run `raw annotations --raw <handle>` or build from an events array.")
        epochs = mne.Epochs(
            raw,
            events=None,
            event_id=p.get("event_id"),
            tmin=float(p["tmin"]),
            tmax=float(p["tmax"]),
            baseline=baseline,
            picks=picks,
            reject=p.get("reject"),
            flat=p.get("flat"),
            decim=p.get("decim") or 1,
            preload=bool(p.get("preload", True)),
            verbose="ERROR",
        )
    else:
        events = _events_from(ctx, p, raw)
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=p.get("event_id"),
            tmin=float(p["tmin"]),
            tmax=float(p["tmax"]),
            baseline=baseline,
            picks=picks,
            reject=p.get("reject"),
            flat=p.get("flat"),
            decim=p.get("decim") or 1,
            preload=bool(p.get("preload", True)),
            verbose="ERROR",
        )

    if len(epochs) == 0:
        raise DataError(
            "no epochs survived",
            hint="Check the event codes against `raw events`, widen --tmin/--tmax, or loosen --reject.",
        )

    summary = epochs_digest(epochs)
    handle = ctx.keep("epochs", epochs, summary=summary)
    ctx.suggest(f"mne-cli epochs.average --epochs {handle} --by event")
    return {**summary, "handle": handle}


def _events_from(ctx: Ctx, p: dict[str, Any], raw):
    import mne
    import numpy as np

    if p.get("events_json") is not None:
        events = np.asarray(p["events_json"], dtype=int)
        if events.ndim != 2 or events.shape[1] != 3:
            raise DataError("--events-json must be [[sample, 0, code], ...]")
        return events

    if p.get("events"):
        return np.asarray(ctx.object("events", p["events"]), dtype=int)

    # Fall back to a stim channel, if there is one.
    try:
        return mne.find_events(raw, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"no events supplied and none could be found ({exc})",
            hint="Run `mne-cli raw events --raw <handle>` first, then pass --events, or use --from-annotations.",
        ) from None


@command(
    "epochs.info",
    "Describe epochs: trial count, time window, per-condition counts, rejections.",
    params=[
        Param("epochs", "str", "epochs handle", required=True, example="epochs_0001"),
        Param("limit", "int", "max trials listed in the preview", default=5),
    ],
    returns="epochs digest + trial preview",
    examples=[{"args": "epochs info --epochs epochs_0001"}],
    tags=("epochs", "info"),
)
def cmd_epochs_info(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    epochs = ctx.object("epochs", p["epochs"])
    payload = epochs_digest(epochs)
    n = int(p.get("limit") or 5)
    payload["trials_preview"] = [
        {"index": i, "condition": epochs.events[i, 2]}
        for i in range(min(n, len(epochs)))
    ]
    payload["handle"] = p["epochs"]
    return payload


@command(
    "epochs.drop-bad",
    "Apply (or re-apply) amplitude rejection to epochs.",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("reject", "json", "peak-to-peak thresholds in volts", required=True, example='{"eeg":150e-6}'),
        Param("flat", "json", "flat thresholds in volts"),
    ],
    returns="epochs digest after rejection + handle",
    examples=[{"args": "epochs drop-bad --epochs epochs_0001 --reject '{\"eeg\":150e-6}'"}],
    tags=("epochs",),
)
def cmd_epochs_drop_bad(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    epochs = ctx.object("epochs", p["epochs"])
    before = len(epochs)
    epochs.drop_bad(reject=p["reject"], flat=p.get("flat"), verbose="ERROR")
    summary = epochs_digest(epochs)
    handle = ctx.keep("epochs", epochs, summary=summary)
    return {**summary, "n_before": before, "n_dropped": before - len(epochs), "handle": handle}


@command(
    "epochs.equalize",
    "Down-sample each condition to the same number of trials.",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("seed", "int", "random seed for selection", default=42),
        Param("method", "choice", "selection scheme", default="truncate", choices=["truncate", "mintime"]),
    ],
    returns="epochs digest + dropped counts",
    examples=[{"args": "epochs equalize --epochs epochs_0001"}],
    tags=("epochs",),
)
def cmd_epochs_equalize(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    epochs = ctx.object("epochs", p["epochs"])
    before = len(epochs)
    dropped = epochs.equalize_event_counts(list(epochs.event_id), method=p.get("method") or "truncate", verbose="ERROR")
    summary = epochs_digest(epochs)
    handle = ctx.keep("epochs", epochs, summary=summary)
    return {
        **summary,
        "n_before": before,
        "n_after": len(epochs),
        "dropped_by_condition": {k: int(len(v)) for k, v in dropped.items()},
        "handle": handle,
    }


@command(
    "epochs.crop",
    "Trim the time window of epochs.",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("tmin", "float", "new start in seconds"),
        Param("tmax", "float", "new end in seconds"),
    ],
    returns="epochs digest + handle",
    examples=[{"args": "epochs crop --epochs epochs_0001 --tmax 0.5"}],
    tags=("epochs",),
)
def cmd_epochs_crop(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    epochs = ctx.object("epochs", p["epochs"]).copy()
    epochs.crop(tmin=p.get("tmin"), tmax=p.get("tmax"), verbose="ERROR")
    summary = epochs_digest(epochs)
    handle = ctx.keep("epochs", epochs, summary=summary)
    return {**summary, "handle": handle}


@command(
    "epochs.array",
    "Export trials to .npy with shape (n_epochs, n_channels, n_times).",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("out", "path", "output .npy path", required=True),
        Param("scale", "float", "multiply the values, e.g. 1e6 for microvolts", default=1.0),
    ],
    returns="{path, shape, y, labels_path}",
    examples=[{"args": "epochs array --epochs epochs_0001 --out X.npy --scale 1e6"}],
    notes="Labels (event codes per trial) are written next to the array as <out>_y.npy.",
    tags=("epochs", "io"),
)
def cmd_epochs_array(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    epochs = ctx.object("epochs", p["epochs"])
    X = epochs.get_data(copy=True) * float(p.get("scale") or 1.0)
    y = epochs.events[:, 2].copy()
    out = __import__("pathlib").Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), X)
    labels = out.with_name(out.stem + "_y.npy")
    np.save(str(labels), y)
    ctx.add_artifact("npy", out, "epochs X")
    ctx.add_artifact("npy", labels, "epochs y")
    return {
        "path": str(out.resolve()),
        "labels_path": str(labels.resolve()),
        "shape": list(X.shape),
        "conditions": {k: int(v) for k, v in epochs.event_id.items()},
    }


@command(
    "epochs.average",
    "Average trials into evoked responses (all trials or one evoked per condition).",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("by", "choice", "average everything, or one evoked per condition", default="event", choices=["event", "all"]),
        Param("conditions", "csv", "restrict to these condition names"),
        Param("picks", "csv", "channels to keep"),
        Param("out", "path", "optional .fif path to also write the first evoked"),
    ],
    returns="{evokeds: [{condition, handle, nave, data_range}], handle}",
    examples=[{"args": "epochs average --epochs epochs_0001 --by event"}],
    tags=("epochs", "evoked"),
)
def cmd_epochs_average(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    from ..utils import evoked_digest

    epochs = ctx.object("epochs", p["epochs"])
    names = p.get("conditions") or list(epochs.event_id)
    if p.get("by") == "all":
        names = [None]  # type: ignore[list-item]

    results = []
    first_handle = None
    for name in names:
        subset = epochs if name is None else epochs[name]
        if len(subset) == 0:
            ctx.warn(f"condition '{name}' has no trials, skipped")
            continue
        evoked = subset.average()
        if name is not None:
            evoked.comment = str(name)
        digest = evoked_digest(evoked)
        handle = ctx.save("evoked", evoked, summary=digest)
        if first_handle is None:
            first_handle = handle
        results.append({"condition": "all" if name is None else str(name), "handle": handle, **digest})

    if not results:
        raise DataError("no condition could be averaged", hint="Run `epochs info` to see which conditions exist.")

    ctx.primary_handle = first_handle
    if p.get("out") and first_handle:
        evoked = ctx.store.load(first_handle, expect="evoked")
        evoked.save(str(p["out"]), overwrite=True, verbose="ERROR")
        ctx.add_artifact("fif", p["out"], "evoked")
    ctx.suggest(f"mne-cli evoked.plot --evoked {first_handle}")
    return {"n_evokeds": len(results), "evokeds": results, "handle": first_handle}


@command(
    "epochs.plot",
    "Render epochs (butterfly or image map) to a PNG.",
    params=[
        Param("epochs", "str", "epochs handle", required=True),
        Param("out", "path", "PNG path (default: store/artifacts)"),
        Param("picks", "csv", "channels to draw"),
        Param("n-epochs", "int", "trials per page", default=5),
        Param("dpi", "int", "image resolution", default=120),
    ],
    returns="{path, n_epochs, n_channels}",
    examples=[{"args": "epochs plot --epochs epochs_0001 --picks Fz,Cz,Pz"}],
    tags=("epochs", "viz"),
)
def cmd_epochs_plot(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    epochs = ctx.object("epochs", p["epochs"])
    names = resolve_picks(epochs.info, p.get("picks"))
    fig = epochs.plot(
        picks=names,
        n_epochs=int(p.get("n_epochs") or 5),
        show=False,
        block=False,
    )
    if isinstance(fig, list):
        fig = fig[0]
    out = ctx.out_path(f"{p['epochs']}_epochs", "png", p.get("out"))
    fig.savefig(str(out), dpi=int(p.get("dpi") or 120), bbox_inches="tight")
    plt.close(fig)
    ctx.add_artifact("png", out, "epochs")
    return {"path": str(out.resolve()), "n_epochs": len(epochs), "n_channels": len(names) if names else len(epochs.ch_names)}
