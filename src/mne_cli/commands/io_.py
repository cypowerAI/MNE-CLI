"""Raw signal I/O and inspection: the entry point of every analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import DataError
from ..readers import read_raw_auto
from ..registry import Param, command
from ..result import array_summary
from ..utils import annotation_summary, raw_digest, resolve_picks

FORMATS = {
    "fif": ("mne.io", "read_raw_fif"),
    "edf": ("mne.io", "read_raw_edf"),
    "bdf": ("mne.io", "read_raw_bdf"),
    "brainvision": ("mne.io", "read_raw_brainvision"),
    "eeglab": ("mne.io", "read_raw_eeglab"),
    "cnt": ("mne.io", "read_raw_cnt"),
    "gdf": ("mne.io", "read_raw_gdf"),
    "kit": ("mne.io", "read_raw_kit"),
    "ctf": ("mne.io", "read_raw_ctf"),
    "egi": ("mne.io", "read_raw_egi"),
    "nxe": ("mne.io", "read_raw_eximia"),
    "curry": ("mne.io", "read_raw_curry"),
    "xdf": ("mne.io", "read_raw_xdf"),
    "persyst": ("mne.io", "read_raw_persyst"),
}


def raw_opt(help_text: str = "raw handle (e.g. raw_0001) or a recording path", required: bool = True) -> Param:
    return Param("raw", "str", help_text, required=required, example="raw_0001")


def _read_with_format(path: Path, fmt: str, preload: bool):
    import importlib

    module_name, func_name = FORMATS[fmt]
    func = getattr(importlib.import_module(module_name), func_name)
    try:
        return func(str(path), preload=preload, verbose="ERROR")
    except TypeError:
        return func(str(path), verbose="ERROR")


@command(
    "raw.load",
    "Load a recording (FIF/EDF/BDF/BrainVision/EEGLAB/CNT/GDF/CTF/KIT...) into the store.",
    params=[
        Param("path", "path", "recording to load", required=True, example="D:/data/sub-01_task.fif"),
        Param("file-format", "choice", "file format; auto sniffs it from the suffix", default="auto", choices=["auto", *FORMATS.keys()]),
        Param("preload", "bool", "read samples into memory now (needed by some ops, slower)", default=False),
        Param("montage", "str", "standard montage to attach, e.g. standard_1020, or a .elc/.bvef file"),
        Param("resample", "float", "resample to this sampling rate (Hz) right after loading"),
        Param("picks", "csv", "only keep these channels"),
        Param("ch-types", "csv", "only keep these channel types, e.g. eeg,stim"),
    ],
    returns="raw digest: sfreq, n_channels, n_times, ch_names, annotations, handle",
    examples=[
        {"args": "raw load --path sub-01_task-motor.fif"},
        {"args": "raw load --path recording.edf --montage standard_1020 --resample 250"},
    ],
    notes="The returned handle is what every other `raw.*` command expects via --raw.",
    tags=("raw", "io"),
)
def cmd_raw_load(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    path = Path(p["path"]).expanduser()
    if not path.exists():
        raise DataError(f"file not found: {path}", hint="Use an absolute path, or `mne-cli dataset download --name sample`.")

    if p.get("file_format") and p["file_format"] != "auto":
        raw = _read_with_format(path, p["file_format"], bool(p.get("preload")))
    else:
        raw = read_raw_auto(path, preload=bool(p.get("preload")))

    if p.get("montage"):
        try:
            raw.set_montage(p["montage"], verbose="ERROR")
        except Exception as exc:  # noqa: BLE001
            raise DataError(
                f"could not apply montage '{p['montage']}': {exc}",
                hint="Try a standard name (standard_1020, standard_1005) or a file path; digitiser points may be missing.",
            ) from None

    picks = resolve_picks(raw.info, p.get("picks"), p.get("ch_types"))
    if picks:
        raw.pick(picks, verbose="ERROR")

    if p.get("resample"):
        raw.resample(float(p["resample"]), verbose="ERROR")

    summary = raw_digest(raw)
    handle = ctx.keep("raw", raw, summary=summary, source=str(path))
    ctx.suggest(f"mne-cli raw info --raw {handle}")
    return {**summary, "handle": handle, "source": str(path)}


@command(
    "raw.info",
    "Describe a raw object: sampling rate, duration, channel mix, annotations, bad channels.",
    params=[raw_opt()],
    returns="raw digest",
    examples=[{"args": "raw info --raw raw_0001"}],
    tags=("raw", "info"),
)
def cmd_raw_info(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    return {**raw_digest(raw), "handle": p["raw"]}


@command(
    "raw.channels",
    "List channels with index, type and unit.",
    params=[
        raw_opt(),
        Param("type", "csv", "filter by channel type, e.g. eeg"),
        Param("limit", "int", "max channels to return", default=200),
    ],
    returns="{n_channels, channels: [{index, name, type, unit}], truncated}",
    examples=[{"args": "raw channels --raw raw_0001 --type eeg --limit 30"}],
    tags=("raw", "info"),
)
def cmd_raw_channels(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    from ..utils import _ch_kind

    raw = ctx.object("raw", p["raw"])
    wanted = {t.strip().lower() for t in (p.get("type") or [])}
    rows = []
    for idx, ch in enumerate(raw.info["chs"]):
        kind = _ch_kind(ch)
        if wanted and kind not in wanted and not (kind == "meg" and "meg" in wanted):
            continue
        rows.append(
            {
                "index": idx,
                "name": raw.ch_names[idx],
                "type": kind,
                "unit": str(ch.get("unit", "")) if ch.get("unit") is not None else "",
            }
        )
    limit = int(p.get("limit") or 200)
    return {
        "n_channels": len(rows),
        "channels": rows[:limit],
        "truncated": len(rows) > limit,
    }


@command(
    "raw.head",
    "Preview the first samples of a recording (values, not statistics).",
    params=[
        raw_opt(),
        Param("tmin", "float", "start time in seconds", default=0.0),
        Param("n-times", "int", "number of samples", default=5),
        Param("picks", "csv", "channels to show"),
        Param("max-channels", "int", "cap on channels returned", default=10),
    ],
    returns="{times_s, channels, data[n_ch][n_times]}",
    examples=[{"args": "raw head --raw raw_0001 --n-times 10"}],
    tags=("raw", "info"),
)
def cmd_raw_head(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    names = resolve_picks(raw.info, p.get("picks")) or list(raw.ch_names)
    names = names[: int(p.get("max_channels") or 10)]
    sfreq = raw.info["sfreq"]
    start = max(0, int(round(float(p.get("tmin") or 0.0) * sfreq)))
    stop = min(raw.n_times, start + int(p.get("n_times") or 5))
    data = raw.get_data(picks=names, start=start, stop=stop)
    times = (raw.first_samp + start + __import__("numpy").arange(stop - start)) / sfreq - raw.first_samp / sfreq
    return {
        "times_s": [round(float(t), 6) for t in times],
        "channels": names,
        "data": [[round(float(v), 6) for v in row] for row in data],
        "unit": "SI volts (as stored by MNE)",
    }


@command(
    "raw.crop",
    "Cut a time window out of a recording and keep the result as a new handle.",
    params=[
        raw_opt(),
        Param("tmin", "float", "start time in seconds", default=0.0),
        Param("tmax", "float", "end time in seconds (default: end of file)"),
        Param("copy", "bool", "keep the original object intact", default=True),
    ],
    returns="raw digest of the cropped copy",
    examples=[{"args": "raw crop --raw raw_0001 --tmin 10 --tmax 70"}],
    tags=("raw",),
)
def cmd_raw_crop(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    if p.get("copy", True):
        raw = raw.copy()
    raw.crop(tmin=float(p.get("tmin") or 0.0), tmax=p.get("tmax"), verbose="ERROR")
    summary = raw_digest(raw)
    handle = ctx.keep("raw", raw, summary=summary)
    return {**summary, "handle": handle}


@command(
    "raw.save",
    "Write a raw object to a FIF file.",
    params=[raw_opt(), Param("out", "path", "output .fif path", required=True)],
    returns="{path, size_mb}",
    examples=[{"args": "raw save --raw raw_0001 --out cleaned_raw.fif"}],
    tags=("raw", "io"),
)
def cmd_raw_save(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(out), overwrite=True, verbose="ERROR")
    ctx.add_artifact("fif", out, "raw")
    return {"path": str(out.resolve()), "size_mb": round(out.stat().st_size / 1e6, 2)}


@command(
    "raw.array",
    "Export the sample matrix to .npy (for external ML tooling).",
    params=[
        raw_opt(),
        Param("out", "path", "output .npy path", required=True),
        Param("picks", "csv", "channels to export"),
        Param("start", "float", "start time in seconds", default=0.0),
        Param("stop", "float", "stop time in seconds"),
        Param("scale", "float", "multiply values, e.g. 1e6 to get microvolts", default=1.0),
    ],
    returns="{path, shape, channels, dtype}",
    examples=[{"args": "raw array --raw raw_0001 --out eeg.npy --scale 1e6"}],
    tags=("raw", "io"),
)
def cmd_raw_array(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    raw = ctx.object("raw", p["raw"])
    names = resolve_picks(raw.info, p.get("picks")) or list(raw.ch_names)
    sfreq = raw.info["sfreq"]
    start = int(round(float(p.get("start") or 0.0) * sfreq))
    stop = int(round(float(p["stop"]) * sfreq)) if p.get("stop") else None
    data = raw.get_data(picks=names, start=start, stop=stop) * float(p.get("scale") or 1.0)
    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), data)
    ctx.add_artifact("npy", out, "raw array")
    return {"path": str(out.resolve()), "shape": list(data.shape), "channels": names, "dtype": str(data.dtype)}


@command(
    "raw.events",
    "Extract stimulus events from a raw object into an events handle.",
    params=[
        raw_opt(),
        Param("stim-channel", "str", "stim channel to read; default: every STIM channel"),
        Param("shortest-event", "int", "ignore pulses shorter than this many samples", default=1),
        Param("min-duration", "float", "minimum event duration in samples", default=0.0),
        Param("event-id", "json", "only keep these events, e.g. '{\"audio\":1}'"),
    ],
    returns="{n_events, event_id, counts, events_preview[][3], handle}",
    examples=[{"args": "raw events --raw raw_0001 --stim-channel STI 014"}],
    notes="If the file has annotations instead of a stim channel, use `raw annotations-from-events` or work from annotations.",
    tags=("raw", "events"),
)
def cmd_raw_events(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = ctx.object("raw", p["raw"])
    stim = p.get("stim_channel")
    if not stim:
        stim_channels = [ch for ch, kind in zip(raw.ch_names, [c["kind"] for c in raw.info["chs"]]) if _is_stim(raw, ch)]
        if not stim_channels:
            raise DataError(
                "this recording has no stimulus channel",
                hint="Inspect `mne-cli raw info` annotations instead, or pass --stim-channel explicitly.",
            )
    events = mne.find_events(
        raw,
        stim_channel=stim,
        shortest_event=int(p.get("shortest_event") or 1),
        min_duration=float(p.get("min_duration") or 0.0),
        verbose="ERROR",
    )
    if events.size == 0:
        raise DataError("no events found", hint="Try another --stim-channel, or check that triggers exist.")

    wanted = p.get("event_id")
    if isinstance(wanted, dict):
        codes = [int(v) for v in wanted.values()]
        mask = [row for row in events if int(row[2]) in codes]
        events = __import__("numpy").array(mask) if mask else events[:0]

    uniq, counts = __import__("numpy").unique(events[:, 2], return_counts=True)
    event_id = {f"event_{int(u)}": int(u) for u in uniq}
    if isinstance(wanted, dict):
        event_id = {str(k): int(v) for k, v in wanted.items()}

    handle = ctx.keep("events", events, summary={"n_events": len(events), "event_id": event_id})
    ctx.suggest(f"mne-cli epochs.build --raw {p['raw']} --events {handle} --tmin -0.2 --tmax 0.8")
    return {
        "n_events": int(len(events)),
        "event_id": event_id,
        "counts": {str(int(u)): int(c) for u, c in zip(uniq, counts)},
        "events_preview": [[int(v) for v in row] for row in events[:10]],
        "sfreq": float(raw.info["sfreq"]),
        "handle": handle,
    }


def _is_stim(raw, name: str) -> bool:
    from mne.io.constants import FIFF

    idx = raw.ch_names.index(name)
    return raw.info["chs"][idx]["kind"] == FIFF.FIFFV_STIM_CH


@command(
    "raw.annotations",
    "List annotations (onset, duration, description) on a raw object.",
    params=[raw_opt(), Param("limit", "int", "max annotations to return", default=50)],
    returns="{n, descriptions, items[]}",
    examples=[{"args": "raw annotations --raw raw_0001"}],
    tags=("raw", "events"),
)
def cmd_raw_annotations(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    ann = raw.annotations
    items = [
        {"onset": round(float(o), 3), "duration": round(float(d), 3), "description": str(desc)}
        for o, d, desc in zip(ann.onset, ann.duration, ann.description)
    ]
    limit = int(p.get("limit") or 50)
    return {
        "n": len(items),
        "descriptions": sorted({i["description"] for i in items})[:30],
        "items": items[:limit],
        "truncated": len(items) > limit,
    }


@command(
    "raw.annotate",
    "Add one annotation (a labelled time window) to a raw object.",
    params=[
        raw_opt(),
        Param("onset", "float", "start time in seconds", required=True),
        Param("duration", "float", "length in seconds", default=0.0),
        Param("description", "str", "label, e.g. bad_segment or blink", required=True),
    ],
    returns="raw digest + n_annotations",
    examples=[{"args": "raw annotate --raw raw_0001 --onset 12.5 --duration 1.0 --description bad_segment"}],
    tags=("raw", "events"),
)
def cmd_raw_annotate(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = ctx.object("raw", p["raw"])
    ann = mne.Annotations(onset=[float(p["onset"])], duration=[float(p.get("duration") or 0.0)], description=[p["description"]])
    raw.set_annotations(raw.annotations + ann, verbose="ERROR")
    summary = raw_digest(raw)
    handle = ctx.keep("raw", raw, summary=summary)
    return {**summary, "n_annotations": int(len(raw.annotations)), "handle": handle}


@command(
    "raw.annotations-from-events",
    "Turn an events array into annotations (so epochs can be built from annotated data).",
    params=[
        raw_opt(),
        Param("events", "str", "events handle from `raw events`", required=True, example="events_0001"),
        Param("event-id", "json", "map code to label, e.g. '{\"audio\":1,\"visual\":2}'"),
    ],
    returns="raw digest + n_annotations",
    examples=[{"args": "raw annotations-from-events --raw raw_0001 --events events_0001"}],
    tags=("raw", "events"),
)
def cmd_raw_annotations_from_events(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = ctx.object("raw", p["raw"])
    events = ctx.object("events", p["events"])
    mapping = p.get("event_id") or None
    if isinstance(mapping, dict):
        mapping = {int(v): str(k) for k, v in mapping.items()}
    ann = mne.annotations_from_events(
        events,
        sfreq=raw.info["sfreq"],
        event_desc=mapping,
        first_samp=raw.first_samp,
        verbose="ERROR",
    )
    raw.set_annotations(ann, verbose="ERROR")
    summary = raw_digest(raw)
    handle = ctx.keep("raw", raw, summary=summary)
    return {**summary, "n_annotations": int(len(ann)), "labels": sorted(set(ann.description))[:20], "handle": handle}


@command(
    "raw.plot",
    "Render a raw trace overview to a PNG (headless, no browser).",
    params=[
        raw_opt(),
        Param("out", "path", "PNG output path (default: store/artifacts)"),
        Param("start", "float", "start time in seconds", default=0.0),
        Param("duration", "float", "window length in seconds", default=10.0),
        Param("n-channels", "int", "channels per page", default=20),
        Param("picks", "csv", "channels to draw"),
        Param("dpi", "int", "image resolution", default=120),
    ],
    returns="{path, duration_s, n_channels}",
    examples=[{"args": "raw plot --raw raw_0001 --duration 10 --picks Fp1,Fp2,O1,O2"}],
    tags=("raw", "viz"),
)
def cmd_raw_plot(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    raw = ctx.object("raw", p["raw"])
    names = resolve_picks(raw.info, p.get("picks"))
    duration = min(float(p.get("duration") or 10.0), max(0.5, float(raw.times[-1]) - float(p.get("start") or 0.0)))
    fig = raw.plot(
        duration=duration,
        start=float(p.get("start") or 0.0),
        n_channels=int(p.get("n_channels") or 20),
        picks=names,
        scalings="auto",
        show=False,
        block=False,
    )
    out = ctx.out_path(f"{p['raw']}_raw", "png", p.get("out"))
    fig.savefig(str(out), dpi=int(p.get("dpi") or 120), bbox_inches="tight")
    plt.close(fig)
    ctx.add_artifact("png", out, "raw traces")
    return {"path": str(out.resolve()), "duration_s": duration, "n_channels": len(names) if names else len(raw.ch_names)}
