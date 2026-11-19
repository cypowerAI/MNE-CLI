"""Preprocessing: filtering, referencing, resampling and channel surgery."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..errors import DataError
from ..registry import Param, command
from ..utils import raw_digest, resolve_picks
from .io_ import raw_opt


def _updated(ctx: Ctx, raw, source: str = "") -> tuple[dict[str, Any], str]:
    summary = raw_digest(raw)
    handle = ctx.keep("raw", raw, summary=summary, source=source)
    return summary, handle


@command(
    "raw.filter",
    "Band-pass / low-pass / high-pass filter a raw object (FIR by default).",
    params=[
        raw_opt(),
        Param("l-freq", "float", "high-pass cut-off in Hz (None disables it)", example=1.0),
        Param("h-freq", "float", "low-pass cut-off in Hz (None disables it)", example=40.0),
        Param("method", "choice", "FIR (linear phase, safe for ERP) or IIR (fast, causal)", default="fir", choices=["fir", "iir"]),
        Param("phase", "choice", "phase response for FIR", default="zero", choices=["zero", "zero-double", "minimum"]),
        Param("iir-order", "int", "Butterworth order when --method iir", default=4),
        Param("picks", "csv", "channels to filter"),
        Param("ch-types", "csv", "channel types to filter, e.g. eeg"),
        Param("skip-bad", "bool", "skip annotated bad segments instead of filtering across them", default=True),
    ],
    returns="raw digest of the filtered copy + handle",
    examples=[
        {"args": "raw filter --raw raw_0001 --l-freq 1 --h-freq 40"},
        {"args": "raw filter --raw raw_0001 --l-freq 0.1 --h-freq 30 --method iir --iir-order 4"},
    ],
    notes="Typical ERP setting: 0.1-30 Hz FIR. Typical resting EEG: 1-40 Hz.",
    tags=("preprocess",),
)
def cmd_raw_filter(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    picks = resolve_picks(raw.info, p.get("picks"), p.get("ch_types"))
    kwargs: dict[str, Any] = {
        "l_freq": p.get("l_freq"),
        "h_freq": p.get("h_freq"),
        "picks": picks,
        "method": p.get("method") or "fir",
        "skip_by_annotation": ("edge", "bad_acq_skip") if p.get("skip_bad", True) else (),
        "verbose": "ERROR",
    }
    if kwargs["l_freq"] is None and kwargs["h_freq"] is None:
        raise DataError("nothing to filter", hint="Pass --l-freq and/or --h-freq.")
    if kwargs["method"] == "fir":
        kwargs["phase"] = p.get("phase") or "zero"
    else:
        kwargs["iir_params"] = {"order": int(p.get("iir_order") or 4), "ftype": "butter"}
    raw.filter(**kwargs)
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "filter": {"l_freq": p.get("l_freq"), "h_freq": p.get("h_freq"), "method": kwargs["method"]}}


@command(
    "raw.notch",
    "Notch-filter power-line noise (50/60 Hz and harmonics).",
    params=[
        raw_opt(),
        Param("freqs", "csv", "frequencies to remove, e.g. 50 or 50,100,150", required=True, example="50"),
        Param("picks", "csv", "channels to filter"),
        Param("ch-types", "csv", "channel types to filter"),
        Param("widths", "csv", "notch width per frequency in Hz"),
    ],
    returns="raw digest of the filtered copy + handle",
    examples=[{"args": "raw notch --raw raw_0001 --freqs 50,100"}],
    tags=("preprocess",),
)
def cmd_raw_notch(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    picks = resolve_picks(raw.info, p.get("picks"), p.get("ch_types"))
    kwargs: dict[str, Any] = {
        "freqs": [float(f) for f in p["freqs"]],
        "picks": picks,
        "verbose": "ERROR",
    }
    if p.get("widths"):
        kwargs["notch_widths"] = [float(w) for w in p["widths"]]
    raw.notch_filter(**kwargs)
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "notched_hz": kwargs["freqs"]}


@command(
    "raw.resample",
    "Resample a raw object to a new sampling rate.",
    params=[
        raw_opt(),
        Param("sfreq", "float", "target sampling rate in Hz", required=True, example=250.0),
        Param("np-gcc", "int", "samples for the FFT padding (advanced)", default=0),
    ],
    returns="raw digest of the resampled copy + handle",
    examples=[{"args": "raw resample --raw raw_0001 --sfreq 250"}],
    tags=("preprocess",),
)
def cmd_raw_resample(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    kwargs: dict[str, Any] = {"sfreq": float(p["sfreq"]), "verbose": "ERROR"}
    if p.get("np_gcc"):
        kwargs["npad"] = int(p["np_gcc"])
    raw.resample(**kwargs)
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle}


@command(
    "raw.reference",
    "Re-reference EEG channels (common average or explicit electrodes).",
    params=[
        raw_opt(),
        Param("ref", "str", "reference: average, or a comma separated list (e.g. M1,M2)", default="average"),
        Param("projection", "bool", "add an SSP projector instead of applying it", default=False),
    ],
    returns="raw digest + handle",
    examples=[{"args": "raw reference --raw raw_0001 --ref average"}],
    tags=("preprocess",),
)
def cmd_raw_reference(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    ref = p.get("ref") or "average"
    if ref not in {"average", "REST"}:
        channels = [c.strip() for c in ref.split(",") if c.strip()]
        raw.set_eeg_reference(ref_channels=channels, projection=bool(p.get("projection")), verbose="ERROR")
    else:
        raw.set_eeg_reference(ref_channels=ref, projection=bool(p.get("projection")), verbose="ERROR")
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "reference": ref}


@command(
    "raw.pick",
    "Keep a subset of channels and drop the rest.",
    params=[
        raw_opt(),
        Param("include", "csv", "channel names to keep"),
        Param("exclude", "csv", "channel names to drop"),
        Param("ch-types", "csv", "channel types to keep, e.g. eeg,stim"),
    ],
    returns="raw digest + handle",
    examples=[{"args": "raw pick --raw raw_0001 --ch-types eeg --exclude EOG061"}],
    tags=("preprocess",),
)
def cmd_raw_pick(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    if not (p.get("include") or p.get("ch_types")):
        if not p.get("exclude"):
            raise DataError("nothing selected", hint="Pass --include, --exclude or --ch-types.")
        raw.drop_channels([c for c in p["exclude"] if c in raw.ch_names], verbose="ERROR")
    else:
        names = resolve_picks(raw.info, p.get("include"), p.get("ch_types"))
        raw.pick(names, verbose="ERROR")
        if p.get("exclude"):
            raw.drop_channels([c for c in p["exclude"] if c in raw.ch_names], verbose="ERROR")
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle}


@command(
    "raw.rename",
    "Rename channels with a mapping, e.g. to fix EDF channel names.",
    params=[
        raw_opt(),
        Param("mapping", "json", "map old name to new name", required=True, example='{"EEG 001":"Fp1"}'),
    ],
    returns="raw digest + handle",
    examples=[{"args": "raw rename --raw raw_0001 --mapping '{\"EEG 001\":\"Fp1\",\"EEG 002\":\"Fp2\"}'"}],
    tags=("preprocess",),
)
def cmd_raw_rename(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    mapping = p["mapping"]
    if not isinstance(mapping, dict):
        raise DataError("--mapping must be a JSON object of old -> new names")
    unknown = [k for k in mapping if k not in raw.ch_names]
    if unknown:
        raise DataError(
            f"channels not present: {', '.join(unknown[:10])}",
            hint="Run `mne-cli raw channels` to see the current names.",
        )
    raw.rename_channels({str(k): str(v) for k, v in mapping.items()}, verbose="ERROR")
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "renamed": len(mapping)}


@command(
    "raw.set-montage",
    "Attach electrode positions (standard montage name or a file).",
    params=[
        raw_opt(),
        Param("montage", "str", "e.g. standard_1020, standard_1005, biosemi64, or a .elc/.bvef/.csv file", required=True),
    ],
    returns="raw digest (montage flag included) + handle",
    examples=[{"args": "raw set-montage --raw raw_0001 --montage standard_1020"}],
    tags=("preprocess", "channels"),
)
def cmd_raw_set_montage(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    try:
        raw.set_montage(p["montage"], verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"montage '{p['montage']}' could not be applied: {exc}",
            hint="Check the name with mne.channels.get_builtin_montages(), or pass a file.",
        ) from None
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle}


@command(
    "raw.set-channel-types",
    "Change channel types (e.g. mark a MISC channel as EOG).",
    params=[
        raw_opt(),
        Param("types", "json", "map channel name to new type", required=True, example='{"MISC001":"eog"}'),
    ],
    returns="raw digest + handle",
    examples=[{"args": "raw set-channel-types --raw raw_0001 --types '{\"MISC001\":\"eog\"}'"}],
    tags=("preprocess", "channels"),
)
def cmd_raw_set_channel_types(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    types = p["types"]
    if not isinstance(types, dict):
        raise DataError("--types must be a JSON object of name -> type")
    raw.set_channel_types({str(k): str(v) for k, v in types.items()}, verbose="ERROR")
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle}


@command(
    "raw.mark-bad",
    "Mark channels as bad (excluded from averages, ICA and plots).",
    params=[raw_opt(), Param("channels", "csv", "channel names to mark", required=True)],
    returns="raw digest + handle",
    examples=[{"args": "raw mark-bad --raw raw_0001 --channels EOG061,ECG062"}],
    tags=("preprocess", "channels"),
)
def cmd_raw_mark_bad(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    missing = [c for c in p["channels"] if c not in raw.ch_names]
    if missing:
        raise DataError(f"channels not present: {', '.join(missing[:10])}", hint="Run `mne-cli raw channels`.")
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(p["channels"]))
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "bad_channels": summary["bad_channels"]}


@command(
    "raw.interpolate-bads",
    "Interpolate previously marked bad channels from their neighbours.",
    params=[raw_opt(), Param("reset-bads", "bool", "clear the bad flag afterwards", default=True)],
    returns="raw digest + handle",
    examples=[{"args": "raw interpolate-bads --raw raw_0001"}],
    notes="Requires electrode positions: run `raw set-montage` first.",
    tags=("preprocess", "channels"),
)
def cmd_raw_interpolate_bads(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    raw = ctx.object("raw", p["raw"])
    if not raw.info["bads"]:
        raise DataError("no channels are marked bad", hint="Use `raw mark-bad` first, or skip interpolation.")
    raw.interpolate_bads(reset_bads=bool(p.get("reset_bads", True)), verbose="ERROR")
    summary, handle = _updated(ctx, raw)
    return {**summary, "handle": handle, "interpolated": True}
