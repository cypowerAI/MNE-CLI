"""Spectral analysis: power spectral density and time-frequency representations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import DataError, UsageError
from ..registry import Param, command
from ..utils import resolve_picks
from ..result import sig


def _one_object(ctx: Ctx, p: dict[str, Any], kinds=("raw", "epochs", "evoked")):
    for kind in kinds:
        if p.get(kind):
            return kind, ctx.object(kind, p[kind])
    raise UsageError(
        "pass one input: --raw, --epochs or --evoked",
        hint="Example: `mne-cli psd --raw raw_0001 --fmin 1 --fmax 40`.",
    )


@command(
    "psd",
    "Compute power spectral density (Welch or multitaper) for raw / epochs / evoked.",
    params=[
        Param("raw", "str", "raw handle or path"),
        Param("epochs", "str", "epochs handle"),
        Param("evoked", "str", "evoked handle"),
        Param("fmin", "float", "lowest frequency in Hz", default=1.0),
        Param("fmax", "float", "highest frequency in Hz", default=40.0),
        Param("method", "choice", "spectral estimator", default="welch", choices=["welch", "multitaper"]),
        Param("n-fft", "int", "FFT length in samples (welch)"),
        Param("picks", "csv", "channels to include"),
        Param("out", "path", "optional .npz path for the full spectrum"),
        Param("plot", "bool", "also render a PNG", default=False),
    ],
    returns="{power_shape, freqs_hz, n_channels, peak_freq_by_channel, mean_power_by_band, path?}",
    examples=[
        {"args": "psd --raw raw_0001 --fmin 1 --fmax 40 --plot"},
        {"args": "psd --epochs epochs_0001 --method multitaper --out psd.npz"},
    ],
    tags=("spectral",),
)
def cmd_psd(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    kind, obj = _one_object(ctx, p)
    names = resolve_picks(obj.info, p.get("picks"))
    kwargs: dict[str, Any] = {
        "fmin": float(p.get("fmin") or 1.0),
        "fmax": float(p.get("fmax") or 40.0),
        "method": p.get("method") or "welch",
        "picks": names,
        "verbose": "ERROR",
    }
    if p.get("n_fft"):
        kwargs["n_fft"] = int(p["n_fft"])
    if kind == "evoked":
        kwargs.pop("picks", None)
        spec = obj.compute_psd(**{k: v for k, v in kwargs.items() if k not in {"picks"}})
        if names:
            idx = [obj.ch_names.index(n) for n in names]
            powers = spec.get_data()[idx]
        else:
            powers = spec.get_data()
        freqs = spec.freqs
        ch_names = [obj.ch_names[i] for i in range(len(obj.ch_names))]
    else:
        spec = obj.compute_psd(**kwargs)
        powers = spec.get_data()
        freqs = spec.freqs
        ch_names = spec.ch_names

    if powers.ndim == 3:  # epochs: average over trials for the summary
        powers_mean = powers.mean(axis=0)
    else:
        powers_mean = powers

    peaks = []
    for name, row in list(zip(ch_names, powers_mean))[:10]:
        peaks.append({"channel": name, "peak_freq_hz": round(float(freqs[int(np.argmax(row))]), 2), "peak_power": sig(float(np.max(row)))})

    bands = {}
    for label, (lo, hi) in {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}.items():
        sel = (freqs >= lo) & (freqs <= hi)
        if sel.any():
            bands[label] = sig(float(powers_mean[:, sel].mean()))

    payload: dict[str, Any] = {
        "input": kind,
        "power_shape": list(powers.shape),
        "n_channels": len(ch_names),
        "freqs_hz": [round(float(f), 2) for f in freqs] if len(freqs) <= 40 else {
            "n": int(len(freqs)),
            "min": round(float(freqs[0]), 2),
            "max": round(float(freqs[-1]), 2),
            "step": round(float(np.median(np.diff(freqs))) if len(freqs) > 1 else 0.0, 3),
            "note": "full list in the optional --out .npz",
        },
        "peak_freq_by_channel": peaks,
        "mean_power_by_band": bands,
        "unit": "power spectral density",
    }

    if p.get("out"):
        out = Path(p["out"]).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out), psd=powers, freqs=freqs, ch_names=np.array(ch_names), powers_mean=powers_mean)
        ctx.add_artifact("npz", out, "psd")
        payload["path"] = str(out.resolve())

    if p.get("plot"):
        import matplotlib.pyplot as plt

        fig = spec.plot(picks=names or ch_names[:6], show=False)
        if isinstance(fig, list):
            fig = fig[0]
        png = ctx.out_path("psd", "png")
        fig.savefig(str(png), dpi=120, bbox_inches="tight")
        plt.close(fig)
        ctx.add_artifact("png", png, "psd")
        payload["plot_path"] = str(png.resolve())

    return payload


@command(
    "tfr.compute",
    "Compute a time-frequency representation (Morlet or multitaper).",
    params=[
        Param("epochs", "str", "epochs handle (recommended)"),
        Param("raw", "str", "raw handle (continuous TFR)"),
        Param("evoked", "str", "evoked handle"),
        Param("fmin", "float", "lowest frequency in Hz", default=4.0),
        Param("fmax", "float", "highest frequency in Hz", default=30.0),
        Param("n-freqs", "int", "number of logarithmically spaced frequencies", default=20),
        Param("freqs", "csv", "explicit frequency list; overrides fmin/fmax/n-freqs"),
        Param("method", "choice", "TFR method", default="morlet", choices=["morlet", "multitaper"]),
        Param("n-cycles", "float", "cycles per wavelet (scalar or 'auto')", example=5.0),
        Param("time-bandwidth", "float", "multitaper bandwidth", default=4.0),
        Param("baseline", "json", "baseline window to correct, e.g. [-0.2,0]"),
        Param("picks", "csv", "channels to include"),
        Param("out", "path", "optional .npz path for the full TFR"),
        Param("plot", "bool", "also render a PNG", default=False),
    ],
    returns="{shape, n_channels, freqs_hz, times_s range, data_range, handle, path?}",
    examples=[
        {"args": "tfr.compute --epochs epochs_0001 --fmin 4 --fmax 30 --n-freqs 20 --plot"},
        {"args": "tfr.compute --epochs epochs_0001 --freqs 8,10,12,14 --n-cycles 5"},
    ],
    tags=("spectral",),
)
def cmd_tfr(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from mne.time_frequency import tfr_morlet, tfr_multitaper

    kind, obj = _one_object(ctx, p)
    if p.get("freqs"):
        freqs = np.array([float(f) for f in p["freqs"]])
    else:
        fmin, fmax, n = float(p.get("fmin") or 4.0), float(p.get("fmax") or 30.0), int(p.get("n_freqs") or 20)
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n)

    if p.get("n_cycles") is None:
        n_cycles = freqs / 2.0
    elif str(p.get("n_cycles")).lower() == "auto":
        n_cycles = freqs / 2.0
    else:
        n_cycles = float(p["n_cycles"])

    names = resolve_picks(obj.info, p.get("picks"))
    kwargs: dict[str, Any] = {
        "freqs": freqs,
        "n_cycles": n_cycles,
        "picks": names,
        "verbose": "ERROR",
    }
    if p.get("method") == "multitaper":
        result = tfr_multitaper(obj, time_bandwidth=float(p.get("time_bandwidth") or 4.0), **kwargs)
    else:
        result = tfr_morlet(obj, **kwargs)
    tfr = result[0] if isinstance(result, tuple) else result

    if p.get("baseline") and kind != "raw":
        tfr.apply_baseline(baseline=tuple(p["baseline"]), mode="mean", verbose="ERROR")

    data = tfr.data
    payload: dict[str, Any] = {
        "input": kind,
        "shape": list(data.shape),
        "n_channels": len(tfr.ch_names),
        "freqs_hz": [round(float(f), 2) for f in tfr.freqs],
        "tmin_s": round(float(tfr.times[0]), 4),
        "tmax_s": round(float(tfr.times[-1]), 4),
        "data_range": [sig(float(np.min(data)), 4), sig(float(np.max(data)), 4)],
        "mean_by_freq": {
            str(round(float(f), 2)): sig(float(v), 4)
            for f, v in zip(tfr.freqs, data.mean(axis=(0, 2)))
        },
    }

    if p.get("out"):
        out = Path(p["out"]).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out), data=data, freqs=tfr.freqs, times=tfr.times, ch_names=np.array(tfr.ch_names))
        ctx.add_artifact("npz", out, "tfr")
        payload["path"] = str(out.resolve())

    handle = ctx.keep("tfr", tfr, summary={"shape": list(data.shape), "freqs_hz": payload["freqs_hz"]})
    payload["handle"] = handle
    ctx.suggest(f"mne-cli tfr.plot --tfr {handle}")

    if p.get("plot"):
        payload.update(_save_tfr_plot(ctx, tfr, None, names))
    return payload


def _save_tfr_plot(ctx: Ctx, tfr, out: str | None, picks=None) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figs = tfr.plot(picks=picks, show=False)
    if not isinstance(figs, list):
        figs = [figs]
    path = ctx.out_path("tfr", "png", out)
    figs[0].savefig(str(path), dpi=120, bbox_inches="tight")
    for fig in figs:
        plt.close(fig)
    ctx.add_artifact("png", path, "tfr")
    return {"plot_path": str(path.resolve())}


@command(
    "tfr.info",
    "Describe a stored TFR: shape, frequencies, time window.",
    params=[Param("tfr", "str", "tfr handle", required=True)],
    returns="{shape, ch_names, freqs_hz, tmin_s, tmax_s}",
    examples=[{"args": "tfr info --tfr tfr_0001"}],
    tags=("spectral", "info"),
)
def cmd_tfr_info(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    tfr = ctx.object("tfr", p["tfr"])
    return {
        "shape": list(tfr.data.shape),
        "n_channels": len(tfr.ch_names),
        "ch_names": tfr.ch_names[:30],
        "freqs_hz": [round(float(f), 2) for f in tfr.freqs],
        "tmin_s": round(float(tfr.times[0]), 4),
        "tmax_s": round(float(tfr.times[-1]), 4),
        "data_range": [sig(float(np.min(tfr.data)), 4), sig(float(np.max(tfr.data)), 4)],
        "handle": p["tfr"],
    }


@command(
    "tfr.plot",
    "Render a TFR as a time-frequency power map.",
    params=[
        Param("tfr", "str", "tfr handle", required=True),
        Param("out", "path", "PNG path (default: store/artifacts)"),
        Param("picks", "csv", "channels to draw"),
    ],
    returns="{path}",
    examples=[{"args": "tfr plot --tfr tfr_0001 --picks Fz,Cz,Pz"}],
    tags=("spectral", "viz"),
)
def cmd_tfr_plot(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    tfr = ctx.object("tfr", p["tfr"])
    names = resolve_picks(tfr.info, p.get("picks"))
    return _save_tfr_plot(ctx, tfr, p.get("out"), names)


@command(
    "tfr.array",
    "Export a TFR to .npz (data, freqs, times, channel names).",
    params=[Param("tfr", "str", "tfr handle", required=True), Param("out", "path", "output .npz path", required=True)],
    returns="{path, shape}",
    examples=[{"args": "tfr array --tfr tfr_0001 --out tfr.npz"}],
    tags=("spectral", "io"),
)
def cmd_tfr_array(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    tfr = ctx.object("tfr", p["tfr"])
    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), data=tfr.data, freqs=tfr.freqs, times=tfr.times, ch_names=np.array(tfr.ch_names))
    ctx.add_artifact("npz", out, "tfr")
    return {"path": str(out.resolve()), "shape": list(tfr.data.shape)}
