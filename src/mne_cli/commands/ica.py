"""ICA: fit, flag artefact components, apply, inspect."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..errors import DataError
from ..registry import Param, command
from ..utils import raw_digest, resolve_picks
from .io_ import raw_opt


def _ica_summary(ica) -> dict[str, Any]:
    return {
        "n_components": int(ica.n_components_),
        "method": str(ica.method),
        "exclude": [int(c) for c in ica.exclude],
    }


@command(
    "ica.fit",
    "Fit ICA on a raw object (use a high-pass filtered, artefact-heavy copy).",
    params=[
        raw_opt(),
        Param("n-components", "int", "number of components; omit to keep all", example=20),
        Param("method", "choice", "ICA algorithm", default="fastica", choices=["fastica", "infomax", "picard", "extended-infomax"]),
        Param("random-state", "int", "seed for reproducibility", default=42),
        Param("max-iter", "int", "maximum iterations", default=500),
        Param("decim", "int", "decimate the data used for fitting (speeds training)", default=1),
        Param("picks", "csv", "channels to fit on"),
        Param("ch-types", "csv", "channel types to fit on, e.g. eeg"),
    ],
    returns="{n_components, method, explained_variance_ratio (first 10), handle}",
    examples=[{"args": "ica fit --raw raw_0001 --n-components 20 --method fastica"}],
    notes="Fit on 1-30 Hz data with bad segments annotated; then flag and apply.",
    tags=("ica", "preprocess"),
)
def cmd_ica_fit(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    from mne.preprocessing import ICA

    raw = ctx.object("raw", p["raw"])
    picks = resolve_picks(raw.info, p.get("picks"), p.get("ch_types")) or "data"
    n_components = p.get("n_components")
    ica = ICA(
        n_components=n_components if n_components else None,
        method=p.get("method") or "fastica",
        random_state=int(p.get("random_state") or 42),
        max_iter=int(p.get("max_iter") or 500),
        verbose="ERROR",
    )
    ica.fit(raw, picks=picks, decim=int(p.get("decim") or 1), verbose="ERROR")
    summary = _ica_summary(ica)
    ratio = getattr(ica, "explained_variance_ratio_", None)
    if ratio is not None:
        summary["explained_variance_ratio_top10"] = [round(float(v), 4) for v in list(ratio)[:10]]
    handle = ctx.keep("ica", ica, summary=summary)
    ctx.suggest(f"mne-cli ica.find-artifacts --ica {handle} --raw {p['raw']}")
    return {**summary, "handle": handle}


@command(
    "ica.find-artifacts",
    "Automatically flag ICA components that look like blinks / cardiac artefacts.",
    params=[
        Param("ica", "str", "ica handle", required=True),
        raw_opt("raw object the ICA was fitted on (needed for detection)"),
        Param("eog", "bool", "detect ocular components from EOG channels", default=True),
        Param("ecg", "bool", "detect cardiac components from ECG channels", default=True),
        Param("threshold", "float", "correlation threshold; lower = more aggressive", example=3.0),
        Param("apply", "bool", "also write the excluded components back into the ICA handle", default=True),
    ],
    returns="{eog_components, ecg_components, exclude, handle}",
    examples=[{"args": "ica find-artifacts --ica ica_0001 --raw raw_0001"}],
    tags=("ica",),
)
def cmd_ica_find_artifacts(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    ica = ctx.object("ica", p["ica"])
    raw = ctx.object("raw", p["raw"])

    found: dict[str, Any] = {"eog_components": [], "ecg_components": []}
    threshold = p.get("threshold") or "auto"
    has_eog = any(ch.lower().startswith("eog") for ch in raw.ch_names)
    has_ecg = any(ch.lower().startswith("ecg") for ch in raw.ch_names)
    if p.get("eog", True):
        if not has_eog:
            ctx.warn("EOG detection skipped: no EOG channel in the raw object")
        else:
            try:
                bads, scores = ica.find_bads_eog(raw, threshold=threshold, verbose="ERROR")
                found["eog_components"] = [int(b) for b in bads]
                found["eog_scores_preview"] = [round(float(s), 3) for s in list(scores)[:5]]
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"EOG detection skipped: {exc}")
    if p.get("ecg", True):
        if not has_ecg:
            ctx.warn("ECG detection skipped: no ECG channel in the raw object")
        else:
            try:
                bads, scores = ica.find_bads_ecg(raw, threshold=threshold, verbose="ERROR")
                found["ecg_components"] = [int(b) for b in bads]
                found["ecg_scores_preview"] = [round(float(s), 3) for s in list(scores)[:5]]
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"ECG detection skipped: {exc}")

    combined = sorted(set(found["eog_components"]) | set(found["ecg_components"]))
    found["exclude"] = combined
    if p.get("apply", True):
        ica.exclude = combined
        handle = ctx.save("ica", ica, summary=_ica_summary(ica))
        ctx.primary_handle = handle
        found["handle"] = handle
    else:
        found["handle"] = p["ica"]

    if not combined:
        ctx.warn("no artefact components were detected automatically; inspect `ica.plot-components` and set them by hand")
    return found


@command(
    "ica.set-exclude",
    "Manually choose which components to remove (e.g. after looking at the component maps).",
    params=[
        Param("ica", "str", "ica handle", required=True),
        Param("components", "csv", "component indices to exclude", required=True, example="0,3,7"),
    ],
    returns="{exclude, handle}",
    examples=[{"args": "ica set-exclude --ica ica_0001 --components 0,3,7"}],
    tags=("ica",),
)
def cmd_ica_set_exclude(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    ica = ctx.object("ica", p["ica"])
    ica.exclude = [int(c) for c in p["components"]]
    handle = ctx.keep("ica", ica, summary=_ica_summary(ica))
    return {"exclude": ica.exclude, "handle": handle}


@command(
    "ica.apply",
    "Remove the flagged components from a raw object (returns a cleaned copy).",
    params=[
        Param("ica", "str", "ica handle", required=True),
        raw_opt("raw object to clean (same channels as during fitting)"),
        Param("components", "csv", "override the excluded components for this application"),
    ],
    returns="raw digest of the cleaned copy + handle",
    examples=[{"args": "ica apply --ica ica_0001 --raw raw_0001"}],
    tags=("ica", "preprocess"),
)
def cmd_ica_apply(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    ica = ctx.object("ica", p["ica"])
    raw = ctx.object("raw", p["raw"])
    if p.get("components"):
        ica.exclude = [int(c) for c in p["components"]]
    if not ica.exclude:
        raise DataError(
            "no components are flagged for removal",
            hint="Run `ica find-artifacts` or `ica set-exclude --components ...` first.",
        )
    cleaned = ica.apply(raw.copy(), verbose="ERROR") or raw
    summary = raw_digest(cleaned)
    handle = ctx.keep("raw", cleaned, summary=summary)
    return {**summary, "handle": handle, "removed_components": [int(c) for c in ica.exclude]}


@command(
    "ica.plot-components",
    "Render the ICA component topographies and time courses to a PNG.",
    params=[
        Param("ica", "str", "ica handle", required=True),
        Param("out", "path", "PNG path (default: store/artifacts)"),
        Param("components", "csv", "components to draw (default: all, max 20)"),
        Param("dpi", "int", "image resolution", default=120),
    ],
    returns="{path, n_components}",
    examples=[{"args": "ica plot-components --ica ica_0001"}],
    tags=("ica", "viz"),
)
def cmd_ica_plot_components(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ica = ctx.object("ica", p["ica"])
    comps = [int(c) for c in p["components"]] if p.get("components") else list(range(min(20, ica.n_components_)))
    figs = ica.plot_components(picks=comps, show=False)
    if not isinstance(figs, list):
        figs = [figs]
    out = ctx.out_path(f"{p['ica']}_components", "png", p.get("out"))
    figs[0].savefig(str(out), dpi=int(p.get("dpi") or 120), bbox_inches="tight")
    for fig in figs[1:]:
        plt.close(fig)
    plt.close(figs[0])
    ctx.add_artifact("png", out, "ica components")
    return {"path": str(out.resolve()), "n_components": len(comps)}


@command(
    "ica.plot-sources",
    "Render the ICA source time series (spot blinks and heartbeat spikes).",
    params=[
        Param("ica", "str", "ica handle", required=True),
        raw_opt("raw object the ICA was fitted on"),
        Param("out", "path", "PNG path (default: store/artifacts)"),
        Param("components", "csv", "components to draw"),
        Param("start", "float", "start time in seconds", default=0.0),
        Param("duration", "float", "window in seconds", default=20.0),
        Param("dpi", "int", "image resolution", default=120),
    ],
    returns="{path, components, duration_s}",
    examples=[{"args": "ica plot-sources --ica ica_0001 --raw raw_0001 --duration 20"}],
    tags=("ica", "viz"),
)
def cmd_ica_plot_sources(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ica = ctx.object("ica", p["ica"])
    raw = ctx.object("raw", p["raw"])
    comps = [int(c) for c in p["components"]] if p.get("components") else list(range(min(10, ica.n_components_)))
    start = float(p.get("start") or 0.0)
    duration = min(float(p.get("duration") or 20.0), float(raw.times[-1]) - start)
    figs = ica.plot_sources(raw, picks=comps, start=start, stop=start + duration, show=False)
    if isinstance(figs, list):
        figs = figs[0]
    out = ctx.out_path(f"{p['ica']}_sources", "png", p.get("out"))
    figs.savefig(str(out), dpi=int(p.get("dpi") or 120), bbox_inches="tight")
    plt.close(figs)
    ctx.add_artifact("png", out, "ica sources")
    return {"path": str(out.resolve()), "components": comps, "duration_s": duration}
