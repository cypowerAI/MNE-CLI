"""Source localisation: BEM, forward model, covariance, inverse, source estimates.

These commands follow the standard MNE pipeline and need a FreeSurfer subject
directory (``SUBJECTS_DIR``) plus a trans file from the MEG/EEG session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import DataError, UsageError
from ..registry import Param, command
from ..utils import ch_type_counts
from ..result import sig
from .io_ import raw_opt


def _subjects_dir(ctx: Ctx, value: str | None) -> str:
    import mne

    if value:
        return str(Path(value).expanduser())
    configured = mne.get_config("MNE_SUBJECTS_DIR")
    if configured:
        return str(configured)
    raise UsageError(
        "no subjects directory",
        hint="Pass --subjects-dir, or set it once with `mne.set_config('MNE_SUBJECTS_DIR', '/path/to/freesurfer/subjects')`.",
    )


@command(
    "source.bem",
    "Build a BEM head model for a FreeSurfer subject.",
    params=[
        Param("subject", "str", "FreeSurfer subject name", required=True, example="sample"),
        Param("subjects-dir", "path", "SUBJECTS_DIR holding the subject"),
        Param("spacing", "choice", "BEM mesh density", default="ico4", choices=["ico3", "ico4", "ico5"]),
        Param("conductivity", "csv", "tissue conductivities (1 or 3 values)", default=[0.3]),
    ],
    returns="{n_vertices, n_triangles, conductivity, handle}",
    examples=[{"args": "source bem --subject sample --subjects-dir D:/freesurfer/subjects"}],
    notes="Requires the watershed output (<subject>/bem/<subject>-head-dense.fif), which MNE creates from the MRI.",
    tags=("source",),
)
def cmd_source_bem(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    subjects_dir = _subjects_dir(ctx, p.get("subjects_dir"))
    ico = int(str(p.get("spacing") or "ico4").replace("ico", ""))
    conductivity = tuple(float(c) for c in (p.get("conductivity") or [0.3]))
    try:
        model = mne.make_bem_model(
            subject=p["subject"],
            ico=ico,
            subjects_dir=subjects_dir,
            conductivity=conductivity,
            verbose="ERROR",
        )
        bem = mne.make_bem_solution(model, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"BEM construction failed: {exc}",
            hint="Check SUBJECTS_DIR, the subject name, and that watershed BEM files exist (`mne watershed_bem`).",
        ) from None

    summary = {
        "subject": p["subject"],
        "n_vertices": int(sum(len(s["rr"]) for s in bem["surfs"])),
        "n_surfaces": len(bem["surfs"]),
        "conductivity": list(conductivity),
        "subjects_dir": subjects_dir,
    }
    handle = ctx.keep("bem", bem, summary=summary)
    return {**summary, "handle": handle}


@command(
    "source.forward",
    "Compute the forward solution (lead field) from a BEM model.",
    params=[
        raw_opt("raw object providing sensor geometry"),
        Param("bem", "str", "bem handle from `source bem`", required=True),
        Param("trans", "path", "head-to-MRI transformation file (-trans.fif)"),
        Param("subject", "str", "FreeSurfer subject name (needed when --src is omitted)"),
        Param("subjects-dir", "path", "SUBJECTS_DIR"),
        Param("src", "path", "source space file; built automatically when omitted"),
        Param("spacing", "choice", "source space density when building --src", default="oct6", choices=["oct5", "oct6", "oct7", "ico3", "ico4"]),
        Param("mindist", "float", "minimum source distance to the inner skull in mm", default=5.0),
        Param("meg", "bool", "include MEG channels", default=True),
        Param("eeg", "bool", "include EEG channels", default=True),
    ],
    returns="{n_sources, n_channels, meg, eeg, handle}",
    examples=[{"args": "source forward --raw raw_0001 --bem bem_0001 --trans sample-trans.fif --subject sample"}],
    tags=("source",),
)
def cmd_source_forward(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = ctx.object("raw", p["raw"])
    bem = ctx.object("bem", p["bem"])

    if p.get("src"):
        src = mne.read_source_spaces(p["src"], verbose="ERROR")
        subject = p.get("subject") or "unknown"
    else:
        if not p.get("subject"):
            raise UsageError("--subject is required when --src is not given")
        subjects_dir = _subjects_dir(ctx, p.get("subjects_dir"))
        src = mne.setup_source_space(
            p["subject"],
            spacing=p.get("spacing") or "oct6",
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose="ERROR",
        )
        subject = p["subject"]

    try:
        fwd = mne.make_forward_solution(
            raw.info,
            trans=p.get("trans"),
            src=src,
            bem=bem,
            meg=bool(p.get("meg", True)),
            eeg=bool(p.get("eeg", True)),
            mindist=float(p.get("mindist") or 5.0),
            verbose="ERROR",
        )
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"forward solution failed: {exc}",
            hint="Check the trans file, sensor positions (montage/digitisation) and BEM.",
        ) from None

    summary = {
        "subject": subject,
        "n_sources": int(fwd["nsource"]),
        "n_channels": int(fwd["nchan"]),
        "ch_types": ch_type_counts(raw.info),
        "meg": bool(p.get("meg", True)),
        "eeg": bool(p.get("eeg", True)),
    }
    handle = ctx.keep("forward", fwd, summary=summary)
    return {**summary, "handle": handle}


@command(
    "source.covariance",
    "Estimate the noise covariance matrix (needed by the inverse operator).",
    params=[
        Param("raw", "str", "raw handle (use a resting/baseline segment)"),
        Param("epochs", "str", "epochs handle (alternative to raw)"),
        Param("method", "choice", "regularisation", default="shrunk", choices=["shrunk", "empirical", "diagonal_fixed", "auto"]),
        Param("tmin", "float", "start of the covariance window in seconds"),
        Param("tmax", "float", "end of the covariance window in seconds"),
    ],
    returns="{n_channels, method, mean_diagonal, handle}",
    examples=[{"args": "source covariance --raw raw_0001 --method shrunk"}],
    tags=("source",),
)
def cmd_source_covariance(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne
    import numpy as np

    if p.get("epochs"):
        obj = ctx.object("epochs", p["epochs"])
    elif p.get("raw"):
        obj = ctx.object("raw", p["raw"])
    else:
        raise UsageError("pass --raw or --epochs")

    kwargs: dict[str, Any] = {"method": p.get("method") or "shrunk", "verbose": "ERROR"}
    if p.get("tmin") is not None:
        kwargs["tmin"] = float(p["tmin"])
    if p.get("tmax") is not None:
        kwargs["tmax"] = float(p["tmax"])
    cov = mne.compute_covariance(obj, **kwargs)

    diag = np.diag(cov["data"])
    summary = {
        "method": p.get("method") or "shrunk",
        "n_channels": int(cov["dim"]),
        "mean_diagonal": sig(float(np.mean(diag)), 4),
        "diagonal_range": [sig(float(np.min(diag)), 4), sig(float(np.max(diag)), 4)],
        "ch_types": ch_type_counts(obj.info),
    }
    handle = ctx.keep("cov", cov, summary=summary)
    return {**summary, "handle": handle}


@command(
    "source.inverse",
    "Build the inverse operator (MNE, dSPM or sLORETA).",
    params=[
        raw_opt("raw object providing sensor geometry"),
        Param("forward", "str", "forward handle", required=True),
        Param("cov", "str", "noise covariance handle", required=True),
        Param("method", "choice", "inverse method", default="dSPM", choices=["MNE", "dSPM", "sLORETA", "eLORETA"]),
        Param("snr", "float", "assumed SNR used to set regularisation", default=3.0),
        Param("loose", "float", "loose orientation constraint (0 = fixed)", default=0.2),
        Param("depth", "float", "depth weighting", default=0.8),
    ],
    returns="{method, n_sources, loose, depth, handle}",
    examples=[{"args": "source inverse --raw raw_0001 --forward forward_0001 --cov cov_0001 --method dSPM"}],
    tags=("source",),
)
def cmd_source_inverse(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    raw = ctx.object("raw", p["raw"])
    fwd = ctx.object("forward", p["forward"])
    cov = ctx.object("cov", p["cov"])
    try:
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info,
            forward=fwd,
            noise_cov=cov,
            loose=float(p.get("loose") if p.get("loose") is not None else 0.2),
            depth=float(p.get("depth") if p.get("depth") is not None else 0.8),
            verbose="ERROR",
        )
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"inverse operator failed: {exc}",
            hint="The forward model, covariance and raw channels must match exactly.",
        ) from None

    summary = {"method": p.get("method") or "dSPM", "n_sources": int(inv["nsource"]), "loose": float(p.get("loose") or 0.2)}
    handle = ctx.keep("inverse", inv, summary=summary)
    return {**summary, "handle": handle}


def _stc_summary(stc) -> dict[str, Any]:
    import numpy as np

    return {
        "n_vertices": int(stc.data.shape[0]),
        "n_times": int(stc.data.shape[1]),
        "tmin_s": round(float(stc.tmin), 4),
        "tmax_s": round(float(stc.times[-1]), 4),
        "data_range": [round(float(np.min(stc.data)), 6), round(float(np.max(stc.data)), 6)],
        "subject": str(stc.subject) if stc.subject else "",
    }


@command(
    "source.apply",
    "Apply the inverse operator to an evoked response, giving a source estimate.",
    params=[
        Param("inverse", "str", "inverse operator handle", required=True),
        Param("evoked", "str", "evoked handle", required=True),
        Param("method", "choice", "inverse method", default="dSPM", choices=["MNE", "dSPM", "sLORETA", "eLORETA"]),
        Param("snr", "float", "SNR used for regularisation (lambda2 = 1/snr^2)", default=3.0),
    ],
    returns="source estimate digest + handle",
    examples=[{"args": "source apply --inverse inverse_0001 --evoked evoked_0001 --method dSPM"}],
    tags=("source",),
)
def cmd_source_apply(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    inv = ctx.object("inverse", p["inverse"])
    evoked = ctx.object("evoked", p["evoked"])
    snr = float(p.get("snr") or 3.0)
    stc = mne.minimum_norm.apply_inverse(
        evoked,
        inv,
        lambda2=1.0 / (snr**2),
        method=p.get("method") or "dSPM",
        verbose="ERROR",
    )
    summary = {**_stc_summary(stc), "method": p.get("method") or "dSPM", "comment": str(evoked.comment)}
    handle = ctx.keep("stc", stc, summary=summary)
    return {**summary, "handle": handle}


@command(
    "source.info",
    "Describe a stored source estimate.",
    params=[Param("stc", "str", "source estimate handle", required=True)],
    returns="source estimate digest",
    examples=[{"args": "source info --stc stc_0001"}],
    tags=("source", "info"),
)
def cmd_source_info(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    stc = ctx.object("stc", p["stc"])
    return {**_stc_summary(stc), "handle": p["stc"]}
