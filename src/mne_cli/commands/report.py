"""HTML analysis report: one shareable file summarising everything so far."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import UsageError
from ..registry import Param, command


@command(
    "report.build",
    "Assemble an HTML report (raw, epochs, evoked, ICA, PSD figures) and save it.",
    params=[
        Param("out", "path", "output .html path", required=True),
        Param("title", "str", "report title", default="MNE analysis report"),
        Param("raw", "csv", "raw handles to include"),
        Param("epochs", "csv", "epochs handles to include"),
        Param("evoked", "csv", "evoked handles to include"),
        Param("ica", "csv", "ica handles to include"),
        Param("psd", "bool", "add power spectral density figures", default=True),
        Param("open-browser", "bool", "open the report after saving", default=False),
    ],
    returns="{path, size_mb, sections: [...]}",
    examples=[
        {"args": "report build --out report.html --raw raw_0001 --epochs epochs_0001 --evoked evoked_0001"},
    ],
    notes="Everything is rendered headless; nothing needs a display.",
    tags=("report",),
)
def cmd_report_build(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne

    report = mne.Report(title=p.get("title") or "MNE analysis report")
    sections: list[str] = []

    for handle in p.get("raw") or []:
        raw = ctx.object("raw", handle)
        report.add_raw(raw, title=f"Raw {handle}", psd=bool(p.get("psd", True)))
        sections.append(f"raw:{handle}")

    for handle in p.get("epochs") or []:
        epochs = ctx.object("epochs", handle)
        report.add_epochs(epochs, title=f"Epochs {handle}", psd=bool(p.get("psd", True)))
        sections.append(f"epochs:{handle}")

    for handle in p.get("evoked") or []:
        evoked = ctx.object("evoked", handle)
        report.add_evokeds([evoked], titles=[f"Evoked {handle}"])
        sections.append(f"evoked:{handle}")

    for handle in p.get("ica") or []:
        ica = ctx.object("ica", handle)
        inst = None
        raw_handles = p.get("raw") or []
        if raw_handles:
            inst = ctx.object("raw", raw_handles[0])
        report.add_ica(ica, title=f"ICA {handle}", inst=inst)
        sections.append(f"ica:{handle}")

    if not sections:
        raise UsageError(
            "nothing to put in the report",
            hint="Pass at least one of --raw, --epochs, --evoked or --ica.",
        )

    out = Path(p["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(out), overwrite=True, open_browser=bool(p.get("open_browser")))
    ctx.add_artifact("html", out, "report")
    return {
        "path": str(out.resolve()),
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "sections": sections,
        "n_sections": len(sections),
    }
