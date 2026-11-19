"""Introspection, environment health and the handle store."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import UsageError
from ..registry import REGISTRY, Param, command, command_schema, get, list_commands


@command(
    "commands",
    "List every available command. This is the discovery entry point.",
    params=[
        Param("group", "str", "restrict to one group, e.g. raw, epochs, ica"),
        Param("tag", "str", "restrict to commands carrying this tag"),
        Param("verbose", "bool", "include parameter counts and return types", default=True),
    ],
    returns="{n_commands, groups: {group: [{command, summary, n_params}]}}",
    examples=[
        {"args": "commands --json", "note": "full catalogue"},
        {"args": "commands --group raw --json", "note": "everything about raw signals"},
    ],
    tags=("meta",),
)
def cmd_commands(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    payload = list_commands()
    group = p.get("group")
    tag = p.get("tag")

    if tag:
        keep = {name for name, cmd in REGISTRY.items() if tag in cmd.tags}
        groups: dict[str, list[dict[str, Any]]] = {}
        for name in sorted(keep):
            cmd = REGISTRY[name]
            groups.setdefault(cmd.group, []).append(
                {"command": cmd.name, "summary": cmd.summary, "n_params": len(cmd.params)}
            )
        payload = {"n_commands": len(keep), "groups": groups}

    if group:
        groups = payload["groups"]
        if group not in groups:
            raise UsageError(
                f"no such group: {group}",
                hint="Available groups: " + ", ".join(sorted(groups)),
            )
        payload = {"n_commands": len(groups[group]), "groups": {group: groups[group]}}

    if not p.get("verbose", True):
        payload = {
            "n_commands": payload["n_commands"],
            "commands": sorted(n for g in payload["groups"].values() for n in [c["command"] for c in g]),
        }
    payload["hint"] = "Use `mne-cli describe <command>` for parameters and examples."
    return payload


@command(
    "describe",
    "Show the machine-readable schema of one command: flags, types, defaults, examples.",
    params=[Param("command", "str", "command name, e.g. raw.filter", required=True, example="raw.filter", positional=True)],
    returns="{command, summary, params[], returns, examples[]}",
    examples=[{"args": "describe raw.filter"}, {"args": "describe epochs.build --json"}],
    tags=("meta",),
)
def cmd_describe(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    return command_schema(get(p["command"]))  # type: ignore[arg-type]


def _mne_data_dir() -> str:
    """Where MNE keeps downloaded datasets, without touching private helpers."""
    import mne

    configured = mne.get_config("MNE_DATASETS_PATH")
    if configured:
        return str(configured)
    return str(Path.home() / "mne_data")


@command(
    "version",
    "Report mne-cli, MNE-Python and core scientific stack versions.",
    returns="{mne_cli, mne, numpy, scipy, python, platform, mne_data_dir}",
    examples=[{"args": "version"}],
    tags=("meta",),
)
def cmd_version(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import mne
    import numpy
    import scipy

    from .. import __version__

    optional = {}
    for module in ("sklearn", "pandas", "matplotlib", "nibabel", "pyvista", "h5py"):
        try:
            optional[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            optional[module] = None

    return {
        "mne_cli": __version__,
        "mne": mne.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mne_data_dir": _mne_data_dir(),
        "optional": optional,
    }


@command(
    "doctor",
    "Check that this machine can actually run MNE analyses (deps, data dir, store, backend).",
    returns="{healthy, checks: [{name, ok, detail, hint}]}",
    examples=[{"args": "doctor"}],
    tags=("meta",),
)
def cmd_doctor(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "hint": hint})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0], "mne-cli needs Python >= 3.10")

    try:
        import mne

        add("mne", True, mne.__version__)
    except Exception as exc:  # noqa: BLE001
        add("mne", False, str(exc), "pip install mne")
        mne = None  # type: ignore[assignment]

    if mne is not None:
        data_dir = _mne_data_dir()
        writable = True
        try:
            from pathlib import Path

            Path(data_dir).mkdir(parents=True, exist_ok=True)
            probe = Path(data_dir) / ".mne_cli_write_test"
            probe.write_text("ok")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            writable = False
            detail = f"{data_dir} ({exc})"
        else:
            detail = data_dir
        add("mne_data_dir_writable", writable, detail, "Set MNE_DATASETS_PATH to a writable folder")

    store_dir = ctx.store.root
    try:
        probe = store_dir / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        add("store_writable", True, str(store_dir))
    except Exception as exc:  # noqa: BLE001
        add("store_writable", False, f"{store_dir} ({exc})", "Pass --store to a writable directory")

    try:
        import matplotlib

        add("matplotlib_agg", matplotlib.get_backend().lower().startswith("agg"), matplotlib.get_backend())
    except Exception as exc:  # noqa: BLE001
        add("matplotlib_agg", False, str(exc), "pip install matplotlib")

    for module, extra in (("sklearn", "decoding"), ("pandas", "reports")):
        try:
            __import__(module)
            add(module, True, "available")
        except Exception:  # noqa: BLE001
            add(module, False, "not installed", f"Only needed for {extra}: pip install mne-cli[full]")

    healthy = all(c["ok"] for c in checks if c["name"] not in {"sklearn", "pandas"})
    return {"healthy": healthy, "checks": checks}


# ------------------------------------------------------------------ the store
@command(
    "store.list",
    "List every handle currently saved in the store.",
    params=[Param("kind", "str", "only show one kind: raw, epochs, evoked, ica, tfr, stc ...")],
    returns="{root, n, handles: [{handle, kind, created, command, source, summary}]}",
    examples=[{"args": "store list"}, {"args": "store list --kind raw"}],
    tags=("meta", "store"),
)
def cmd_store_list(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    items = ctx.store.listing(p.get("kind"))
    return {"root": str(ctx.store.root), "n": len(items), "handles": items}


@command(
    "store.show",
    "Show the provenance and summary of one handle (what produced it, from which file).",
    params=[Param("handle", "str", "handle name, e.g. raw_0001", required=True)],
    returns="{handle, kind, command, source, params, summary}",
    examples=[{"args": "store show --handle raw_0001"}],
    tags=("meta", "store"),
)
def cmd_store_show(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    entry = ctx.store.entry(p["handle"])
    entry["exists_on_disk"] = ctx.store.path_of(p["handle"]).exists()
    return entry


@command(
    "store.drop",
    "Delete one handle and its file.",
    params=[Param("handle", "str", "handle to remove", required=True)],
    returns="{dropped}",
    examples=[{"args": "store drop --handle raw_0001"}],
    tags=("meta", "store", "destructive"),
)
def cmd_store_drop(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    return ctx.store.drop(p["handle"])


@command(
    "store.clear",
    "Delete every handle in this store. Requires --confirm.",
    params=[Param("confirm", "bool", "must be passed explicitly; guards against accidents", default=False)],
    returns="{dropped}",
    examples=[{"args": "store clear --confirm"}],
    tags=("meta", "store", "destructive"),
)
def cmd_store_clear(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    if not p.get("confirm"):
        raise UsageError(
            "store.clear is destructive and needs an explicit --confirm",
            hint="Re-run with `--confirm` if you really want to delete all handles in this store.",
        )
    return ctx.store.clear()
