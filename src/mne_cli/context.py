"""Shared context handed to every command handler."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import HandleError, UsageError
from .result import artifact as artifact_entry
from .store import Store


@dataclass
class Ctx:
    store: Store
    command: str
    fmt: str = "json"
    log_level: str = "WARNING"
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    primary_handle: str | None = None
    _counter: int = 0

    # ------------------------------------------------------------- arguments
    def p(self, name: str, default: Any = None):
        return self.params.get(name, default)

    def require(self, name: str):
        value = self.params.get(name)
        if value is None or value == "":
            raise UsageError(
                f"missing required option --{name}",
                hint=f"Run `mne-cli describe {self.command}` for the full parameter list.",
            )
        return value

    # --------------------------------------------------------------- objects
    def object(self, kind: str, value: Any, *, required: bool = True, what: str | None = None, preload: bool = True):
        """Resolve a handle **or** a file path into a live MNE object."""
        label = what or f"--{kind}"
        if value is None or value == "":
            if required:
                raise UsageError(f"{label} is required", hint=f"Pass a handle from `mne-cli store list`, or a file path.")
            return None

        if self.store.exists(str(value)):
            obj = self.store.load(str(value), expect=kind)
        else:
            candidate = Path(str(value)).expanduser()
            if candidate.exists():
                from .readers import read_object

                obj = read_object(kind, candidate)
            elif str(value).startswith(f"{kind}_"):
                raise HandleError(
                    f"{label}: handle '{value}' is not in this store ({self.store.root})",
                    hint="Run `mne-cli store list`. Handles are per-store: pass --store to reuse another directory.",
                )
            else:
                raise UsageError(
                    f"{label}: '{value}' is neither a handle in {self.store.root} nor an existing file",
                    hint="Load the file first, e.g. `mne-cli raw load --path <file>`, then pass the returned handle.",
                )

        return self._ensure_loaded(obj) if preload else obj

    def _ensure_loaded(self, obj):
        """Most analyses need samples in memory; MNE raises a cryptic error otherwise."""
        if getattr(obj, "preload", True) is False and hasattr(obj, "load_data"):
            try:
                obj.load_data(verbose="ERROR")
            except TypeError:
                obj.load_data()
        return obj

    def save(self, kind: str, obj: Any, *, summary: dict[str, Any] | None = None, source: str = "") -> str:
        handle = self.store.save(
            kind,
            obj,
            command=self.command,
            source=source,
            summary=summary,
            params=self.clean_params(),
        )
        self.primary_handle = handle
        return handle

    def keep(self, kind: str, obj: Any, *, summary: dict[str, Any] | None = None, source: str = "") -> str:
        """Alias of :meth:`save` that also makes the handle the command result."""
        return self.save(kind, obj, summary=summary, source=source)

    def clean_params(self) -> dict[str, Any]:
        """Parameters worth remembering in the index (no gigantic JSON)."""
        keep = {}
        for key, value in self.params.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, list)):
                text = str(value)
                keep[key] = value if len(text) < 200 else f"<{type(value).__name__} len={len(text)}>"
            else:
                keep[key] = f"<{type(value).__name__}>"
        return keep

    # ------------------------------------------------------------- artifacts
    @property
    def out_dir(self) -> Path:
        path = self.store.root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def out_path(self, stem: str, ext: str, explicit: str | None = None) -> Path:
        if explicit:
            path = Path(explicit).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        stamp = datetime.now().strftime("%H%M%S")
        path = self.out_dir / f"{safe}_{stamp}.{ext}"
        n = 1
        while path.exists():
            self._counter += 1
            path = self.out_dir / f"{safe}_{stamp}_{self._counter}.{ext}"
            n += 1
            if n > 100:
                break
        return path

    def add_artifact(self, kind: str, path: str | Path, label: str = "") -> dict[str, Any]:
        entry = artifact_entry(kind, path, label)
        self.artifacts.append(entry)
        return entry

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def suggest(self, command: str) -> None:
        self.next_steps.append(command)


def default_store_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("MNE_CLI_STORE")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / ".mne-cli"


def make_ctx(command: str, params: dict[str, Any], store_dir: str | None, fmt: str, log_level: str) -> Ctx:
    return Ctx(
        store=Store(default_store_dir(store_dir)),
        command=command,
        fmt=fmt,
        log_level=log_level,
        params=params,
    )
