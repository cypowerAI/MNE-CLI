"""Persistent handle store.

A CLI process is short-lived, so every object an agent creates has to survive
on disk. The store keeps serialised MNE objects plus an index, and hands back a
short handle (``raw_0001``) that later commands refer to via ``--raw``,
``--epochs`` and friends. That removes the need for an agent to carry large
arrays or file paths through a conversation.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import HandleError, KindError, MneCliError
from .result import to_jsonable

INDEX_NAME = "index.json"
KINDS = ("raw", "epochs", "evoked", "ica", "tfr", "stc", "inverse", "forward", "cov", "bem", "events", "array", "report", "meta")

EXTENSIONS = {
    "raw": "fif",
    "epochs": "fif",
    "evoked": "fif",
    "ica": "fif",
    "tfr": "h5",
    "stc": "h5",
    "inverse": "fif",
    "forward": "fif",
    "cov": "fif",
    "bem": "fif",
    "events": "npy",
    "array": "npy",
    "report": "html",
    "meta": "json",
}

HANDLE_RE = re.compile(r"^(?P<kind>[a-z]+)_(?P<num>\d{4,})$")


class Store:
    """Directory-backed object store."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / INDEX_NAME
        self._index = self._read_index()

    # ---------------------------------------------------------------- index
    def _read_index(self) -> dict[str, Any]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raise MneCliError(
                    f"store index is corrupted: {self._index_path}",
                    hint="Delete the store directory (or set --store to another path) to start clean.",
                ) from None
        return {"handles": {}, "counters": {}}

    def _flush(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._index, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._index_path)

    # -------------------------------------------------------------- handles
    def next_handle(self, kind: str) -> str:
        num = int(self._index["counters"].get(kind, 0)) + 1
        self._index["counters"][kind] = num
        width = max(4, len(str(num)))
        return f"{kind}_{num:0{width}d}"

    def kind_of(self, handle: str) -> str | None:
        if handle in self._index["handles"]:
            return self._index["handles"][handle]["kind"]
        match = HANDLE_RE.match(handle)
        return match.group("kind") if match else None

    def path_of(self, handle: str) -> Path:
        entry = self._index["handles"].get(handle)
        if entry is None:
            raise HandleError(
                f"unknown handle: {handle}",
                hint="Run `mne-cli store list` to see what exists in this store.",
            )
        return self.root / entry["file"]

    def exists(self, handle: str) -> bool:
        return handle in self._index["handles"]

    # ------------------------------------------------------------------ io
    def save(
        self,
        kind: str,
        obj: Any,
        handle: str | None = None,
        *,
        command: str = "",
        source: str = "",
        summary: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        if kind not in KINDS:
            raise MneCliError(f"cannot store unknown kind '{kind}'")
        handle = handle or self.next_handle(kind)
        ext = EXTENSIONS[kind]
        filename = f"{handle}.{ext}"
        path = self.root / filename
        _write_object(kind, obj, path)

        self._index["handles"][handle] = {
            "kind": kind,
            "file": filename,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": command,
            "source": source,
            "summary": to_jsonable(summary or {}),
            "params": to_jsonable(params or {}),
        }
        self._flush()
        return handle

    def load(self, handle: str, expect: str | None = None) -> Any:
        kind = self.kind_of(handle)
        if kind is None:
            raise HandleError(f"unknown handle: {handle}", hint="Run `mne-cli store list`.")
        if expect and kind != expect:
            raise KindError(
                f"handle {handle} holds a '{kind}', but this command needs a '{expect}'",
                hint=f"Create a {expect} first, e.g. `mne-cli {expect} info --{expect} <handle>`.",
            )
        path = self.path_of(handle)
        if not path.exists():
            raise HandleError(
                f"the file behind handle {handle} is gone ({path.name})",
                hint="Recreate it, or run `mne-cli store drop --handle " + handle + "` to clean the index.",
            )
        return _read_object(kind, path)

    def entry(self, handle: str) -> dict[str, Any]:
        entry = self._index["handles"].get(handle)
        if entry is None:
            raise HandleError(f"unknown handle: {handle}")
        return {"handle": handle, **entry}

    def listing(self, kind: str | None = None) -> list[dict[str, Any]]:
        items = []
        for handle, entry in self._index["handles"].items():
            if kind and entry["kind"] != kind:
                continue
            items.append({"handle": handle, **entry})
        return sorted(items, key=lambda x: x["created"])

    def drop(self, handle: str) -> dict[str, Any]:
        entry = self._index["handles"].pop(handle, None)
        if entry is None:
            raise HandleError(f"unknown handle: {handle}")
        path = self.root / entry["file"]
        if path.exists():
            path.unlink()
        self._flush()
        return {"dropped": handle}

    def clear(self) -> dict[str, Any]:
        count = len(self._index["handles"])
        for entry in self._index["handles"].values():
            path = self.root / entry["file"]
            if path.exists():
                path.unlink()
        self._index = {"handles": {}, "counters": {}}
        self._flush()
        return {"dropped": count}


# --------------------------------------------------------------- serialisers
def _write_object(kind: str, obj: Any, path: Path) -> None:
    import mne

    with mne.utils.use_log_level("ERROR"):
        if kind == "raw":
            obj.save(str(path), overwrite=True, verbose="ERROR")
        elif kind == "epochs":
            obj.save(str(path), overwrite=True, verbose="ERROR")
        elif kind == "evoked":
            obj.save(str(path), overwrite=True, verbose="ERROR")
        elif kind == "ica":
            obj.save(str(path), overwrite=True, verbose="ERROR")
        elif kind == "tfr":
            obj.save(str(path), overwrite=True, verbose="ERROR")
        elif kind == "stc":
            obj.save(str(path), ftype="h5", overwrite=True, verbose="ERROR")
        elif kind == "inverse":
            mne.minimum_norm.write_inverse_operator(str(path), obj, overwrite=True, verbose="ERROR")
        elif kind == "forward":
            mne.write_forward_solution(str(path), obj, overwrite=True, verbose="ERROR")
        elif kind == "cov":
            mne.write_cov(str(path), obj, overwrite=True, verbose="ERROR")
        elif kind == "bem":
            mne.write_bem_solution(str(path), obj, overwrite=True, verbose="ERROR")
        elif kind in {"events", "array"}:
            import numpy as np

            np.save(str(path), obj)
        elif kind in {"report", "meta"}:
            if isinstance(obj, (dict, list)):
                path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                path.write_text(str(obj), encoding="utf-8")
        else:  # pragma: no cover - defensive
            raise MneCliError(f"no serialiser for kind '{kind}'")


def _read_object(kind: str, path: Path) -> Any:
    import mne

    with mne.utils.use_log_level("ERROR"):
        if kind == "raw":
            return mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
        if kind == "epochs":
            return mne.read_epochs(str(path), preload=False, verbose="ERROR")
        if kind == "evoked":
            return mne.read_evokeds(str(path), verbose="ERROR")[0]
        if kind == "ica":
            return mne.preprocessing.read_ica(str(path), verbose="ERROR")
        if kind == "tfr":
            tfrs = mne.time_frequency.read_tfrs(str(path), verbose="ERROR")
            return tfrs[0] if isinstance(tfrs, (list, tuple)) else tfrs
        if kind == "stc":
            return mne.read_source_estimate(str(path), verbose="ERROR")
        if kind == "inverse":
            return mne.minimum_norm.read_inverse_operator(str(path), verbose="ERROR")
        if kind == "forward":
            return mne.read_forward_solution(str(path), verbose="ERROR")
        if kind == "cov":
            return mne.read_cov(str(path), verbose="ERROR")
        if kind == "bem":
            return mne.read_bem_solution(str(path), verbose="ERROR")
        if kind in {"events", "array"}:
            import numpy as np

            return np.load(str(path), allow_pickle=False)
        if kind == "meta":
            return json.loads(path.read_text(encoding="utf-8"))
        raise MneCliError(f"no reader for kind '{kind}'")
