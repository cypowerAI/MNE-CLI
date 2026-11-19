"""Format sniffing readers.

``--raw`` may point at a handle or straight at a recording; a language model
seldom knows the exact MNE reader function, so the suffix decides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import DataError


READERS = {
    ".fif": ("mne.io", "read_raw_fif"),
    ".gz": ("mne.io", "read_raw_fif"),
    ".edf": ("mne.io", "read_raw_edf"),
    ".bdf": ("mne.io", "read_raw_bdf"),
    ".gdf": ("mne.io", "read_raw_gdf"),
    ".vhdr": ("mne.io", "read_raw_brainvision"),
    ".vmrk": ("mne.io", "read_raw_brainvision"),
    ".eeg": ("mne.io", "read_raw_brainvision"),
    ".set": ("mne.io", "read_raw_eeglab"),
    ".cnt": ("mne.io", "read_raw_cnt"),
    ".egi": ("mne.io", "read_raw_egi"),
    ".mff": ("mne.io", "read_raw_egi"),
    ".bti": ("mne.io", "read_raw_bti"),
    ".pdf": ("mne.io", "read_raw_kit"),
    ".sqd": ("mne.io", "read_raw_kit"),
    ".con": ("mne.io", "read_raw_kit"),
    ".ds": ("mne.io", "read_raw_ctf"),
    ".nxe": ("mne.io", "read_raw_eximia"),
    ".ahk": ("mne.io", "read_raw_persyst"),
    ".lay": ("mne.io", "read_raw_persyst"),
    ".xdf": ("mne.io", "read_raw_xdf"),
}


def reader_for(path: Path) -> tuple[str, str]:
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        raise DataError(
            f"cannot infer a reader for {path.name!r} (no suffix)",
            hint="Rename with the proper extension or convert the file to .fif.",
        )
    # .fif.gz -> read_raw_fif
    if suffixes[-1] == ".gz" and len(suffixes) > 1:
        return "mne.io", "read_raw_fif"
    key = suffixes[-1]
    if key not in READERS:
        supported = ", ".join(sorted({k for k in READERS if k != ".gz"}))
        raise DataError(
            f"no reader registered for '{key}' files",
            hint=f"Supported extensions: {supported}. Convert with the vendor tool, or open an issue.",
        )
    return READERS[key]


def read_raw_auto(path: str | Path, *, preload: bool = False, verbose: str = "ERROR", extra: dict[str, Any] | None = None):
    import importlib

    path = Path(path).expanduser()
    if not path.exists():
        raise DataError(f"file not found: {path}", hint="Check the path; use `mne-cli dataset download sample` for demo data.")
    module_name, func_name = reader_for(path)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    kwargs: dict[str, Any] = {"preload": preload, "verbose": verbose}
    if func_name == "read_raw_eeglab":
        kwargs.pop("preload")
        kwargs["preload"] = preload
    if extra:
        kwargs.update(extra)
    try:
        return func(str(path), **kwargs)
    except TypeError:
        kwargs.pop("verbose", None)
        return func(str(path), **kwargs)


def read_object(kind: str, path: str | Path):
    """Read a plain file as the object kind a command expects."""
    import mne

    path = Path(path).expanduser()
    with mne.utils.use_log_level("ERROR"):
        if kind == "raw":
            return read_raw_auto(path)
        if kind == "epochs":
            return mne.read_epochs(str(path), preload=False, verbose="ERROR")
        if kind == "evoked":
            evokeds = mne.read_evokeds(str(path), verbose="ERROR")
            return evokeds[0] if len(evokeds) == 1 else evokeds
        if kind == "tfr":
            return mne.time_frequency.read_tfrs(str(path), verbose="ERROR")[0]
        if kind == "stc":
            return mne.read_source_estimate(str(path), verbose="ERROR")
        if kind == "ica":
            return mne.preprocessing.read_ica(str(path), verbose="ERROR")
        if kind == "inverse":
            return mne.minimum_norm.read_inverse_operator(str(path), verbose="ERROR")
        if kind == "forward":
            return mne.read_forward_solution(str(path), verbose="ERROR")
        if kind == "cov":
            return mne.read_cov(str(path), verbose="ERROR")
        if kind == "events":
            import numpy as np

            return np.load(str(path))
    raise DataError(f"cannot read {path.name} as a {kind}")
