"""Importing this package registers every command in the registry."""

from __future__ import annotations

import importlib

MODULES = (
    "system",
    "dataset",
    "io_",
    "preprocess",
    "epochs",
    "evoked",
    "ica",
    "spectral",
    "stats",
    "source",
    "report",
    "mcp_",
    "pipeline",
)


_loaded = False


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
    _loaded = True
