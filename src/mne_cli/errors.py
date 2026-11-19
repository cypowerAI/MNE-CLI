"""Structured, agent-readable errors.

Every failure surfaces as a machine-readable ``code`` plus a human ``message``
and, where possible, a ``hint`` that tells the caller what to do next. Agents
are expected to branch on ``code``, not to parse prose.
"""

from __future__ import annotations

from typing import Any


class MneCliError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "MNE_CLI_ERROR"
    exit_code = 1

    def __init__(self, message: str, hint: str = "", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


class UsageError(MneCliError):
    code = "USAGE_ERROR"
    exit_code = 2


class HandleError(MneCliError):
    code = "UNKNOWN_HANDLE"


class KindError(MneCliError):
    code = "WRONG_OBJECT_KIND"


class DataError(MneCliError):
    code = "DATA_ERROR"


class DependencyError(MneCliError):
    code = "MISSING_DEPENDENCY"


class FileError(MneCliError):
    code = "FILE_ERROR"


def classify(exc: BaseException) -> MneCliError:
    """Wrap an arbitrary exception into something an agent can act on."""
    import os

    if isinstance(exc, MneCliError):
        return exc

    name = type(exc).__name__
    text = str(exc)
    low = text.lower()

    if isinstance(exc, FileNotFoundError):
        missing = getattr(exc, "filename", None) or ""
        return FileError(
            f"file not found: {missing or text}",
            hint="Check the path, or pass an absolute path. Use `mne-cli dataset download sample` to get demo data.",
        )
    if isinstance(exc, PermissionError):
        return FileError(f"permission denied: {text}", hint="Close any program holding the file, or choose another --out path.")
    if isinstance(exc, ImportError) or name == "ModuleNotFoundError":
        module = getattr(exc, "name", None) or text
        return DependencyError(
            f"missing optional dependency: {module}",
            hint=f"Install it with: pip install \"mne-cli[full]\" (or pip install {module})",
        )
    if isinstance(exc, MemoryError):
        return DataError("out of memory", hint="Crop/resample the data, or load fewer channels (use --picks).")
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        return DataError(text or name, hint="Re-check the arguments with `mne-cli describe <command>`.")
    if name in {"RuntimeError", "AssertionError"} and ("no " in low and "channel" in low):
        return DataError(text, hint="Select channels that exist; run `mne-cli raw channels --raw <handle>` first.")

    hint = ""
    if os.path.isfile(text):  # MNE often raises with a bare path-ish message
        hint = "The file could not be read as the expected format; try --format."
    return MneCliError(f"{name}: {text}", hint=hint)
