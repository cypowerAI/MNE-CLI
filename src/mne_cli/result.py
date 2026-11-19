"""JSON-safe rendering of scientific objects.

The CLI never streams raw data matrices. Arrays above a small threshold are
replaced by a statistical digest so that a language model can reason about
them without blowing its context window, while still being able to point at
the artifact file that holds the full array.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

SMALL_ARRAY = 64          # arrays up to this size are serialised in full
PREVIEW_ROWS = 3


def _array_summary(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    if arr.size <= SMALL_ARRAY:
        return {"values": arr.tolist(), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    flat = arr.astype(float, copy=False).ravel() if arr.dtype.kind in "biuf" else None
    summary: dict[str, Any] = {
        "kind": "array_summary",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_elements": int(arr.size),
    }
    if flat is not None and flat.size:
        finite = flat[np.isfinite(flat)]
        if finite.size:
            summary.update(
                min=_r(float(finite.min())),
                max=_r(float(finite.max())),
                mean=_r(float(finite.mean())),
                std=_r(float(finite.std())),
            )
        n_nan = int(np.isnan(flat).sum())
        if n_nan:
            summary["n_nan"] = n_nan
    if arr.dtype.kind in "UO":
        summary["preview"] = [str(v) for v in arr.ravel()[:PREVIEW_ROWS]]
    if arr.ndim == 1 and arr.size <= 4096:
        summary["head"] = [to_jsonable(v) for v in arr[:PREVIEW_ROWS]]
        summary["tail"] = [to_jsonable(v) for v in arr[-PREVIEW_ROWS:]]
    return summary


def _r(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    # Significant digits, not decimal places: PSD/TFR powers are ~1e-10 and
    # round(1e-10, 6) would silently collapse them to 0.
    rounded = float(f"{value:.10g}")
    return 0.0 if rounded == 0 else rounded


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert numpy / MNE / python values into strict JSON values."""
    if _depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _r(float(obj))
    if isinstance(obj, np.ndarray):
        return _array_summary(obj)
    if isinstance(obj, (np.void,)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_jsonable(obj.to_dict(), _depth + 1)
        except Exception:  # noqa: BLE001 - fall through to repr
            pass
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return to_jsonable(asdict(obj), _depth + 1)
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    return str(obj)


ARRAY_SUMMARY = _array_summary


def artifact(kind: str, path: str | Path, label: str = "") -> dict[str, Any]:
    """Describe a file a command produced."""
    entry = {"kind": kind, "path": str(Path(path).resolve())}
    if label:
        entry["label"] = label
    return entry


def array_summary(arr: Any) -> dict[str, Any]:
    """Public helper for commands that want a digest without full data."""
    return _array_summary(np.asarray(arr))


def sig(value: float, digits: int = 5) -> float | None:
    """Round to significant digits — PSD values are ~1e-10, not ~0."""
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return float(f"{value:.{digits}g}")
