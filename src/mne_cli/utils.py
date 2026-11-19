"""Helpers shared by command modules: channel selection and object digests."""

from __future__ import annotations

from typing import Any, Iterable

from .errors import DataError, UsageError
from .result import to_jsonable

CH_TYPE_KEYS = ("meg", "mag", "grad", "eeg", "csd", "seeg", "ecog", "dbs", "fnirs", "eog", "ecg", "emg", "misc", "stim", "resp", "bio", "chpi", "exci", "ias", "syst", "gofs", "temperature", "gsr", "eyetrack")


def ch_type_counts(info) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in info["chs"]:
        kind = _ch_kind(ch)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _ch_kind(ch) -> str:
    from mne.io.constants import FIFF

    kind = ch["kind"]
    mapping = {
        FIFF.FIFFV_MEG_CH: "meg",
        FIFF.FIFFV_REF_MEG_CH: "ref_meg",
        FIFF.FIFFV_EEG_CH: "eeg",
        FIFF.FIFFV_EOG_CH: "eog",
        FIFF.FIFFV_ECG_CH: "ecg",
        FIFF.FIFFV_EMG_CH: "emg",
        FIFF.FIFFV_STIM_CH: "stim",
        FIFF.FIFFV_MISC_CH: "misc",
        FIFF.FIFFV_RESP_CH: "resp",
        FIFF.FIFFV_SEEG_CH: "seeg",
        FIFF.FIFFV_ECOG_CH: "ecog",
        FIFF.FIFFV_FNIRS_CH: "fnirs",
        FIFF.FIFFV_SYST_CH: "syst",
        FIFF.FIFFV_IAS_CH: "ias",
        FIFF.FIFFV_EXCI_CH: "exci",
        FIFF.FIFFV_QUAT_0: "quat",
    }
    return mapping.get(kind, str(kind))


def resolve_picks(info, names: Iterable[str] | None = None, types: Iterable[str] | None = None) -> list[str] | None:
    """Translate ``--picks``/``--ch-types`` into channel names."""
    import mne

    picked: set[str] | None = None

    if types:
        kwargs = {key: False for key in CH_TYPE_KEYS}
        for token in types:
            token = token.strip().lower()
            if token in {"all", "any"}:
                kwargs = {key: True for key in CH_TYPE_KEYS}
                continue
            if token == "meg":
                kwargs["meg"] = True
                continue
            if token not in kwargs:
                raise UsageError(
                    f"unknown channel type '{token}'",
                    hint="Use one of: " + ", ".join(k for k in CH_TYPE_KEYS if k != "ref_meg"),
                )
            kwargs[token] = True
        if not any(kwargs.values()):
            raise UsageError("--ch-types selected nothing")
        try:
            idx = mne.pick_types(info, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DataError(str(exc), hint="Check the channel types with `mne-cli raw channels`.") from None
        picked = {info["ch_names"][i] for i in idx}

    if names:
        names = [n.strip() for n in names if n.strip()]
        missing = [n for n in names if n not in info["ch_names"]]
        if missing:
            available = ", ".join(info["ch_names"][:25])
            raise DataError(
                f"channel(s) not found: {', '.join(missing[:10])}",
                hint=f"Available channels start with: {available} ...",
            )
        picked = set(names) if picked is None else picked & set(names)

    if picked is None:
        return None
    result = [name for name in info["ch_names"] if name in picked]
    if not result:
        raise DataError("the channel selection is empty", hint="Loosen --picks or --ch-types.")
    return result


def safe_picks(raw_or_epochs, names: list[str] | None):
    """Return a picks argument MNE accepts, or None for 'all data channels'."""
    return names


def info_digest(info, max_names: int = 64) -> dict[str, Any]:
    counts = ch_type_counts(info)
    payload: dict[str, Any] = {
        "sfreq": float(info["sfreq"]),
        "n_channels": len(info["ch_names"]),
        "ch_types": counts,
        "bad_channels": [info["ch_names"][i] for i in info["bads"]] if info["bads"] else [],
        "highpass_hz": float(info.get("highpass", 0) or 0),
        "lowpass_hz": float(info.get("lowpass", 0) or 0),
        "n_proj": len(info.get("projs", [])),
    }
    if counts.get("eeg"):
        payload["montage"] = "set" if _has_montage(info) else "missing"
    if len(info["ch_names"]) <= max_names:
        payload["ch_names"] = list(info["ch_names"])
    else:
        payload["ch_names_preview"] = list(info["ch_names"][:10])
    return payload


def _has_montage(info) -> bool:
    try:
        return info.get_montage() is not None
    except Exception:  # noqa: BLE001
        return False


def raw_digest(raw, max_names: int = 64) -> dict[str, Any]:
    payload = info_digest(raw.info, max_names=max_names)
    payload.update(
        n_times=int(raw.n_times),
        duration_s=round(float(raw.times[-1]) - float(raw.times[0]) + 1.0 / raw.info["sfreq"], 3) if raw.n_times else 0.0,
        first_time_s=round(float(raw.times[0]), 3) if raw.n_times else 0.0,
        first_samp=int(raw.first_samp),
        preload=bool(raw.preload),
    )
    payload["annotations"] = annotation_summary(raw)
    return payload


def annotation_summary(raw, limit: int = 10) -> dict[str, Any]:
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return {"n": 0}
    descriptions = sorted({str(d) for d in ann.description})
    items = [
        {"onset": round(float(o), 3), "duration": round(float(d), 3), "description": str(desc)}
        for o, d, desc in zip(ann.onset[:limit], ann.duration[:limit], ann.description[:limit])
    ]
    return {"n": len(ann), "descriptions": descriptions[:20], "first": items}


def epochs_digest(epochs, max_names: int = 64) -> dict[str, Any]:
    payload = {
        "n_epochs": len(epochs),
        "tmin_s": float(epochs.tmin),
        "tmax_s": float(epochs.tmax),
        "sfreq": float(epochs.info["sfreq"]),
        "n_channels": len(epochs.ch_names),
        "ch_types": ch_type_counts(epochs.info),
        "event_id": {k: int(v) for k, v in epochs.event_id.items()},
        "n_events_by_condition": _epoch_counts(epochs),
        "baseline": [round(float(v), 4) for v in epochs.baseline] if epochs.baseline else None,
        "reject": to_jsonable(epochs.reject),
        "drop_log_rejected": int(sum(1 for log in epochs.drop_log if log == ("DROP",))),
        "preload": bool(epochs.preload),
    }
    if len(epochs.ch_names) <= max_names:
        payload["ch_names"] = list(epochs.ch_names)
    return payload


def _epoch_counts(epochs) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code in epochs.events[:, 2]:
        key = next((k for k, v in epochs.event_id.items() if v == code), str(int(code)))
        counts[key] = counts.get(key, 0) + 1
    return counts


def evoked_digest(evoked, max_names: int = 64) -> dict[str, Any]:
    import numpy as np

    data = evoked.data
    payload = {
        "comment": str(evoked.comment),
        "nave": int(evoked.nave),
        "n_times": int(data.shape[1]),
        "tmin_s": float(evoked.tmin),
        "tmax_s": float(evoked.tmax),
        "sfreq": float(evoked.info["sfreq"]),
        "n_channels": len(evoked.ch_names),
        "ch_types": ch_type_counts(evoked.info),
        "data_range": [round(float(np.min(data)), 6), round(float(np.max(data)), 6)],
        "baseline": [round(float(v), 4) for v in evoked.baseline] if evoked.baseline else None,
    }
    if len(evoked.ch_names) <= max_names:
        payload["ch_names"] = list(evoked.ch_names)
    return payload


def event_id_from(value: Any, epochs_or_events=None) -> dict[str, int] | None:
    """Normalise ``--event-id``: dict, 'name:code,name:code', or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(item): i + 1 for i, item in enumerate(value)}
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        out = {}
        for part in text.split(","):
            key, _, code = part.partition(":")
            out[key.strip()] = int(code)
        return out
    raise UsageError(
        f"cannot parse event id: {value!r}",
        hint='Pass JSON like \'{"audio":1,"visual":2}\' or "audio:1,visual:2".',
    )


def condition_names(epochs) -> list[str]:
    return list(epochs.event_id.keys())
