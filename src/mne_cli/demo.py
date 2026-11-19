"""Synthetic EEG generation, so an agent can try the CLI without downloading data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4"]


def make_demo_raw(
    *,
    n_channels: int = 16,
    sfreq: float = 250.0,
    duration: float = 60.0,
    n_events: int = 40,
    seed: int = 0,
    with_eog: bool = True,
):
    """Build an EEG recording with alpha rhythm, blinks and two event types.

    Returns ``(raw, events)`` — events are in the (sample, 0, code) format MNE
    expects, with code 1 = "audio" and code 2 = "visual".
    """
    import mne

    names = CHANNELS[:n_channels]
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq

    info = mne.create_info(names, sfreq, ch_types="eeg")
    data = rng.standard_normal((len(names), n_samples)) * 15e-6  # ~15 uV noise
    data += 20e-6 * np.sin(2 * np.pi * 10.0 * t)  # alpha
    data += 5e-6 * np.sin(2 * np.pi * 2.0 * t)  # slow drift

    # Event-related deflection: a 100 ms bump after every event, stronger for code 2.
    events = np.zeros((n_events, 3), dtype=int)
    event_samples = np.linspace(int(2 * sfreq), n_samples - int(1.5 * sfreq), n_events).astype(int)
    for i, sample in enumerate(event_samples):
        code = 1 if i % 2 == 0 else 2
        events[i] = [sample, 0, code]
        width = int(0.1 * sfreq)
        shape = np.exp(-0.5 * ((np.arange(width) - width / 2) / (width / 6)) ** 2)
        amp = 8e-6 if code == 1 else 14e-6
        seg = slice(sample, min(sample + width, n_samples))
        data[:, seg] += amp * shape[: seg.stop - seg.start]

    raw = mne.io.RawArray(data, info, verbose="ERROR")
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore", verbose="ERROR")
    except Exception:  # noqa: BLE001 - montage is a nicety, not a requirement
        pass

    # Blink artefacts: big frontal deflections, mirrored in an EOG channel.
    blink_times = rng.uniform(1.0, duration - 1.0, max(4, n_events // 4))
    eog_trace = np.zeros(n_samples)
    for bt in blink_times:
        start = int(bt * sfreq)
        width = int(0.25 * sfreq)
        shape = np.exp(-0.5 * ((np.arange(width) - width / 3) / (width / 5)) ** 2)
        end = min(start + width, n_samples)
        blink = 120e-6 * shape[: end - start]
        eog_trace[start:end] += blink
        frontal = [i for i, name in enumerate(names) if name.startswith("F")]
        for idx in frontal:
            data[idx, start:end] += 0.6 * blink

    all_names = names + (["EOG061"] if with_eog else [])
    all_types = ["eeg"] * len(names) + (["eog"] if with_eog else [])
    matrix = data if not with_eog else np.vstack([data, eog_trace])
    raw = mne.io.RawArray(matrix, mne.create_info(all_names, sfreq, ch_types=all_types), verbose="ERROR")
    if with_eog:
        try:
            raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore", verbose="ERROR")
        except Exception:  # noqa: BLE001
            pass

    stim = np.zeros((1, n_samples))
    for sample, _, code in events:
        stim[0, sample] += code
    stim_raw = mne.io.RawArray(stim, mne.create_info(["STI 014"], sfreq, ch_types=["stim"]), verbose="ERROR")
    raw.add_channels([stim_raw])
    return raw, events


def write_demo(path: str | Path, **kwargs: Any) -> Path:
    raw, _ = make_demo_raw(**kwargs)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True, verbose="ERROR")
    return path
