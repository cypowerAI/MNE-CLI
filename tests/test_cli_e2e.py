"""End-to-end tests for the mne-cli command surface.

Everything runs through the real CLI entry point in a throw-away store, so the
tests cover argument parsing, the JSON envelope, handle persistence and the
MNE glue code — not just the individual handlers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def store(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mne-cli-store")


def run(store: Path, *args: str, expect_ok: bool = True) -> dict:
    env = os.environ.copy()
    env["MNE_CLI_STORE"] = str(store)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "mne_cli", *args],
        cwd=str(store),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout.strip().splitlines()
    assert out, f"no JSON on stdout for {args}\nstderr: {proc.stderr[-2000:]}"
    payload = json.loads(out[-1])
    if expect_ok:
        assert payload["ok"] is True, f"{args} failed: {payload}\nstderr: {proc.stderr[-2000:]}"
        assert proc.returncode == 0
    else:
        assert payload["ok"] is False
        assert proc.returncode != 0
    return payload


@pytest.fixture(scope="session")
def demo(store: Path) -> dict:
    return run(store, "dataset", "demo", "--duration", "20", "--n-events", "12")


@pytest.fixture(scope="session")
def raw_handle(store: Path, demo: dict) -> str:
    return run(store, "raw", "load", "--path", demo["data"]["path"])["handle"]


@pytest.fixture(scope="session")
def epochs_handle(store: Path, raw_handle: str) -> str:
    filtered = run(store, "raw", "filter", "--raw", raw_handle, "--l-freq", "1", "--h-freq", "40")
    events = run(store, "raw", "events", "--raw", filtered["handle"])
    built = run(
        store,
        "epochs",
        "build",
        "--raw",
        filtered["handle"],
        "--events",
        events["handle"],
        "--tmin",
        "-0.2",
        "--tmax",
        "0.8",
        "--event-id",
        '{"audio":1,"visual":2}',
    )
    return built["handle"]


def test_version_and_doctor(store: Path) -> None:
    version = run(store, "version")
    assert version["data"]["mne"]
    doctor = run(store, "doctor")
    assert "checks" in doctor["data"] or "store" in doctor["data"]


def test_commands_and_describe(store: Path) -> None:
    commands = run(store, "commands")
    names = {c["command"] for group in commands["data"]["groups"].values() for c in group}
    assert {"raw.load", "epochs.build", "tfr.compute", "dataset.demo"} <= names

    described = run(store, "describe", "epochs.build")
    assert any(p["name"] == "tmin" for p in described["data"]["params"])


def test_dotted_invocation(store: Path, raw_handle: str) -> None:
    dotted = run(store, "raw.info", "--raw", raw_handle)
    spaced = run(store, "raw", "info", "--raw", raw_handle)
    assert dotted["data"]["sfreq"] == spaced["data"]["sfreq"]


def test_raw_pipeline(store: Path, raw_handle: str) -> None:
    info = run(store, "raw", "info", "--raw", raw_handle)["data"]
    assert info["n_channels"] == 18
    assert info["ch_types"]["eeg"] == 16

    filtered = run(store, "raw", "filter", "--raw", raw_handle, "--l-freq", "1", "--h-freq", "40")["data"]
    assert filtered["highpass_hz"] == 1.0
    assert filtered["lowpass_hz"] == 40.0

    events = run(store, "raw", "events", "--raw", filtered["handle"])["data"]
    assert events["n_events"] >= 1

    stored = run(store, "store", "list")
    assert stored["data"]["n"] >= 3


def test_epochs_average_and_psd(store: Path, epochs_handle: str) -> None:
    averaged = run(store, "epochs", "average", "--epochs", epochs_handle, "--by", "event")["data"]
    assert averaged["n_evokeds"] == 2
    assert averaged["evokeds"][0]["n_channels"] == 16

    psd = run(store, "psd", "--raw", "raw_0002")["data"]
    assert psd["power_shape"][0] == 16
    # Powers are ~1e-10: the digest must not collapse them to zero.
    assert any(v > 0 for v in psd["mean_power_by_band"].values())


def test_tfr_and_ica(store: Path, epochs_handle: str, raw_handle: str) -> None:
    tfr = run(
        store, "tfr.compute", "--epochs", epochs_handle, "--fmin", "4", "--fmax", "30", "--n-freqs", "8"
    )["data"]
    assert tfr["shape"][0] == 16
    assert tfr["data_range"][1] > 0

    ica = run(store, "ica", "fit", "--raw", "raw_0002", "--n-components", "5")["data"]
    assert ica["n_components"] == 5


def test_stats_and_decode(store: Path, epochs_handle: str) -> None:
    stats = run(
        store, "epochs", "stats", "--epochs", epochs_handle, "--a", "audio", "--b", "visual", "--mode", "ttest"
    )["data"]
    assert stats["n_tests"] > 0

    decoded = run(store, "epochs", "decode", "--epochs", epochs_handle)["data"]
    assert 0.0 <= decoded["accuracy_mean"] <= 1.0


def test_report(store: Path, epochs_handle: str) -> None:
    report_path = store / "report.html"
    payload = run(
        store, "report", "build", "--epochs", epochs_handle, "--out", str(report_path)
    )
    assert report_path.exists()
    assert payload["data"]["size_mb"] > 0


def test_pipeline(store: Path, demo: dict) -> None:
    plan = store / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "steps": [
                    {"command": "raw.load", "args": {"path": demo["data"]["path"]}, "as": "raw"},
                    {"command": "raw.filter", "args": {"raw": "{{raw}}", "l_freq": 1, "h_freq": 40}, "as": "clean"},
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = run(store, "run", "--spec", str(plan))
    assert payload["data"]["n_failed"] == 0
    assert payload["data"]["bindings"]["clean"].startswith("raw_")


def test_mcp_manifest(store: Path) -> None:
    manifest = run(store, "mcp", "manifest")
    tools = manifest["data"]["tools"]
    assert any(tool["name"] == "mne_cli_epochs_build" for tool in tools)


def test_error_envelope(store: Path) -> None:
    missing = run(store, "raw", "info", "--raw", "raw_9999", expect_ok=False)
    assert missing["error"]["code"] == "UNKNOWN_HANDLE"
    assert missing["error"]["hint"]

    wrong_kind = run(store, "epochs", "info", "--epochs", "raw_0001", expect_ok=False)
    assert wrong_kind["error"]["code"] in {"WRONG_OBJECT_KIND", "UNKNOWN_HANDLE"}

    bad_args = run(store, "raw", "nope", expect_ok=False)
    assert bad_args["error"]["code"]
