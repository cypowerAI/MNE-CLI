---
name: mne-cli
description: Run EEG/MEG analysis (MNE-Python) through JSON-speaking commands. Use when the user asks to inspect, filter, epoch, average, decode, time-frequency analyse, ICA-clean, plot or report on .fif/.edf/.bdf/.vhdr/.set recordings — or to build a reproducible pipeline — without writing Python. Objects travel between commands as handles such as raw_0001.
---

# MNE-CLI — operating instructions for an LLM agent

You are driving **MNE-Python** through a CLI where every step is one command and
every object is a short opaque handle. Read this file once; after that, discover
the exact arguments from the CLI itself, never from memory.

## 1. Ground rules

1. **Never guess a signature.** `mne-cli describe <command>` returns every flag,
   type, default, example and the shape of the returned data.
2. **Never carry arrays between turns.** Anything you need later must live in the
   store and be referenced by its handle (`raw_0002`, `epochs_0001`, …).
3. **Read the envelope, not the logs.** stdout is exactly one JSON object. MNE's
   own logging goes to stderr and is irrelevant unless a command failed.
4. **Units are SI.** Voltages are volts: 100 µV is `150e-6`, not `150`.
5. **Both spellings work.** `mne-cli epochs build …` and `mne-cli epochs.build …`
   are equivalent; use whichever the catalogue prints.

## 2. First three calls in any new session

```bash
mne-cli commands                # every command, grouped, with one-line summaries
mne-cli describe raw.filter     # exact parameters for the command you need
mne-cli doctor                  # optional: is MNE importable, is the store writable?
```

`mne-cli commands --group raw` narrows to one group. Bare `mne-cli` is the same
as `mne-cli commands`.

## 3. The envelope

Success:

```json
{"ok":true,"command":"epochs.build",
 "data":{"n_epochs":40,"event_id":{"audio":1,"visual":2},"handle":"epochs_0001"},
 "handle":"epochs_0001",
 "warnings":["..."],
 "next":["mne-cli epochs.average --epochs epochs_0001 --by event"],
 "store":"D:/work/.mne-cli","elapsed_ms":586}
```

Failure:

```json
{"ok":false,"command":"raw.load",
 "error":{"code":"FILE_ERROR","message":"file not found: D:/data/x.fif",
          "hint":"Use an absolute path, or `mne-cli dataset demo` for synthetic data."},
 "elapsed_ms":120}
```

Fields to act on: `handle` (what to pass next), `next` (the recommended next
command), `warnings` (non-fatal caveats), `artifacts` (files produced, with
absolute paths), `error.code` / `error.hint`.

| code | meaning | recovery |
| --- | --- | --- |
| `USAGE_ERROR` | unknown flag/command, missing argument | run `describe <command>` |
| `UNKNOWN_HANDLE` | handle not in this store | `store list`; handles are per-store |
| `WRONG_OBJECT_KIND` | passed `raw_0001` where epochs was expected | check the `--<kind>` flag |
| `DATA_ERROR` | MNE rejected the data or arguments | read `hint`, adjust parameters |
| `MISSING_DEPENDENCY` | optional package absent (e.g. scikit-learn) | `pip install "mne-cli[full]"` |
| `FILE_ERROR` | path missing/unreadable | fix path or generate demo data |
| `MNE_CLI_ERROR` | unexpected MNE exception | message carries the exception text |

## 4. Handles: the only state

Objects are serialised into a store (default `./.mne-cli`, override with
`--store DIR` or `$MNE_CLI_STORE`) and indexed in `index.json`.

| kind | produced by | typical next command |
| --- | --- | --- |
| `raw` | `raw load`, `raw filter`, `raw crop`, `ica apply` | `raw info`, `epochs build` |
| `events` | `raw events`, `raw annotations-from-events` | `epochs build` |
| `epochs` | `epochs build`, `epochs crop/equalize/drop-bad` | `epochs average`, `psd`, `tfr compute` |
| `evoked` | `epochs average` | `evoked plot`, `evoked values`, `report build` |
| `ica` | `ica fit`, `ica find-artifacts`, `ica set-exclude` | `ica apply`, `ica plot-components` |
| `tfr` | `tfr compute` | `tfr plot`, `tfr info`, `tfr array` |
| `cov`, `forward`, `inverse`, `stm`/`stc`, `bem` | `source *` | `source inverse/apply` |

Useful housekeeping: `mne-cli store list` (everything with provenance),
`mne-cli store show --handle raw_0002`, `mne-cli store drop --handle raw_0002`,
`mne-cli store clear`.

## 5. The canonical ERP recipe

```bash
mne-cli dataset demo --out demo_raw.fif                    # or raw load --path <file>
mne-cli raw load --path demo_raw.fif                      # -> raw_0001
mne-cli raw filter --raw raw_0001 --l-freq 1 --h-freq 40  # -> raw_0002
mne-cli raw events --raw raw_0002                         # -> events_0001
mne-cli epochs.build --raw raw_0002 --events events_0001 \
        --tmin -0.2 --tmax 0.8 \
        --baseline '[-0.2,0]' --reject '{"eeg":150e-6}' \
        --event-id '{"audio":1,"visual":2}'               # -> epochs_0001
mne-cli epochs.average --epochs epochs_0001 --by event     # -> evoked_0001, evoked_0002
mne-cli evoked.plot --evoked evoked_0001 --kind butterfly
mne-cli epochs.stats --epochs epochs_0001 --a audio --b visual --mode cluster
```

Notes that save a round trip:

- `--baseline` takes a JSON pair, `[-0.2,0]`; omit it for no baseline.
- `--reject` and `--event-id` take JSON objects, quoted for the shell.
- `epochs average --by event` returns one evoked per condition; the first one
  becomes the primary `handle`, all of them are listed in `data.evokeds`.
- Non-preloaded files are loaded automatically when an analysis needs samples.
- `epochs.stats --mode cluster` defaults to spatio-temporal clustering when a
  channel adjacency exists and returns the 5 most significant clusters.

## 6. Spectral, ICA, decoding, reports

```bash
mne-cli psd --raw raw_0002 --plot                        # alpha peak + band powers
mne-cli tfr.compute --epochs epochs_0001 --fmin 4 --fmax 30 --n-freqs 12 --baseline '[-0.2,0]' --plot
mne-cli ica.fit --raw raw_0002 --n-components 10          # -> ica_0001
mne-cli ica.find-artifacts --ica ica_0001 --raw raw_0002  # threshold defaults to "auto"
mne-cli ica.plot-components --ica ica_0001
mne-cli ica.apply --ica ica_0001 --raw raw_0002           # only if components are flagged
mne-cli epochs.decode --epochs epochs_0001 --pipeline csp-lda
mne-cli report build --raw raw_0002 --epochs epochs_0001 --evoked evoked_0001 --out report.html
```

- PSD/TFR powers are tiny (`1e-10` scale); the digest uses significant digits,
  so `1.18e-09` is a real value, not a rounding artefact.
- `ica find-artifacts` only flags components when the recording has an EOG/ECG
  channel and the correlation passes the threshold; if nothing is flagged, look
  at `ica plot-components` and use `ica set-exclude --components 0,3`.
- `report build` writes a self-contained HTML with figures embedded.

## 7. Plotting

Every `*-plot` command writes a PNG into `.mne-cli/artifacts/` (or `--out`) and
returns the absolute path in `data.path` / `artifacts`. Matplotlib is forced to
the `Agg` backend, so nothing opens a window. Present the PNG to the user; do not
try to describe pixel content you cannot see.

## 8. Run a whole plan in one call

```json
{
  "steps": [
    {"command": "raw.load", "args": {"path": "demo_raw.fif"}, "as": "raw"},
    {"command": "raw.filter", "args": {"raw": "{{raw}}", "l_freq": 1, "h_freq": 40}, "as": "clean"},
    {"command": "raw.events", "args": {"raw": "{{clean}}"}, "as": "events"},
    {"command": "epochs.build", "args": {"raw": "{{clean}}", "events": "{{events}}", "tmin": -0.2, "tmax": 0.8}, "as": "epochs"},
    {"command": "epochs.average", "args": {"epochs": "{{epochs}}", "by": "event"}}
  ]
}
```

```bash
mne-cli run --spec plan.json --dry-run   # validate first
mne-cli run --spec plan.json             # same process, shared store
```

`--stop-on-error` (default on) stops at the first failing step; the trace lists
each step's handle, artifacts and warnings. A step that returns numbers but no
object (e.g. `psd`) cannot be bound with `as` — omit `as` for those.

## 9. MCP integration

- `mne-cli mcp serve` — newline-delimited JSON-RPC 2.0 over stdio, protocol
  versions 2025-06-18 / 2025-03-26 / 2024-11-05. stdout carries only protocol
  frames; diagnostics go to stderr.
- `mne-cli mcp manifest` — the same tool list as JSON, for HTTP runtimes.
- Tool names are `mne_cli_<group>_<action>`, e.g. `mne_cli_raw_load`,
  `mne_cli_epochs_build`, `mne_cli_dataset_demo`. Each schema is generated from
  the same registry as the CLI, so a tool and its command never drift apart.

Claude Desktop style config:

```json
{"mcpServers": {"mne": {"command": "mne-cli", "args": ["mcp", "serve", "--store", "D:/work/mne-store"]}}}
```

## 10. Failure playbook

| symptom | what to do |
| --- | --- |
`UNKNOWN_HANDLE` | `store list`; you are probably in a different `--store` directory |
`WRONG_OBJECT_KIND` | the flag and the handle kind must match (`--epochs epochs_0001`) |
`RuntimeError: ... requires preload` | should not happen; report it — preload is automatic |
`filter` fails on a stim channel | pass `--skip-bad` or `--picks` to select data channels |
`ica apply` says nothing is flagged | run `ica find-artifacts` or `ica set-exclude` first |
`epochs decode` says sklearn missing | `pip install "mne-cli[decoding]"` |
`tfr` save complains about h5io | `pip install h5io` (declared dependency; a broken env causes this) |
`source *` fails at BEM/forward | needs FreeSurfer subjects dir and a `-trans` file; not a CLI bug |

## 11. What the CLI deliberately does not do

- No interactive browsers (`raw.plot` GUI) — static PNG only.
- No 3-D rendering (pyvista) — source estimates are returned numerically.
- No live streams (LSL) — files only.
- No silent unit conversion — volts in, volts out.
