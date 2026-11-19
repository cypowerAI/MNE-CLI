# MNE-CLI

An **LLM-friendly command line interface for [MNE-Python](https://github.com/mne-tools/mne-python)**.
Every EEG/MEG analysis step becomes a single, JSON-speaking command — so a language
model (or a shell script, or an MCP client) can run a full neuroscience pipeline
by conversation instead of by writing Python.

```bash
mne-cli dataset demo --out demo_raw.fif             # -> a 60 s synthetic EEG file
mne-cli raw load    --path demo_raw.fif             # -> raw_0001
mne-cli raw filter  --raw raw_0001 --l-freq 1 --h-freq 40
mne-cli raw events  --raw raw_0002                  # -> events_0001
mne-cli epochs.build --raw raw_0002 --events events_0001 --tmin -0.2 --tmax 0.8
mne-cli epochs.average --epochs epochs_0001 --by event
mne-cli evoked.plot --evoked evoked_0001
```

Both `mne-cli epochs build` and `mne-cli epochs.build` are accepted, so prompts
written either way keep working.

Each call returns one JSON envelope:

```json
{"ok":true,"command":"raw.filter","data":{"sfreq":1000.0,"n_channels":64,"duration_s":300.0,
 "handle":"raw_0002"},"handle":"raw_0002","elapsed_ms":812}
```

Failures are equally machine-readable — a stable `code`, a `message`, and a `hint`
that says what to do next:

```json
{"ok":false,"command":"raw.load","error":{"code":"DATA_ERROR",
 "message":"file not found: D:/data/x.fif",
 "hint":"Use an absolute path, or `mne-cli dataset download --name sample`."}}
```

## Why a CLI instead of an API

| Problem for an LLM | How mne-cli solves it |
| --- | --- |
| Long Python snippets with implicit state | One command per step, state lives in handles |
| `raw`, `epochs`, `evoked` objects cannot cross turns | Every object is serialised into a store and referenced as `raw_0001` |
| Guessing function signatures | `mne-cli describe <command>` returns flags, types, defaults, examples |
| Unparseable log noise | One JSON object on stdout, all MNE logs on stderr |
| Plotting needs a display | Matplotlib runs on `Agg`, figures are written as PNG |
| Silent failures | Errors carry `code` + `hint`; exit codes are stable |
| Many round trips | `mne-cli run --spec plan.json` executes a whole plan in one call |

## Install

```bash
pip install mne-cli                 # core: raw, epochs, evoked, ICA, TFR, reports
pip install "mne-cli[decoding]"     # + scikit-learn for epochs.decode
pip install "mne-cli[full]"         # + pandas, matplotlib extras
```

Python ≥ 3.10. MNE ≥ 1.6 is pulled in automatically.

Run from a checkout without installing:

```bash
python -m mne_cli commands
```

## Design contract

1. **One command, one result.** No command mutates hidden global state; anything
   you want to keep is written to the store and returned as a handle.
2. **JSON by default.** `--format json` is the default, `--text` is for humans.
   `--json`/`--text` work as shortcuts and may be placed before or after the group.
3. **Handles, not arrays.** `raw_0001`, `epochs_0003`, `evoked_0002`… are valid
   only inside the store that produced them (`./.mne-cli` by default, `--store`
   or `$MNE_CLI_STORE` to relocate). `mne-cli store list` shows everything.
4. **Large matrices are summarised.** Arrays above 64 elements come back as
   `{shape, min, max, mean, std}`; the full data goes to `.npy`/`.npz` on request.
5. **Stable vocabulary.** `command` names, parameter names and error codes never
   change within a major version, so prompts and agent policies keep working.
6. **Volts, not microvolts.** MNE stores SI units; thresholds like 100 µV are
   written `150e-6`. The CLI does not silently rescale.

## Discover everything

```bash
mne-cli commands                      # full catalogue, grouped
mne-cli commands --group raw          # one group
mne-cli describe raw.filter           # flags, types, defaults, examples
mne-cli describe epochs.build --json  # same, as JSON schema
mne-cli doctor                        # is this machine able to run analyses?
mne-cli version
```

`mne-cli` with no arguments is equivalent to `mne-cli commands`.

## Command reference

| Group | Commands |
| --- | --- |
| `raw` | `load` `info` `channels` `head` `crop` `save` `array` `events` `annotations` `annotate` `annotations-from-events` `plot` |
| `raw` (preprocess) | `filter` `notch` `resample` `reference` `pick` `rename` `set-montage` `set-channel-types` `mark-bad` `interpolate-bads` |
| `epochs` | `build` `info` `drop-bad` `equalize` `crop` `array` `average` `plot` `stats` `decode` |
| `evoked` | `info` `values` `crop` `save` `array` `plot` |
| `psd` `tfr` | PSD (welch/multitaper) and TFR (morlet/multitaper) for raw, epochs and evoked; `tfr.compute` `tfr.info` `tfr.plot` `tfr.array` |
| `ica` | `fit` `find-artifacts` `set-exclude` `apply` `plot-components` `plot-sources` |
| `source` | `bem` `forward` `covariance` `inverse` `apply` `info` |
| `dataset` | `list` `download` `demo` |
| `store` | `list` `show` `drop` `clear` |
| `report` | `build` |
| `mcp` | `serve` `manifest` |
| top-level | `commands` `describe` `run` `version` `doctor` |

`mne-cli commands` is the authoritative list; the table above is a summary.

## End-to-end recipes

### Event-related potentials

```bash
mne-cli raw load --path sub-01_task-auditory.fif                    # raw_0001
mne-cli raw filter --raw raw_0001 --l-freq 0.1 --h-freq 30         # raw_0002
mne-cli raw events --raw raw_0002 --stim-channel STI 014            # events_0001
mne-cli epochs.build --raw raw_0002 --events events_0001 \
        --tmin -0.2 --tmax 0.8 --baseline '[-0.2,0]' \
        --reject '{"eeg":150e-6}'                                   # epochs_0001
mne-cli epochs.average --epochs epochs_0001 --by event              # evoked_0001...
mne-cli evoked.plot --evoked evoked_0001 --kind butterfly
mne-cli epochs.stats --epochs epochs_0001 --a audio --b visual --mode cluster
```

### Artefact removal with ICA

```bash
mne-cli raw filter --raw raw_0001 --l-freq 1 --h-freq 30            # raw_0002
mne-cli ica.fit --raw raw_0002 --n-components 20                    # ica_0001
mne-cli ica.find-artifacts --ica ica_0001 --raw raw_0002            # ica_0002
mne-cli ica.plot-components --ica ica_0002
mne-cli ica.apply --ica ica_0002 --raw raw_0001                     # raw_0003 (clean)
```

### Single-trial decoding

```bash
mne-cli epochs.decode --epochs epochs_0001 --pipeline csp-lda --cv 5
mne-cli epochs.decode --epochs epochs_0001 --pipeline time-lda --scoring roc_auc
```

### Time-frequency

```bash
mne-cli tfr.compute --epochs epochs_0001 --fmin 4 --fmax 30 --n-freqs 20 \
        --method morlet --baseline '[-0.2,0]' --plot                  # tfr_0001 + PNG
```

### Source localisation (needs FreeSurfer)

```bash
mne-cli source.bem --subject sample --subjects-dir D:/freesurfer/subjects   # bem_0001
mne-cli source.forward --raw raw_0001 --bem bem_0001 \
        --trans sample-trans.fif --subject sample                           # forward_0001
mne-cli source.covariance --raw raw_0001                                    # cov_0001
mne-cli source.inverse --raw raw_0001 --forward forward_0001 --cov cov_0001 # inverse_0001
mne-cli source.apply --inverse inverse_0001 --evoked evoked_0001            # stc_0001
```

### One call for a whole plan

`plan.json`:

```json
{
  "steps": [
    {"command": "raw.load", "args": {"path": "sub-01_task.fif"}, "as": "r"},
    {"command": "raw.filter", "args": {"raw": "{{r}}", "l_freq": 1, "h_freq": 40}, "as": "f"},
    {"command": "raw.events", "args": {"raw": "{{f}}", "stim_channel": "STI 014"}, "as": "ev"},
    {"command": "epochs.build", "args": {"raw": "{{f}}", "events": "{{ev}}", "tmin": -0.2, "tmax": 0.8}, "as": "ep"},
    {"command": "epochs.average", "args": {"epochs": "{{ep}}", "by": "event"}}
  ]
}
```

```bash
mne-cli run --spec plan.json            # executes all five steps, returns the trace
mne-cli run --spec plan.json --dry-run  # validate first
```

### HTML report

```bash
mne-cli report build --out report.html --raw raw_0003 --epochs epochs_0001 --evoked evoked_0001 --ica ica_0002
```

### Demo data

No network needed — generate a small synthetic recording with an alpha rhythm,
blinks and two event types:

```bash
mne-cli dataset demo --out demo_raw.fif --duration 60 --n-events 40
# 18 channels (16 EEG + EOG + stim), 250 Hz, 40 events: audio / visual
```

Real public datasets:

```bash
mne-cli dataset list
mne-cli dataset download --name eegbci --subjects 1 --runs 3
```

## Using MNE-CLI from an LLM agent

Three integration paths, in increasing order of coupling:

1. **Shell tools.** Give the agent a bash/powershell tool; the JSON envelope is
   enough to act on. A ready-made agent skill lives in [`SKILL.md`](SKILL.md).
2. **MCP tools.** `mne-cli mcp serve` speaks JSON-RPC over stdio. Example
   Claude Desktop config:

   ```json
   {
     "mcpServers": {
       "mne": {
         "command": "mne-cli",
         "args": ["mcp", "serve", "--store", "D:/work/mne-store"]
       }
     }
   }
   ```

   Every command appears as a tool (`mne_cli_raw_load`, `mne_cli_epochs_build`,
   …) with a JSON schema. `mne-cli mcp manifest` prints the same list for
   HTTP-based runtimes, and `mne-cli mcp serve` keeps stdout clean for JSON-RPC
   (all MNE logs go to stderr).
3. **Python API.** `from mne_cli.registry import REGISTRY` and call
   `REGISTRY["raw.filter"].handler(ctx, params)` if you build your own orchestrator.

## Where files go

| Path | Contents |
| --- | --- |
| `./.mne-cli/` | Store: serialised objects + `index.json` with provenance |
| `./.mne-cli/artifacts/` | Auto-named PNG/NPZ outputs of plotting/export commands |
| `--out` | Explicit destination, used verbatim |

Nothing is written outside these paths unless you pass `--out`/`--path`.

## Comparison with plain MNE

```python
# MNE-Python
raw = mne.io.read_raw_fif("sub-01.fif", preload=True)
raw.filter(1, 40)
events = mne.find_events(raw, stim_channel="STI 014")
epochs = mne.Epochs(raw, events, event_id, -0.2, 0.8, baseline=(-0.2, 0),
                    preload=True, reject={"eeg": 150e-6})
evoked = epochs["audio"].average()
```

```bash
# mne-cli
mne-cli raw load --path sub-01.fif --preload
mne-cli raw filter --raw raw_0001 --l-freq 1 --h-freq 40
mne-cli raw events --raw raw_0002 --stim-channel "STI 014"
mne-cli epochs.build --raw raw_0002 --events events_0001 --tmin -0.2 --tmax 0.8 --reject '{"eeg":150e-6}'
mne-cli epochs.average --epochs epochs_0001 --conditions audio
```

Same numerics — mne-cli calls MNE under the hood. The difference is that every
intermediate object is addressable and every result is described.

## Limits

- Interactive browsing (`raw.plot()` with keyboard/mouse) is unavailable: all
  plotting is static PNG.
- 3-D source visualisation needs `pyvista` and a GPU-friendly display stack; the
  CLI returns numerical source estimates and leaves rendering to dedicated tools.
- Streams (LSL, BrainVision Live) are out of scope; mne-cli works on files.
- The store is a working directory, not a database: handles are not portable
  across stores unless you re-load the underlying file.

## 中文快速开始

```bash
pip install mne-cli
mne-cli dataset demo --out demo_raw.fif                         # 生成一份合成 EEG（无需联网）
mne-cli raw load --path demo_raw.fif                            # 返回 raw_0001
mne-cli raw info --raw raw_0001                                 # 看采样率/通道/时长
mne-cli raw filter --raw raw_0001 --l-freq 1 --h-freq 40        # 带通滤波
mne-cli raw plot --raw raw_0002                                 # 生成 PNG 波形图
```

- 所有命令默认输出 JSON，便于大模型解析；加 `--text` 看人类可读版本。
- 每条命令的返回值里若含 `handle`，后续命令就用这个句柄（如 `--raw raw_0001`）传递对象。
- `mne-cli commands` 查看全部能力，`mne-cli describe <命令>` 查看精确参数。
- 多步流程可写成 JSON，用 `mne-cli run --spec plan.json` 一次执行。
- MCP 客户端接入：`mne-cli mcp serve`。

## Acknowledgements

mne-cli is an independent wrapper around MNE-Python (BSD-3-Clause), which does
all of the numerical work. Please cite MNE-Python when you publish results:

> Gramfort A, et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 2013.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
