<div align="center">

<img src="assets/tiles.png" width="920" alt="GPU Control Panel dashboard tiles: speed, the three GPU temperatures, VRAM, power, GPU load, PCIe, CPU and disk">

# GPU Control Panel

**AMD GPU fan control in the terminal, with live llama.cpp / Strata inference telemetry.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey)
![Textual 8.2](https://img.shields.io/badge/Textual-8.2-5a56e0)

</div>

GPU Control Panel talks to the `amdgpu` hwmon interface in `/sys/class/drm/`
directly: no daemon, no root service, no plotting library. One Textual app
draws a dashboard of eight live tiles, a btop-style temperature graph, a strip
of system meters and the recent inference requests, an nvtop-style process
table, and it drives the fan itself when you want a real curve.

<img src="assets/overview.png" width="920" alt="GPU Control Panel main screen: eight live tiles, temperature graph, system meters and recent requests">

## What it does

- **Dashboard tiles.** Two rows of four tiles, each with a big value, a context
  line, a three-row sparkline and the session's min/max: speed (decode and
  prefill), the three GPU temperatures, VRAM, power, GPU load, PCIe link,
  CPU and disk. See [The dashboard](#the-dashboard).
- **Live temperature graph.** Edge / junction / memory as layered ridges: a
  bright edge per sensor fading into its own gradient body, on an axis that
  hugs the data. Each curve ends in its live value, the window's min and max
  sit in the title, round time ticks run back from "now", and the critical
  limit appears as a dashed line once a sensor gets within 10 degrees of it.
  Switchable timeframes (1m / 5m / 15m / 1h); the last one is remembered
  between runs.
- **Probe rate.** The `-` / `+` buttons next to the timeframes set how often
  hwmon is read, from 500 ms to 10 s in 500 ms steps (2 s by default). The
  value is remembered between runs.
- **System meters.** Context fill, system RAM, swap and fan duty as one-line
  bars, with the fan mode and the curve profile buttons underneath and the
  recent requests beside them.
- **Fan control.** Firmware auto, manual, or the app's own fan curve.
  Manual speed via `+` / `-` (5% steps), `0`-`9` (0%-90%), or `s` for an exact
  value. Any manual key stops the curve.
- **GPU processes.** `v` opens a table of the processes with a DRM fd on the
  card: user, engine type, GPU%, VRAM, CPU% and command line.

The UI never blocks on a read or a redraw: hwmon, the inference endpoint, the
`/proc` reads and the process scan all run on worker threads or on short
timers, and the charts are block glyphs rendered in a few milliseconds.

## Screenshots

<table>
<tr>
<td width="50%"><img src="assets/overview.png" alt="Main screen"></td>
<td width="50%"><img src="assets/processes.png" alt="GPU process table"></td>
</tr>
<tr>
<td><em>Tiles, temperature graph, meters and recent requests.</em></td>
<td><em><code>v</code> opens the process table.</em></td>
</tr>
</table>

## Requirements

- Linux with the `amdgpu` driver and a fan exposed through `pwm1`
  (`/sys/class/drm/card*/device/hwmon/hwmon*/pwm1`). Monitoring works without
  it; fan control does not.
- Python 3.10 or newer.
- `sudo` for fan writes. Read-only monitoring needs no privileges.

## Run it

```sh
git clone https://github.com/yaminerl/gpu-control-panel
cd gpu-control-panel
./run.sh
```

`run.sh` creates `.venv/` on first run and installs `textual` into it. To do it
by hand:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Pass `--debug` to show the discovered hwmon path as a startup toast.

## Keys

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit |
| `c` | Fan curve on / off |
| `p` | Next fan curve profile |
| `r` | Reload the fan curve config |
| `a` | Fan mode: firmware auto |
| `m` | Fan mode: manual |
| `+` / `-` | Fan speed +5% / -5% |
| `0`-`9` | Jump fan speed to 0%-90% |
| `s` | Type an exact fan speed |
| `[` / `]` | Previous / next graph timeframe |
| `v` | Show / hide the GPU process list |
| `i` | Show / hide the dashboard tiles |

## The dashboard

Two rows of four tiles sit above the graph. Every tile has the same shape: a
big value, one line of context, a three-row sparkline (the last two minutes)
and the session's min/max underneath. The sparkline scale comes from the
tile's own data, so a 34 tok/s decode curve fills the box instead of running
flat along the bottom of a fixed 120 tok/s ceiling.

| Tile | Value | Context line |
|------|-------|--------------|
| `SPEED tok/s` | decode rate now (the last request's rate when idle) | prefill rate, and `reading` while a prompt is processed |
| `GPU TEMP` | edge, junction and memory in one line, coloured by distance to each critical limit | the three critical limits |
| `VRAM` | used / total GB | GTT use, memory clock |
| `POWER` | watts now | the cap, core clock |
| `GPU LOAD` | busy % | memory-controller %, core clock |
| `PCIe` | link generation and width | one-way bandwidth of the negotiated link, `idle link` when PCIe has dropped it |
| `CPU` | host CPU % | physical cores, threads, load average |
| `DISK` | host read rate | write rate |

The `PCIe` tile is the one the driver does not fully answer: amdgpu exposes
the link's generation and width but no traffic counter, so the value line is
the link and the graph is the memory controller. When the memory controller is
busy, data is moving. The tile never shows a `--` next to a dead graph.

Under the tiles, `i` hides the whole grid for a fan-only view.

### Meters and requests

The right column carries four one-line bars and the fan control:

- `CONTEXT`: the inference server's context use.
- `SYSMEM`: used / total system RAM, from `/proc/meminfo`.
- `SWAP`: used / total swap, marked `no swap` when the machine has none.
- `FAN`: duty cycle and RPM.

`MODE` shows the fan's mode (`AUTO (firmware)`, `MANUAL`, `CURVE`, `FULL
SPEED`), the current duty, the curve's target and a `HOT` marker when the
curve is in its emergency state. The three profile buttons under it are the
same as `p`.

`Recent requests` lists the newest first and uses every row the panel has:
time, state (`run` / `done` / `stop` / `len` / `canc`), prompt tokens, tokens
reused from the cache, output tokens, tokens per second and duration. Strata
serves its own request list under `/metrics`; for llama.cpp, which keeps no
history, the monitor records each request as it finishes and when the slot
goes idle.

Numbers come from llama.cpp's `/slots` endpoint or Strata's `/metrics` JSON,
polled twice a second on a worker thread. llama.cpp updates its counters once
per batch, so a rate is the counter delta divided by the time since that
counter last moved; dividing by the poll interval would report 2,048 tok/s for
a 1,024-token prefill batch that arrived after 2.5 s. Strata reports its own
rates (`live.tok_s`, `live.prefill_tok_s_mean`), so those are used as-is.

Nothing needs configuring. The server is found in this order:

1. `GPU_CTL_LLAMA_URL` (with `GPU_CTL_LLAMA_KEY` or `GPU_CTL_LLAMA_KEY_FILE`)
2. a `llama-server` process owned by you: `--host` / `--port`,
   `--api-key-file` and `-m` are read from `/proc/<pid>/cmdline`, which also
   works when llama-swap or a script started it; or a Strata server
   (`serve/server.py --engine strata`), whose port and key come from its flags,
   its `--config` JSON, or the CLI's `~/.config/strata/api-key`
3. llama-swap on `127.0.0.1:8091` (`/running` lists the child's URL and
   command; `sh -c` wrappers are unwrapped)
4. the usual ports: llama.cpp's 8080 / 8096, Strata's 8092 / 8095

Without a server the hardware tiles and the meters still fill in from hwmon
and `/proc`; the speed tile, the context bar and the request table say `--`
or `no server`, and discovery keeps running.

## Fan curve

RDNA2 cards have no editable fan curve in sysfs (`gpu_od/fan_ctrl/` is RDNA3+
only), and `pwm1_enable=2` is just the SMU firmware's fixed behaviour. So `c`
runs a curve inside the app: the GPU goes to manual mode and the app writes
`pwm1` once a second from the temperature readings.

- Three profiles, switched with `p` or the buttons under the meters: `quiet`
  (zero-RPM until about 55C junction, slow spin-down), `balanced`, and `cool`
  (fan always on, steep ramp, reacts early to a fast rise).
- Each profile has a curve per sensor (junction and memory by default). The fan
  follows whichever sensor asks for the most.
- Readings are smoothed, so a short spike does not make the fan jump; a rising
  temperature makes it ramp early (look-ahead, capped at 8C).
- Hysteresis plus a spin-down delay: the fan rises at once but only slows after
  the temperature has really dropped, then ramps down gently. That also stops
  zero-RPM on/off cycling.
- Fail safe: 100% at once if the junction reading is missing or any sensor
  comes within `emergency_margin_c` (default 10C) of its critical temperature.
- One root helper process does the writes (one password prompt per session, no
  PAM login per write). It accepts only numbers 0-255. When the app exits,
  crashes or is killed, the helper's pipe closes and it puts the fan back on
  firmware auto. If something else changes `pwm1_enable` while the curve runs,
  the curve stops.

## Configuration

Three small state files live next to the checkout and are gitignored:

| File | Contents |
|------|----------|
| `.gpuctl_state` | last graph timeframe, in seconds |
| `.gpuctl_probe_interval` | hwmon poll interval in ms (500-10000; set with the `-` / `+` buttons) |
| `.gpuctl_fancurve.json` | fan curve profiles (written with the defaults on first run) |

To start from a clean, fully commented example instead of the defaults:

```sh
cp examples/fancurve.example.json .gpuctl_fancurve.json
```

Edit it and press `r` to reload. A bad file is reported and ignored, and the
built-in defaults are used. Fields per profile:

```json
{
  "junction": [[55, 0], [60, 22], [70, 30], [80, 45], [100, 100]],
  "mem": [[75, 0], [80, 25], [94, 100]]
}
```

A profile's `curves` map is `sensor -> [[temp_c, pct], ...]` (ascending
temperature). The same profile holds `min_spin_pct`, `hysteresis_c`,
`spin_down_delay_s`, `ramp_down_pct_per_s`, `ramp_up_pct_per_s`, `smoothing_s`
and `lookahead_s`. The top-level keys are `active`, `autostart` and
`emergency_margin_c`.

The curve starts when the app opens, so you get the password prompt at
startup. Cancel it to stay on firmware auto, or set `"autostart": false`. The
curve runs only while the app is open.

## Sysfs layout this relies on

Base: `/sys/class/drm/card*/device/hwmon/hwmon*/` where `name` is `amdgpu`.
It is discovered at runtime, not hardcoded to a card or hwmon index, because
those change across reboots and driver reloads.

- `pwm1`, `pwm1_enable` (0 = no control / full speed, 1 = manual,
  2 = firmware auto), `pwm1_min`, `pwm1_max`, `fan1_input`, `fan1_min`,
  `fan1_max`
- `temp{1,2,3}_input` / `_label` / `_crit` / `_crit_hyst` / `_emergency`
  (edge / junction / mem)
- `power1_average`, `power1_cap*` (display only, never written)

From the sibling device directory (`card*/device/`, not inside `hwmon/`):
`gpu_busy_percent`, `mem_busy_percent`, `mem_info_vram_{used,total}`,
`mem_info_gtt_{used,total}`, `pp_dpm_sclk`, `pp_dpm_mclk`, and the PCIe link
pair `current_link_speed` / `current_link_width` (plus the `max_*` pair, to
tell a power-managed idle link from a narrowed one). The process table reads
`fdinfo` of the processes holding a DRM file descriptor on the card.

The host tiles and the memory bars come from `/proc/stat` (CPU jiffies),
`/proc/meminfo` (RAM and swap) and `/proc/diskstats` (sector counters, whole
disks only), sampled once a second.

## Development

```sh
PYTHONPATH=. .venv/bin/python tests/test_app.py
pytest tests/ -q                    # same checks, if pytest is installed
```

The chart, sparkline and inference-rate checks run anywhere. The app checks
need an `amdgpu` hwmon node and skip themselves without one; they restore the
state files they touch.

The README screenshots are generated from synthetic data, so they never
contain anything from the machine that renders them:

```sh
.venv/bin/python tools/render_screenshots.py        # needs rsvg-convert
```

```
main.py                 entry point
run.sh                  venv setup + launch wrapper
gpuctl/
  hwmon.py              hwmon discovery + sysfs reads (no writes)
  host.py               CPU / RAM / swap / disk from /proc
  fancurve.py           curve profiles, config file, curve controller
  gpu_processes.py      DRM fdinfo scan for the GPU process table
  privileged.py         sudo-authenticated writes + the root fan helper
  graphs.py             block-glyph gradient charts + sparklines (no plotting library)
  dashboard.py          the tile grid, the meters, the status strip, the request table
  llama.py              inference-server discovery, live token rates, request history
  widgets.py            the GPU process table
  screens.py            password modal, set-speed modal, "no GPU found" screen
  app.py                the Textual App: layout, workers, keybindings
  styles.tcss           Textual CSS
tests/test_app.py       headless smoke checks
tools/
  render_screenshots.py README image generator (fake hwmon + fake snapshots)
examples/
  fancurve.example.json fan curve config to start from
```

## Scope

Fan control and monitoring only: no overclocking, power-limit editing or
power-profile switching. Built and used on RDNA2 (Navi 21), where the kernel
offers no fan curve of its own.

## License

[MIT](LICENSE).
