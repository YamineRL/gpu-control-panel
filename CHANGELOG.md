# Changelog

Notable changes to GPU Control Panel. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-25

First tagged release. The dashboard is new; fan control, the temperature
graph, the process table and the keys that reach them work as they did on the
development line, with no state files to migrate.

### Added

- **Eight dashboard tiles**, two rows of four: speed, GPU temperature, VRAM,
  power, GPU load, PCIe, CPU and disk. Each tile has one shape: the current
  value, a context line, a two-minute sparkline three rows tall, and the
  session's min/max underneath. The tile scale comes from its own ceiling
  (VRAM against total, power against the cap, load against 100%) or from the
  window's maximum when the quantity has no ceiling.
- **Split speed readout.** Decode tokens per second is the tile's value and
  the prefill rate sits on the context line, marked `reading` while a prompt
  is being processed. The two rates are no longer averaged into one number.
- **The three GPU temperatures in one tile**, edge, junction and memory on
  the value line, each coloured by its distance to its own critical limit,
  with the three limits on the context line and a min/max that covers all
  three sensors.
- **PCIe link reporting.** Generation, width and the one-way bandwidth of the
  negotiated link (`32.0 GB/s` for Gen4 x16), with `idle link` when the
  kernel reports the link powered down or narrowed. `PcieLink` also carries
  the link's maximum, so a power-managed idle link can be told apart from a
  genuinely narrowed one.
- **Memory-controller load** (`mem_busy_percent`) sampled from hwmon, and the
  graph the PCIe tile draws, because `amdgpu` exposes no PCIe traffic
  counter.
- **Host sampling** from `/proc/stat` (CPU jiffies), `/proc/meminfo` (RAM and
  swap) and `/proc/diskstats` (sector counters, whole disks only, resolved
  through `/sys/block`), once a second. Paths are injectable so the parsers
  are tested against fixtures.
- **System meters.** Context fill, system RAM, swap and fan duty as one-line
  bars, with the fan's mode, duty, curve target and `HOT` state on the `MODE`
  line, and the three curve profile buttons under them.
- **Recent requests table.** Newest first, using every row the panel has
  (eleven at 172x48): time, state, prompt tokens, tokens reused from the
  cache, output tokens, tokens per second and duration. Strata serves its own
  list under `/metrics`; for llama.cpp, which keeps no history, the monitor
  records each request as it starts, polls and finishes.
- **Status strip** across the top: inference state, model name, context use,
  request count, server kind, and host CPU and RAM at a glance.
- **Multi-series sparklines** (`render_sparkline_series`) with an explicit y
  range, replacing the single hardcoded 0-100 scale.
- Tests for the `/proc` parsers, the PCIe arithmetic, the request history
  mapping, the table's row count and the session min/max reset.

### Changed

- The main screen is a dashboard rather than a tile strip above three sidebar
  panels. The temperature graph and the process table keep their places; the
  process table now sits under the main area and is still toggled with `v`.
- `i` hides the whole tile grid instead of just the inference strip, and the
  binding is labelled `Tiles`.
- Sparklines are three rows tall and no longer leave blank rows inside a
  tile.
- The temperature tile draws only the junction curve. Three filled ridges on
  the same axis read as noise; the value line still carries all three
  readings and the min/max still covers them.
- The sidebar is 52 columns wide, sized for the request table.
- README reorganised around the dashboard, with a tile table, the `/proc` and
  `sysfs` sources, the module map and regenerated screenshots.

### Removed

- `gpuctl/inference.py`, the four-tile inference strip it drew.
- `TempPanel`, `FanPanel` and `PowerPanel`, and their `ProgressBar` rules in
  `styles.tcss`.

[Unreleased]: https://github.com/YamineRL/gpu-control-panel/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/YamineRL/gpu-control-panel/releases/tag/v1.0.0
