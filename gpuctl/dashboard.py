"""The dashboard: two rows of live tiles, the meters column, the request table.

Every tile carries the same four lines, top to bottom: a big value, a line of
context, a sparkline, and the session's min/max under it. The sparkline is
three rows tall so it has room to show a shape, and a tile's scale comes from
its own data (a 34 tok/s decode curve against a fixed 120 ceiling was a flat
line at the bottom of the box).

The meters are the bars the tile grid has no room for: context fill, system
RAM, swap and the fan. The request table lists the newest requests first and
shows as many as the panel is tall.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from .fancurve import PROFILE_ORDER
from .graphs import Series, Sparkline
from .hwmon import GpuLimits, PcieLink, Sample
from .host import HostSample
from .llama import InferenceSnapshot, RequestRecord, format_rate, format_tokens

GIB = 1024 ** 3
MIB = 1024 ** 2
# Sparklines show the last two minutes, the same window the tiles always had.
SPARK_SPAN = 120.0
BAR_WIDTH = 12

SPEED_LINE = "spring_green2"
SPEED_FILL = (0, 60, 40)
TEMP_LINE = "#ff9a3c"
TEMP_FILL = (56, 26, 8)
PREFILL_COLOR = "#5fafff"
POWER_LINE = "#f5c56b"
POWER_FILL = (52, 40, 8)
BUSY_LINE = "#5aa9ff"
BUSY_FILL = (0, 34, 60)
VRAM_LINE = "#b48cff"
VRAM_FILL = (40, 22, 66)
LINK_LINE = "#7fd6c8"
LINK_FILL = (0, 44, 40)
CPU_LINE = "#f28b82"
CPU_FILL = (56, 16, 14)
DISK_LINE = "#c0cad4"
DISK_FILL = (34, 40, 46)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _fmt_gb(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:,.{digits}f}"


def _fmt_rate(value: float | None, unit: str = "/s") -> str:
    """Bytes per second as KB/s, MB/s or GB/s."""
    if value is None:
        return "--"
    if value >= GIB:
        return f"{value / GIB:,.1f} GB{unit}"
    if value >= MIB:
        return f"{value / MIB:,.1f} MB{unit}"
    return f"{value / 1024:,.0f} KB{unit}"


def _temp_style(value: float | None, crit: float | None) -> str:
    if value is None:
        return "dim"
    if crit:
        ratio = value / crit
        if ratio >= 0.95:
            return "bold red"
        if ratio >= 0.80:
            return "bold yellow"
    return "bold green"


def _bar(pct: float | None, width: int = BAR_WIDTH, style: str = "") -> Text:
    """A block bar. No percentage, no fill: the caller says what it means."""
    text = Text()
    if pct is None:
        text.append("░" * width, style="grey30")
        return text
    filled = round(width * _clamp(pct) / 100.0)
    text.append("█" * filled, style=style or "green")
    if filled < width:
        text.append("░" * (width - filled), style="grey30")
    return text


def _series(
    points: Sequence[tuple[float, float | None]],
) -> list[tuple[float, float]]:
    return [(t, v) for t, v in points if v is not None]


def _pick(samples: Sequence[Sample], attr: str) -> list[tuple[float, float | None]]:
    return [(sample.t, getattr(sample, attr, None)) for sample in samples]


def _pick_host(
    samples: Sequence[HostSample], attr: str
) -> list[tuple[float, float | None]]:
    return [(sample.t, getattr(sample, attr, None)) for sample in samples]


def _window_range(values: Sequence[float], pad: float) -> tuple[float, float]:
    """Visible low/high plus a little air, or a sane band when there is none."""
    usable = [v for v in values if math.isfinite(v)]
    if not usable:
        return 0.0, 1.0
    low, high = min(usable), max(usable)
    if high - low < pad:
        mid = (high + low) / 2
        return mid - pad, mid + pad
    return low - pad, high + pad


class Tile(Vertical):
    """Value, context line, sparkline, session min/max."""

    DEFAULT_CSS = """
    Tile {
        width: 1fr;
        min-width: 0;
        height: 8;
        border: round $primary;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    Tile .tile-value {
        height: 1;
        width: 100%;
        text-overflow: ellipsis;
    }
    Tile .tile-sub {
        height: 1;
        width: 100%;
        color: $text-muted;
        text-overflow: ellipsis;
    }
    Tile Sparkline {
        height: 3;
        width: 100%;
    }
    Tile .tile-range {
        height: 1;
        width: 100%;
        color: $text-muted;
        text-overflow: ellipsis;
    }
    """

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = title
        self._low: float | None = None
        self._high: float | None = None

    def compose(self) -> ComposeResult:
        yield Static("--", classes="tile-value", markup=False)
        yield Static("", classes="tile-sub", markup=False)
        yield Sparkline()
        yield Static("", classes="tile-range", markup=False)

    def reset_extremes(self) -> None:
        self._low = None
        self._high = None

    @property
    def extremes(self) -> tuple[float | None, float | None]:
        return self._low, self._high

    def update_tile(
        self,
        *,
        value: Text | str,
        sub: Text | str = "",
        series: Sequence[Series] = (),
        y_range: tuple[float, float] | None = None,
        now: float,
        extremes: Sequence[float | None] = (),
        range_unit: str = "",
        range_digits: int = 1,
        span: float = SPARK_SPAN,
    ) -> None:
        if isinstance(value, str):
            value = Text(value, style="bold")
        self.query_one(".tile-value", Static).update(value)
        self.query_one(".tile-sub", Static).update(
            sub if isinstance(sub, Text) else Text(sub)
        )
        self.query_one(Sparkline).set_series(series, now, span, y_range)

        for item in extremes:
            if item is None or not math.isfinite(item):
                continue
            self._low = item if self._low is None else min(self._low, item)
            self._high = item if self._high is None else max(self._high, item)

        if self._low is None or self._high is None:
            range_text = "min --  max --"
        else:
            range_text = (
                f"min {self._low:,.{range_digits}f}  "
                f"max {self._high:,.{range_digits}f}{range_unit}"
            )
        self.query_one(".tile-range", Static).update(Text(range_text))


class TileGrid(Vertical):
    """Two rows of four tiles: inference on top, hardware below."""

    DEFAULT_CSS = """
    TileGrid {
        height: 16;
        width: 100%;
        padding: 0 1;
    }
    TileGrid Horizontal {
        height: 8;
        width: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot = InferenceSnapshot()
        self._decode: list[tuple[float, float]] = []
        # When the current session started. Session min/max ignore anything
        # older, so a new conversation does not inherit the last one's range.
        self._session_started = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tile("SPEED tok/s", id="tile-speed")
            yield Tile("GPU TEMP", id="tile-temp")
            yield Tile("VRAM", id="tile-vram")
            yield Tile("POWER", id="tile-power")
        with Horizontal():
            yield Tile("GPU LOAD", id="tile-busy")
            yield Tile("PCIe", id="tile-pcie")
            yield Tile("CPU", id="tile-cpu")
            yield Tile("DISK", id="tile-disk")

    def tile(self, name: str) -> Tile:
        return self.query_one(f"#tile-{name}", Tile)

    def start_session(self, now: float) -> None:
        """Start a fresh decode/prefill min/max, keeping the sparkline."""
        self._session_started = now
        self.tile("speed").reset_extremes()

    def push_inference(self, snapshot: InferenceSnapshot, now: float) -> None:
        """Keep the decode sparkline fed even when the screen is not redrawn."""
        if snapshot.gen_tps is not None:
            self._decode.append((now, snapshot.gen_tps))
        while self._decode and now - self._decode[0][0] > SPARK_SPAN * 1.5:
            self._decode.pop(0)
        self._snapshot = snapshot

    # ------------------------------------------------------------- drawing

    def update_tiles(
        self,
        *,
        samples: Sequence[Sample],
        hosts: Sequence[HostSample],
        limits: GpuLimits,
        link: PcieLink,
        now: float,
    ) -> None:
        window = [s for s in samples if now - s.t <= SPARK_SPAN]
        host_window = [s for s in hosts if now - s.t <= SPARK_SPAN]
        self._update_speed(now)
        self._update_temp(window, limits, now)
        self._update_vram(window, now)
        self._update_power(window, limits, now)
        self._update_busy(window, now)
        self._update_pcie(window, link, now)
        self._update_cpu(host_window, now)
        self._update_disk(host_window, now)

    def _update_speed(self, now: float) -> None:
        snapshot = self._snapshot
        live, last = snapshot.gen_tps, snapshot.last_gen_tps
        decode = live if live is not None else last
        if snapshot.state == "generate" and live is not None:
            style = "bold spring_green2"
        elif decode is not None:
            style = "spring_green2"
        else:
            style = "dim"

        value = Text()
        value.append(format_rate(decode), style=style)
        value.append(" tok/s", style="dim")

        prefill = snapshot.prefill_tps
        if prefill is None:
            prefill = snapshot.last_prefill_tps
        sub = Text()
        sub.append("prefill ", style="grey62")
        sub.append(format_rate(prefill), style="bold " + PREFILL_COLOR)
        sub.append(" tok/s", style="dim")
        if snapshot.state == "prefill":
            sub.append("  reading", style=PREFILL_COLOR)

        points = _series(self._decode)
        # The y scale follows the data: a fixed 120 ceiling drew a 34 tok/s
        # session as a line along the bottom of the tile.
        if points:
            top = max(y for _, y in points)
            y_range = (0.0, max(top * 1.15, 1.0))
        else:
            y_range = (0.0, 1.0)
        session = [y for at, y in points if at >= self._session_started]
        self.tile("speed").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, SPEED_LINE, SPEED_FILL)],
            y_range=y_range,
            now=now,
            extremes=session,
            range_unit=" tok/s",
        )

    def _temp_number(
        self, value: float | None, crit: float | None
    ) -> Text:
        chunk = Text()
        if value is None:
            chunk.append("--", style="dim")
        else:
            chunk.append(f"{value:.0f}", style=_temp_style(value, crit))
            chunk.append("°", style=_temp_style(value, crit))
        return chunk

    def _update_temp(
        self, window: Sequence[Sample], limits: GpuLimits, now: float
    ) -> None:
        latest = window[-1] if window else Sample(t=now)
        value = Text()
        for index, (label, key, reading) in enumerate(
            (
                ("edge", "edge", latest.edge_c),
                ("jct", "junction", latest.junction_c),
                ("mem", "mem", latest.mem_c),
            )
        ):
            if index:
                value.append("  ")
            value.append(f"{label} ", style="grey62")
            value += self._temp_number(reading, limits.temp_crit.get(key))
        if latest.edge_c is None and latest.junction_c is None and latest.mem_c is None:
            value = Text("--", style="dim")

        crit = limits.temp_crit
        sub = Text()
        sub.append(
            "crit "
            + " / ".join(
                f"{crit[key]:.0f}" for key in ("edge", "junction", "mem") if key in crit
            )
            + " °C",
            style="grey62",
        )
        if not crit:
            sub = Text("no critical limit reported", style="dim")

        edges = _series(_pick(window, "edge_c"))
        junctions = _series(_pick(window, "junction_c"))
        mems = _series(_pick(window, "mem_c"))
        values = [y for _, y in edges + junctions + mems]
        # One curve, not three: at three rows tall, three filled ridges read
        # as noise. The junction is the sensor the fan curve follows, so the
        # shape is that one, while the min/max under it covers all three.
        shape = junctions or edges or mems
        y_range = _window_range(values, pad=2.0) if values else (0.0, 100.0)
        self.tile("temp").update_tile(
            value=value,
            sub=sub,
            series=[Series(shape, TEMP_LINE, TEMP_FILL)],
            y_range=y_range,
            now=now,
            extremes=values,
            range_unit=" °C",
            range_digits=0,
        )

    def _update_vram(self, window: Sequence[Sample], now: float) -> None:
        latest = window[-1] if window else Sample(t=now)
        used = latest.vram_used / GIB if latest.vram_used is not None else None
        total = latest.vram_total / GIB if latest.vram_total is not None else None
        value = Text()
        value.append(_fmt_gb(used), style="bold " + VRAM_LINE)
        value.append(" / ", style="dim")
        value.append(_fmt_gb(total), style="")
        value.append(" GB", style="dim")

        sub = Text()
        if latest.gtt_used is not None:
            sub.append(f"GTT {_fmt_gb(latest.gtt_used / GIB)} GB", style="grey62")
        if latest.mclk_mhz:
            sub.append("  ·  " if sub.plain else "")
            sub.append(f"mclk {latest.mclk_mhz / 1000:.2f} GHz", style="grey62")

        points = _series(
            [(s.t, s.vram_used / GIB if s.vram_used is not None else None) for s in window]
        )
        values = [y for _, y in points]
        # A card has a hard ceiling, so the tile is drawn against the whole
        # VRAM, not against the last two minutes of drift.
        ceiling = latest.vram_total / GIB if latest.vram_total else 1.0
        self.tile("vram").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, VRAM_LINE, VRAM_FILL)],
            y_range=(0.0, ceiling),
            now=now,
            extremes=values,
            range_unit=" GB",
        )

    def _update_power(
        self, window: Sequence[Sample], limits: GpuLimits, now: float
    ) -> None:
        latest = window[-1] if window else Sample(t=now)
        value = Text()
        value.append(
            f"{latest.power_w:.0f}" if latest.power_w is not None else "--",
            style="bold " + POWER_LINE,
        )
        value.append(" W", style="dim")

        sub = Text()
        cap = limits.power_cap
        if cap:
            sub.append(f"of {cap:.0f} W limit", style="grey62")
        if latest.sclk_mhz:
            if sub.plain:
                sub.append("  ·  ")
            sub.append(f"sclk {latest.sclk_mhz / 1000:.2f} GHz", style="grey62")
        if not sub.plain:
            sub = Text("no power limit reported", style="dim")

        points = _series(_pick(window, "power_w"))
        values = [y for _, y in points]
        if cap:
            y_range = (0.0, cap)
        else:
            y_range = (0.0, max(values) * 1.15) if values else (0.0, 1.0)
        self.tile("power").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, POWER_LINE, POWER_FILL)],
            y_range=y_range,
            now=now,
            extremes=values,
            range_unit=" W",
            range_digits=0,
        )

    def _update_busy(self, window: Sequence[Sample], now: float) -> None:
        latest = window[-1] if window else Sample(t=now)
        busy = latest.gpu_busy_pct
        value = Text()
        value.append(f"{busy}" if busy is not None else "--", style="bold " + BUSY_LINE)
        value.append(" %", style="dim")

        sub = Text()
        if latest.mem_busy_pct is not None:
            sub.append(f"mem {latest.mem_busy_pct}%", style="grey62")
        if latest.sclk_mhz:
            if sub.plain:
                sub.append("  ·  ")
            sub.append(f"sclk {latest.sclk_mhz / 1000:.2f} GHz", style="grey62")

        points = _series(
            [(s.t, float(s.gpu_busy_pct) if s.gpu_busy_pct is not None else None) for s in window]
        )
        values = [y for _, y in points]
        self.tile("busy").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, BUSY_LINE, BUSY_FILL)],
            y_range=(0.0, 100.0),
            now=now,
            extremes=values,
            range_unit=" %",
            range_digits=0,
        )

    def _update_pcie(
        self, window: Sequence[Sample], link: PcieLink, now: float
    ) -> None:
        generation = link.generation
        value = Text()
        if generation is not None and link.width:
            value.append(f"Gen{generation}", style="bold " + LINK_LINE)
            value.append(f" x{link.width}", style="")
        else:
            value.append("link --", style="dim")

        bandwidth = link.bandwidth_gbs
        sub = Text()
        if bandwidth is not None:
            sub.append(f"{bandwidth:,.1f} GB/s one way", style="grey62")
        if link.downgraded:
            sub.append("  ·  idle link", style="grey62")
        if not sub.plain:
            sub = Text("the kernel reports no PCIe link", style="dim")

        # No driver counter exposes traffic over the link, so the graph is the
        # memory controller: when it is busy, data is moving. Never a dead
        # graph with a "--" beside it.
        points = _series(
            [(s.t, float(s.mem_busy_pct) if s.mem_busy_pct is not None else None) for s in window]
        )
        values = [y for _, y in points]
        self.tile("pcie").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, LINK_LINE, LINK_FILL)],
            y_range=(0.0, 100.0),
            now=now,
            extremes=values,
            range_unit=" % mem",
            range_digits=0,
        )

    def _update_cpu(self, window: Sequence[HostSample], now: float) -> None:
        latest = window[-1] if window else HostSample(t=now)
        value = Text()
        value.append(
            f"{latest.cpu_pct:.0f}" if latest.cpu_pct is not None else "--",
            style="bold " + CPU_LINE,
        )
        value.append(" %", style="dim")

        sub = Text()
        if latest.cores:
            sub.append(
                f"{latest.cores} cores · {latest.threads} threads", style="grey62"
            )
        if latest.load1 is not None:
            if sub.plain:
                sub.append("  ·  ")
            sub.append(f"load {latest.load1:.2f}", style="grey62")

        points = _series(_pick_host(window, "cpu_pct"))
        values = [y for _, y in points]
        self.tile("cpu").update_tile(
            value=value,
            sub=sub,
            series=[Series(points, CPU_LINE, CPU_FILL)],
            y_range=(0.0, 100.0),
            now=now,
            extremes=values,
            range_unit=" %",
            range_digits=0,
        )

    def _update_disk(self, window: Sequence[HostSample], now: float) -> None:
        latest = window[-1] if window else HostSample(t=now)
        value = Text()
        value.append(_fmt_rate(latest.disk_read_bps), style="bold " + DISK_LINE)
        sub = Text()
        sub.append("write ", style="grey62")
        sub.append(_fmt_rate(latest.disk_write_bps), style="grey62")

        points = _series(_pick_host(window, "disk_read_bps"))
        values = [v / MIB for _, v in points]
        scaled = [(t, v / MIB) for t, v in points]
        self.tile("disk").update_tile(
            value=value,
            sub=sub,
            series=[Series(scaled, DISK_LINE, DISK_FILL)],
            y_range=(0.0, max(values) * 1.15) if values else (0.0, 1.0),
            now=now,
            extremes=values,
            range_unit=" MB/s",
        )


class Meters(Vertical):
    """The bars plus the fan control: context, RAM, swap, fan, profiles."""

    DEFAULT_CSS = """
    Meters {
        height: 8;
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    Meters > Static {
        height: 1;
        width: 100%;
    }
    Meters #fan-profiles {
        height: 1;
        width: 100%;
        margin-top: 0;
    }
    Meters #fan-profiles Button {
        width: 1fr;
        min-width: 0;
        height: 1;
        margin: 0 1 0 0;
        border: none;
        padding: 0 1;
        background: $panel;
    }
    Meters #fan-profiles Button.profile-selected {
        background: $success;
        text-style: bold;
    }
    """

    MODE_LABELS = {0: "FULL SPEED", 1: "MANUAL", 2: "AUTO (firmware)"}
    MODE_STYLES = {0: "bold red", 1: "bold yellow", 2: "cyan"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "System"

    def compose(self) -> ComposeResult:
        yield Static(id="meter-context", markup=False)
        yield Static(id="meter-ram", markup=False)
        yield Static(id="meter-swap", markup=False)
        yield Static(id="meter-fan", markup=False)
        yield Static(id="fan-mode", markup=False)
        with Horizontal(id="fan-profiles"):
            for name in PROFILE_ORDER:
                button = Button(name, id=f"profile-{name}", compact=True)
                # Keyboard access is via the `p` binding (same reason as the
                # timeframe buttons: no focus ring that looks like a choice).
                button.can_focus = False
                yield button

    def set_profile(self, active: str) -> None:
        for name in PROFILE_ORDER:
            self.query_one(f"#profile-{name}", Button).set_class(
                name == active, "profile-selected"
            )

    def _row(
        self,
        label: str,
        pct: float | None,
        detail: str,
        *,
        style: str = "green",
    ) -> Text:
        row = Text()
        row.append(f"{label:<8}", style="grey62")
        row.append(" ")
        row += _bar(pct, style=style)
        row.append(" ")
        row.append(f"{pct:3.0f}%" if pct is not None else " --", style="bold")
        row.append("  ")
        row.append(detail, style="grey62")
        return row

    def update_meters(
        self,
        sample: Sample,
        host: HostSample,
        snapshot: InferenceSnapshot,
        pct: float = 0.0,
        *,
        curve_target: float | None = None,
        curve_emergency: bool = False,
    ) -> None:
        context_pct = (
            snapshot.ctx_used / snapshot.ctx_total * 100
            if snapshot.ctx_total
            else None
        )
        context_detail = (
            f"{format_tokens(snapshot.ctx_used)}/{format_tokens(snapshot.ctx_total)}"
            if snapshot.ctx_total
            else "no server"
        )
        self.query_one("#meter-context", Static).update(
            self._row(
                "CONTEXT",
                context_pct,
                context_detail,
                style="green" if (context_pct or 0) < 90 else "red",
            )
        )

        ram_pct = host.ram_pct
        ram_detail = (
            f"{_fmt_gb((host.ram_used or 0) / GIB)}/{_fmt_gb((host.ram_total or 0) / GIB)} GB"
            if host.ram_total
            else "n/a"
        )
        self.query_one("#meter-ram", Static).update(
            self._row(
                "SYSMEM",
                ram_pct,
                ram_detail,
                style="green" if (ram_pct or 0) < 80 else ("yellow" if (ram_pct or 0) < 92 else "red"),
            )
        )

        swap_pct = host.swap_pct
        if not host.swap_total:
            swap_row = self._row("SWAP", None, "no swap")
        else:
            swap_row = self._row(
                "SWAP",
                swap_pct,
                f"{_fmt_gb((host.swap_used or 0) / GIB)}/{_fmt_gb(host.swap_total / GIB)} GB",
                style="green" if (swap_pct or 0) < 25 else ("yellow" if (swap_pct or 0) < 60 else "red"),
            )
        self.query_one("#meter-swap", Static).update(swap_row)

        rpm = f"{sample.fan_rpm} RPM" if sample.fan_rpm is not None else "no tach"
        self.query_one("#meter-fan", Static).update(
            self._row("FAN", pct, rpm, style="cyan")
        )
        self.query_one("#fan-mode", Static).update(
            self._mode_line(sample, pct, curve_target, curve_emergency)
        )

    def _mode_line(
        self,
        sample: Sample,
        pct: float,
        curve_target: float | None,
        curve_emergency: bool,
    ) -> Text:
        mode = sample.pwm_enable
        if curve_target is not None and mode == 1:
            label, style = "CURVE", "bold green"
        else:
            label = self.MODE_LABELS.get(mode, f"unknown ({mode})")
            style = self.MODE_STYLES.get(mode, "white")
        line = Text()
        line.append("MODE ", style="grey62")
        line.append(label, style=style)
        line.append(f"  {pct:.0f}%", style="bold")
        if curve_target is not None:
            line.append(f" to {curve_target:.0f}%", style="grey62")
            if curve_emergency:
                line.append(" HOT", style="bold red")
        line.append("  p = profile", style="grey62")
        return line


class StatusStrip(Static):
    """One line over the tiles: what the model is doing, and from where."""

    DEFAULT_CSS = """
    StatusStrip {
        height: 1;
        width: 100%;
        padding: 0 2;
    }
    """

    STATE_STYLES = {
        "generate": ("spring_green2", "generating"),
        "prefill": ("#5fafff", "reading prompt"),
        "idle": ("grey62", "idle"),
        "offline": ("indian_red", "offline"),
    }

    def update_status(
        self, snapshot: InferenceSnapshot, host: HostSample | None = None
    ) -> None:
        color, label = self.STATE_STYLES.get(snapshot.state, ("white", snapshot.state))
        line = Text()
        line.append("● ", style=color)
        line.append(label, style=color)
        if snapshot.model:
            line.append("   ", style="grey62")
            line.append(_short_model(snapshot.model), style="bold")
        if snapshot.ctx_total:
            line.append("   ", style="grey62")
            line.append(
                f"{format_tokens(snapshot.ctx_used)}/{format_tokens(snapshot.ctx_total)} ctx",
                style="grey62",
            )
        if snapshot.requests:
            line.append("   ", style="grey62")
            line.append(f"{snapshot.requests} req", style="grey62")
        if snapshot.origin:
            line.append("   ", style="grey62")
            line.append(snapshot.origin, style="grey62")
        if not snapshot.ok and snapshot.error:
            line.append("   ", style="grey62")
            line.append(snapshot.error, style="indian_red")
        if host is not None and host.ram_total:
            line.append("   ", style="grey62")
            cpu = f"{host.cpu_pct:.0f}% · " if host.cpu_pct is not None else ""
            line.append(
                f"host {cpu}"
                f"{_fmt_gb((host.ram_used or 0) / GIB)}/{_fmt_gb(host.ram_total / GIB)} GB",
                style="grey62",
            )
        self.update(line)


def _short_model(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    return base if len(base) <= 40 else base[:39] + "…"


class RequestTable(Vertical):
    """The newest requests, as many rows as the panel has lines for."""

    DEFAULT_CSS = """
    RequestTable {
        height: 1fr;
        min-height: 5;
        border: round $primary;
        padding: 0 1;
    }
    RequestTable Static {
        height: 1fr;
        width: 100%;
    }
    """

    COLUMNS = (
        ("TIME", 8, "left"),
        ("STATE", 6, "left"),
        ("PROMPT", 6, "right"),
        ("REUSE", 5, "right"),
        ("OUT", 5, "right"),
        ("TOK/S", 5, "right"),
        ("DUR", 6, "right"),
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "Recent requests"
        self._records: tuple[RequestRecord, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static("waiting for a server", markup=False)

    def on_resize(self) -> None:
        self.refresh_table()

    def update_records(self, records: Sequence[RequestRecord]) -> None:
        self._records = tuple(records)
        self.refresh_table()

    def refresh_table(self) -> None:
        """Fit the table to the panel; called again when the panel resizes."""
        if not self._records:
            self.query_one(Static).update(
                Text("no requests yet", style="grey62")
            )
            return
        table = Text()
        header = Text()
        for index, (name, width, justify) in enumerate(self.COLUMNS):
            if index:
                header.append(" ")
            header.append(f"{name:>{width}}" if justify == "right" else f"{name:<{width}}",
                          style="bold grey62")
        table.append_text(header)

        rows = max(1, self.size.height - 3)  # border, header, one spare
        for record in self._records[:rows]:
            table.append("\n")
            table.append_text(self._row(record))
        self.query_one(Static).update(table)

    # What the table calls a request's state. Long words would push the
    # columns apart in a 52-column panel.
    STATE_LABELS = {
        "running": ("run", "spring_green2"),
        "done": ("done", "grey74"),
        "stop": ("stop", "grey74"),
        "length": ("len", "yellow"),
        "cancelled": ("canc", "indian_red"),
        "error": ("err", "indian_red"),
    }

    def _row(self, record: RequestRecord) -> Text:
        stamp = (
            time.strftime("%H:%M:%S", time.localtime(record.at))
            if record.at > 0
            else "   --   "
        )
        label, style = self.STATE_LABELS.get(
            record.status, (record.status[:6], "grey74")
        )
        row = Text()
        row.append(f"{stamp:<8}", style="grey62")
        row.append(" ")
        row.append(f"{label:<6}", style=style)
        row.append(" ")
        row.append(f"{format_tokens(record.prompt_tokens):>6}", style="")
        row.append(" ")
        row.append(
            f"{format_tokens(record.reused_tokens):>5}" if record.reused_tokens else f"{'--':>5}",
            style="grey62",
        )
        row.append(" ")
        row.append(f"{format_tokens(record.output_tokens):>5}")
        row.append(" ")
        row.append(f"{format_rate(record.tok_s):>5}", style="grey62")
        row.append(" ")
        duration = (
            f"{record.duration_s:,.1f}s" if record.duration_s is not None else "--"
        )
        row.append(f"{duration:>6}", style="grey62")
        return row
