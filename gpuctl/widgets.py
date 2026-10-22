"""Small presentational widgets: the bordered live-stat side panels."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, ProgressBar, Static

from .fancurve import PROFILE_ORDER
from .gpu_processes import GpuProcess
from .hwmon import GpuLimits, Sample


def _temp_style(value: float | None, crit: float | None) -> str:
    """Colour a temperature green -> yellow -> red as it nears critical."""
    if value is None:
        return "dim"
    if crit:
        ratio = value / crit
        if ratio >= 0.95:
            return "bold red"
        if ratio >= 0.80:
            return "bold yellow"
    return "bold green"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    gib = n / (1024**3)
    return f"{gib:.1f} GiB"


class StatPanel(Vertical):
    """A bordered, titled panel. Subclasses fill in the body content."""

    DEFAULT_CSS = """
    StatPanel {
        border: round $primary;
        padding: 0 1;
        height: auto;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = title


class TempPanel(StatPanel):
    def __init__(self, **kwargs) -> None:
        super().__init__("Temperatures", **kwargs)

    def compose(self):
        yield Static(id="temp-body")

    def update_sample(self, sample: Sample, limits: GpuLimits) -> None:
        table = Table.grid(padding=(0, 2, 0, 0), expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right")
        for label, value, key in (
            ("Edge", sample.edge_c, "edge"),
            ("Junction", sample.junction_c, "junction"),
            ("Mem", sample.mem_c, "mem"),
        ):
            crit = limits.temp_crit.get(key)
            style = _temp_style(value, crit)
            text = f"{value:.1f}°C" if value is not None else "n/a"
            suffix = f" / {crit:.0f}°C" if crit else ""
            row_text = Text(text, style=style)
            row_text.append(suffix, style="dim")
            table.add_row(label, row_text)
        self.query_one("#temp-body", Static).update(table)


class FanPanel(StatPanel):
    # amdgpu pwm1_enable: 0 = no control (fan at full speed), 1 = manual,
    # 2 = SMU firmware auto. A software curve runs in mode 1.
    MODE_LABELS = {0: "FULL SPEED", 1: "MANUAL", 2: "AUTO (firmware)"}
    MODE_STYLES = {0: "bold red", 1: "bold yellow", 2: "cyan"}

    DEFAULT_CSS = """
    FanPanel #fan-profiles {
        height: 1;
        margin-top: 1;
    }
    FanPanel #fan-profiles Button {
        width: 1fr;
        min-width: 0;
        margin: 0;
    }
    FanPanel #fan-profiles Button.profile-selected {
        background: $success;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Fan", **kwargs)

    def compose(self):
        yield Static(id="fan-body")
        yield ProgressBar(total=100, show_eta=False, id="fan-gauge")
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

    def update_sample(
        self,
        sample: Sample,
        pct: float,
        curve_target: float | None = None,
        curve_emergency: bool = False,
    ) -> None:
        mode = sample.pwm_enable
        if curve_target is not None and mode == 1:
            mode_label, mode_style = "CURVE", "bold green"
        else:
            mode_label = self.MODE_LABELS.get(mode, f"unknown ({mode})")
            mode_style = self.MODE_STYLES.get(mode, "white")

        table = Table.grid(padding=(0, 2, 0, 0), expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right")
        table.add_row("Mode", Text(mode_label, style=mode_style))
        table.add_row("Speed", Text(f"{pct:.0f}%", style="bold"))
        if curve_target is not None:
            target = Text(f"{curve_target:.0f}%")
            if curve_emergency:
                target.append(" HOT", style="bold red")
            table.add_row("Target", target)
        rpm = f"{sample.fan_rpm} RPM" if sample.fan_rpm is not None else "n/a"
        table.add_row("RPM", Text(rpm))
        self.query_one("#fan-body", Static).update(table)
        self.query_one("#fan-gauge", ProgressBar).update(progress=pct)


class PowerPanel(StatPanel):
    def __init__(self, **kwargs) -> None:
        super().__init__("Power & Load", **kwargs)

    def compose(self):
        yield Static(id="power-body")

    def update_sample(self, sample: Sample, limits: GpuLimits) -> None:
        table = Table.grid(padding=(0, 2, 0, 0), expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right")

        busy = f"{sample.gpu_busy_pct}%" if sample.gpu_busy_pct is not None else "n/a"
        busy_style = "bold red" if (sample.gpu_busy_pct or 0) >= 90 else "bold"
        table.add_row("GPU busy", Text(busy, style=busy_style))

        if sample.power_w is not None:
            cap = f" / {limits.power_cap:.0f}W" if limits.power_cap else ""
            table.add_row("Power", Text(f"{sample.power_w:.0f}W{cap}"))
        else:
            table.add_row("Power", Text("n/a"))

        if sample.vram_used is not None and sample.vram_total:
            vram_pct = sample.vram_used / sample.vram_total * 100
            table.add_row(
                "VRAM",
                Text(f"{_fmt_bytes(sample.vram_used)} ({vram_pct:.0f}%)"),
            )
        else:
            table.add_row("VRAM", Text("n/a"))

        if sample.gtt_used is not None and sample.gtt_total:
            gtt_pct = sample.gtt_used / sample.gtt_total * 100
            table.add_row("GTT", Text(f"{_fmt_bytes(sample.gtt_used)} ({gtt_pct:.0f}%)"))
        else:
            table.add_row("GTT", Text("n/a"))

        if sample.sclk_mhz:
            table.add_row("SCLK", Text(f"{sample.sclk_mhz} MHz"))
        if sample.mclk_mhz:
            table.add_row("MCLK", Text(f"{sample.mclk_mhz} MHz"))

        self.query_one("#power-body", Static).update(table)


def _fmt_kib(n: int) -> str:
    if n <= 0:
        return "0MiB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}GiB"
    return f"{n / 1024:.0f}MiB"


class ProcessPanel(Vertical):
    """Full-width nvtop-style table of processes currently using the GPU."""

    DEFAULT_CSS = """
    ProcessPanel {
        border: round $primary;
        padding: 0 1;
    }
    ProcessPanel DataTable {
        height: 1fr;
    }
    ProcessPanel DataTable > .datatable--cursor {
        background: $primary;
        color: $text;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "GPU Processes"

    def compose(self):
        yield DataTable(id="process-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#process-table", DataTable)
        table.add_columns(
            "PID", "USER", "DEV", "TYPE", "GPU%", "GPU MEM", "CPU%", "HOST MEM", "Command"
        )

    def update_processes(self, processes: list[GpuProcess]) -> None:
        table = self.query_one("#process-table", DataTable)
        table.clear()
        for p in processes:
            table.add_row(
                str(p.pid),
                p.user,
                str(p.dev),
                p.type_label,
                f"{p.gpu_pct:.0f}%",
                _fmt_kib(p.vram_kib),
                f"{p.cpu_pct:.0f}%",
                _fmt_kib(p.host_mem_kib),
                p.command,
                key=str(p.pid),
            )
        if not processes:
            table.add_row("", "", "", "", "", "", "", "", "(no processes found using the GPU)")
