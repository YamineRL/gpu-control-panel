"""The GPU process table: the one full-width panel that is not a tile."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import DataTable

from .gpu_processes import GpuProcess


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
