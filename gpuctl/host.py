"""Host readings the GPU's hwmon does not carry: CPU load, RAM and swap, disk.

Everything comes from /proc (stat, meminfo, diskstats), so the app still has
no third-party dependency, needs no root and never blocks on a device. The
counters are cumulative, so the source keeps the previous read and turns the
difference into a rate. A file that cannot be read yields None, never a zero
that would read as a real measurement.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

SECTOR_BYTES = 512
# /proc/stat column order after the "cpu" label.
CPU_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
    "guest",
    "guest_nice",
)


@dataclass(slots=True)
class HostSample:
    """One point-in-time reading of the machine around the card."""

    t: float = 0.0

    cpu_pct: float | None = None
    cores: int = 0
    threads: int = 0
    load1: float | None = None

    ram_used: int | None = None
    ram_total: int | None = None
    swap_used: int | None = None
    swap_total: int | None = None

    disk_read_bps: float | None = None
    disk_write_bps: float | None = None

    @property
    def ram_pct(self) -> float | None:
        if self.ram_used is None or not self.ram_total:
            return None
        return self.ram_used / self.ram_total * 100.0

    @property
    def swap_pct(self) -> float | None:
        if self.swap_used is None or not self.swap_total:
            return None
        return self.swap_used / self.swap_total * 100.0


def _read(path: str) -> str | None:
    try:
        with open(path, "r") as handle:
            return handle.read()
    except OSError:
        return None


def _meminfo(path: str) -> dict[str, int]:
    """MemTotal: 36847848 kB -> {"MemTotal": 37732200 (bytes), ...}"""
    text = _read(path)
    values: dict[str, int] = {}
    if not text:
        return values
    for line in text.splitlines():
        label, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else ""
        values[label.strip()] = value * 1024 if unit == "kb" else value
    return values


def _cpu_totals(path: str) -> tuple[int, int] | None:
    """Aggregate (busy, total) jiffies from the first /proc/stat line."""
    text = _read(path)
    if not text:
        return None
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()[1:]
        try:
            values = [int(part) for part in parts]
        except ValueError:
            return None
        total = sum(values)
        idle = values[3] if len(values) > 3 else 0
        iowait = values[4] if len(values) > 4 else 0
        return total - idle - iowait, total
    return None


def _disk_fields(path: str, names: set[str]) -> tuple[int, int] | None:
    """Sectors read and written by the whole disks named in *names*."""
    text = _read(path)
    if not text:
        return None
    read_sectors = written_sectors = 0
    seen = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or parts[2] not in names:
            continue
        try:
            read_sectors += int(parts[5])
            written_sectors += int(parts[9])
        except ValueError:
            continue
        seen = True
    return (read_sectors, written_sectors) if seen else None


def _cpu_shape(sys_cpu_root: str) -> tuple[int, int]:
    """(physical cores, hardware threads) from the kernel's own view."""
    threads = os.cpu_count() or 0
    cores = threads
    try:
        names = os.listdir(sys_cpu_root)
    except OSError:
        return cores, threads
    physical = set()
    for name in names:
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        base = os.path.join(sys_cpu_root, name, "topology")
        package = _read(os.path.join(base, "physical_package_id"))
        core = _read(os.path.join(base, "core_id"))
        if package is None or core is None:
            continue
        physical.add((package.strip(), core.strip()))
    if physical:
        cores = len(physical)
    return cores, threads


class HostSource:
    """Samples the host's CPU, memory and disk counters.

    The paths are injectable so the tests can feed fixture files instead of
    whatever machine happens to run them.
    """

    def __init__(
        self,
        stat_path: str = "/proc/stat",
        meminfo_path: str = "/proc/meminfo",
        diskstats_path: str = "/proc/diskstats",
        block_dir: str = "/sys/block",
        cpu_root: str = "/sys/devices/system/cpu",
    ) -> None:
        self.stat_path = stat_path
        self.meminfo_path = meminfo_path
        self.diskstats_path = diskstats_path
        self.block_dir = block_dir
        self.cores, self.threads = _cpu_shape(cpu_root)
        self._prev_cpu: tuple[float, int, int] | None = None
        self._prev_disk: tuple[float, int, int] | None = None

    def _disks(self) -> set[str]:
        try:
            return set(os.listdir(self.block_dir))
        except OSError:
            return set()

    def _load(self) -> float | None:
        try:
            return os.getloadavg()[0]
        except OSError:
            return None

    def read(self, t: float | None = None) -> HostSample:
        now = time.monotonic() if t is None else t
        sample = HostSample(t=now, cores=self.cores, threads=self.threads)

        cpu = _cpu_totals(self.stat_path)
        if cpu is not None:
            if self._prev_cpu is not None:
                prev_t, prev_busy, prev_total = self._prev_cpu
                busy_delta = cpu[0] - prev_busy
                total_delta = cpu[1] - prev_total
                elapsed = now - prev_t
                if total_delta > 0 and elapsed > 0:
                    sample.cpu_pct = max(
                        0.0, min(100.0, busy_delta / total_delta * 100.0)
                    )
            self._prev_cpu = (now, cpu[0], cpu[1])
        sample.load1 = self._load()

        memory = _meminfo(self.meminfo_path)
        if memory:
            total = memory.get("MemTotal")
            available = memory.get("MemAvailable")
            if total is not None and available is not None:
                sample.ram_total = total
                sample.ram_used = max(0, total - available)
            swap_total = memory.get("SwapTotal")
            swap_free = memory.get("SwapFree")
            if swap_total is not None and swap_free is not None:
                sample.swap_total = swap_total
                sample.swap_used = max(0, swap_total - swap_free)

        disk = _disk_fields(self.diskstats_path, self._disks())
        if disk is not None:
            if self._prev_disk is not None:
                prev_t, prev_read, prev_written = self._prev_disk
                elapsed = now - prev_t
                if elapsed > 0:
                    sample.disk_read_bps = max(
                        0.0,
                        (disk[0] - prev_read) * SECTOR_BYTES / elapsed,
                    )
                    sample.disk_write_bps = max(
                        0.0,
                        (disk[1] - prev_written) * SECTOR_BYTES / elapsed,
                    )
            self._prev_disk = (now, disk[0], disk[1])

        return sample
