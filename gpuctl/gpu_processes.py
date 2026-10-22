"""Per-process GPU usage, nvtop-style, via /proc/<pid>/fdinfo DRM accounting.

Every process that holds an open file descriptor onto a DRM device node
(/dev/dri/cardN or /dev/dri/renderDN) gets driver-populated accounting lines
in /proc/<pid>/fdinfo/<fd> -- things like drm-memory-vram, drm-memory-gtt and
drm-engine-* (cumulative nanoseconds of engine time). This is exactly how
nvtop/amdgpu_top attribute VRAM and engine usage to processes without needing
root: each process's own fdinfo is readable by that process's owner, and this
is a single-user box so cross-user PermissionErrors (skipped silently) aren't
a real limitation here.

Read-only, no privileged access needed -- mirrors hwmon.py's separation of
concerns (discovery + reading only, no writing).
"""

from __future__ import annotations

import glob
import os
import pwd
import time
from dataclasses import dataclass

_UNIT_TO_KIB = {"B": 1 / 1024, "KIB": 1, "MIB": 1024, "GIB": 1024 * 1024}


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError:
        return None


def _parse_size_kib(raw: str) -> int:
    """Parse a fdinfo size value like '13816 KiB' / '2 MiB' into KiB."""
    parts = raw.split()
    if not parts:
        return 0
    try:
        value = float(parts[0])
    except ValueError:
        return 0
    unit = parts[1].upper() if len(parts) > 1 else "KIB"
    return int(value * _UNIT_TO_KIB.get(unit, 1))


def _username_for_uid(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _proc_uid(pid: int) -> int | None:
    text = _read_text(f"/proc/{pid}/status")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    return None


def _proc_rss_kib(pid: int) -> int | None:
    text = _read_text(f"/proc/{pid}/status")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    return None


def _proc_cpu_ticks(pid: int) -> int | None:
    """Sum of utime+stime (in clock ticks) from /proc/<pid>/stat."""
    text = _read_text(f"/proc/{pid}/stat")
    if not text:
        return None
    # comm (2nd field) is parenthesised and may itself contain ')', so split
    # on the *last* ')' to safely skip past it before splitting on spaces.
    try:
        after_comm = text.rsplit(")", 1)[1]
    except IndexError:
        return None
    fields = after_comm.split()
    # fields[0] is state; utime is field 14 overall -> index 11 here, stime 12.
    if len(fields) < 13:
        return None
    try:
        return int(fields[11]) + int(fields[12])
    except ValueError:
        return None


def _proc_command(pid: int) -> str:
    raw = None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        raw = None
    if raw:
        parts = [p.decode(errors="replace") for p in raw.split(b"\x00") if p]
        if parts:
            return " ".join(parts)
    comm = _read_text(f"/proc/{pid}/comm")
    return comm.strip() if comm else f"pid {pid}"


@dataclass
class GpuProcess:
    """One row of the process table: one pid using our GPU."""

    pid: int
    user: str
    dev: int
    type_label: str
    gpu_pct: float
    vram_kib: int
    gtt_kib: int
    cpu_pct: float
    host_mem_kib: int
    command: str


class GpuProcessMonitor:
    """Scans /proc for processes with a DRM fd into our GPU.

    Owns the previous-scan state (per-pid engine-ns and cpu-ticks counters,
    plus the wall-clock time of the previous scan) needed to turn the
    cumulative counters fdinfo/proc expose into instantaneous percentages.
    """

    def __init__(self) -> None:
        self._prev_engine_ns: dict[int, int] = {}
        self._prev_cpu_ticks: dict[int, int] = {}
        self._prev_t: float | None = None
        try:
            self._clk_tck = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError):
            self._clk_tck = 100

    def scan(self, pci_slot: str, dev_index: int = 0) -> list[GpuProcess]:
        now = time.monotonic()
        dt = None if self._prev_t is None else max(now - self._prev_t, 1e-6)

        vram_kib: dict[int, int] = {}
        gtt_kib: dict[int, int] = {}
        engine_ns: dict[int, int] = {}
        has_gfx: dict[int, bool] = {}
        has_compute: dict[int, bool] = {}

        for fdinfo_dir in glob.glob("/proc/[0-9]*/fdinfo"):
            try:
                pid = int(fdinfo_dir.split("/")[2])
            except (IndexError, ValueError):
                continue
            try:
                entries = os.listdir(fdinfo_dir)
            except OSError:
                continue  # process exited between glob() and listdir()

            for entry in entries:
                text = _read_text(f"{fdinfo_dir}/{entry}")
                if not text or "amdgpu" not in text or pci_slot not in text:
                    continue  # cheap pre-filter before the full parse below

                fields: dict[str, str] = {}
                for line in text.splitlines():
                    if ":" not in line:
                        continue
                    key, _, value = line.partition(":")
                    fields[key.strip()] = value.strip()

                if fields.get("drm-driver") != "amdgpu":
                    continue
                if fields.get("drm-pdev") != pci_slot:
                    continue

                vram_kib[pid] = vram_kib.get(pid, 0) + _parse_size_kib(
                    fields.get("drm-memory-vram", "0")
                )
                gtt_kib[pid] = gtt_kib.get(pid, 0) + _parse_size_kib(
                    fields.get("drm-memory-gtt", "0")
                )

                pid_engine_ns = engine_ns.get(pid, 0)
                for key, value in fields.items():
                    if not key.startswith("drm-engine-"):
                        continue
                    try:
                        pid_engine_ns += int(value.split()[0])
                    except (ValueError, IndexError):
                        continue
                    if key == "drm-engine-gfx":
                        has_gfx[pid] = True
                    elif key == "drm-engine-compute":
                        has_compute[pid] = True
                engine_ns[pid] = pid_engine_ns

        seen_pids = set(vram_kib) | set(gtt_kib) | set(engine_ns)
        new_cpu_ticks: dict[int, int] = {}
        results: list[GpuProcess] = []

        for pid in seen_pids:
            uid = _proc_uid(pid)
            user = _username_for_uid(uid) if uid is not None else "?"

            gpu_pct = 0.0
            if dt is not None and pid in self._prev_engine_ns:
                delta_ns = engine_ns.get(pid, 0) - self._prev_engine_ns[pid]
                if delta_ns > 0:
                    gpu_pct = max(0.0, min(100.0, delta_ns / (dt * 1e9) * 100))

            cpu_pct = 0.0
            ticks = _proc_cpu_ticks(pid)
            if ticks is not None:
                new_cpu_ticks[pid] = ticks
                if dt is not None and pid in self._prev_cpu_ticks:
                    delta_ticks = ticks - self._prev_cpu_ticks[pid]
                    if delta_ticks > 0:
                        cpu_pct = max(
                            0.0, min(100.0, delta_ticks / (dt * self._clk_tck) * 100)
                        )

            if has_gfx.get(pid) and has_compute.get(pid):
                type_label = "Both G+C"
            elif has_compute.get(pid):
                type_label = "Compute"
            else:
                type_label = "Graphic"

            rss = _proc_rss_kib(pid)
            results.append(
                GpuProcess(
                    pid=pid,
                    user=user,
                    dev=dev_index,
                    type_label=type_label,
                    gpu_pct=gpu_pct,
                    vram_kib=vram_kib.get(pid, 0),
                    gtt_kib=gtt_kib.get(pid, 0),
                    cpu_pct=cpu_pct,
                    host_mem_kib=rss or 0,
                    command=_proc_command(pid),
                )
            )

        self._prev_engine_ns = engine_ns
        self._prev_cpu_ticks = new_cpu_ticks
        self._prev_t = now

        results.sort(key=lambda p: (p.vram_kib, p.gpu_pct), reverse=True)
        return results
