"""Discovery and reading of the amdgpu hwmon sysfs interface.

No writing happens in this module -- privileged writes to pwm1 / pwm1_enable
live in privileged.py, since they need a sudo password.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field


class HwmonNotFoundError(RuntimeError):
    """Raised when no amdgpu hwmon directory can be found on this machine."""


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class Sample:
    """One point-in-time reading of everything the UI cares about."""

    t: float

    edge_c: float | None = None
    junction_c: float | None = None
    mem_c: float | None = None

    fan_rpm: int | None = None
    pwm: int | None = None
    pwm_enable: int | None = None

    gpu_busy_pct: int | None = None
    mem_busy_pct: int | None = None

    power_w: float | None = None

    vram_used: int | None = None
    vram_total: int | None = None
    gtt_used: int | None = None
    gtt_total: int | None = None

    sclk_mhz: int | None = None
    mclk_mhz: int | None = None


PCIE_GENERATIONS: tuple[tuple[float, int], ...] = (
    (2.5, 1),
    (5.0, 2),
    (8.0, 3),
    (16.0, 4),
    (32.0, 5),
    (64.0, 6),
)


def _parse_link_rate(raw: str | None) -> float | None:
    """"16.0 GT/s PCIe" -> 16.0. None when the file is missing or odd."""
    if not raw:
        return None
    token = raw.strip().split(" ", 1)[0]
    try:
        return float(token)
    except ValueError:
        return None


@dataclass(frozen=True)
class PcieLink:
    """The card's current and maximum PCIe link, as the kernel reports it."""

    speed_gt: float | None = None
    width: int | None = None
    max_speed_gt: float | None = None
    max_width: int | None = None

    @property
    def generation(self) -> int | None:
        if self.speed_gt is None:
            return None
        for rate, generation in PCIE_GENERATIONS:
            if abs(self.speed_gt - rate) < 0.25:
                return generation
        return None

    @property
    def max_generation(self) -> int | None:
        if self.max_speed_gt is None:
            return None
        for rate, generation in PCIE_GENERATIONS:
            if abs(self.max_speed_gt - rate) < 0.25:
                return generation
        return None

    @property
    def bandwidth_gbs(self) -> float | None:
        """One-direction payload rate of the negotiated link, in GB/s."""
        if self.speed_gt is None or self.width is None:
            return None
        return self.speed_gt * self.width / 8.0

    @property
    def downgraded(self) -> bool:
        """True when the link is narrower or slower than the card can do.

        PCIe power management drops the link while the card is idle, so this
        is a fact about the moment, not a fault.
        """
        if self.generation is None or self.max_generation is None:
            return False
        if self.generation < self.max_generation:
            return True
        return bool(
            self.width is not None
            and self.max_width is not None
            and self.width < self.max_width
        )


@dataclass
class GpuLimits:
    """Static bounds/labels discovered once at startup."""

    pwm_min: int
    pwm_max: int
    fan_min: int
    fan_max: int

    temp_labels: dict[str, str] = field(default_factory=dict)  # e.g. temp1 -> "edge"
    temp_crit: dict[str, float] = field(default_factory=dict)  # in C, keyed by "edge"/"junction"/"mem"
    temp_emergency: dict[str, float] = field(default_factory=dict)

    power_cap: float | None = None  # watts
    power_cap_min: float | None = None
    power_cap_max: float | None = None


class HwmonSource:
    """Finds the amdgpu hwmon + device sysfs directories and reads samples."""

    def __init__(self, hwmon_dir: str, device_dir: str) -> None:
        self.hwmon_dir = hwmon_dir
        self.device_dir = device_dir
        self.limits = self._discover_limits()

    @classmethod
    def discover(cls) -> "HwmonSource":
        for hwmon_path in sorted(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")):
            name = _read_text(f"{hwmon_path}/name")
            if name == "amdgpu":
                device_dir = f"{hwmon_path}/../.."
                device_dir = os.path.normpath(device_dir)
                return cls(hwmon_path, device_dir)
        raise HwmonNotFoundError(
            "No amdgpu hwmon device found under /sys/class/drm/card*/device/hwmon/.\n"
            "This tool only works on a machine with an amdgpu-driven GPU."
        )

    @property
    def pci_slot(self) -> str:
        """PCI slot string (e.g. '0000:09:00.0') for this GPU.

        Used to attribute /proc/<pid>/fdinfo DRM entries to this card
        specifically (see gpu_processes.py), in case the machine ever has
        more than one GPU.
        """
        return os.path.basename(os.path.realpath(self.device_dir))

    @property
    def card_index(self) -> int:
        """The N in /dev/dri/cardN, derived from device_dir (.../cardN/device)."""
        card_dir_name = os.path.basename(os.path.dirname(self.device_dir))
        try:
            return int(card_dir_name.removeprefix("card"))
        except ValueError:
            return 0

    def _hw(self, key: str) -> str:
        return f"{self.hwmon_dir}/{key}"

    def _dev(self, key: str) -> str:
        return f"{self.device_dir}/{key}"

    @property
    def link(self) -> PcieLink:
        """The negotiated PCIe link, read once (the files are cheap but static).

        Lazy so a source built without ``__init__`` (the screenshot renderer)
        still answers.
        """
        cached = getattr(self, "_link", None)
        if cached is not None:
            return cached
        device = getattr(self, "device_dir", "")
        link = PcieLink(
            speed_gt=_parse_link_rate(_read_text(f"{device}/current_link_speed")),
            width=_read_int(f"{device}/current_link_width"),
            max_speed_gt=_parse_link_rate(_read_text(f"{device}/max_link_speed")),
            max_width=_read_int(f"{device}/max_link_width"),
        )
        self._link = link
        return link

    @property
    def pwm1_path(self) -> str:
        return self._hw("pwm1")

    @property
    def pwm1_enable_path(self) -> str:
        return self._hw("pwm1_enable")

    def _discover_limits(self) -> GpuLimits:
        pwm_min = _read_int(self._hw("pwm1_min")) or 0
        pwm_max = _read_int(self._hw("pwm1_max")) or 255
        fan_min = _read_int(self._hw("fan1_min")) or 0
        fan_max = _read_int(self._hw("fan1_max")) or 0

        temp_labels: dict[str, str] = {}
        temp_crit: dict[str, float] = {}
        temp_emergency: dict[str, float] = {}
        for n in (1, 2, 3):
            label = _read_text(self._hw(f"temp{n}_label"))
            if label is None:
                continue
            temp_labels[f"temp{n}"] = label
            crit = _read_int(self._hw(f"temp{n}_crit"))
            if crit is not None:
                temp_crit[label] = crit / 1000
            emer = _read_int(self._hw(f"temp{n}_emergency"))
            if emer is not None:
                temp_emergency[label] = emer / 1000

        cap = _read_int(self._hw("power1_cap"))
        cap_min = _read_int(self._hw("power1_cap_min"))
        cap_max = _read_int(self._hw("power1_cap_max"))

        return GpuLimits(
            pwm_min=pwm_min,
            pwm_max=pwm_max,
            fan_min=fan_min,
            fan_max=fan_max,
            temp_labels=temp_labels,
            temp_crit=temp_crit,
            temp_emergency=temp_emergency,
            power_cap=(cap / 1_000_000 if cap is not None else None),
            power_cap_min=(cap_min / 1_000_000 if cap_min is not None else None),
            power_cap_max=(cap_max / 1_000_000 if cap_max is not None else None),
        )

    def _temp_by_label(self, label: str) -> float | None:
        for n, lbl in self.limits.temp_labels.items():
            if lbl == label:
                raw = _read_int(self._hw(f"{n}_input"))
                return raw / 1000 if raw is not None else None
        return None

    @staticmethod
    def _parse_active_clock(raw: str | None) -> int | None:
        """Parse pp_dpm_sclk / pp_dpm_mclk text, return the active clock in MHz."""
        if not raw:
            return None
        for line in raw.splitlines():
            line = line.strip()
            if line.endswith("*"):
                # e.g. "1: 2510Mhz *"
                part = line.split(":", 1)[1].strip().rstrip("*").strip()
                digits = "".join(c for c in part if c.isdigit())
                if digits:
                    return int(digits)
        return None

    def read_sample(self) -> Sample:
        pwm = _read_int(self._hw("pwm1"))
        pwm_enable = _read_int(self._hw("pwm1_enable"))
        fan_rpm = _read_int(self._hw("fan1_input"))

        power_raw = _read_int(self._hw("power1_average"))
        power_w = power_raw / 1_000_000 if power_raw is not None else None

        sclk = self._parse_active_clock(_read_text(self._dev("pp_dpm_sclk")))
        mclk = self._parse_active_clock(_read_text(self._dev("pp_dpm_mclk")))

        return Sample(
            t=time.monotonic(),
            edge_c=self._temp_by_label("edge"),
            junction_c=self._temp_by_label("junction"),
            mem_c=self._temp_by_label("mem"),
            fan_rpm=fan_rpm,
            pwm=pwm,
            pwm_enable=pwm_enable,
            gpu_busy_pct=_read_int(self._dev("gpu_busy_percent")),
            mem_busy_pct=_read_int(self._dev("mem_busy_percent")),
            power_w=power_w,
            vram_used=_read_int(self._dev("mem_info_vram_used")),
            vram_total=_read_int(self._dev("mem_info_vram_total")),
            gtt_used=_read_int(self._dev("mem_info_gtt_used")),
            gtt_total=_read_int(self._dev("mem_info_gtt_total")),
            sclk_mhz=sclk,
            mclk_mhz=mclk,
        )

    def pct_to_pwm(self, pct: float) -> int:
        pct = max(0.0, min(100.0, pct))
        lo, hi = self.limits.pwm_min, self.limits.pwm_max
        return round(lo + (hi - lo) * pct / 100)

    def pwm_to_pct(self, pwm: int) -> float:
        lo, hi = self.limits.pwm_min, self.limits.pwm_max
        if hi == lo:
            return 0.0
        return max(0.0, min(100.0, (pwm - lo) / (hi - lo) * 100))
