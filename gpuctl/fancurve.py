"""Software fan curve: temperature -> fan % with hysteresis, smoothing,
spin-down delay and a trend look-ahead.

Why this exists: on RDNA2 (Navi 21) the kernel exposes no editable fan curve
(`gpu_od/fan_ctrl/` is RDNA3+ only). `pwm1_enable=2` just hands the fan to
the SMU firmware's fixed behaviour. So "curve" mode here is a control loop
in the app: the GPU is put in manual mode (`pwm1_enable=1`) and the app
writes `pwm1` itself from the readings. The root helper that does the
writes (privileged.RootFanHelper) restores firmware auto when the app exits.

This module is pure logic plus the JSON config file. No sysfs access.

How one control step works (FanCurveController.step):

1. Each sensor's reading is smoothed with an EMA (`smoothing_s`), so a
   one-sample spike does not make the fan jump.
2. A look-ahead adds `max(0, slope) * lookahead_s` (at most
   MAX_LOOKAHEAD_C) to the smoothed temp, so the fan starts to ramp while
   the temp is still climbing, not after. The slope comes from the smoothed
   series, so a spike does not count as a climb.
3. Each sensor has its own curve. The demand is the max over sensors.
4. Hysteresis: the speed rises to curve(T), but only falls to
   curve(T + hysteresis_c). Between the two, the speed holds. The same rule
   gives zero-RPM a start/stop band for free.
5. A fall must be wanted for `spin_down_delay_s` before it starts, and then
   it ramps at `ramp_down_pct_per_s`. Rises are limited by
   `ramp_up_pct_per_s`.
6. Fail safe: a missing junction reading, or any sensor within
   `emergency_margin_c` of its critical temp, gives 100% at once.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent.parent / ".gpuctl_fancurve.json"

PROFILE_ORDER = ("quiet", "balanced", "cool")

# Seconds of smoothed history used to estimate how fast a temp is rising.
SLOPE_WINDOW_S = 6.0
# Upper bound on what the look-ahead may add, so a sharp but short rise
# cannot push the fan to full speed by itself.
MAX_LOOKAHEAD_C = 8.0


@dataclass
class CurveProfile:
    # Per-sensor curves: sensor label -> [[temp_c, pct], ...], temp ascending.
    # Labels match hwmon temp*_label ("edge", "junction", "mem").
    curves: dict[str, list[list[float]]]
    # Lowest non-zero speed. Below this the fan may stall or click.
    min_spin_pct: float
    hysteresis_c: float
    spin_down_delay_s: float
    ramp_down_pct_per_s: float
    ramp_up_pct_per_s: float
    smoothing_s: float
    lookahead_s: float


DEFAULT_PROFILES: dict[str, CurveProfile] = {
    # Zero-RPM until ~55C junction, slow to react, slow to spin down.
    "quiet": CurveProfile(
        curves={
            "junction": [[55, 0], [60, 22], [70, 30], [80, 45], [90, 68], [100, 100]],
            "mem": [[75, 0], [80, 25], [88, 50], [94, 100]],
        },
        min_spin_pct=20,
        hysteresis_c=5,
        spin_down_delay_s=15,
        ramp_down_pct_per_s=1.5,
        ramp_up_pct_per_s=15,
        smoothing_s=6,
        lookahead_s=0,
    ),
    "balanced": CurveProfile(
        curves={
            "junction": [[48, 0], [50, 25], [65, 38], [80, 60], [90, 82], [98, 100]],
            "mem": [[70, 0], [72, 28], [85, 55], [92, 100]],
        },
        min_spin_pct=22,
        hysteresis_c=4,
        spin_down_delay_s=8,
        ramp_down_pct_per_s=3,
        ramp_up_pct_per_s=25,
        smoothing_s=4,
        lookahead_s=3,
    ),
    # Fan always on, steep ramp, reacts early to a fast rise.
    "cool": CurveProfile(
        curves={
            "junction": [[0, 30], [50, 35], [65, 50], [75, 70], [85, 90], [92, 100]],
            "mem": [[0, 30], [70, 40], [82, 70], [90, 100]],
        },
        min_spin_pct=25,
        hysteresis_c=3,
        spin_down_delay_s=5,
        ramp_down_pct_per_s=4,
        ramp_up_pct_per_s=50,
        smoothing_s=2,
        lookahead_s=6,
    ),
}


@dataclass
class CurveConfig:
    active: str = "quiet"
    # Start the curve when the app opens (asks for the sudo password then).
    autostart: bool = True
    emergency_margin_c: float = 10
    profiles: dict[str, CurveProfile] = field(
        default_factory=lambda: dict(DEFAULT_PROFILES)
    )

    @property
    def profile(self) -> CurveProfile:
        return self.profiles[self.active]

    def next_profile(self) -> str:
        names = [n for n in PROFILE_ORDER if n in self.profiles]
        names += [n for n in self.profiles if n not in names]
        idx = names.index(self.active) if self.active in names else -1
        return names[(idx + 1) % len(names)]


def _validate_profile(name: str, raw: dict) -> CurveProfile:
    profile = CurveProfile(**raw)
    if not profile.curves:
        raise ValueError(f"profile '{name}': no curves")
    for sensor, points in profile.curves.items():
        if not points:
            raise ValueError(f"profile '{name}': curve '{sensor}' is empty")
        temps = [float(p[0]) for p in points]
        if temps != sorted(temps):
            raise ValueError(f"profile '{name}': curve '{sensor}' temps not ascending")
        for p in points:
            if len(p) != 2 or not 0 <= float(p[1]) <= 100:
                raise ValueError(f"profile '{name}': curve '{sensor}' has a bad point {p}")
    return profile


def load_config() -> tuple[CurveConfig, str | None]:
    """Read the config file. Write the defaults first if it does not exist.

    Returns (config, error). On a bad file the defaults are returned with the
    error text, and the file is left alone so the user can fix it.
    """
    if not CONFIG_FILE.exists():
        config = CurveConfig()
        save_config(config)
        return config, None
    try:
        raw = json.loads(CONFIG_FILE.read_text())
        profiles = {
            name: _validate_profile(name, p) for name, p in raw["profiles"].items()
        }
        active = raw.get("active", "quiet")
        if active not in profiles:
            active = next(iter(profiles))
        config = CurveConfig(
            active=active,
            autostart=bool(raw.get("autostart", True)),
            emergency_margin_c=float(raw.get("emergency_margin_c", 10)),
            profiles=profiles,
        )
        return config, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return CurveConfig(), f"{CONFIG_FILE.name}: {exc}"


def save_config(config: CurveConfig) -> None:
    data = {
        "active": config.active,
        "autostart": config.autostart,
        "emergency_margin_c": config.emergency_margin_c,
        "profiles": {name: asdict(p) for name, p in config.profiles.items()},
    }
    text = json.dumps(data, indent=2)
    # Keep each curve point on one line ("[55, 0]") so the file is easy to edit.
    text = re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", text)
    try:
        CONFIG_FILE.write_text(text + "\n")
    except OSError:
        pass


def interpolate(points: list[list[float]], temp: float) -> float:
    """Piecewise-linear lookup, flat beyond the first and last point."""
    if temp <= points[0][0]:
        return float(points[0][1])
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if temp <= t1:
            if t1 == t0:
                return float(p1)
            return p0 + (p1 - p0) * (temp - t0) / (t1 - t0)
    return float(points[-1][1])


class FanCurveController:
    """Stateful curve evaluator. Feed it readings, get back a target %."""

    def __init__(
        self,
        profile: CurveProfile,
        crit: dict[str, float],
        emergency_margin_c: float,
        start_pct: float,
    ) -> None:
        self.profile = profile
        self.crit = crit
        self.emergency_margin_c = emergency_margin_c
        self.pct = start_pct
        self.emergency = False
        self._smoothed: dict[str, float] = {}
        self._history: dict[str, deque[tuple[float, float]]] = {}
        self._last_t: float | None = None
        self._down_since: float | None = None

    def set_profile(self, profile: CurveProfile, emergency_margin_c: float) -> None:
        self.profile = profile
        self.emergency_margin_c = emergency_margin_c
        self._down_since = None

    def _effective_temp(self, sensor: str, now: float, raw: float, dt: float) -> float:
        prev = self._smoothed.get(sensor)
        if prev is None or self.profile.smoothing_s <= 0:
            smoothed = raw
        else:
            alpha = min(1.0, dt / self.profile.smoothing_s)
            smoothed = prev + alpha * (raw - prev)
        self._smoothed[sensor] = smoothed

        hist = self._history.setdefault(sensor, deque())
        hist.append((now, smoothed))
        while hist and now - hist[0][0] > SLOPE_WINDOW_S:
            hist.popleft()

        slope = 0.0
        if len(hist) >= 2 and hist[-1][0] - hist[0][0] >= 2.0:
            slope = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
        return smoothed + min(MAX_LOOKAHEAD_C, max(0.0, slope) * self.profile.lookahead_s)

    def _demand(self, temps: dict[str, float], offset: float) -> float:
        demand = 0.0
        for sensor, points in self.profile.curves.items():
            if sensor in temps:
                demand = max(demand, interpolate(points, temps[sensor] + offset))
        if 0 < demand < self.profile.min_spin_pct:
            demand = self.profile.min_spin_pct
        return demand

    def step(self, now: float, readings: dict[str, float | None]) -> float:
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        self.emergency = readings.get("junction") is None or any(
            value is not None
            and sensor in self.crit
            and value >= self.crit[sensor] - self.emergency_margin_c
            for sensor, value in readings.items()
        )
        temps = {
            sensor: self._effective_temp(sensor, now, value, dt)
            for sensor, value in readings.items()
            if value is not None
        }
        if self.emergency:
            self.pct = 100.0
            self._down_since = None
            return self.pct

        up = self._demand(temps, 0.0)
        down = self._demand(temps, self.profile.hysteresis_c)

        if up > self.pct:
            self._down_since = None
            if dt == 0:
                self.pct = up
            else:
                self.pct = min(up, self.pct + self.profile.ramp_up_pct_per_s * dt)
        elif down < self.pct:
            if self._down_since is None:
                self._down_since = now
            if now - self._down_since >= self.profile.spin_down_delay_s:
                self.pct = max(down, self.pct - self.profile.ramp_down_pct_per_s * dt)
                # Do not crawl through the stall zone: once the curve wants
                # the fan off and we are at the lowest spin speed, stop.
                if down == 0 and self.pct < self.profile.min_spin_pct:
                    self.pct = 0.0
        else:
            self._down_since = None
        return self.pct
