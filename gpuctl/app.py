"""GPU Control Panel main application.

A Textual TUI for monitoring and controlling the fan of an AMD GPU via the
amdgpu driver's hwmon sysfs interface (pwm1 / pwm1_enable / fan1_input /
temp*_input / power1_average / gpu_busy_percent / mem_info_*).

Run via ./run.sh from the project root (sets up/uses the local .venv), or
manually:

    .venv/bin/python main.py
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Static

from .fancurve import FanCurveController, load_config, save_config
from .gpu_processes import GpuProcessMonitor
from .graphs import ChartView, Series, snap_window
from .hwmon import HwmonNotFoundError, HwmonSource, Sample
from .inference import InferenceBar
from .llama import InferenceSnapshot, LlamaMonitor
from .privileged import RootFanHelper, sudo_write
from .screens import HwmonMissingApp, PasswordModal, SetSpeedModal
from .widgets import FanPanel, PowerPanel, ProcessPanel, TempPanel

TIMEFRAMES: list[tuple[str, int]] = [
    ("1m", 60),
    ("5m", 300),
    ("15m", 900),
    ("1h", 3600),
]

PROBE_INTERVAL_DEFAULT_MS = 2000
PROBE_INTERVAL_MIN_MS = 500
PROBE_INTERVAL_MAX_MS = 10000
PROBE_INTERVAL_STEP_MS = 500

# How often the llama.cpp server is asked for its slot counters. This runs
# in a worker thread, so it never competes with rendering or mouse events.
INFERENCE_INTERVAL_S = 0.5

# The fan curve runs on its own timer so it stays responsive whatever probe
# rate the graph uses.
CURVE_INTERVAL_S = 1.0
# The driver may round pwm1 on read-back. Rewrite only when it drifts by
# more than this (e.g. the firmware took the fan back).
PWM_DRIFT_TOLERANCE = 3

# Sized for the fastest possible probe rate so the sample buffer can hold a
# full "1h" window's worth of history no matter what rate is picked later.
MAX_SAMPLES = int(TIMEFRAMES[-1][1] / (PROBE_INTERVAL_MIN_MS / 1000)) + 10

STATE_FILE = Path(__file__).resolve().parent.parent / ".gpuctl_state"
PROBE_STATE_FILE = Path(__file__).resolve().parent.parent / ".gpuctl_probe_interval"

TEMP_COLORS = {"edge": "cyan", "junction": "#ff8700", "mem": "magenta"}
# Dimmer variants used for the filled area under each line (btop style: a
# bright top edge over a dim body). RGB tuples so they don't depend on the
# terminal's 16-colour palette.
TEMP_FILL_COLORS = {
    "edge": (0, 46, 46),
    "junction": (58, 36, 0),
    "mem": (46, 0, 46),
}


def _short_duration(seconds: int) -> str:
    """300 -> 5m, 90 -> 1m30s, 3600 -> 1h."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}m" if rest == 0 else f"{minutes}m{rest}s"


class TimeframeBar(Horizontal):
    """Row of buttons to pick how much graph history is shown."""

    DEFAULT_CSS = """
    TimeframeBar {
        height: 3;
        align: center middle;
    }
    TimeframeBar Button {
        min-width: 8;
        margin: 0 1;
    }
    TimeframeBar Button.tf-selected {
        text-style: bold;
        background: $success;
        color: $text;
        border-top: tall $success;
        border-bottom: tall $success;
    }
    """

    def compose(self) -> ComposeResult:
        for label, seconds in TIMEFRAMES:
            button = Button(label, id=f"tf-{seconds}")
            # Keyboard access is via [ / ] bindings; keep these mouse-only so
            # a stray keyboard focus ring can't be mistaken for the
            # "currently selected timeframe" highlight below.
            button.can_focus = False
            yield button

    def set_active(self, seconds: int) -> None:
        for _, s in TIMEFRAMES:
            self.query_one(f"#tf-{s}", Button).remove_class("tf-selected")
        self.query_one(f"#tf-{seconds}", Button).add_class("tf-selected")


class ProbeRateBar(Horizontal):
    """"- 2000ms +" style control for how often hwmon is polled."""

    DEFAULT_CSS = """
    ProbeRateBar {
        height: 3;
        align: center middle;
    }
    ProbeRateBar Button {
        min-width: 5;
        margin: 0 1;
    }
    ProbeRateBar #probe-rate-label {
        width: auto;
        min-width: 10;
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        dec = Button("-", id="probe-dec")
        dec.can_focus = False
        yield dec
        yield Static(f"{PROBE_INTERVAL_DEFAULT_MS}ms", id="probe-rate-label")
        inc = Button("+", id="probe-inc")
        inc.can_focus = False
        yield inc

    def set_value(self, ms: int) -> None:
        self.query_one("#probe-rate-label", Static).update(f"{ms}ms")


class InferenceUpdated(Message):
    """One llama.cpp snapshot, posted from the poll worker thread.

    Workers use ``post_message`` rather than ``call_from_thread``: posting is
    thread-safe and never blocks, so a worker that happens to finish while
    the app is closing cannot wedge shutdown waiting for a loop that stopped.
    """

    def __init__(self, snapshot: InferenceSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot


class GpuProcessesScanned(Message):
    """Result of one /proc scan, posted from the scan worker thread."""

    def __init__(self, processes: list) -> None:
        super().__init__()
        self.processes = processes


class GpuCtlApp(App):
    """The main GPU Control Panel application."""

    TITLE = "GPU Control Panel"
    SUB_TITLE = "AMD GPU fan control"
    CSS_PATH = "styles.tcss"
    # Textual's built-in command palette (ctrl+p) is unrelated to fan control
    # and its theme picker crashes with ScreenStackError on rapid double-select
    # in this Textual version. Disable it rather than carry that crash risk.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("c", "toggle_curve", "Curve"),
        Binding("p", "cycle_profile", "Profile"),
        Binding("r", "reload_curve", "Reload curve", show=False),
        Binding("a", "set_auto", "Firmware auto"),
        Binding("m", "set_manual", "Manual"),
        Binding("plus", "fan_increase", "Fan +5%"),
        Binding("equals_sign", "fan_increase", "Fan +5%", show=False),
        Binding("minus", "fan_decrease", "Fan -5%"),
        Binding("underscore", "fan_decrease", "Fan -5%", show=False),
        Binding("s", "set_speed", "Set %"),
        Binding("left_square_bracket", "timeframe_prev", "Timeframe -"),
        Binding("right_square_bracket", "timeframe_next", "Timeframe +"),
        Binding("v", "toggle_processes", "Processes"),
        Binding("i", "toggle_inference", "Inference"),
        Binding("0", "jump_pct(0)", "0%", show=False),
        Binding("1", "jump_pct(10)", "10%", show=False),
        Binding("2", "jump_pct(20)", "20%", show=False),
        Binding("3", "jump_pct(30)", "30%", show=False),
        Binding("4", "jump_pct(40)", "40%", show=False),
        Binding("5", "jump_pct(50)", "50%", show=False),
        Binding("6", "jump_pct(60)", "60%", show=False),
        Binding("7", "jump_pct(70)", "70%", show=False),
        Binding("8", "jump_pct(80)", "80%", show=False),
        Binding("9", "jump_pct(90)", "90%", show=False),
    ]

    timeframe_seconds: reactive[int] = reactive(TIMEFRAMES[1][1])  # default 5m
    probe_interval_ms: reactive[int] = reactive(PROBE_INTERVAL_DEFAULT_MS)
    show_processes: reactive[bool] = reactive(False)
    show_inference: reactive[bool] = reactive(True)

    def __init__(self, hwmon: HwmonSource, debug: bool = False) -> None:
        super().__init__()
        self.hwmon = hwmon
        self.debug_mode = debug
        self.samples: deque[Sample] = deque(maxlen=MAX_SAMPLES)
        self.sudo_password: str | None = None
        self._poll_timer = None
        self.timeframe_seconds = self._load_last_timeframe()
        self.probe_interval_ms = self._load_probe_interval()
        self.gpu_process_monitor = GpuProcessMonitor()
        self._scanning_processes = False
        self.inference = LlamaMonitor()
        self._inference_busy = False
        self._inference_seen = False
        self._inference_timer = None
        self.curve_config, self._curve_config_error = load_config()
        self.fan_helper: RootFanHelper | None = None
        self.curve: FanCurveController | None = None
        self._curve_timer = None
        self._curve_busy = False
        self._last_pwm_written: int | None = None

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield InferenceBar(id="inference-bar")
        with Horizontal(id="body"):
            with Vertical(id="graph-panel"):
                with Horizontal(id="graph-toolbar"):
                    yield TimeframeBar(id="timeframe-bar")
                    yield ProbeRateBar(id="probe-rate-bar")
                yield ChartView(id="graph")
            with Vertical(id="sidebar"):
                yield TempPanel(id="temp-panel")
                yield FanPanel(id="fan-panel")
                yield PowerPanel(id="power-panel")
        yield ProcessPanel(id="process-panel")
        with Vertical(id="control-bar"):
            yield Static(
                "c = fan curve on/off  |  p = curve profile  |  a = firmware auto  |  "
                "0-9 / s = manual %  |  [ / ] = timeframe  |  v = processes  |  "
                "i = inference tiles",
                id="control-hint",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(TimeframeBar).set_active(self.timeframe_seconds)
        self.query_one(ProbeRateBar).set_value(self.probe_interval_ms)
        self.query_one(ProcessPanel).display = self.show_processes
        self.query_one(FanPanel).set_profile(self.curve_config.active)
        # Hidden until a llama.cpp server answers, so a fan-only user does
        # not lose six rows to empty tiles. Polling continues while hidden.
        self.query_one(InferenceBar).display = False
        if self._curve_config_error:
            self.notify(
                f"Fan curve config not loaded, using defaults: {self._curve_config_error}",
                severity="error",
                timeout=10,
            )
        if self.curve_config.autostart:
            self._autostart_curve()
        if self.debug_mode:
            self.notify(f"hwmon: {self.hwmon.hwmon_dir}", timeout=5)
        self._poll_once()
        self._restart_poll_timer()
        self._schedule_inference()
        self._inference_timer = self.set_interval(
            INFERENCE_INTERVAL_S, self._schedule_inference
        )

    # -------------------------------------------------------------- polling

    def _restart_poll_timer(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._poll_timer = self.set_interval(
            self.probe_interval_ms / 1000, self._poll_once
        )

    def _poll_once(self) -> None:
        try:
            sample = self.hwmon.read_sample()
        except OSError:
            return
        self.samples.append(sample)
        self._update_panels(sample)
        self._update_graph()
        if self.show_processes:
            self._scan_processes()

    def _update_panels(self, sample: Sample) -> None:
        limits = self.hwmon.limits
        self.query_one(TempPanel).update_sample(sample, limits)
        pct = self.hwmon.pwm_to_pct(sample.pwm or 0)
        self.query_one(FanPanel).update_sample(
            sample,
            pct,
            curve_target=self.curve.pct if self.curve else None,
            curve_emergency=self.curve.emergency if self.curve else False,
        )
        self.query_one(PowerPanel).update_sample(sample, limits)

    def _update_graph(self) -> None:
        graph = self.query_one("#graph", ChartView)

        now = time.monotonic()
        window = self.timeframe_seconds
        pts = [s for s in self.samples if now - s.t <= window]

        series: list[Series] = []
        if len(pts) >= 2:
            for name, pick in (
                ("edge", lambda p: p.edge_c),
                ("junction", lambda p: p.junction_c),
                ("mem", lambda p: p.mem_c),
            ):
                points = [
                    (p.t, value) for p in pts if (value := pick(p)) is not None
                ]
                if len(points) >= 2:
                    series.append(
                        Series(points, TEMP_COLORS[name], TEMP_FILL_COLORS[name])
                    )

        junction_crit = self.hwmon.limits.temp_crit.get("junction")
        top = max(90.0, (junction_crit or 100) + 10)

        title = Text("GPU Temperatures   ", style="bold")
        for name in ("edge", "junction", "mem"):
            title.append("● ", style=TEMP_COLORS[name])
            title.append(f"{name}  ", style="grey62")

        label = self._timeframe_label()
        # One dot column per half cell; snapping the window to that grid is
        # what makes the history scroll left instead of being re-interpolated
        # from scratch on every probe tick.
        x_range = snap_window(now, window, graph.size.width * 2)
        graph.set_series(
            series,
            x_range=x_range,
            y_range=(0.0, top),
            title=title,
            ref_line=(junction_crit, "red3") if junction_crit else None,
            x_labels=(
                (x_range[0], f"-{label}"),
                (x_range[0] + (x_range[1] - x_range[0]) / 2, f"-{_short_duration(window // 2)}"),
                (x_range[1], "now"),
            ),
            y_fmt=lambda value: f"{value:.0f}",
        )

    def _timeframe_label(self) -> str:
        for label, seconds in TIMEFRAMES:
            if seconds == self.timeframe_seconds:
                return label
        return f"{self.timeframe_seconds}s"

    # ------------------------------------------------------------- inference

    def _schedule_inference(self) -> None:
        """Ask for one slot snapshot, in a thread so the UI never waits."""
        if self._inference_busy:
            return
        self._inference_busy = True
        try:
            self.run_worker(
                self._inference_worker, name="llama-poll", group="llama", thread=True
            )
        except Exception:
            # Worker submission can fail while the app is shutting down; do
            # not leave the guard set forever.
            self._inference_busy = False

    def _inference_worker(self) -> None:
        try:
            snapshot = self.inference.poll()
        except Exception as exc:  # a broken endpoint must not stop the poller
            snapshot = InferenceSnapshot(ok=False, error=str(exc))
        self._inference_busy = False
        # Thread-safe and non-blocking; a no-op if the app already closed.
        self.post_message(InferenceUpdated(snapshot))

    def on_inference_updated(self, message: InferenceUpdated) -> None:
        self._apply_inference(message.snapshot)

    def _apply_inference(self, snapshot: InferenceSnapshot) -> None:
        try:
            bar = self.query_one(InferenceBar)
        except Exception:
            return  # not mounted yet
        if not snapshot.ok and not self._inference_seen:
            # No server yet: keep the strip hidden and skip painting tiles
            # nobody can see. Discovery keeps running in the worker.
            return
        bar.update_snapshot(snapshot)
        self._inference_seen = True
        self._apply_inference_visibility()

    def _apply_inference_visibility(self) -> None:
        try:
            bar = self.query_one(InferenceBar)
        except Exception:
            return
        bar.display = self.show_inference and self._inference_seen

    def action_toggle_inference(self) -> None:
        self.show_inference = not self.show_inference
        if self.show_inference and not self._inference_seen:
            # Discovery does network and /proc work; the poller runs it in a
            # worker, so only nudge it instead of probing from here.
            self.notify("Looking for a llama.cpp server...", timeout=3)
            self._schedule_inference()
        self._apply_inference_visibility()

    # ------------------------------------------------------------ timeframe

    def watch_timeframe_seconds(self, seconds: int) -> None:
        try:
            self.query_one(TimeframeBar).set_active(seconds)
        except Exception:
            pass  # widget not mounted yet
        self._save_last_timeframe(seconds)

    def action_timeframe_next(self) -> None:
        self._cycle_timeframe(1)

    def action_timeframe_prev(self) -> None:
        self._cycle_timeframe(-1)

    def _cycle_timeframe(self, direction: int) -> None:
        seconds_list = [s for _, s in TIMEFRAMES]
        idx = seconds_list.index(self.timeframe_seconds)
        idx = (idx + direction) % len(seconds_list)
        self.timeframe_seconds = seconds_list[idx]
        self._update_graph()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("tf-"):
            self.timeframe_seconds = int(event.button.id.removeprefix("tf-"))
            self._update_graph()
        elif event.button.id and event.button.id.startswith("profile-"):
            self._select_profile(event.button.id.removeprefix("profile-"))
        elif event.button.id == "probe-inc":
            self._adjust_probe_interval(PROBE_INTERVAL_STEP_MS)
        elif event.button.id == "probe-dec":
            self._adjust_probe_interval(-PROBE_INTERVAL_STEP_MS)

    @staticmethod
    def _load_last_timeframe() -> int:
        try:
            seconds = int(STATE_FILE.read_text().strip())
            if seconds in [s for _, s in TIMEFRAMES]:
                return seconds
        except (OSError, ValueError):
            pass
        return TIMEFRAMES[1][1]

    @staticmethod
    def _save_last_timeframe(seconds: int) -> None:
        try:
            STATE_FILE.write_text(str(seconds))
        except OSError:
            pass

    # ------------------------------------------------------------ probe rate

    def watch_probe_interval_ms(self, ms: int) -> None:
        try:
            self.query_one(ProbeRateBar).set_value(ms)
        except Exception:
            pass  # widget not mounted yet
        if self._poll_timer is not None:
            self._restart_poll_timer()
        self._save_probe_interval(ms)

    def _adjust_probe_interval(self, delta_ms: int) -> None:
        new_ms = self.probe_interval_ms + delta_ms
        new_ms = max(PROBE_INTERVAL_MIN_MS, min(PROBE_INTERVAL_MAX_MS, new_ms))
        self.probe_interval_ms = new_ms

    @staticmethod
    def _load_probe_interval() -> int:
        try:
            ms = int(PROBE_STATE_FILE.read_text().strip())
            if PROBE_INTERVAL_MIN_MS <= ms <= PROBE_INTERVAL_MAX_MS:
                return ms
        except (OSError, ValueError):
            pass
        return PROBE_INTERVAL_DEFAULT_MS

    @staticmethod
    def _save_probe_interval(ms: int) -> None:
        try:
            PROBE_STATE_FILE.write_text(str(ms))
        except OSError:
            pass

    # -------------------------------------------------------------- processes

    def action_toggle_processes(self) -> None:
        self.show_processes = not self.show_processes

    def watch_show_processes(self, visible: bool) -> None:
        try:
            self.query_one(ProcessPanel).display = visible
        except Exception:
            pass  # widget not mounted yet
        if visible:
            self._scan_processes()

    def _scan_processes(self) -> None:
        # One pass over /proc takes ~25ms here. That is short but it is still
        # the UI thread, so it runs in a worker and the result is handed
        # back through the event loop.
        if self._scanning_processes:
            return
        self._scanning_processes = True
        try:
            self.run_worker(
                self._scan_processes_worker, name="proc-scan", group="proc", thread=True
            )
        except Exception:
            self._scanning_processes = False

    def _scan_processes_worker(self) -> None:
        try:
            processes = self.gpu_process_monitor.scan(
                self.hwmon.pci_slot, self.hwmon.card_index
            )
        except Exception:
            processes = []
        self._scanning_processes = False
        self.post_message(GpuProcessesScanned(processes))

    def on_gpu_processes_scanned(self, message: GpuProcessesScanned) -> None:
        self._apply_processes(message.processes)

    def _apply_processes(self, processes: list) -> None:
        try:
            self.query_one(ProcessPanel).update_processes(processes)
        except Exception:
            pass  # widget not mounted / not visible yet

    # ------------------------------------------------------------- fan control

    def _current_pwm_enable(self) -> int | None:
        return self.samples[-1].pwm_enable if self.samples else None

    def _current_pct(self) -> float:
        if not self.samples:
            return 0.0
        return self.hwmon.pwm_to_pct(self.samples[-1].pwm or 0)

    @work
    async def action_set_auto(self) -> None:
        if self.fan_helper is not None:
            # Stopping the helper is what restores firmware auto.
            await self._stop_curve("Switched to firmware auto")
            return
        await self._set_pwm_enable(2, "Switched to firmware auto")

    @work
    async def action_set_manual(self) -> None:
        await self._leave_curve()
        await self._set_pwm_enable(1, "Switched to manual")

    async def _set_pwm_enable(self, mode: int, description: str) -> bool:
        return await self._privileged_write(
            self.hwmon.pwm1_enable_path, str(mode), description
        )

    @work
    async def action_fan_increase(self) -> None:
        await self._adjust_fan(5)

    @work
    async def action_fan_decrease(self) -> None:
        await self._adjust_fan(-5)

    async def _adjust_fan(self, delta_pct: float) -> None:
        await self._leave_curve()
        if self._current_pwm_enable() != 1:
            if not await self._set_pwm_enable(1, "Switched to manual"):
                return
        new_pct = max(0.0, min(100.0, self._current_pct() + delta_pct))
        await self._set_fan_pct(new_pct)

    @work
    async def action_jump_pct(self, pct: int) -> None:
        await self._leave_curve()
        if self._current_pwm_enable() != 1:
            if not await self._set_pwm_enable(1, "Switched to manual"):
                return
        await self._set_fan_pct(float(pct))

    @work
    async def action_set_speed(self) -> None:
        pct = await self.push_screen_wait(SetSpeedModal())
        if pct is None:
            return
        await self._leave_curve()
        if self._current_pwm_enable() != 1:
            if not await self._set_pwm_enable(1, "Switched to manual"):
                return
        await self._set_fan_pct(float(pct))

    async def _set_fan_pct(self, pct: float) -> bool:
        pwm_val = self.hwmon.pct_to_pwm(pct)
        return await self._privileged_write(
            self.hwmon.pwm1_path, str(pwm_val), f"Fan set to {round(pct)}%"
        )

    async def _ensure_password(
        self,
        cancel_message: str = "Cancelled - sudo password needed to change fan settings.",
    ) -> bool:
        if self.sudo_password is None:
            password = await self.push_screen_wait(PasswordModal())
            if password is None:
                self.notify(cancel_message, severity="warning", timeout=6)
                return False
            self.sudo_password = password
        return True

    async def _privileged_write(self, path: str, value: str, description: str) -> bool:
        if not await self._ensure_password():
            return False

        result = await sudo_write(path, value, self.sudo_password)
        if not result.ok:
            if "password" in (result.message or "").lower():
                self.sudo_password = None
            self.notify(f"{description} failed: {result.message}", severity="error", timeout=6)
            return False

        self.notify(description, timeout=2)
        self._poll_once()  # refresh immediately so the UI reflects the change
        return True

    # --------------------------------------------------------------- fan curve

    @work
    async def action_toggle_curve(self) -> None:
        if self.fan_helper is not None:
            await self._stop_curve("Fan curve off, back on firmware auto")
        else:
            await self._start_curve()

    @work
    async def _autostart_curve(self) -> None:
        # Ask here so a cancel gives one notice that says what happens next.
        # Other start failures are reported by _start_curve itself.
        if await self._ensure_password(
            "No password, so the fan curve is off and the fan stays on "
            "firmware auto. Press c to start it."
        ):
            await self._start_curve()

    def action_cycle_profile(self) -> None:
        self._select_profile(self.curve_config.next_profile())

    def _select_profile(self, name: str) -> None:
        if name not in self.curve_config.profiles:
            return
        self.curve_config.active = name
        save_config(self.curve_config)
        self.query_one(FanPanel).set_profile(name)
        if self.curve is not None:
            self.curve.set_profile(
                self.curve_config.profile, self.curve_config.emergency_margin_c
            )
            self.notify(f"Fan curve profile: {name}", timeout=2)
        else:
            self.notify(f"Fan curve profile: {name} (press c to run it)", timeout=3)

    def action_reload_curve(self) -> None:
        config, error = load_config()
        if error:
            self.notify(f"Fan curve config not reloaded: {error}", severity="error", timeout=10)
            return
        self.curve_config = config
        self._select_profile(config.active)

    async def _start_curve(self) -> bool:
        if self.fan_helper is not None:
            return True
        if not await self._ensure_password():
            return False
        helper, result = await RootFanHelper.start(
            self.hwmon.pwm1_enable_path, self.hwmon.pwm1_path, self.sudo_password
        )
        if helper is None:
            if "password" in (result.message or "").lower():
                self.sudo_password = None
            self.notify(f"Fan curve failed to start: {result.message}", severity="error", timeout=6)
            return False

        self.fan_helper = helper
        self.curve = FanCurveController(
            self.curve_config.profile,
            crit=self.hwmon.limits.temp_crit,
            emergency_margin_c=self.curve_config.emergency_margin_c,
            start_pct=self._current_pct(),
        )
        self._last_pwm_written = None
        self._curve_timer = self.set_interval(CURVE_INTERVAL_S, self._curve_tick)
        await self._curve_tick()
        self.notify(f"Fan curve on ({self.curve_config.active})", timeout=2)
        return True

    async def _stop_curve(self, message: str | None, severity: str = "information") -> None:
        helper, self.fan_helper = self.fan_helper, None
        self.curve = None
        if self._curve_timer is not None:
            self._curve_timer.stop()
            self._curve_timer = None
        if helper is not None:
            await helper.stop()
        if message:
            self.notify(message, severity=severity, timeout=4 if severity == "information" else 8)
        self._poll_once()

    async def _leave_curve(self) -> None:
        """Manual fan actions take over from the curve."""
        if self.fan_helper is not None:
            await self._stop_curve(None)

    async def _curve_tick(self) -> None:
        helper, curve = self.fan_helper, self.curve
        if helper is None or curve is None or self._curve_busy:
            return
        self._curve_busy = True
        try:
            if not helper.alive:
                await self._stop_curve(
                    "Fan curve helper exited. The fan is on firmware auto.", "error"
                )
                return
            try:
                sample = self.hwmon.read_sample()
            except OSError:
                sample = None
            if sample is not None and sample.pwm_enable != 1:
                await self._stop_curve(
                    "Fan mode was changed outside GPU Control Panel. Fan curve stopped.", "warning"
                )
                return

            readings = {
                "edge": sample.edge_c if sample else None,
                "junction": sample.junction_c if sample else None,
                "mem": sample.mem_c if sample else None,
            }
            pct = curve.step(time.monotonic(), readings)
            pwm = self.hwmon.pct_to_pwm(pct)
            drifted = (
                sample is not None
                and sample.pwm is not None
                and abs(sample.pwm - pwm) > PWM_DRIFT_TOLERANCE
            )
            if pwm != self._last_pwm_written or drifted:
                if await helper.write_pwm(pwm):
                    self._last_pwm_written = pwm
        finally:
            self._curve_busy = False

    async def action_quit(self) -> None:
        if self.fan_helper is not None:
            await self._stop_curve(None)
        self.exit()


def main(debug: bool = False) -> None:
    try:
        hwmon = HwmonSource.discover()
    except HwmonNotFoundError as exc:
        HwmonMissingApp(str(exc)).run()
        return

    GpuCtlApp(hwmon, debug=debug).run()


if __name__ == "__main__":
    import sys

    main(debug="--debug" in sys.argv)
