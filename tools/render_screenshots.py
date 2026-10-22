#!/usr/bin/env python3
"""Render the README screenshots from synthetic data.

The app is driven by a fake hwmon backend and synthetic llama.cpp
snapshots, so the images never contain readings, paths, process lists or
model names from the machine that renders them. No GPU, no llama.cpp
server and no root access are involved.

Run from the repository root:

    python tools/render_screenshots.py
    python tools/render_screenshots.py --svg     # keep the SVG sources too

Needs ``rsvg-convert`` (librsvg) to turn the Textual SVG export into PNG.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lact_tui.app as app_module  # noqa: E402
import lact_tui.fancurve as fancurve  # noqa: E402
from lact_tui.app import LactApp  # noqa: E402
from lact_tui.fancurve import CurveConfig  # noqa: E402
from lact_tui.gpu_processes import GpuProcess  # noqa: E402
from lact_tui.hwmon import GpuLimits, Sample  # noqa: E402
from lact_tui.inference import InferenceBar  # noqa: E402
from lact_tui.llama import InferenceSnapshot  # noqa: E402
from lact_tui.widgets import ProcessPanel  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402

ASSETS = ROOT / "assets"
SCREEN = (172, 48)
SCREEN_PROCESSES = (172, 62)
TILES = (128, 6)
KEEP_SVG = False

# Everything below is invented; keep it generic so the screenshots stay
# portable and leak nothing about the machine that produced them.
MODEL = "Qwen4-27B-Q4_K_M"
ORIGIN = "llama-server"
CTX_TOTAL = 98_304
GIB = 1024**3


class DemoHwmon:
    """A hardware backend that only supplies limits; readings are injected."""

    hwmon_dir = Path("/sys/class/drm/card0/device/hwmon/hwmon0")
    pci_slot = "0000:03:00.0"
    limits = GpuLimits(
        pwm_min=0,
        pwm_max=255,
        fan_min=0,
        fan_max=3400,
        temp_labels={"temp1": "edge", "temp2": "junction", "temp3": "mem"},
        temp_crit={"edge": 100.0, "junction": 110.0, "mem": 95.0},
        power_cap=300.0,
        power_cap_min=100.0,
        power_cap_max=340.0,
    )

    def read_sample(self) -> Sample:  # pragma: no cover - polls are disabled
        raise RuntimeError("the screenshot renderer has no sensor backend")

    @staticmethod
    def pwm_to_pct(pwm: int) -> float:
        return max(0.0, min(100.0, pwm / 255 * 100))

    @staticmethod
    def pct_to_pwm(pct: float) -> int:
        return round(max(0.0, min(100.0, pct)) * 255 / 100)


def _sample(t: float, edge: float, junction: float, mem: float, pct: float) -> Sample:
    return Sample(
        t=t,
        edge_c=round(edge, 1),
        junction_c=round(junction, 1),
        mem_c=round(mem, 1),
        fan_rpm=round(540 + pct * 21.4),
        pwm=round(pct * 255 / 100),
        pwm_enable=1,
        gpu_busy_pct=99,
        power_w=267.4,
        vram_used=19_327_352_832,
        vram_total=25_769_803_776,
        gtt_used=2_411_724_800,
        gtt_total=17_179_869_184,
        sclk_mhz=2418,
        mclk_mhz=1000,
    )


def _history(now: float, points: int = 150, step: float = 2.0) -> list[Sample]:
    """Five minutes at 2 s: idle, a ramp into a long inference session."""
    rng = random.Random(731)
    out: list[Sample] = []
    for index in range(points):
        x = index / (points - 1)
        if x < 0.16:
            load = 0.04
        elif x < 0.30:
            load = 0.04 + (x - 0.16) / 0.14 * 0.96
        elif x < 0.88:
            load = 1.0 - 0.09 * math.sin((x - 0.3) * 12.0)
        else:
            load = 0.72 + 0.28 * (x - 0.88) / 0.12
        jitter = rng.uniform(-0.5, 0.5)
        t = now - (points - 1 - index) * step
        out.append(
            _sample(
                t,
                55.0 + 9.2 * load + jitter,
                64.0 + 15.3 * load + 1.3 * jitter,
                53.0 + 6.1 * load + 0.7 * jitter,
                52.0,
            )
        )
    return out


def _inference(
    now: float, count: int = 236, step: float = 0.5
) -> list[tuple[float, InferenceSnapshot]]:
    """Two minutes of decode with three short prefill bursts."""
    rng = random.Random(5150)
    start = now - (count - 1) * step
    bursts = {38, 39, 40, 122, 123, 124, 125, 196, 197, 198}
    snapshots: list[tuple[float, InferenceSnapshot]] = []
    for index in range(count):
        frac = index / (count - 1)
        used = round(CTX_TOTAL * (0.575 + 0.05 * frac))
        common = dict(
            ok=True,
            model=MODEL,
            origin=ORIGIN,
            prompt_tokens=used,
            prompt_processed=used,
            prompt_cached=used * 3 // 4,
            prompt_total=used,
            ctx_used=used,
            ctx_total=CTX_TOTAL,
            requests=9 + index // 24,
        )
        if index in bursts:
            rate = rng.uniform(1620.0, 1910.0)
            snapshot = InferenceSnapshot(
                state="prefill",
                prefill_tps=rate,
                last_prefill_tps=rate,
                last_gen_tps=34.1,
                decoded_tokens=1180,
                **common,
            )
        else:
            rate = 33.4 + 2.4 * math.sin(index / 8.5) + rng.uniform(-1.3, 1.3)
            snapshot = InferenceSnapshot(
                state="generate",
                gen_tps=rate,
                last_prefill_tps=1842.0,
                last_gen_tps=rate,
                decoded_tokens=(index - 44) * 12 if index > 44 else 96,
                **common,
            )
        snapshots.append((start + index * step, snapshot))
    # The frame that is on screen when the shot is taken.
    snapshots.append(
        (
            now,
            InferenceSnapshot(
                ok=True,
                model=MODEL,
                origin=ORIGIN,
                state="generate",
                gen_tps=34.6,
                last_prefill_tps=1842.0,
                last_gen_tps=34.6,
                prompt_tokens=61_440,
                prompt_processed=61_440,
                prompt_cached=49_152,
                prompt_total=61_440,
                decoded_tokens=3136,
                ctx_used=61_440,
                ctx_total=CTX_TOTAL,
                requests=13,
            ),
        )
    )
    return snapshots


def _processes() -> list[GpuProcess]:
    return [
        GpuProcess(
            pid=1842,
            user="dev",
            dev=0,
            type_label="Compute",
            gpu_pct=97.0,
            vram_kib=18_380_800,
            gtt_kib=2_236_416,
            cpu_pct=11.0,
            host_mem_kib=2_797_568,
            command=f"llama-server -m {MODEL}.gguf --port 8080",
        ),
        GpuProcess(
            pid=1103,
            user="dev",
            dev=0,
            type_label="Graphic",
            gpu_pct=3.0,
            vram_kib=524_288,
            gtt_kib=196_608,
            cpu_pct=2.0,
            host_mem_kib=262_144,
            command="Xorg -nolisten tcp",
        ),
        GpuProcess(
            pid=2987,
            user="dev",
            dev=0,
            type_label="Graphic",
            gpu_pct=1.0,
            vram_kib=389_120,
            gtt_kib=98_304,
            cpu_pct=4.0,
            host_mem_kib=634_880,
            command="firefox",
        ),
    ]


def _prepare(app: LactApp) -> None:
    """Freeze timers and stop the app from touching the user's state files."""
    for timer in (app._poll_timer, app._inference_timer):
        if timer is not None:
            timer.stop()


def _freeze_state_files() -> None:
    """Read fixed values and never write the real state files.

    The app normally remembers the timeframe, the probe interval and the
    selected curve profile between runs; a documentation render must not
    overwrite any of them.
    """
    LactApp._load_last_timeframe = staticmethod(lambda: 300)  # type: ignore[method-assign]
    LactApp._load_probe_interval = staticmethod(lambda: 2000)  # type: ignore[method-assign]
    LactApp._save_last_timeframe = staticmethod(lambda seconds: None)  # type: ignore[method-assign]
    LactApp._save_probe_interval = staticmethod(lambda ms: None)  # type: ignore[method-assign]


def _new_app() -> LactApp:
    _freeze_state_files()
    app_module.load_config = lambda: (
        CurveConfig(autostart=False, active="balanced"),
        None,
    )
    # A render must never rewrite the local curve config: pin the real ones.
    app_module.save_config = lambda config: None  # type: ignore[assignment]
    fancurve.save_config = app_module.save_config  # type: ignore[assignment]
    app = LactApp(DemoHwmon())  # type: ignore[arg-type]
    app._poll_once = lambda: None  # type: ignore[method-assign]
    app._schedule_inference = lambda: None  # type: ignore[method-assign]
    app._scan_processes = lambda: None  # type: ignore[method-assign]
    return app


def _fill(app: LactApp) -> None:
    """Put the synthetic history, readings and inference snapshots on screen."""
    now = time.monotonic()
    app.samples.clear()
    app.samples.extend(_history(now))
    app.curve = SimpleNamespace(pct=52.0, emergency=False)  # type: ignore[assignment]
    app._update_panels(app.samples[-1])
    app._update_graph()

    bar = app.query_one(InferenceBar)
    for timestamp, snapshot in _inference(now):
        bar.update_snapshot(snapshot, now=timestamp)
    app._inference_seen = True
    app._apply_inference_visibility()


async def _render_overview() -> None:
    app = _new_app()
    async with app.run_test(size=SCREEN) as pilot:
        _prepare(app)
        _fill(app)
        await pilot.pause()
        await pilot.pause()
        _save("overview", app)


async def _render_processes() -> None:
    app = _new_app()
    async with app.run_test(size=SCREEN_PROCESSES) as pilot:
        _prepare(app)
        _fill(app)
        app.show_processes = True
        app.query_one(ProcessPanel).update_processes(_processes())
        await pilot.pause()
        await pilot.pause()
        _save("processes", app)


class TilesOnly(App[None]):
    """Just the inference strip, for a legible close-up."""

    CSS = "Screen { background: $background; }"

    def compose(self) -> ComposeResult:
        yield InferenceBar()


async def _render_tiles() -> None:
    app = TilesOnly()
    async with app.run_test(size=TILES) as pilot:
        now = time.monotonic()
        bar = app.query_one(InferenceBar)
        for timestamp, snapshot in _inference(now):
            bar.update_snapshot(snapshot, now=timestamp)
        await pilot.pause()
        await pilot.pause()
        _save("tiles", app)


def _save(name: str, app: App) -> None:
    ASSETS.mkdir(exist_ok=True)
    svg_path = ASSETS / f"{name}.svg"
    svg_path.write_text(app.export_screenshot(title="lact-tui"))
    png_path = ASSETS / f"{name}.png"
    subprocess.run(
        ["rsvg-convert", "-o", str(png_path), str(svg_path)],
        check=True,
    )
    if not KEEP_SVG:
        svg_path.unlink()
    print(f"wrote {png_path.relative_to(ROOT)}")


async def main() -> None:
    global KEEP_SVG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", action="store_true", help="keep the SVG sources")
    KEEP_SVG = parser.parse_args().svg

    if shutil.which("rsvg-convert") is None:
        raise SystemExit("rsvg-convert not found (install librsvg2-bin)")
    print(f"rendering at {datetime.now():%H:%M:%S}")
    await _render_overview()
    await _render_processes()
    await _render_tiles()


if __name__ == "__main__":
    asyncio.run(main())
