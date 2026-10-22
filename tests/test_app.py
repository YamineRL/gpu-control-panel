"""Headless checks: app boots, graph renders, clicks switch timeframe fast.

Run with:  PYTHONPATH=. .venv/bin/python tests/test_app.py
       or: PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

The app checks need an amdgpu hwmon node and skip themselves without one;
the chart and inference-rate checks run anywhere.
"""

from __future__ import annotations

import asyncio
import time

from gpuctl.app import GpuCtlApp
from gpuctl.graphs import Series, render_chart, render_sparkline, snap_window
from gpuctl.hwmon import HwmonNotFoundError, HwmonSource
from gpuctl.llama import InferenceSnapshot


def test_snapped_window_scrolls() -> None:
    """The window must snap to the dot grid, or history gets re-plotted.

    With a free-running window ("now - 5m, now") every column's centre moves
    by a fraction of a column on every refresh, so the interpolation weights
    for the whole history change each tick and the curve boils. Snapping the
    right edge to the grid pins column centres to fixed clock times, so a
    column only changes when a new sample actually lands in it.
    """
    window = 300.0
    columns = 260  # dot columns for a 130-cell-wide graph
    step = window / columns
    base = 1000.0  # arbitrary absolute clock

    # Samples every 2 s, the default probe rate.
    points = [(base - window + i * 2.0, 60.0 + 10.0 * ((i * 7) % 11 / 10)) for i in range(151)]

    def column_values(t_now: float, snapped: bool) -> str:
        x0, x1 = (
            snap_window(t_now, window, columns) if snapped else (t_now - window, t_now)
        )
        return render_chart(
            [Series(points, "cyan", (0, 46, 46))],
            columns // 2,
            20,
            x_range=(x0, x1),
            y_range=(0.0, 120.0),
            gutter=False,
            x_axis=False,
        ).plain

    # A tick from an unaligned instant: the worst case for a free window.
    # The data must be a *sloped* line for a re-interpolation to show up; a
    # flat series interpolates to the same value and would prove nothing.
    points.clear()
    points.extend(
        ((base - window + i * 2.0), 20.0 + 0.35 * i) for i in range(151)
    )

    t0 = base + 0.37 * step
    snapped_diff = sum(
        a != b
        for a, b in zip(column_values(t0, True), column_values(t0 + 2.0, True))
    )
    free_diff = sum(
        a != b
        for a, b in zip(column_values(t0, False), column_values(t0 + 2.0, False))
    )
    print(f"    tick moved: snapped {snapped_diff} chars, free {free_diff} chars")
    # Unsnapped the whole diagonal is re-drawn; snapped only the columns that
    # the new samples actually entered, plus the rightmost cell.
    assert snapped_diff < free_diff * 0.8, (
        f"snapping did not stabilise history: {snapped_diff} vs {free_diff}"
    )
    # Snapped, only the samples that entered this tick's columns move, so the
    # redrawn area stays a small fraction of the curve.
    assert snapped_diff < columns // 2 * 0.5, f"too much redrawn: {snapped_diff}"
    assert snapped_diff < free_diff, (
        f"free window redrew no more than the snapped one: {free_diff} vs {snapped_diff}"
    )


def test_session_extremes() -> None:
    """Tiles track a session min/max that only a new session resets."""
    from textual.app import App

    from gpuctl.inference import InferenceBar

    bar = InferenceBar()

    def snap(*, gen: float | None, prefill: float | None = None, new_session=False):
        return InferenceSnapshot(
            ok=True,
            model="demo",
            state="generate" if gen is not None else "idle",
            gen_tps=gen,
            prefill_tps=prefill,
            ctx_used=50,
            ctx_total=100,
            new_session=new_session,
        )

    class Harness(App):
        def compose(self):
            yield bar

    async def run() -> None:
        async with Harness().run_test(size=(120, 8)) as pilot:
            for index, rate in enumerate((40.0, 12.0, 31.0)):
                bar.update_snapshot(snap(gen=rate, new_session=index == 0), now=1000.0 + index)
            await pilot.pause()
            assert bar._extremes["decode"] == (12.0, 40.0), bar._extremes
            # A later, narrower value must not shrink the extremes.
            bar.update_snapshot(snap(gen=20.0), now=1004.0)
            await pilot.pause()
            assert bar._extremes["decode"] == (12.0, 40.0), bar._extremes
            # A new session starts clean instead of carrying old numbers.
            bar.update_snapshot(snap(gen=20.0, new_session=True), now=1005.0)
            await pilot.pause()
            assert bar._extremes["decode"] == (20.0, 20.0), bar._extremes

    asyncio.run(run())


def test_graph_render_is_fast() -> None:
    now = time.monotonic()
    series = [
        Series([(now - 300 + i, 60 + i % 20) for i in range(600)], "cyan", (0, 46, 46)),
        Series([(now - 300 + i, 70 + i % 15) for i in range(600)], "#ff8700", (58, 36, 0)),
        Series([(now - 300 + i, 50 + i % 10) for i in range(600)], "magenta", (46, 0, 46)),
    ]
    start = time.perf_counter()
    for _ in range(20):
        render_chart(
            series,
            160,
            24,
            x_range=(now - 300, now),
            y_range=(0.0, 110.0),
            ref_line=(95, "red3"),
            x_labels=((now - 300, "-5m"), (now - 150, "-2m30s"), (now, "now")),
        )
    average_ms = (time.perf_counter() - start) / 20 * 1000
    assert average_ms < 20, f"chart render too slow: {average_ms:.1f}ms"


def test_sparkline_small() -> None:
    now = time.monotonic()
    points = [(now - 120 + i * 0.5, float(i % 40)) for i in range(240)]
    out = render_sparkline(points, 24, 2, x_range=(now - 120, now), line="cyan", fill=(0, 60, 60))
    assert out.plain.count("\n") <= 2


def _slot(*, task: int, processed: int, cached: int, prompt: int, decoded: int, processing: bool = True) -> dict:
    return {
        "id": 0,
        "n_ctx": 32768,
        "is_processing": processing,
        "id_task": task,
        "n_prompt_tokens": prompt,
        "n_prompt_tokens_processed": processed,
        "n_prompt_tokens_cache": cached,
        "next_token": [{"n_decoded": decoded}],
    }


def test_inference_rates() -> None:
    """Prefill/decode rates come from counter deltas over the right window.

    llama.cpp only moves ``n_prompt_tokens_processed`` once per batch, so a
    batch of 1,024 tokens seen after 2.5 s has to read as ~410 tok/s, not
    2,048.
    """
    from gpuctl import llama

    clock = {"t": 100.0}
    payloads: list[list[dict]] = []

    def fetch(target):
        return (payloads.pop(0) if payloads else []), None

    monitor = llama.LlamaMonitor()
    monitor._target = llama.LlamaTarget(base_url="http://fake", model="test.gguf")
    monitor._next_discovery = float("inf")

    original_time = llama.time
    original_fetch = llama.fetch_slots

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return clock["t"]

    llama.time = FakeTime
    llama.fetch_slots = fetch
    try:
        # Prompt of 2,048 new tokens, processed in 1,024-token batches.
        # n_prompt_tokens counts prompt + generated tokens on current
        # llama.cpp builds, so it grows with n_decoded.
        steps = [
            (0.0, _slot(task=1, processed=0, cached=0, prompt=2048, decoded=0)),
            (2.5, _slot(task=1, processed=1024, cached=0, prompt=2048, decoded=0)),
            (5.0, _slot(task=1, processed=2048, cached=0, prompt=2048, decoded=0)),
            # Prefill done (processed + cached == prompt - decoded): decoding.
            (5.5, _slot(task=1, processed=2048, cached=0, prompt=2072, decoded=24)),
            (6.0, _slot(task=1, processed=2048, cached=0, prompt=2092, decoded=44)),
            (6.5, _slot(task=1, processed=2048, cached=0, prompt=2112, decoded=64)),
            (7.0, _slot(task=1, processed=0, cached=0, prompt=0, decoded=0, processing=False)),
        ]
        rate_at: dict[float, tuple[str, float | None, float | None]] = {}
        for moment, slot in steps:
            clock["t"] = 100.0 + moment
            payloads.append([slot])
            snapshot = monitor.poll()
            rate_at[moment] = (snapshot.state, snapshot.prefill_tps, snapshot.gen_tps)

        state, prefill, _gen = rate_at[2.5]
        assert state == "prefill", state
        assert prefill is not None and 400 <= prefill <= 420, prefill

        state, _prefill, gen = rate_at[6.0]
        assert state == "generate", state
        assert gen is not None and 35 <= gen <= 45, gen

        idle = monitor.poll()
        assert idle.state == "idle"
        assert idle.last_prefill_tps is not None and 400 <= idle.last_prefill_tps <= 420
        assert idle.last_gen_tps is not None and 35 <= idle.last_gen_tps <= 45
    finally:
        llama.time = original_time
        llama.fetch_slots = original_fetch


async def _app_checks() -> None:
    try:
        hwmon = HwmonSource.discover()
    except HwmonNotFoundError:
        hwmon = None
    if hwmon is None:
        print("no hwmon source on this machine, skipping app checks")
        return

    # The local config autostarts the fan curve, which opens the sudo modal
    # over everything. Tests want the plain screen.
    import gpuctl.app as app_module
    from gpuctl.app import PROBE_STATE_FILE, STATE_FILE
    from gpuctl.fancurve import CurveConfig

    original = app_module.load_config
    app_module.load_config = lambda: (CurveConfig(autostart=False), None)
    # Clicking a timeframe or a probe button writes the user's remembered
    # state; put it back afterwards.
    saved = {
        path: path.read_text() if path.exists() else None
        for path in (STATE_FILE, PROBE_STATE_FILE)
    }
    try:
        app = GpuCtlApp(hwmon)
    finally:
        app_module.load_config = original
    try:
        await _drive_app(app, hwmon)
    finally:
        for path, content in saved.items():
            try:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(content)
            except OSError:
                pass


async def _drive_app(app: GpuCtlApp, hwmon: HwmonSource) -> None:
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        # Feed a few samples so the graph has data.
        for _ in range(5):
            app.samples.append(hwmon.read_sample())
        app._update_graph()
        await pilot.pause()
        graph = app.query_one("#graph")
        assert graph.size.height > 2

        # The click path that used to block for seconds on a 1h window.
        app.probe_interval_ms = 500
        start = time.perf_counter()
        await pilot.click("#tf-3600")
        await pilot.pause()
        elapsed = time.perf_counter() - start
        assert app.timeframe_seconds == 3600
        assert elapsed < 0.5, f"timeframe click took {elapsed:.2f}s"

        # Inference tiles: feed a snapshot the way the worker would.
        bar = app.query_one("#inference-bar")
        bar.display = True
        app._apply_inference(
            InferenceSnapshot(
                ok=True,
                model="Qwen4-27B-Q4_K_M",
                state="generate",
                gen_tps=41.2,
                last_prefill_tps=1183.0,
                ctx_used=3200,
                ctx_total=98304,
                decoded_tokens=512,
                requests=3,
            )
        )
        await pilot.pause()
        value = bar.query_one("#tile-decode .metric-value").render().plain
        assert "41.2" in value, value
        prefill_value = bar.query_one("#tile-prefill .metric-value").render().plain
        assert "1,183" in prefill_value, prefill_value

        # Pressing i hides the tiles again.
        await pilot.press("i")
        await pilot.pause()
        assert app.show_inference is False


def test_app() -> None:
    asyncio.run(_app_checks())


if __name__ == "__main__":
    test_snapped_window_scrolls()
    test_session_extremes()
    test_graph_render_is_fast()
    test_sparkline_small()
    test_inference_rates()
    test_app()
    print("all checks passed")
