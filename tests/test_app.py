"""Headless checks: app boots, graph renders, clicks switch timeframe fast.

Run with:  PYTHONPATH=. .venv/bin/python tests/test_app.py
       or: PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

The app checks need an amdgpu hwmon node and skip themselves without one;
the chart and inference-rate checks run anywhere.
"""

from __future__ import annotations

import asyncio
import shlex
import tempfile
import time
from pathlib import Path

from gpuctl.app import GpuCtlApp, _graph_range, _graph_span, _time_ticks
from gpuctl.dashboard import Meters
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

    from gpuctl.dashboard import TileGrid
    from gpuctl.hwmon import GpuLimits, PcieLink

    grid = TileGrid()

    def snap(rate: float) -> InferenceSnapshot:
        return InferenceSnapshot(
            ok=True,
            model="demo",
            state="generate",
            gen_tps=rate,
            ctx_used=50,
            ctx_total=100,
        )

    class Harness(App):
        def compose(self):
            yield grid

    async def run() -> None:
        limits = GpuLimits(pwm_min=0, pwm_max=255, fan_min=0, fan_max=0)
        async with Harness().run_test(size=(120, 18)) as pilot:

            def paint(rate: float, at: float) -> None:
                grid.push_inference(snap(rate), at)
                grid.update_tiles(
                    samples=[], hosts=[], limits=limits, link=PcieLink(), now=at
                )

            for index, rate in enumerate((40.0, 12.0, 31.0)):
                paint(rate, 1000.0 + index)
            await pilot.pause()
            assert grid.tile("speed").extremes == (12.0, 40.0), grid.tile("speed").extremes
            # A later, narrower value must not shrink the extremes.
            paint(20.0, 1004.0)
            await pilot.pause()
            assert grid.tile("speed").extremes == (12.0, 40.0)
            # A new session starts clean instead of carrying old numbers.
            grid.start_session(1005.0)
            paint(20.0, 1005.0)
            await pilot.pause()
            assert grid.tile("speed").extremes == (20.0, 20.0)

    asyncio.run(run())


def test_host_readings() -> None:
    """CPU, RAM, swap and disk are read from /proc-style files, not guessed."""
    from gpuctl.host import HostSource

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stat = root / "stat"
        meminfo = root / "meminfo"
        diskstats = root / "diskstats"
        block = root / "block"
        block.mkdir()
        (block / "nvme0n1").mkdir()
        # /sys/block lists whole disks only; the partition line below must be
        # ignored even though /proc/diskstats reports it.

        stat.write_text("cpu  100 0 100 800 0 0 0 0 0 0\n")
        meminfo.write_text(
            "MemTotal:       1000 kB\n"
            "MemAvailable:    250 kB\n"
            "SwapTotal:       500 kB\n"
            "SwapFree:        300 kB\n"
        )
        diskstats.write_text(
            "259       0 nvme0n1 10 0 100 0 5 0 40 0 0 0 0\n"
            "259       1 nvme0n1p1 9 0 90 0 4 0 30 0 0 0 0\n"
        )
        source = HostSource(
            stat_path=str(stat),
            meminfo_path=str(meminfo),
            diskstats_path=str(diskstats),
            block_dir=str(block),
            cpu_root=str(block),
        )
        first = source.read(t=10.0)
        # No previous counters yet: a rate needs two reads.
        assert first.cpu_pct is None
        assert first.disk_read_bps is None
        assert first.ram_used == 750 * 1024
        assert first.ram_total == 1000 * 1024
        assert first.swap_used == 200 * 1024
        assert first.swap_pct == 40.0

        # Two seconds later: 100 busy jiffies of 1000 total, 6000 sectors read
        # (a partition's counters must not be added to its disk's).
        stat.write_text("cpu  200 0 200 1600 0 0 0 0 0 0\n")
        diskstats.write_text(
            "259       0 nvme0n1 10 0 6100 0 5 0 8040 0 0 0 0\n"
            "259       1 nvme0n1p1 9 0 6090 0 4 0 8030 0 0 0 0\n"
        )
        second = source.read(t=12.0)
        assert second.cpu_pct == 20.0, second.cpu_pct
        assert second.disk_read_bps == 1_536_000.0, second.disk_read_bps
        assert second.disk_write_bps == 2_048_000.0, second.disk_write_bps

        # No swap configured: a blank bar, not a division by zero.
        meminfo.write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
        third = source.read(t=14.0)
        assert third.swap_pct is None
        assert third.swap_total is None


def test_pcie_link() -> None:
    """Gen, width, one-way bandwidth and the idle-link flag come from the rate."""
    from gpuctl.hwmon import PcieLink, _parse_link_rate

    assert _parse_link_rate("16.0 GT/s PCIe") == 16.0
    assert _parse_link_rate(None) is None
    full = PcieLink(
        speed_gt=16.0, width=16, max_speed_gt=16.0, max_width=16
    )
    assert full.generation == 4
    assert full.bandwidth_gbs == 32.0
    assert full.downgraded is False
    # PCIe power management drops the link when the card is idle.
    idle = PcieLink(speed_gt=2.5, width=16, max_speed_gt=16.0, max_width=16)
    assert idle.generation == 1
    assert idle.downgraded is True
    assert PcieLink().generation is None
    assert PcieLink().bandwidth_gbs is None


def test_request_history() -> None:
    """Strata's own request list, and its running request, fill the table."""
    from gpuctl.llama import _history_from_metrics

    recent = [
        {
            "time": 1_700_000_000.0,
            "finish": "stop",
            "prompt_tokens": 100,
            "reused": 80,
            "output_tokens": 20,
            "decode_tok_s": 25.0,
            "duration_s": 3.2,
        }
    ]
    idle = _history_from_metrics(recent, "idle", {}, 0, 0)
    assert len(idle) == 1
    assert idle[0].status == "stop"
    assert idle[0].reused_tokens == 80
    assert idle[0].tok_s == 25.0

    live = {"elapsed_s": 4.0, "reused": 5, "tok_s": 30.0}
    busy = _history_from_metrics(recent, "generate", live, 100, 50)
    assert busy[0].status == "running", busy[0]
    assert busy[0].output_tokens == 50
    assert busy[1].status == "stop"


def test_request_table_fits_its_panel() -> None:
    """More room under the meters means more request rows, never a spill."""
    from gpuctl.dashboard import RequestTable
    from gpuctl.llama import RequestRecord
    from textual.app import App
    from textual.widgets import Static

    table = RequestTable()

    class Harness(App):
        def compose(self):
            yield table

    async def run() -> None:
        async with Harness().run_test(size=(56, 12)) as pilot:
            records = tuple(
                RequestRecord(
                    at=1_700_000_000.0 + index,
                    status="stop",
                    prompt_tokens=1000 + index,
                    reused_tokens=900,
                    output_tokens=120,
                    tok_s=24.0,
                    duration_s=5.0,
                )
                for index in range(40)
            )
            table.update_records(records)
            await pilot.pause()
            await pilot.pause()
            content = table.query_one(Static).render().plain
            lines = content.splitlines()
            # Border rows are not content; the panel is 12 tall.
            assert len(lines) <= 12 - 2, len(lines)
            assert len(lines) >= 5, len(lines)

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


def test_chart_body_is_solid() -> None:
    """The area under the curves reaches the axis in solid block cells.

    The half-block rewrite once painted every sub-row of a band with its own
    partial glyph, so each cell ended up as a bottom eighth (``▁``) and the
    lower half of the chart looked empty. A filled body must be ``█`` all the
    way down, not a sprinkle of partials.
    """
    from gpuctl.graphs import Series, render_chart

    now = 100.0
    pts = [(now - 60 + i, 50.0 + (i % 5)) for i in range(61)]
    series = [
        Series(pts, "cyan", (0, 46, 46)),
        Series([(t, value + 8) for t, value in pts], "#ff8700", (58, 36, 0)),
    ]
    chart = render_chart(
        series,
        40,
        12,
        x_range=(now - 60, now),
        y_range=(0.0, 120.0),
        gutter=False,
        x_axis=False,
    )
    rows = chart.plain.splitlines()
    assert set(rows[-1]) == {"█"}, rows[-1]
    assert set(rows[-3]) == {"█"}, rows[-3]


def test_chart_body_has_no_black_gaps() -> None:
    """An inner curve's edge must not punch a hole in the filled body.

    A curve that runs inside a taller curve's body used to be stroked with a
    bare partial block glyph. The empty part of that cell showed the terminal
    background, so the fill read as black spots. The cell has to carry the
    body colour as its background instead.
    """
    from gpuctl.graphs import Series, render_chart

    lo = [(float(i), 40.0) for i in range(11)]
    hi = [(float(i), 90.0) for i in range(11)]
    chart = render_chart(
        [Series(lo, "cyan", (0, 46, 46)), Series(hi, "magenta", (46, 0, 46))],
        24,
        14,
        x_range=(0.0, 10.0),
        y_range=(0.0, 100.0),
        gutter=False,
        x_axis=False,
    )
    lines = chart.plain.split("\n")
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1

    def style_at(row: int, column: int):
        position = offsets[row] + column
        for span in chart.spans:
            if span.start <= position < span.end:
                return span.style
        return None

    checked = 0
    for column in range(len(lines[0])):
        body_top = next(
            (row for row, line in enumerate(lines) if line[column] != " "), None
        )
        if body_top is None:
            continue
        for row in range(body_top + 1, len(lines)):
            if column >= len(lines[row]):
                continue
            cell = lines[row][column]
            if cell in (" ", "\u2588"):
                continue
            style = style_at(row, column)
            assert style is not None and style.bgcolor is not None, (
                row,
                column,
                cell,
            )
            checked += 1
    assert checked, "fixture should have an inner edge inside the body"


def _cell_styles(chart):
    """(plain rows, style_at(row, column)) for a rendered chart."""
    lines = chart.plain.split("\n")
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1

    def style_at(row: int, column: int):
        position = offsets[row] + column
        found = None
        for span in chart.spans:
            if span.start <= position < span.end:
                found = span.style
        return found

    return lines, style_at


def test_lower_curve_keeps_its_own_body() -> None:
    """A curve below another one is drawn as its own ridge, down to the floor.

    The old renderer let the tallest curve's fill win every column, so a
    lower curve was only a stroke floating in someone else's colour. Here
    the cells under the low curve must carry the low curve's colours, and
    the cells between the two curves the high curve's.
    """
    from rich.color import Color

    from gpuctl.graphs import Series, render_chart

    lo = [(float(i), 30.0) for i in range(11)]
    hi = [(float(i), 80.0) for i in range(11)]
    chart = render_chart(
        [Series(hi, "#ff0000", (60, 0, 0)), Series(lo, "#0000ff", (0, 0, 60))],
        20,
        10,
        x_range=(0.0, 10.0),
        y_range=(0.0, 100.0),
        gutter=False,
        x_axis=False,
    )
    lines, style_at = _cell_styles(chart)

    def hue(row: int, column: int) -> str:
        style = style_at(row, column)
        color = style.bgcolor or style.color
        red, _, blue = color.get_truecolor()
        return "red" if red > blue else "blue"

    column = 10
    # 30 of 100 over 10 rows: the low edge sits in row 7, the high one in row 2.
    assert hue(9, column) == "blue", lines
    assert hue(8, column) == "blue", lines
    assert hue(5, column) == "red", lines
    assert hue(3, column) == "red", lines
    assert isinstance(style_at(9, column).bgcolor, Color), "body needs a background"


def test_graph_range_hugs_data() -> None:
    """The y-axis wraps the data tightly, on round numbers.

    The fill used to run to zero on a fixed scale, so a card idling around
    45 C left most of the graph empty.
    """
    for low, high in ((40.0, 45.0), (53.0, 81.0), (70.0, 96.0)):
        y_min, y_max = _graph_range(low, high)
        assert y_min < low and high < y_max, (low, high, y_min, y_max)
        assert y_min % 5 == 0 and y_max % 5 == 0, (y_min, y_max)
        assert (high - low) / (y_max - y_min) >= 0.55 or y_max - y_min == 20.0
    # A critical limit far above the data stays off the axis...
    assert _graph_range(53.0, 81.0, (110.0,))[1] < 100.0
    # ...and comes into view once the peak gets within 10 degrees of it.
    assert _graph_range(53.0, 101.0, (110.0,))[1] >= 110.0


def test_crit_line_is_drawn_in_range() -> None:
    """A threshold inside the y-range shows as a dashed line across the plot."""
    from gpuctl.graphs import Series, Threshold, render_chart

    pts = [(float(i), 90.0 + i) for i in range(11)]
    chart = render_chart(
        [Series(pts, "#ff9a3c", (56, 26, 6))],
        40,
        12,
        x_range=(0.0, 10.0),
        y_range=(70.0, 115.0),
        thresholds=[Threshold(110.0)],
    )
    dashed = [line for line in chart.plain.splitlines() if line.count("\u254c") > 10]
    assert len(dashed) == 1, chart.plain


def test_time_ticks_are_round_and_end_at_now() -> None:
    """Time labels fall on round steps back from "now" and do not crowd."""
    ticks = _time_ticks((1000.0 - 300.0, 1000.0), 120)
    labels = [label for _, label in ticks]
    assert labels[-1] == "now"
    assert labels == ["-5m", "-4m", "-3m", "-2m", "-1m", "now"], labels
    narrow = _time_ticks((1000.0 - 300.0, 1000.0), 40)
    assert len(narrow) <= 40 // 16 + 1, narrow


def test_axes_chart_scrolls_without_boiling() -> None:
    """With axis gutter and end labels, snap=True still scrolls the history.

    The plot is narrower than the widget (axis labels left, value labels
    right). Snapping the window to the widget width instead of the plot
    width moves every column's bucket on each tick, so the curve boils.
    """
    from gpuctl.graphs import Series, render_chart

    base = 1000.0
    # Sloped with a wobble, so any re-sampling moves some edge glyphs.
    points = [
        (base - 300.0 + i * 2.0, 40.0 + 0.2 * i + 1.5 * (i % 3)) for i in range(151)
    ]

    def plot(now: float, snap: bool) -> list[str]:
        chart = render_chart(
            [Series(points, "#38d6f5", (6, 38, 52))],
            90,
            16,
            x_range=(now - 300.0, now),
            y_range=(35.0, 75.0),
            y_fmt=lambda v: f"{v:.0f}\u00b0",
            end_labels=True,
            snap=snap,
        )
        return chart.plain.splitlines()

    def scrolled(before: list[str], after: list[str]) -> bool:
        """True if ``after`` is ``before`` moved left by 0-2 whole columns."""
        rows = range(1, 15)  # plot rows only: skip the axis and labels
        left, right = 4, 80  # inside the axis gutter and the value labels
        return any(
            all(after[row][left : right - shift] == before[row][left + shift : right] for row in rows)
            for shift in (0, 1, 2)
        )

    for start in (base - 0.73, base + 1.1, base + 2.9):
        assert scrolled(plot(start, True), plot(start + 2.0, True)), start
    # Negative control: a free-running window re-samples the history.
    assert not scrolled(plot(base - 0.73, False), plot(base + 1.27, False))


def test_graph_span_fills_width() -> None:
    """A young history is stretched over the whole width, not a sliver."""
    assert _graph_span(1000.0, 300.0, None) == 300.0
    assert _graph_span(1000.0, 300.0, 990.0) == 10.0
    assert _graph_span(1000.0, 300.0, 100.0) == 300.0  # capped at timeframe
    assert _graph_span(1000.0, 300.0, 1000.0) == 1.0  # never a zero-width axis


def test_llama_swap_shell_wrapper() -> None:
    """A ``sh -c '... --port 8092'`` command must yield the real port.

    llama-swap wraps Strata that way, so the flags sit inside one quoted
    argument where an argv parser cannot see them. Guessing the llama.cpp
    default (8080) there produced a dead candidate and an empty tile strip.
    """
    from gpuctl import llama

    cmd = (
        "sh -c 'cd /srv/strata && exec /srv/strata/.venv/bin/python "
        "serve/server.py --engine strata --config strata.json "
        "--port 8092 --api-key test-key'"
    )
    argv = llama._unwrap_shell(shlex.split(cmd))
    target = llama._target_from_argv(argv, origin="llama-swap", default_port=None)
    assert target is not None
    assert target.base_url == "http://127.0.0.1:8092", target
    assert target.engine == "strata", target
    assert target.api_key == "test-key", target

    # No --port and no proxy: return nothing rather than inventing 8080.
    none = llama._target_from_argv(
        shlex.split("python serve/server.py --engine strata"),
        origin="llama-swap",
        default_port=None,
    )
    assert none is None, none

    # A relative --config resolves against the process working directory.
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "cfg.json").write_text('{"api_key": "from-config"}')
        target = llama._target_from_argv(
            ["python", "serve/server.py", "--engine", "strata", "--config", "cfg.json"],
            origin="strata",
            cwd=tmp,
        )
        assert target is not None and target.api_key == "from-config", target


def test_strata_metrics_snapshot() -> None:
    """Strata's /metrics fills the same tiles from its own live rates."""
    from gpuctl import llama

    def metrics(
        state: str,
        *,
        prompt=0,
        read=0,
        generated=0,
        tok_s=None,
        prefill=None,
        last=None,
    ) -> dict:
        return {
            "engine": {"model": "qwen3.8-flash-next-iq3_xxs", "max_context": 98304},
            "live": {
                "state": state,
                "prompt_tokens": prompt if state != "idle" else None,
                "prompt_read": read if state == "reading" else None,
                "prompt_total": prompt if state == "reading" else None,
                "generated": generated if state != "idle" else None,
                "elapsed_s": 10.0,
                "tok_s": tok_s,
                "tok_s_mean": tok_s,
                "prefill_tok_s_mean": prefill,
            },
            "requests": [last] if last else [],
        }

    payloads = [
        metrics("reading", prompt=2048, read=1024, prefill=410.0),
        metrics("generating", prompt=2064, generated=16, tok_s=42.5, prefill=410.0),
        metrics(
            "idle",
            last={
                "prompt_tokens": 2048,
                "prompt_total": 2048,
                "output_tokens": 96,
                "decode_tok_s": 40.0,
            },
        ),
    ]
    monitor = llama.LlamaMonitor()
    monitor._target = llama.LlamaTarget(
        base_url="http://fake", origin="strata", engine="strata"
    )
    monitor._next_discovery = float("inf")
    original = llama.fetch_strata_metrics
    llama.fetch_strata_metrics = lambda target: (payloads.pop(0), None)
    try:
        first = monitor.poll()
        assert first.ok and first.model == "qwen3.8-flash-next-iq3_xxs", first
        assert first.state == "prefill", first.state
        assert first.prefill_tps == 410.0 and first.gen_tps is None
        assert first.prompt_processed == 1024 and first.prompt_total == 2048
        assert first.ctx_used == 2048 and first.ctx_total == 98304
        assert first.requests == 1 and first.new_session

        second = monitor.poll()
        assert second.state == "generate", second.state
        assert second.gen_tps == 42.5 and second.prefill_tps is None
        assert second.decoded_tokens == 16
        assert second.ctx_used == 2080
        assert second.requests == 1 and not second.new_session

        third = monitor.poll()
        assert third.state == "idle", third.state
        assert third.gen_tps is None and third.last_gen_tps == 40.0
        # The reading rate was banked when the prompt finished, and the idle
        # view falls back to the newest finished request for context.
        assert third.last_prefill_tps == 410.0
        assert third.ctx_used == 2144
        assert third.new_session is False

        # A failed engine is not a healthy idle server.
        llama.fetch_strata_metrics = lambda target: (
            {"engine": {"error": "model unavailable"}, "live": {"state": "error"}},
            None,
        )
        broken = monitor.poll()
        assert not broken.ok and broken.error == "model unavailable", broken
    finally:
        llama.fetch_strata_metrics = original


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

        # Inference tiles: feed a snapshot the way the worker would. Stop the
        # live poller first: if a server answers on this machine, its next
        # snapshot would overwrite these fixed numbers mid-assertion.
        if app._inference_timer is not None:
            app._inference_timer.stop()
        while app._inference_busy:
            await asyncio.sleep(0.05)
        await pilot.pause()

        grid = app.query_one("#tiles")
        grid.display = True
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
        speed = grid.tile("speed")
        value = speed.query_one(".tile-value").render().plain
        assert "41.2" in value, value
        sub = speed.query_one(".tile-sub").render().plain
        assert "1,183" in sub, sub
        context = app.query_one(Meters).query_one("#meter-context").render().plain
        assert "3,200" in context, context

        # Pressing i hides the tiles again.
        await pilot.press("i")
        await pilot.pause()
        assert app.show_inference is False


def test_app() -> None:
    asyncio.run(_app_checks())


if __name__ == "__main__":
    test_snapped_window_scrolls()
    test_session_extremes()
    test_host_readings()
    test_pcie_link()
    test_request_history()
    test_request_table_fits_its_panel()
    test_graph_render_is_fast()
    test_sparkline_small()
    test_chart_body_is_solid()
    test_chart_body_has_no_black_gaps()
    test_lower_curve_keeps_its_own_body()
    test_graph_range_hugs_data()
    test_crit_line_is_drawn_in_range()
    test_time_ticks_are_round_and_end_at_now()
    test_axes_chart_scrolls_without_boiling()
    test_graph_span_fills_width()
    test_inference_rates()
    test_llama_swap_shell_wrapper()
    test_strata_metrics_snapshot()
    test_app()
    print("all checks passed")
