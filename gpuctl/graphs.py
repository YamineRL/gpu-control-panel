"""Braille area charts, drawn straight into Rich Text.

The GPU temperature graph used to go through plotext with ``fillx=True``.
That looks nice but the fill loop is slow: measured on an RDNA2 card a single
redraw costs ~0.3 s at 150 points (5m window / 2s polling) and ~2 s at 600
points (5m / 500ms), rising to 30 s for the 1h window. It runs on the
Textual UI thread inside the widget render, so every probe tick froze the
interface: button clicks were dropped or took seconds to register.

This module draws the same btop-style picture - a dim filled area under a
bright top edge - in a couple of milliseconds, plus the small sparklines
used by the inference tiles. It has no dependency beyond Rich.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Sequence

from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.widget import Widget

# Braille dot bit for [row inside the cell][column inside the cell].
# A cell is 2 dots wide and 4 dots tall.
_DOTS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)
_BLANK = 0x2800  # U+2800 BRAILLE PATTERN BLANK

_DIM = "grey37"
_AXIS = "grey30"
_LABEL = "grey58"

ColorLike = str | Style | tuple[int, int, int]


def snap_window(now: float, window: float, columns: int) -> tuple[float, float]:
    """A right-edge-anchored window whose columns fall on fixed clock times.

    ``render_chart`` draws one value per braille dot column, i.e. two per
    text column, and a column's value is interpolated at its centre. If the
    window slides continuously with the clock (``now - window, now``), every
    column's centre moves by a fraction of a column on every refresh, so the
    interpolation weights for the *whole* history change each frame and the
    curve visibly boils instead of scrolling.

    Rounding the right edge down to a multiple of one dot column, in absolute
    time, pins dot ``k`` to the same instant on every refresh: the oldest
    columns become bit-identical and the graph advances left one dot at a
    time. The cost is that the right edge lags by up to one dot column.
    """
    columns = max(int(columns), 2)
    if window <= 0.0:
        return (now - 1.0, now)
    step = window / columns
    right = step * math.floor(now / step)
    return (right - window, right)


@dataclass(frozen=True)
class Series:
    """One curve plus the colours for its top edge and its filled body."""

    points: Sequence[tuple[float, float]]
    line: ColorLike
    fill: ColorLike


def _style(value: ColorLike) -> Style:
    if isinstance(value, Style):
        return value
    if isinstance(value, tuple):
        return Style(color=Color.from_rgb(*value))
    return Style.parse(value)


# ------------------------------------------------------------------ maths


def _sample_columns(
    points: Sequence[tuple[float, float]],
    count: int,
    x_min: float,
    x_max: float,
) -> list[float | None]:
    """Linear-interpolate one value per column centre; None outside the data."""
    if count <= 0 or not points:
        return [None] * max(count, 0)
    xs = [p[0] for p in points]
    first, last = xs[0], xs[-1]
    span = x_max - x_min
    step = span / count
    values: list[float | None] = []
    for column in range(count):
        x = x_min + (column + 0.5) * step
        if x < first or x > last:
            values.append(None)
            continue
        index = bisect_left(xs, x)
        if index < len(xs) and xs[index] == x:
            values.append(points[index][1])
        elif index == 0:
            values.append(points[0][1])
        elif index >= len(xs):
            values.append(points[-1][1])
        else:
            xa, ya = points[index - 1]
            xb, yb = points[index]
            values.append(ya if xb <= xa else ya + (yb - ya) * (x - xa) / (xb - xa))
    return values


def _paint(
    bits: list[list[int]],
    styles: list[list[Style | None]],
    points: Sequence[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    line: Style,
    fill: Style,
) -> None:
    rows = len(bits)
    if rows < 1 or not bits[0] or not points:
        return
    # The resampler assumes ascending x. Callers normally hand over already
    # ordered samples; sorting here keeps a stray out-of-order point (a clock
    # hiccup, a test fixture) from blanking the whole chart.
    ascending = all(points[i][0] <= points[i + 1][0] for i in range(len(points) - 1))
    if not ascending:
        points = sorted(points)
    columns = len(bits[0])
    dot_columns = columns * 2
    values = _sample_columns(points, dot_columns, x_min, x_max)
    y_span = (y_max - y_min) or 1.0
    total_dots = rows * 4

    for dot_column, value in enumerate(values):
        if value is None or not math.isfinite(value):
            continue
        level = (value - y_min) / y_span
        if level <= 0.0:
            continue
        level = 1.0 if level > 1.0 else level
        column = dot_column // 2
        dot_x = dot_column & 1
        top = int(round((1.0 - level) * (total_dots - 1)))
        first_row = top >> 2
        edge_offset = top & 3
        for row in range(first_row, rows):
            if row == first_row:
                mask = 0
                for dot_y in range(edge_offset, 4):
                    mask |= _DOTS[dot_y][dot_x]
                bits[row][column] |= mask
                # The bright cell is the top of the fill: that is the curve's
                # visible edge, the way btop draws it.
                styles[row][column] = line
            else:
                bits[row][column] |= 0xFF
                if styles[row][column] is None:
                    styles[row][column] = fill


def _paint_reference(
    bits: list[list[int]],
    styles: list[list[Style | None]],
    value: float,
    y_min: float,
    y_max: float,
    style: Style,
    period: int = 6,
) -> None:
    rows = len(bits)
    if rows < 1 or not bits[0] or not (y_min < value < y_max):
        return
    columns = len(bits[0])
    total_dots = rows * 4
    level = (value - y_min) / ((y_max - y_min) or 1.0)
    dot = int(round((1.0 - level) * (total_dots - 1)))
    row, offset = divmod(dot, 4)
    for column in range(columns):
        if column % period >= period // 2:  # dashed
            continue
        bits[row][column] |= _DOTS[offset][0] | _DOTS[offset][1]
        styles[row][column] = style


def _grid_to_lines(
    bits: list[list[int]],
    styles: list[list[Style | None]],
) -> list[Text]:
    """Merge runs of equal style so Rich gets a handful of spans per row."""
    lines: list[Text] = []
    for bit_row, style_row in zip(bits, styles):
        line = Text()
        run_text: list[str] = []
        run_style: Style | None = None
        for bit, style in zip(bit_row, style_row):
            char = _BLANK if not bit else _BLANK + bit
            if style is not run_style and run_text:
                line.append("".join(run_text), style=run_style)
                run_text = []
            run_style = style
            run_text.append(chr(char))
        if run_text:
            line.append("".join(run_text), style=run_style)
        lines.append(line)
    return lines


# ------------------------------------------------------------- rendering


def render_chart(
    series: Sequence[Series],
    width: int,
    height: int,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: Text | None = None,
    y_ticks: int = 3,
    y_fmt: Callable[[float], str] | None = None,
    ref_line: tuple[float, ColorLike] | None = None,
    x_labels: Sequence[tuple[float, str]] = (),
    gutter: bool = True,
    x_axis: bool = True,
) -> Text:
    """Render a chart of exactly ``height`` lines and ``width`` columns.

    ``x_range``/``y_range`` are data coordinates; ``x_labels`` are
    ``(data_x, text)`` pairs placed under the axis.
    """
    y_fmt = y_fmt or (lambda value: f"{value:.0f}")
    width = max(int(width), 4)
    height = max(int(height), 1)

    reserved = 0
    if title is not None:
        reserved += 1
    if x_axis:
        reserved += 1
    if x_labels:
        reserved += 1
    plot_height = max(height - reserved, 1)
    plot_width = width

    tick_values: list[float] = []
    gutter_width = 0
    if gutter:
        y_min, y_max = y_range
        count = max(int(y_ticks), 2)
        tick_values = [
            y_min + (y_max - y_min) * index / (count - 1) for index in range(count)
        ]
        label_width = max(len(y_fmt(value)) for value in tick_values)
        gutter_width = label_width + 1
        plot_width = max(width - gutter_width, 2)

    bits = [[0] * plot_width for _ in range(plot_height)]
    styles: list[list[Style | None]] = [
        [None] * plot_width for _ in range(plot_height)
    ]

    # Bigger areas first so the smaller curves stay visible on top, the same
    # ordering the old plotext code used.
    def area(item: Series) -> float:
        values = [y for _, y in item.points if math.isfinite(y)]
        return sum(values) / len(values) if values else 0.0

    x_min, x_max = x_range
    y_min, y_max = y_range
    for item in sorted(series, key=area, reverse=True):
        _paint(
            bits,
            styles,
            item.points,
            x_min,
            x_max,
            y_min,
            y_max,
            _style(item.line),
            _style(item.fill),
        )
    if ref_line is not None:
        _paint_reference(
            bits, styles, ref_line[0], y_min, y_max, _style(ref_line[1])
        )

    grid_lines = _grid_to_lines(bits, styles)
    out = Text()

    if title is not None:
        out.append_text(title)
        out.append("\n")

    if gutter:
        # Tick rows: top, middle, bottom of the plot area.
        tick_rows: dict[int, str] = {}
        count = len(tick_values)
        for index, value in enumerate(tick_values):
            row = round((count - 1 - index) * (plot_height - 1) / (count - 1))
            tick_rows[row] = y_fmt(value)
        label_width = gutter_width - 1
        for row, line in enumerate(grid_lines):
            label = tick_rows.get(row, "")
            out.append(f"{label:>{label_width}}", style=_LABEL)
            out.append("│", style=_AXIS)
            out.append_text(line)
            out.append("\n")
    else:
        for line in grid_lines:
            out.append_text(line)
            out.append("\n")

    if x_axis:
        if gutter:
            out.append(" " * (gutter_width - 1), style=_AXIS)
            out.append("└", style=_AXIS)
        out.append("─" * plot_width, style=_AXIS)
        out.append("\n")

    if x_labels:
        pad = " " * gutter_width
        row = Text(pad)
        for value, label in x_labels:
            if x_max <= x_min:
                offset = 0
            else:
                offset = round((value - x_min) / (x_max - x_min) * (plot_width - 1))
            start = max(0, min(offset - len(label) // 2, plot_width - len(label)))
            if start > row.cell_len - gutter_width:
                gap = start - (row.cell_len - gutter_width)
                row.append(" " * gap)
            row.append(label, style=_LABEL)
        out.append_text(row)

    return out


def render_sparkline(
    points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    *,
    x_range: tuple[float, float],
    line: ColorLike,
    fill: ColorLike,
    y_floor: float = 1.0,
    y_max: float | None = None,
) -> Text:
    """Tiny axes-free chart for a tile; scales to the data's own maximum.

    ``y_max`` fixes the scale instead of deriving it from the current data,
    which keeps old points from being rescaled (and the line from jumping
    vertically) on every update.
    """
    values = [y for _, y in points if math.isfinite(y)]
    if y_max is not None and math.isfinite(y_max):
        top = y_max
    else:
        top = max(values) if values else 0.0
        top = max(top, y_floor) * 1.15
    return render_chart(
        [Series(points, line, fill)],
        width,
        height,
        x_range=x_range,
        y_range=(0.0, top),
        gutter=False,
        x_axis=False,
    )


# ------------------------------------------------------------- widgets


class ChartView(Widget):
    """A plotting widget whose data is pushed in, not pulled on render."""

    can_focus = False

    def __init__(self, *, id: str | None = None, name: str | None = None) -> None:
        super().__init__(id=id, name=name)
        self._series: list[Series] = []
        self._x_range: tuple[float, float] = (0.0, 1.0)
        self._y_range: tuple[float, float] = (0.0, 1.0)
        self._title: Text | None = None
        self._ref_line: tuple[float, ColorLike] | None = None
        self._x_labels: tuple[tuple[float, str], ...] = ()
        self._y_fmt: Callable[[float], str] | None = None
        self._gutter = True
        self._empty_text = "waiting for samples..."
        self._has_data = False

    def set_series(
        self,
        series: Sequence[Series],
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        title: Text | None = None,
        ref_line: tuple[float, ColorLike] | None = None,
        x_labels: Sequence[tuple[float, str]] = (),
        y_fmt: Callable[[float], str] | None = None,
        gutter: bool = True,
    ) -> None:
        self._series = list(series)
        self._x_range = x_range
        self._y_range = y_range
        self._title = title
        self._ref_line = ref_line
        self._x_labels = tuple(x_labels)
        self._y_fmt = y_fmt
        self._gutter = gutter
        self._has_data = any(item.points for item in self._series)
        self.refresh()

    def render(self) -> Text:
        width, height = self.size.width, self.size.height
        if width < 4 or height < 1:
            return Text("")
        if not self._has_data:
            return Text(self._empty_text, style=_DIM)
        return render_chart(
            self._series,
            width,
            height,
            x_range=self._x_range,
            y_range=self._y_range,
            title=self._title,
            ref_line=self._ref_line,
            x_labels=self._x_labels,
            y_fmt=self._y_fmt,
            gutter=self._gutter,
        )


class Sparkline(Widget):
    """A small scrolling area chart, used by the inference tiles."""

    can_focus = False
    DEFAULT_CSS = """
    Sparkline {
        height: 2;
        width: 100%;
    }
    """

    def __init__(
        self,
        *,
        line: ColorLike = "cyan",
        fill: ColorLike = (0, 70, 70),
        span: float = 120.0,
        y_max: float | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._line = line
        self._fill = fill
        self._span = span
        self._y_max = y_max
        self._points: list[tuple[float, float]] = []
        self._now = 0.0

    def update_points(
        self, points: list[tuple[float, float]], now: float, span: float | None = None
    ) -> None:
        self._points = points
        self._now = now
        if span is not None:
            self._span = span
        self.refresh()

    def render(self) -> Text:
        width, height = self.size.width, self.size.height
        if width < 2 or height < 1:
            return Text("")
        if not self._points:
            return Text("")
        # Same snapping as the big graph (two dot columns per cell), so the
        # sparkline scrolls instead of being resampled into a boil.
        x_range = snap_window(self._now, self._span, width * 2)
        return render_sparkline(
            self._points,
            width,
            height,
            x_range=x_range,
            line=self._line,
            fill=self._fill,
            y_floor=1.0,
            y_max=self._y_max,
        )
