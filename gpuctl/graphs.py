"""Layered gradient area charts, drawn straight into Rich Text.

The GPU temperature graph used to go through plotext with ``fillx=True``.
That looks nice but the fill loop is slow: measured on an RDNA2 card a single
redraw costs ~0.3 s at 150 points (5m window / 2s polling) and ~2 s at 600
points (5m / 500ms), rising to 30 s for the 1h window. It runs on the
Textual UI thread inside the widget render, so every probe tick froze the
interface: button clicks were dropped or took seconds to register.

This module draws the picture in a few milliseconds, plus the small
sparklines used by the inference tiles. It has no dependency beyond Rich.

The look: every curve is a ridge. Its edge is drawn in the bright line
colour and its body fades down into the series' dark fill colour. Curves are
layered like a mountain range: each body runs from its own edge down to the
next lower curve, so a curve that runs below another one is still a ridge of
its own instead of a stroke buried in someone else's fill.

Cells are drawn with the block characters (U+2581-U+2588), not braille.
Braille gives four sub-rows per cell, but plenty of terminals have no braille
glyphs and silently fall back to a font whose "filled" cells are a grid of
separated dots. Blocks are present in every monospace font. A block cell
holds two colours (glyph and background), so an edge lands on its own eighth
of a cell with the body above it as the background: no black notches.
Full cells set the same colour as glyph and background, so there are no
seams between rows either.
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

# A cell is one column wide and eight sub-rows tall: a value maps to one of
# the eight bottom-aligned block glyphs, so the edge lands on its own eighth
# of the cell. All of these exist in every monospace font.
_FULL = "█"      # U+2588 full block
_LOWER = "▄"     # U+2584 lower half block
_NINTHS = (
    " ",           # blank
    "▁",      # lower one eighth  (U+2581)
    "▂",      # lower quarter
    "▃",      # lower three eighths
    _LOWER,        # lower half
    "▅",      # lower five eighths
    "▆",      # lower three quarters
    "▇",      # lower seven eighths
    _FULL,         # full
)

_GRID_GLYPH = "┈"       # U+2508 light quadruple dash horizontal
_THRESHOLD_GLYPH = "╌"  # U+254C light double dash horizontal

_DIM = "grey37"
_AXIS = "grey35"
_LABEL = "grey58"
_GRID = Color.from_rgb(48, 52, 58)

ColorLike = str | Style | tuple[int, int, int]
XLabels = (
    Sequence[tuple[float, str]]
    | Callable[[tuple[float, float], int], Sequence[tuple[float, str]]]
)


def snap_window(now: float, window: float, columns: int) -> tuple[float, float]:
    """A right-edge-anchored window whose columns fall on fixed clock times.

    ``render_chart`` draws one value per text column. If the window slides
    continuously with the clock (``now - window, now``), every column's
    bucket moves by a fraction of a column on every refresh, so the values
    for the *whole* history change each frame and the curve visibly boils
    instead of scrolling.

    Rounding the right edge down to a multiple of one column, in absolute
    time, pins column ``k`` to the same instants on every refresh: the oldest
    columns become bit-identical and the graph advances left one column at a
    time. The cost is that the right edge lags by up to one column.
    """
    columns = max(int(columns), 2)
    if window <= 0.0:
        return (now - 1.0, now)
    step = window / columns
    right = step * math.floor(now / step)
    return (right - window, right)


@dataclass(frozen=True)
class Series:
    """One curve plus the colours for its bright edge and its dark body."""

    points: Sequence[tuple[float, float]]
    line: ColorLike
    fill: ColorLike


@dataclass(frozen=True)
class Threshold:
    """A horizontal marker line, e.g. a critical temperature."""

    value: float
    color: ColorLike = "#ef4444"


def _color(value: ColorLike) -> Color:
    if isinstance(value, tuple):
        return Color.from_rgb(*value)
    style = value if isinstance(value, Style) else Style.parse(value)
    return style.color or Color.parse("white")


def _rgb(color: Color) -> tuple[int, int, int]:
    triplet = color.get_truecolor()
    return (triplet.red, triplet.green, triplet.blue)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> Color:
    return Color.from_rgb(
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


class _Palette:
    """Per-chart caches: gradient colours and the Style for each pair."""

    def __init__(
        self, series: Sequence[Series], fade_rows: float, edge_dim: float = 0.0
    ) -> None:
        self.lines = [_color(item.line) for item in series]
        self._line_rgb = [_rgb(color) for color in self.lines]
        self._fill_rgb = [_rgb(_color(item.fill)) for item in series]
        self._fade8 = max(fade_rows, 1.0) * 8.0
        self._edge_dim = edge_dim
        self._gradient: dict[tuple[int, int], Color] = {}
        self._styles: dict[tuple[Color | None, Color | None], Style] = {}

    def body(self, index: int, depth8: float) -> Color:
        """Body colour ``depth8`` eighths below the series' edge.

        The cell right under the edge starts a bit under halfway from the
        line colour to the fill, so the edge reads as a bright ridge with a
        glow; deeper cells settle on the fill colour.
        """
        key = (index, int(depth8))
        color = self._gradient.get(key)
        if color is None:
            depth = max(key[1] - 4, 0) / self._fade8
            t = 0.45 + 0.55 * min(depth, 1.0) ** 0.8
            color = _mix(self._line_rgb[index], self._fill_rgb[index], t)
            self._gradient[key] = color
        return color

    def edge(self, index: int, thickness: int) -> Color:
        """Edge colour for an edge glyph ``thickness`` eighths tall.

        A thick edge glyph next to a thin one reads as a tooth sticking out
        of the ridge. Dimming the glyph a little as it gets thicker keeps the
        ridge looking like a line of even weight.
        """
        if thickness <= 2 or self._edge_dim <= 0.0:
            return self.lines[index]
        key = (index, -thickness)
        color = self._gradient.get(key)
        if color is None:
            t = self._edge_dim * (thickness - 2) / 6
            color = _mix(self._line_rgb[index], self._fill_rgb[index], t)
            self._gradient[key] = color
        return color

    def style(self, fg: Color | None, bg: Color | None) -> Style:
        key = (fg, bg)
        style = self._styles.get(key)
        if style is None:
            style = Style(color=fg, bgcolor=bg)
            self._styles[key] = style
        return style


# ------------------------------------------------------------------ maths


def _sample_columns(
    points: Sequence[tuple[float, float]],
    count: int,
    x_min: float,
    x_max: float,
) -> list[float | None]:
    """One value per column: the mean of the samples in the column's bucket.

    A bucket without samples (a sparse history, a short window) is linearly
    interpolated at the column centre instead. Averaging keeps a long window
    smooth: picking one sample per column aliases jitter into spikes.
    None outside the data.
    """
    if count <= 0 or not points:
        return [None] * max(count, 0)
    xs = [p[0] for p in points]
    first, last = xs[0], xs[-1]
    step = (x_max - x_min) / count
    values: list[float | None] = []
    start = bisect_left(xs, x_min)
    for column in range(count):
        right = x_min + (column + 1) * step
        end = bisect_left(xs, right, start)
        if end > start:
            total = 0.0
            used = 0
            for index in range(start, end):
                value = points[index][1]
                if math.isfinite(value):
                    total += value
                    used += 1
            values.append(total / used if used else None)
            start = end
            continue
        x = right - step * 0.5
        if x < first or x > last:
            values.append(None)
            continue
        index = bisect_left(xs, x)
        if index == 0:
            values.append(points[0][1])
        elif index >= len(xs):
            values.append(points[-1][1])
        else:
            xa, ya = points[index - 1]
            xb, yb = points[index]
            values.append(ya if xb <= xa else ya + (yb - ya) * (x - xa) / (xb - xa))
    return values


def nice_ticks(y_min: float, y_max: float, max_count: int) -> list[float]:
    """Round tick values (steps of 1, 2, 2.5 or 5 x 10^n) inside the range."""
    span = y_max - y_min
    if span <= 0 or max_count < 2:
        return [y_min, y_max] if span > 0 else [y_min]
    raw = span / (max_count - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude * 10
    for factor in (1, 2, 2.5, 5, 10):
        if factor * magnitude >= raw:
            step = factor * magnitude
            break
    first = math.ceil(y_min / step - 1e-9) * step
    ticks: list[float] = []
    value = first
    while value <= y_max + 1e-9:
        ticks.append(round(value, 9))
        value += step
    return ticks


def _row_of(value: float, y_min: float, y_max: float, rows: int) -> int:
    """The text row (0 = top) whose cell holds ``value``."""
    span = (y_max - y_min) or 1.0
    eighths = (value - y_min) / span * rows * 8
    cell = int(min(max(eighths, 0.0), rows * 8 - 1) // 8)
    return rows - 1 - cell


# ------------------------------------------------------------- rendering


def _plot(
    series: Sequence[Series],
    rows: int,
    columns: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    palette: _Palette,
) -> tuple[list[list[str]], list[list[Style | None]], list[list[Color | None]], list[float | None]]:
    """Paint the ridges. Returns glyphs, styles, cell backgrounds and the
    last finite value of each series (for the end labels)."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    y_span = (y_max - y_min) or 1.0
    height8 = rows * 8

    glyphs = [[" "] * columns for _ in range(rows)]
    styles: list[list[Style | None]] = [[None] * columns for _ in range(rows)]
    backs: list[list[Color | None]] = [[None] * columns for _ in range(rows)]

    # Top of each series per column, in eighths above the plot floor.
    tops: list[list[int]] = []
    last_values: list[float | None] = []
    for item in series:
        points = item.points
        # The resampler assumes ascending x. A stray out-of-order point (a
        # clock hiccup, a test fixture) must not blank the whole chart.
        if any(points[i][0] > points[i + 1][0] for i in range(len(points) - 1)):
            points = sorted(points)
        sampled = _sample_columns(points, columns, x_min, x_max)
        heights: list[int] = []
        last: float | None = None
        for value in sampled:
            if value is None or not math.isfinite(value):
                heights.append(0)
                continue
            last = value
            level = (value - y_min) / y_span
            heights.append(int(round(min(max(level, 0.0), 1.0) * height8)))
        tops.append(heights)
        last_values.append(last)

    for column in range(columns):
        # (height, series index), tallest first.
        stack = sorted(
            ((tops[k][column], k) for k in range(len(series)) if tops[k][column] > 0),
            reverse=True,
        )
        if not stack:
            continue
        ascending = stack[::-1]

        def band(height: float) -> tuple[int, int] | None:
            # The body that covers ``height``: the lowest edge above it.
            for edge, index in ascending:
                if edge > height:
                    return edge, index
            return None

        first_row = rows - 1 - (stack[0][0] - 1) // 8
        for row in range(first_row, rows):
            floor = (rows - 1 - row) * 8
            ceiling = floor + 8
            edge_here: tuple[int, int] | None = None
            for edge, index in stack:
                if edge <= ceiling:
                    if edge > floor:
                        edge_here = (edge, index)
                    break
            if edge_here is not None:
                edge, index = edge_here
                fg = palette.edge(index, edge - floor)
                if edge == ceiling:
                    glyphs[row][column] = _FULL
                    styles[row][column] = palette.style(fg, fg)
                    backs[row][column] = fg
                    continue
                middle = (edge + ceiling) / 2
                above = band(middle)
                bg = palette.body(above[1], above[0] - middle) if above else None
                glyphs[row][column] = _NINTHS[edge - floor]
                styles[row][column] = palette.style(fg, bg)
                backs[row][column] = bg
                continue
            middle = floor + 4
            covering = band(middle)
            if covering is None:
                continue
            color = palette.body(covering[1], covering[0] - middle)
            glyphs[row][column] = _FULL
            styles[row][column] = palette.style(color, color)
            backs[row][column] = color
    return glyphs, styles, backs, last_values


def _grid_to_lines(
    bits: list[list[str]],
    styles: list[list[Style | None]],
) -> list[Text]:
    """Merge runs of equal style so Rich gets a handful of spans per row."""
    lines: list[Text] = []
    for cell_row, style_row in zip(bits, styles):
        line = Text()
        run_text: list[str] = []
        run_style: Style | None = None
        for cell, style in zip(cell_row, style_row):
            if style is not run_style and run_text:
                line.append("".join(run_text), style=run_style)
                run_text = []
            run_style = style
            run_text.append(cell)
        if run_text:
            line.append("".join(run_text), style=run_style)
        lines.append(line)
    return lines


def render_chart(
    series: Sequence[Series],
    width: int,
    height: int,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: Text | None = None,
    title_right: Text | None = None,
    y_fmt: Callable[[float], str] | None = None,
    x_labels: XLabels = (),
    gutter: bool = True,
    x_axis: bool = True,
    end_labels: bool = False,
    thresholds: Sequence[Threshold] = (),
    snap: bool = False,
) -> Text:
    """Render a chart of exactly ``height`` lines and ``width`` columns.

    ``x_range``/``y_range`` are data coordinates; ``x_labels`` are
    ``(data_x, text)`` pairs placed under the axis, or a function of the
    final ``(x_range, plot_columns)`` that returns them. ``end_labels``
    prints each series' latest value at the right end of its curve.

    ``snap`` treats ``x_range`` as ``(now - span, now)`` and snaps it with
    :func:`snap_window` to the plot's own column count. Only the renderer
    knows that count (the axis and label columns take some of the width),
    and snapping to any other grid would make the history boil again.
    """
    y_fmt = y_fmt or (lambda value: f"{value:.0f}")
    width = max(int(width), 4)
    height = max(int(height), 1)
    x_min, x_max = x_range
    y_min, y_max = y_range

    reserved = (title is not None) + bool(x_axis) + bool(x_labels)
    rows = max(height - reserved, 1)

    ticks: list[float] = []
    gutter_width = 0
    if gutter:
        ticks = nice_ticks(y_min, y_max, max(2, rows // 3 + 1))
        gutter_width = max((len(y_fmt(value)) for value in ticks), default=1) + 1

    # Peek at the latest values to size the right-hand label column.
    right_width = 0
    if end_labels:
        latest = [
            next((y for _, y in reversed(item.points) if math.isfinite(y)), None)
            for item in series
        ]
        texts = [y_fmt(value) for value in latest if value is not None]
        if texts:
            right_width = max(len(text) for text in texts) + 2
    columns = max(width - gutter_width - right_width, 2)
    if snap:
        x_range = snap_window(x_max, x_max - x_min, columns)
        x_min, x_max = x_range
    if callable(x_labels):
        x_labels = x_labels(x_range, columns)

    # Small charts (the tile sparklines) keep a flat edge colour: at two rows
    # the thickness dimming reads as stripes, not as an even line.
    palette = _Palette(
        series,
        fade_rows=max(2.0, rows * 0.45),
        edge_dim=0.16 if rows >= 6 else 0.0,
    )
    glyphs, styles, backs, last_values = _plot(
        series, rows, columns, x_range, y_range, palette
    )

    # Grid lines and thresholds go only where they do not hide a curve edge:
    # empty cells and solid body cells.
    tick_rows: dict[int, str] = {}
    for value in ticks:
        row = _row_of(value, y_min, y_max, rows)
        tick_rows.setdefault(row, y_fmt(value))
    grid_style = Style(color=_GRID)
    for row in tick_rows:
        for column in range(columns):
            if glyphs[row][column] == " ":
                glyphs[row][column] = _GRID_GLYPH
                styles[row][column] = grid_style
    for threshold in thresholds:
        if not (y_min <= threshold.value <= y_max):
            continue
        color = _color(threshold.color)
        row = _row_of(threshold.value, y_min, y_max, rows)
        for column in range(columns):
            cell = glyphs[row][column]
            if cell in (" ", _GRID_GLYPH, _FULL):
                glyphs[row][column] = _THRESHOLD_GLYPH
                styles[row][column] = palette.style(color, backs[row][column])

    # Right-hand labels: one per series at the row of its latest edge,
    # nudged down when two curves end in the same row.
    end_rows: dict[int, Text] = {}
    if right_width:
        placed: list[tuple[int, int]] = []
        for index, value in enumerate(last_values):
            if value is not None:
                placed.append((_row_of(value, y_min, y_max, rows), index))
        placed.sort()
        used = -1
        for row, index in placed:
            row = min(max(row, used + 1), rows - 1)
            if row in end_rows:
                continue
            used = row
            label = Text(" ")
            label.append("◂", style=palette.style(palette.lines[index], None))
            label.append(
                f"{y_fmt(last_values[index]):<{right_width - 2}}",
                style=Style(color=palette.lines[index], bold=True),
            )
            end_rows[row] = label

    out = Text()
    if title is not None:
        out.append_text(title)
        if title_right is not None:
            gap = width - title.cell_len - title_right.cell_len
            if gap >= 1:
                out.append(" " * gap)
                out.append_text(title_right)
        out.append("\n")

    label_width = gutter_width - 1
    for row, line in enumerate(_grid_to_lines(glyphs, styles)):
        if gutter:
            label = tick_rows.get(row, "")
            out.append(f"{label:>{label_width}}", style=_LABEL)
            out.append("┤" if label else "│", style=_AXIS)
        out.append_text(line)
        if right_width:
            out.append_text(end_rows.get(row, Text(" " * right_width)))
        if row < rows - 1 or x_axis or x_labels:
            out.append("\n")

    def column_of(value: float) -> int:
        if x_max <= x_min:
            return 0
        return round((value - x_min) / (x_max - x_min) * (columns - 1))

    if x_axis:
        axis = ["─"] * columns
        for value, _ in x_labels:
            position = column_of(value)
            if 0 <= position < columns:
                axis[position] = "┴"
        if gutter:
            # Unstyled padding: SVG exports squash styled leading spaces.
            out.append(" " * label_width)
            out.append("└", style=_AXIS)
        out.append("".join(axis), style=_AXIS)
        if x_labels:
            out.append("\n")

    if x_labels:
        row = Text(" " * gutter_width)
        for value, label in x_labels:
            offset = column_of(value)
            start = max(0, min(offset - len(label) // 2, columns - len(label)))
            used = row.cell_len - gutter_width
            if start < used + (1 if used else 0):
                continue  # would collide with the previous label
            row.append(" " * (start - used))
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
        self._title_right: Text | None = None
        self._x_labels: XLabels = ()
        self._snap = False
        self._y_fmt: Callable[[float], str] | None = None
        self._gutter = True
        self._end_labels = False
        self._thresholds: tuple[Threshold, ...] = ()
        self._empty_text = "waiting for samples..."
        self._has_data = False

    def set_series(
        self,
        series: Sequence[Series],
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        title: Text | None = None,
        title_right: Text | None = None,
        x_labels: XLabels = (),
        y_fmt: Callable[[float], str] | None = None,
        gutter: bool = True,
        end_labels: bool = False,
        thresholds: Sequence[Threshold] = (),
        snap: bool = False,
    ) -> None:
        self._series = list(series)
        self._x_range = x_range
        self._y_range = y_range
        self._title = title
        self._title_right = title_right
        self._x_labels = x_labels if callable(x_labels) else tuple(x_labels)
        self._snap = snap
        self._y_fmt = y_fmt
        self._gutter = gutter
        self._end_labels = end_labels
        self._thresholds = tuple(thresholds)
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
            title_right=self._title_right,
            x_labels=self._x_labels,
            y_fmt=self._y_fmt,
            gutter=self._gutter,
            end_labels=self._end_labels,
            thresholds=self._thresholds,
            snap=self._snap,
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
        # Same snapping as the big graph, so the sparkline scrolls instead of
        # being resampled into a boil.
        x_range = snap_window(self._now, self._span, width)
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
