"""Inference tiles: decode / prefill tokens per second, context, model.

The layout is the same idea as the Strata terminal monitor: a labelled tile
per metric, a big number, and a small live graph underneath.
"""

from __future__ import annotations

import time
from collections import deque

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from .graphs import Sparkline
from .llama import InferenceSnapshot, format_rate, format_tokens

# Roughly two minutes of history at a 0.5 s poll.
HISTORY = 240

DECODE_LINE = "spring_green2"
DECODE_FILL = (0, 60, 40)
PREFILL_LINE = "#5fafff"
PREFILL_FILL = (0, 45, 80)
CONTEXT_LINE = "yellow3"
CONTEXT_FILL = (60, 50, 0)

# Sparkline ceilings. Decode is tens of tokens/second and prefill is
# thousands, so they get their own scales; context use is always a percent.
DECODE_Y_MAX = 120.0
PREFILL_Y_MAX = 4000.0

STATE_STYLES = {
    "generate": ("spring_green2", "generating"),
    "prefill": ("#5fafff", "prefill"),
    "idle": ("grey62", "idle"),
    "offline": ("indian_red", "offline"),
}


def _short_model(name: str) -> str:
    """Qwen4-27B-Q4_K_M.gguf -> Qwen4-27B-Q4_K_M (any directory, clipped)."""
    if not name:
        return "no model"
    base = name.rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[: -5]
    if len(base) > 26:
        base = base[:25] + "…"
    return base


class MetricTile(Vertical):
    """Label in the border, a big value, a small graph, a dim footnote."""

    DEFAULT_CSS = """
    MetricTile {
        width: 1fr;
        min-width: 0;
        height: 7;
        border: round $primary;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    MetricTile .metric-value {
        height: 1;
        width: 100%;
        text-overflow: ellipsis;
    }
    MetricTile .metric-note {
        height: 1;
        width: 100%;
        color: $text-muted;
        text-overflow: ellipsis;
    }
    MetricTile .metric-extremes {
        height: 1;
        width: 100%;
        color: $text-muted;
        text-overflow: ellipsis;
    }
    MetricTile .metric-body {
        height: 2;
        width: 100%;
    }
    """

    def __init__(
        self,
        label: str,
        *,
        value_style: str = "bold",
        line: str = "cyan",
        fill: tuple[int, int, int] = (0, 60, 60),
        span: float = 120.0,
        y_max: float | None = None,
        placeholder: str = "--",
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.border_title = label
        self._value_style = value_style
        self._line = line
        self._fill = fill
        self._span = span
        # A fixed ceiling keeps the sparkline's scale (and so every old
        # point in it) stable from one update to the next.
        self._y_max = y_max
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder, classes="metric-value", markup=False)
        yield Sparkline(
            line=self._line,
            fill=self._fill,
            span=self._span,
            y_max=self._y_max,
            classes="metric-body",
        )
        yield Static("", classes="metric-note", markup=False)
        yield Static("", classes="metric-extremes", markup=False)

    def update_tile(
        self,
        value: Text | str,
        note: Text | str | None = None,
        *,
        extremes: Text | str | None = None,
        points: list[tuple[float, float]] | None = None,
        now: float | None = None,
        span: float | None = None,
        value_style: str | None = None,
    ) -> None:
        if isinstance(value, str):
            value = Text(value, style=value_style or self._value_style)
        elif value_style:
            value.stylize(value_style)
        self.query_one(".metric-value", Static).update(value)
        if note is not None:
            self.query_one(".metric-note", Static).update(
                note if isinstance(note, Text) else Text(note)
            )
        if extremes is not None:
            self.query_one(".metric-extremes", Static).update(
                extremes if isinstance(extremes, Text) else Text(extremes)
            )
        if points is not None:
            self.query_one(Sparkline).update_points(
                points, now if now is not None else time.monotonic(), span
            )


class StateTile(Vertical):
    """Model name, live state and where the numbers come from."""

    DEFAULT_CSS = """
    StateTile {
        width: 1fr;
        min-width: 0;
        height: 7;
        border: round $primary;
        padding: 0 1;
    }
    StateTile .metric-value {
        height: 1;
        width: 100%;
        text-overflow: ellipsis;
    }
    StateTile .state-line {
        height: 1;
        width: 100%;
        text-overflow: ellipsis;
    }
    StateTile .metric-note {
        height: 1;
        width: 100%;
        color: $text-muted;
        text-overflow: ellipsis;
    }
    StateTile .state-spare {
        height: 1;
        width: 100%;
    }
    """

    def __init__(self, name: str | None = None, id: str | None = None) -> None:
        super().__init__(name=name, id=id)
        self.border_title = "MODEL"

    def compose(self) -> ComposeResult:
        yield Static("no model", classes="metric-value", markup=False)
        yield Static("", classes="state-line", markup=False)
        yield Static("", classes="metric-note", markup=False)
        yield Static("", classes="state-spare", markup=False)

    def update_tile(self, snapshot: InferenceSnapshot) -> None:
        color, state_label = STATE_STYLES.get(snapshot.state, ("grey62", snapshot.state))
        model = _short_model(snapshot.model)
        self.query_one(".metric-value", Static).update(Text(model, style="bold"))
        state = Text()
        state.append("● ", style=color)
        state.append(state_label, style=color)
        if snapshot.state in ("prefill", "generate"):
            # Keep the padding inside the coloured run: a run that *starts*
            # with spaces is exported to SVG with the spaces as zero-width
            # glyphs, which costs the gap in screenshots.
            state.append("  ", style=color)
            if snapshot.state == "prefill" and snapshot.prompt_total:
                done = min(
                    snapshot.prompt_processed + snapshot.prompt_cached,
                    snapshot.prompt_total,
                )
                state.append(f"{done / snapshot.prompt_total:.0%}", style="grey62")
            elif snapshot.state == "generate" and snapshot.decoded_tokens:
                state.append(f"{format_tokens(snapshot.decoded_tokens)} tok", style="grey62")
        self.query_one(".state-line", Static).update(state)
        note = f"{snapshot.requests} req"
        if snapshot.origin:
            note += f" · {snapshot.origin}"
        if not snapshot.ok and snapshot.error:
            note = snapshot.error
        self.query_one(".metric-note", Static).update(Text(note))


class InferenceBar(Horizontal):
    """The four tiles above the graph. Hidden when no server answers."""

    DEFAULT_CSS = """
    InferenceBar {
        height: 7;
        width: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, name: str | None = None, id: str | None = None) -> None:
        super().__init__(name=name, id=id)
        self._gen: deque[tuple[float, float]] = deque(maxlen=HISTORY)
        self._prefill: deque[tuple[float, float]] = deque(maxlen=HISTORY)
        self._ctx: deque[tuple[float, float]] = deque(maxlen=HISTORY)
        self._span = 120.0
        # Session-scoped extremes: they span the whole session (not just the
        # sparkline window) and are never cleared while the session runs.
        self._extremes: dict[str, tuple[float, float]] = {}  # key -> (min, max)

    def compose(self) -> ComposeResult:
        yield MetricTile(
            "DECODE tok/s",
            line=DECODE_LINE,
            fill=DECODE_FILL,
            span=self._span,
            y_max=DECODE_Y_MAX,
            id="tile-decode",
        )
        yield MetricTile(
            "PREFILL tok/s",
            line=PREFILL_LINE,
            fill=PREFILL_FILL,
            span=self._span,
            y_max=PREFILL_Y_MAX,
            id="tile-prefill",
        )
        yield MetricTile(
            "CONTEXT",
            line=CONTEXT_LINE,
            fill=CONTEXT_FILL,
            span=self._span,
            y_max=100.0,
            placeholder="0%",
            id="tile-context",
        )
        yield StateTile(id="tile-model")

    # -- updates -------------------------------------------------------

    def _extend_extremes(self, key: str, value: float) -> None:
        low, high = self._extremes.get(key, (value, value))
        self._extremes[key] = (min(low, value), max(high, value))

    def _extremes_text(self, key: str, *, percent: bool = False) -> Text:
        """``min 28.4 · max 41.9`` for the session, or dashes when empty.

        The separator deliberately has no leading spaces: a run that starts
        with a space exports to SVG as a run with zero-width space glyphs, so
        the gap disappears in screenshots and any non-terminal render.
        """
        if key not in self._extremes:
            text = Text("min ", style="grey58")
            text.append("--", style="bold grey74")
            text.append(" · max ", style="grey58")
            text.append("--", style="bold grey74")
            return text
        low, high = self._extremes[key]
        text = Text("min ", style="grey58")
        text.append(
            f"{low:.0f}%" if percent else format_rate(low), style="bold grey74"
        )
        text.append(" · max ", style="grey58")
        text.append(
            f"{high:.0f}%" if percent else format_rate(high), style="bold grey74"
        )
        return text

    def _note_for(
        self, snapshot: InferenceSnapshot, live: float | None, last: float | None,
        idle_word: str,
    ) -> str:
        if live is not None:
            return "live"
        if last is not None:
            return f"last {format_rate(last)} tok/s"
        if snapshot.ok and snapshot.state != "idle":
            return idle_word
        return "no request yet" if snapshot.ok else "offline"

    def update_snapshot(self, snapshot: InferenceSnapshot, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        decode = self.query_one("#tile-decode", MetricTile)
        prefill = self.query_one("#tile-prefill", MetricTile)
        context = self.query_one("#tile-context", MetricTile)
        model = self.query_one("#tile-model", StateTile)

        if snapshot.new_session:
            self._extremes.clear()

        if snapshot.gen_tps is not None:
            self._gen.append((now, snapshot.gen_tps))
            self._extend_extremes("decode", snapshot.gen_tps)
        if snapshot.prefill_tps is not None:
            self._prefill.append((now, snapshot.prefill_tps))
            self._extend_extremes("prefill", snapshot.prefill_tps)
        pct = 0.0
        if snapshot.ctx_total:
            pct = 100.0 * snapshot.ctx_used / snapshot.ctx_total
        if snapshot.ok:
            self._ctx.append((now, pct))
            self._extend_extremes("ctx", pct)

        # Drop points that scrolled out of the tile's time span so the
        # deque cannot grow without bound on a slow probe.
        horizon = now - self._span * 1.5
        for history in (self._gen, self._prefill, self._ctx):
            while history and history[0][0] < horizon:
                history.popleft()

        if not snapshot.ok:
            for tile, key in ((decode, "decode"), (prefill, "prefill"), (context, "ctx")):
                tile.update_tile(
                    Text("--", style="grey42"),
                    "offline",
                    extremes=self._extremes_text(key, percent=(key == "ctx")),
                    points=[],
                    now=now,
                )
            model.update_tile(snapshot)
            return

        decode.update_tile(
            Text(
                format_rate(
                    snapshot.gen_tps
                    if snapshot.gen_tps is not None
                    else snapshot.last_gen_tps
                ),
                style="bold "
                + (DECODE_LINE if snapshot.gen_tps is not None else "grey62"),
            ),
            self._note_for(snapshot, snapshot.gen_tps, snapshot.last_gen_tps,
                           "waiting for decode"),
            extremes=self._extremes_text("decode"),
            points=list(self._gen),
            now=now,
        )
        prefill.update_tile(
            Text(
                format_rate(
                    snapshot.prefill_tps
                    if snapshot.prefill_tps is not None
                    else snapshot.last_prefill_tps
                ),
                style="bold "
                + (PREFILL_LINE if snapshot.prefill_tps is not None else "grey62"),
            ),
            self._note_for(snapshot, snapshot.prefill_tps, snapshot.last_prefill_tps,
                           "waiting for prefill"),
            extremes=self._extremes_text("prefill"),
            points=list(self._prefill),
            now=now,
        )

        if snapshot.ctx_total:
            value = f"{pct:.0f}%"
            note = (
                f"{format_tokens(snapshot.ctx_used)} / "
                f"{format_tokens(snapshot.ctx_total)} tok"
            )
        else:
            value = "--"
            note = "context unknown"
        context.update_tile(
            Text(value, style="bold " + CONTEXT_LINE),
            note,
            extremes=self._extremes_text("ctx", percent=True),
            points=list(self._ctx),
            now=now,
        )
        model.update_tile(snapshot)

