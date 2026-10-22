"""Live decode / prefill throughput from a local llama.cpp server.

LACT-TUI is a fan-control TUI, but the machine it runs on often has a
llama.cpp server (`llama-server`, optionally behind `llama-swap`) using
the same GPU. This module turns that server's `/slots` endpoint into two
numbers people actually stare at while a model runs:

- **decode** (generation) tokens per second, sampled live;
- **prefill** (prompt processing) tokens per second, sampled live;

plus context usage, the model name and a request counter. The last finished
request's rates are kept so an idle server still shows what it last did.

Discovery is best effort and needs no configuration:

1. ``LACT_TUI_LLAMA_URL`` (+ ``LACT_TUI_LLAMA_KEY`` / ``..._KEY_FILE``) wins.
2. Any ``llama-server`` process owned by the current user: its ``--port``,
   ``--api-key-file`` and ``-m`` are read straight from ``/proc/<pid>/cmdline``.
   That also works when the server was started by hand or by llama-swap.
3. llama-swap on 127.0.0.1:8091: ``/running`` lists the proxy URL and the
   child command line.
4. The usual llama.cpp ports (8080, 8096) with the usual key-file locations.

All HTTP is blocking on purpose: the app calls :meth:`LlamaMonitor.poll`
from a worker thread, never from the UI thread.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HTTP_TIMEOUT = 1.0
# Re-run process/port discovery at most this often while nothing answers.
REDISCOVER_S = 10.0
# A couple of timeouts must pass before dropping a known-good target, so one
# slow poll does not throw away the model name and request counter.
DROP_AFTER_FAILURES = 2

DEFAULT_PORTS = (8080, 8096)
KEY_FILE_HINTS = (
    "~/.config/llama-server/api-key.swap",
    "~/.config/llama-server/api-key",
    "~/.config/llama-server/api-key.line1",
    "~/.llama-server-api-key",
)

# llama.cpp returns 401 without the key, and some builds answer 403.
AUTH_ERRORS = ("HTTP 401", "HTTP 403")


@dataclass(frozen=True)
class LlamaTarget:
    """A llama.cpp server we can talk to."""

    base_url: str
    api_key: str | None = None
    model: str = ""
    origin: str = ""


@dataclass(slots=True)
class InferenceSnapshot:
    """One poll's worth of inference telemetry (all rates in tokens/second)."""

    ok: bool = False
    error: str | None = None
    origin: str = ""
    model: str = ""
    state: str = "offline"  # offline | idle | prefill | generate
    gen_tps: float | None = None
    prefill_tps: float | None = None
    last_gen_tps: float | None = None
    last_prefill_tps: float | None = None
    prompt_tokens: int = 0
    prompt_processed: int = 0
    prompt_cached: int = 0
    prompt_total: int = 0
    decoded_tokens: int = 0
    ctx_used: int = 0
    ctx_total: int = 0
    requests: int = 0


# --------------------------------------------------------------- plumbing


def _http_json(url: str, api_key: str | None, timeout: float = HTTP_TIMEOUT):
    """GET a JSON document. Returns ``(data, error)``; never raises."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(getattr(exc, "reason", exc))
    except ValueError as exc:  # malformed URL
        return None, str(exc)
    try:
        return json.loads(body.decode("utf-8", "replace")), None
    except (ValueError, UnicodeDecodeError):
        return None, "invalid JSON"


def _read_key_file(path: str) -> str | None:
    try:
        text = Path(path).expanduser().read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _arg(argv: list[str], *names: str) -> str | None:
    """Find ``--flag value`` or ``--flag=value`` in an argv list."""
    for index, item in enumerate(argv):
        for name in names:
            if item == name and index + 1 < len(argv):
                return argv[index + 1]
            prefix = f"{name}="
            if item.startswith(prefix):
                return item[len(prefix):]
    return None


def _target_from_argv(argv: list[str], origin: str, model: str = "") -> LlamaTarget | None:
    """Build a target from a llama-server command line."""
    if not argv:
        return None
    # Only long flags: on llama-server `-p` is --prompt, not --port, and a
    # short-flag guess would read a prompt as a port number.
    host = _arg(argv, "--host", "--listen") or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = _arg(argv, "--port") or "8080"
    if not port.isdigit():
        return None

    key: str | None = None
    key_file = _arg(argv, "--api-key-file")
    if key_file:
        key = _read_key_file(key_file)
    if key is None:
        key = _arg(argv, "--api-key")

    if not model:
        model_path = _arg(argv, "-m", "--model") or ""
        model = Path(model_path).name if model_path else ""
    if model.endswith(".gguf"):
        model = model[: -len(".gguf")]

    return LlamaTarget(
        base_url=f"http://{host}:{port}",
        api_key=key,
        model=model,
        origin=origin,
    )


def _candidate_from_env() -> list[LlamaTarget]:
    url = os.environ.get("LACT_TUI_LLAMA_URL", "").strip()
    if not url:
        return []
    if "://" not in url:
        url = f"http://{url}"
    key = os.environ.get("LACT_TUI_LLAMA_KEY") or None
    key_file = os.environ.get("LACT_TUI_LLAMA_KEY_FILE")
    if key is None and key_file:
        key = _read_key_file(key_file)
    return [LlamaTarget(base_url=url.rstrip("/"), api_key=key, origin="env")]


def _candidate_from_processes() -> list[LlamaTarget]:
    """Every llama-server owned by this user, newest first."""
    found: list[tuple[float, LlamaTarget]] = []
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    uid = os.getuid()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"llama-server" not in raw:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not argv:
            continue
        exe = os.path.basename(argv[0])
        if exe not in ("llama-server", "llama-server.exe", "rpc-server"):
            continue
        target = _target_from_argv(argv, origin="llama-server")
        if target is None:
            continue
        try:
            started = entry.stat().st_mtime
        except OSError:
            started = 0.0
        found.append((started, target))
    found.sort(key=lambda item: item[0], reverse=True)
    return [target for _, target in found]


def _candidate_from_llama_swap() -> list[LlamaTarget]:
    """Ask llama-swap what it currently runs (its /running is always local)."""
    data, _ = _http_json("http://127.0.0.1:8091/running", None, timeout=0.4)
    if not isinstance(data, dict):
        return []
    targets: list[LlamaTarget] = []
    for item in data.get("running") or []:
        if not isinstance(item, dict):
            continue
        state = str(item.get("state", "")).lower()
        if state and state not in ("ready", "starting", "running"):
            continue
        cmd = item.get("cmd")
        name = str(item.get("name") or item.get("model") or "")
        target = None
        if isinstance(cmd, str) and cmd:
            try:
                target = _target_from_argv(shlex.split(cmd), origin="llama-swap", model=name)
            except ValueError:
                target = None
        if target is None:
            proxy = str(item.get("proxy") or "").rstrip("/")
            if proxy.startswith("http"):
                target = LlamaTarget(base_url=proxy, model=name, origin="llama-swap")
        if target is not None:
            targets.append(target)
    return targets


def _candidate_from_defaults() -> list[LlamaTarget]:
    targets: list[LlamaTarget] = []
    for port in DEFAULT_PORTS:
        for hint in KEY_FILE_HINTS:
            path = Path(hint).expanduser()
            if not path.exists():
                continue
            target = LlamaTarget(
                base_url=f"http://127.0.0.1:{port}",
                api_key=_read_key_file(str(path)),
                origin="default port",
            )
            targets.append(target)
            break
        else:
            targets.append(
                LlamaTarget(base_url=f"http://127.0.0.1:{port}", origin="default port")
            )
    return targets


def discover_targets() -> list[LlamaTarget]:
    """Candidate servers in priority order, duplicates removed."""
    seen: set[str] = set()
    ordered: list[LlamaTarget] = []
    for target in (
        *_candidate_from_env(),
        *_candidate_from_processes(),
        *_candidate_from_llama_swap(),
        *_candidate_from_defaults(),
    ):
        key = (target.base_url, target.api_key)
        marker = f"{key[0]}|{key[1] or ''}"
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(target)
    return ordered


def fetch_slots(target: LlamaTarget) -> tuple[list[dict] | None, str | None]:
    data, error = _http_json(f"{target.base_url}/slots", target.api_key)
    if error is None and not isinstance(data, list):
        return None, "unexpected /slots payload"
    return data, error


def fetch_props(target: LlamaTarget) -> dict:
    data, _ = _http_json(f"{target.base_url}/props", target.api_key, timeout=0.6)
    return data if isinstance(data, dict) else {}


def _slot_decoded(slot: dict) -> int:
    """``n_decoded`` lives in ``next_token`` (a list on current builds)."""
    next_token = slot.get("next_token")
    if isinstance(next_token, list):
        if not next_token:
            return 0
        next_token = next_token[0]
    if isinstance(next_token, dict):
        try:
            return int(next_token.get("n_decoded") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------ monitor


@dataclass(slots=True)
class _Prev:
    t: float
    task: object
    processed: int
    decoded: int
    processed_at: float  # when ``processed`` last changed
    decoded_at: float  # when ``decoded`` last changed


class LlamaMonitor:
    """Polls one llama.cpp server and derives live token rates.

    Not thread-safe by accident: :meth:`poll` takes an internal lock so the
    Textual worker thread can call it while the UI thread reads
    :attr:`last_snapshot`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: LlamaTarget | None = None
        self._model = ""
        self._next_discovery = 0.0
        self._failures = 0
        self._prev: _Prev | None = None
        self._gen_ema = 0.0
        self._prefill_ema = 0.0
        self._live_gen: float | None = None
        self._live_prefill: float | None = None
        self._last_gen: float | None = None
        self._last_prefill: float | None = None
        self._requests = 0
        self._state = "offline"
        self.last_snapshot = InferenceSnapshot()

    # -- discovery -----------------------------------------------------

    @property
    def target(self) -> LlamaTarget | None:
        return self._target

    def _select_target(self) -> LlamaTarget | None:
        for candidate in discover_targets():
            slots, error = fetch_slots(candidate)
            if slots is None:
                continue
            if not candidate.model:
                props = fetch_props(candidate)
                model = str(props.get("model_alias") or props.get("model_path") or "")
                name = Path(model).name
                if name.endswith(".gguf"):
                    name = name[: -len(".gguf")]
                candidate = LlamaTarget(
                    base_url=candidate.base_url,
                    api_key=candidate.api_key,
                    model=name,
                    origin=candidate.origin,
                )
            self._target = candidate
            self._model = candidate.model
            self._prev = None
            self._live_gen = self._live_prefill = None
            self._last_gen = self._last_prefill = None
            self._requests = 0
            return candidate
        return None

    # -- polling -------------------------------------------------------

    def poll(self) -> InferenceSnapshot:
        with self._lock:
            return self._poll_locked()

    def _offline(self, error: str, ctx_total: int = 0) -> InferenceSnapshot:
        self._state = "offline"
        snapshot = InferenceSnapshot(
            ok=False,
            error=error,
            origin=self._target.origin if self._target else "",
            model=self._model,
            state="offline",
            last_gen_tps=self._last_gen,
            last_prefill_tps=self._last_prefill,
            requests=self._requests,
            ctx_total=ctx_total,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _poll_locked(self) -> InferenceSnapshot:
        now = time.monotonic()
        if self._target is None:
            # Rediscovery is /proc + a few HTTP probes; do it on an interval,
            # not on every poll, and only while nothing is answering.
            if now < self._next_discovery:
                return self._offline(self.last_snapshot.error or "no llama.cpp server found")
            self._next_discovery = now + REDISCOVER_S
            if self._select_target() is None:
                return self._offline("no llama.cpp server found")

        assert self._target is not None
        slots, error = fetch_slots(self._target)
        if slots is None:
            self._failures += 1
            if self._failures >= DROP_AFTER_FAILURES:
                # Server restarted on another port / with another key.
                self._target = None
                self._failures = 0
                # Give it a moment before scanning /proc and probing again.
                self._next_discovery = time.monotonic() + 2.0
            return self._offline(error or "server did not answer")
        self._failures = 0

        processing = False
        prompt_tokens = prompt_processed = prompt_cached = decoded = 0
        ctx_used = ctx_total = 0
        task: object = None
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            if slot.get("is_processing"):
                processing = True
            slot_prompt = max(_as_int(slot.get("n_prompt_tokens")), 0)
            slot_processed = max(_as_int(slot.get("n_prompt_tokens_processed")), 0)
            slot_cached = max(_as_int(slot.get("n_prompt_tokens_cache")), 0)
            slot_decoded = max(_slot_decoded(slot), 0)
            prompt_tokens += slot_prompt
            prompt_processed += slot_processed
            prompt_cached += slot_cached
            decoded += slot_decoded
            slot_ctx = max(_as_int(slot.get("n_ctx")), 0)
            ctx_total = max(ctx_total, slot_ctx)
            # Each slot has its own context window; report the fullest one.
            # `n_prompt_tokens` already counts generated tokens on current
            # llama.cpp builds, so it *is* the context in use.
            ctx_used = max(ctx_used, slot_prompt)
            if slot.get("is_processing"):
                task = slot.get("id_task")

        # Phase detection. `n_prompt_tokens` counts prompt + generated
        # tokens, so the prompt itself is `prompt - decoded`; whatever was
        # not reused from the cache still has to be processed. Decoding
        # starts once that remainder is done. (Checked against a live
        # llama.cpp b12044: processed 401 + cached 36,907 + decoded 1,683
        # == n_prompt_tokens 38,991.)
        prompt_total = max(prompt_tokens - decoded, 0)
        prefill_done = min(prompt_processed + prompt_cached, prompt_total)
        state = "idle"
        if processing:
            state = "prefill" if prefill_done < prompt_total else "generate"

        if state == "prefill":
            self._gen_ema = 0.0
        elif state == "generate":
            self._prefill_ema = 0.0

        prev = self._prev
        task_changed = prev is not None and prev.task != task
        # Prefill -> generate (or idle -> busy) changes which counter moves;
        # the first sample after the switch only sets a baseline.
        phase_changed = state != self._state and prev is not None
        if task_changed:
            # A new request took the slot: bank the finished one's numbers.
            self._commit_live()
            if processing and task is not None:
                self._requests += 1
        elif phase_changed:
            self._commit_live()

        reset_baseline = task_changed or phase_changed
        if reset_baseline:
            prev = None

        processed_at = prev.processed_at if prev is not None else now
        decoded_at = prev.decoded_at if prev is not None else now

        if prev is not None and now > prev.t:
            # llama.cpp updates its counters per batch, not per poll, so a
            # delta divided by the poll interval overstates the rate (a
            # 1,024-token prefill batch arriving after 2.5 s would read as
            # 2,048 tok/s). Divide by the time since the counter last moved
            # instead: that is the average speed over the batch.
            if state == "prefill" and prompt_processed > prev.processed:
                span = max(now - prev.processed_at, 0.05)
                instant = (prompt_processed - prev.processed) / span
                self._prefill_ema = (
                    instant if self._prefill_ema == 0.0
                    else self._prefill_ema * 0.6 + instant * 0.4
                )
                self._live_prefill = self._prefill_ema
                processed_at = now
            elif state == "generate" and decoded > prev.decoded:
                span = max(now - prev.decoded_at, 0.05)
                instant = (decoded - prev.decoded) / span
                self._gen_ema = (
                    instant if self._gen_ema == 0.0
                    else self._gen_ema * 0.6 + instant * 0.4
                )
                self._live_gen = self._gen_ema
                decoded_at = now

        self._prev = _Prev(
            t=now,
            task=task,
            processed=prompt_processed,
            decoded=decoded,
            processed_at=processed_at,
            decoded_at=decoded_at,
        )
        if not processing:
            self._commit_live()
        self._state = state

        snapshot = InferenceSnapshot(
            ok=True,
            error=None,
            origin=self._target.origin,
            model=self._model or "llama.cpp",
            state=state,
            gen_tps=self._live_gen if state == "generate" else None,
            prefill_tps=self._live_prefill if state == "prefill" else None,
            last_gen_tps=self._last_gen,
            last_prefill_tps=self._last_prefill,
            prompt_tokens=prompt_tokens,
            prompt_processed=prompt_processed,
            prompt_cached=prompt_cached,
            prompt_total=prompt_total,
            decoded_tokens=decoded,
            ctx_used=ctx_used,
            ctx_total=ctx_total,
            requests=self._requests,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _commit_live(self) -> None:
        if self._live_gen is not None:
            self._last_gen = self._live_gen
        if self._live_prefill is not None:
            self._last_prefill = self._live_prefill
        self._live_gen = self._live_prefill = None


def format_rate(value: float | None, *, compact: bool = False) -> str:
    """Human token rate: 45.1, 1.2K, 12.3K."""
    if value is None:
        return "--"
    if value < 0:
        value = 0.0
    if compact and value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def format_tokens(value: int) -> str:
    """Token counts the way llama.cpp logs them: 3,197 / 96K."""
    if value >= 100_000:
        return f"{value / 1000:.0f}K"
    if value >= 10_000:
        return f"{value / 1000:.1f}K"
    return f"{value:,}"
