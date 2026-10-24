"""Live decode / prefill throughput from a local inference server.

GPU Control Panel is a fan-control TUI, but the machine it runs on often has
an inference server (`llama-server` or Strata, optionally behind `llama-swap`)
using the same GPU. This module turns that server's telemetry into two numbers
people actually stare at while a model runs:

- **decode** (generation) tokens per second, sampled live;
- **prefill** (prompt processing) tokens per second, sampled live;

plus context usage, the model name and a request counter. The last finished
request's rates are kept so an idle server still shows what it last did.

llama.cpp is read from ``/slots``; Strata (its Python server, ``server.py
--engine strata``) from ``/metrics``, whose ``live`` block has the same
counters. The tiles do not care which engine answered.

Discovery is best effort and needs no configuration:

1. ``GPU_CTL_LLAMA_URL`` (+ ``GPU_CTL_LLAMA_KEY`` / ``..._KEY_FILE``) wins.
2. Any ``llama-server`` or Strata server process owned by the current user: its
   ``--port``, key (``--api-key-file`` / ``--api-key`` / the Strata config or
   CLI key file) and ``-m`` are read straight from ``/proc/<pid>``. That also
   works when the server was started by hand or by llama-swap.
3. llama-swap on 127.0.0.1:8091: ``/running`` lists the proxy URL and the
   child command line, unwrapped from ``sh -c`` when needed.
4. The usual ports: llama.cpp's 8080 / 8096, Strata's 8092 / 8095.

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
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import time as _wall_time

HTTP_TIMEOUT = 1.0
# How many finished requests the monitor remembers for the table. One row is
# one request, so this caps scrollback, not screen space.
HISTORY_RECORDS = 40
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
# Strata's own default port is 8095; 8092 is where llama-swap runs it here.
STRATA_DEFAULT_PORTS = (8092, 8095)
# The file the `strata` CLI itself reads (bin/strata: STRATA_API_KEY).
STRATA_KEY_HINTS = ("~/.config/strata/api-key",)

# llama.cpp returns 401 without the key, and some builds answer 403.
AUTH_ERRORS = ("HTTP 401", "HTTP 403")


@dataclass(frozen=True)
class LlamaTarget:
    """An inference server we can talk to (llama.cpp or Strata)."""

    base_url: str
    api_key: str | None = None
    model: str = ""
    origin: str = ""
    engine: str = "llama"  # llama | strata


@dataclass(slots=True)
class RequestRecord:
    """One inference request, finished or still running, for the table."""

    at: float = 0.0  # wall-clock start, epoch seconds
    status: str = "done"  # running | done | stop | length | cancelled
    prompt_tokens: int = 0
    reused_tokens: int = 0
    output_tokens: int = 0
    tok_s: float | None = None
    duration_s: float | None = None


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
    # Newest first. Strata serves its own list; for llama.cpp the monitor
    # records each request as it finishes.
    history: tuple[RequestRecord, ...] = ()
    # True on the first request of a session, so the tiles can start a fresh
    # min/max instead of carrying the previous session's numbers.
    new_session: bool = False


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


def _key_from_hints(hints: tuple[str, ...]) -> str | None:
    """The first key in the first key file that exists."""
    for hint in hints:
        key = _read_key_file(hint)
        if key:
            return key
    return None


def _unwrap_shell(argv: list[str]) -> list[str]:
    """Flatten ``sh -c 'cd DIR && exec CMD'`` into CMD's arguments.

    llama-swap wraps its children that way, and the flags that matter
    (``--port``, ``--config``) then sit inside one quoted argument where
    :func:`_arg` cannot see them. ``cd DIR`` is kept so a relative
    ``--config`` can still be resolved.
    """
    shells = ("sh", "bash", "dash", "zsh", "ash")
    for index, item in enumerate(argv):
        if item != "-c" or index + 1 >= len(argv):
            continue
        if index == 0 or Path(argv[index - 1]).name not in shells:
            continue
        inner = argv[index + 1]
        tokens: list[str] = []
        for part in inner.replace(";", "&&").split("&&"):
            try:
                tokens.extend(shlex.split(part))
            except ValueError:
                return argv
        if tokens:
            return tokens
    return argv


def _is_strata_argv(argv: list[str]) -> bool:
    """True for Strata's Python server: ``serve/server.py --engine strata``."""
    if _arg(argv, "--engine") != "strata":
        return False
    return any(Path(token).name == "server.py" for token in argv)


def _strata_key(argv: list[str], cwd: str | None = None) -> str | None:
    """The key Strata accepts, from its own flag, config, or the CLI key file.

    Preferring the server's config over the client file keeps gpuctl working
    when the two have drifted apart.
    """
    key = _arg(argv, "--api-key")
    if key:
        # Strata accepts a comma-separated list ("k1,k2"); any one is enough.
        return key.split(",")[0].strip() or None
    config = _arg(argv, "--config")
    if config:
        path = Path(config).expanduser()
        if not path.is_absolute() and cwd:
            path = Path(cwd) / path
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            key = data.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.split(",")[0].strip()
    return _key_from_hints(STRATA_KEY_HINTS)


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


def _target_from_argv(
    argv: list[str],
    origin: str,
    model: str = "",
    default_port: str | None = "8080",
    cwd: str | None = None,
) -> LlamaTarget | None:
    """Build a target from a server command line.

    ``default_port`` is llama-server's 8080 for a real process. llama-swap
    passes ``None``: its entries carry an explicit proxy URL, and guessing
    8080 for a child that uses another port produces a dead candidate.
    """
    if not argv:
        return None
    # Only long flags: on llama-server `-p` is --prompt, not --port, and a
    # short-flag guess would read a prompt as a port number.
    host = _arg(argv, "--host", "--listen") or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = _arg(argv, "--port") or default_port
    if port is None or not port.isdigit():
        return None

    engine = "strata" if _is_strata_argv(argv) else "llama"
    if engine == "strata":
        key = _strata_key(argv, cwd)
    else:
        key = None
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
        engine=engine,
    )


def _candidate_from_env() -> list[LlamaTarget]:
    url = os.environ.get("GPU_CTL_LLAMA_URL", "").strip()
    if not url:
        return []
    if "://" not in url:
        url = f"http://{url}"
    key = os.environ.get("GPU_CTL_LLAMA_KEY") or None
    key_file = os.environ.get("GPU_CTL_LLAMA_KEY_FILE")
    if key is None and key_file:
        key = _read_key_file(key_file)
    return [LlamaTarget(base_url=url.rstrip("/"), api_key=key, origin="env")]


def _candidate_from_processes() -> list[LlamaTarget]:
    """Every llama-server or Strata server owned by this user, newest first."""
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
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not argv:
            continue
        exe = os.path.basename(argv[0])
        if exe in ("llama-server", "llama-server.exe", "rpc-server"):
            target = _target_from_argv(argv, origin="llama-server")
        elif _is_strata_argv(argv):
            # Strata names --config relative to its working directory.
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                cwd = None
            target = _target_from_argv(argv, origin="strata", cwd=cwd)
        else:
            continue
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
        argv: list[str] = []
        if isinstance(cmd, str) and cmd:
            try:
                argv = _unwrap_shell(shlex.split(cmd))
            except ValueError:
                argv = []
        target = None
        if argv:
            # The proxy URL is authoritative for llama-swap children; a
            # missing --port must not fall back to 8080.
            target = _target_from_argv(
                argv, origin="llama-swap", model=name, default_port=None
            )
        if target is None:
            proxy = str(item.get("proxy") or "").rstrip("/")
            if not proxy.startswith("http"):
                continue
            target = LlamaTarget(
                base_url=proxy,
                model=name,
                origin="llama-swap",
                engine="strata" if _is_strata_argv(argv) else "llama",
            )
        if target.api_key is None:
            key = (
                _strata_key(argv)
                if target.engine == "strata"
                else _key_from_hints(KEY_FILE_HINTS)
            )
            if key:
                target = replace(target, api_key=key)
        targets.append(target)
    return targets


def _candidate_from_defaults() -> list[LlamaTarget]:
    targets: list[LlamaTarget] = []
    for port in DEFAULT_PORTS:
        targets.append(
            LlamaTarget(
                base_url=f"http://127.0.0.1:{port}",
                api_key=_key_from_hints(KEY_FILE_HINTS),
                origin="default port",
            )
        )
    for port in STRATA_DEFAULT_PORTS:
        targets.append(
            LlamaTarget(
                base_url=f"http://127.0.0.1:{port}",
                api_key=_key_from_hints(STRATA_KEY_HINTS),
                origin="strata port",
                engine="strata",
            )
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


def fetch_strata_metrics(target: LlamaTarget) -> tuple[dict | None, str | None]:
    """Strata's JSON /metrics: engine facts under ``engine``, live counters
    under ``live``. (``/metrics/prometheus`` is the text format, not this.)"""
    data, error = _http_json(f"{target.base_url}/metrics", target.api_key)
    if error is None and not (
        isinstance(data, dict) and isinstance(data.get("live"), dict)
    ):
        return None, "unexpected /metrics payload"
    return data, error


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


def _as_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # JSON has no NaN/Infinity, but a stale or hand-written document can.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


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
    """Polls one inference server and derives live token rates.

    Not thread-safe by accident: :meth:`poll` takes an internal lock so the
    Textual worker thread can call it while the UI thread reads
    :attr:`last_snapshot`. llama.cpp counters are diffed between polls;
    Strata reports its own rates in ``/metrics`` and needs no diffing.
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
        self._history: deque[RequestRecord] = deque(maxlen=HISTORY_RECORDS)
        self._pending: RequestRecord | None = None
        self._pending_started = 0.0
        self.last_snapshot = InferenceSnapshot()

    # -- discovery -----------------------------------------------------

    @property
    def target(self) -> LlamaTarget | None:
        return self._target

    def _select_target(self) -> LlamaTarget | None:
        for candidate in discover_targets():
            if candidate.engine == "strata":
                metrics, _ = fetch_strata_metrics(candidate)
                if metrics is None:
                    continue
                if not candidate.model:
                    engine = metrics.get("engine")
                    model = str((engine or {}).get("model") or "")
                    candidate = replace(candidate, model=model)
            else:
                slots, _ = fetch_slots(candidate)
                if slots is None:
                    continue
                if not candidate.model:
                    props = fetch_props(candidate)
                    model = str(
                        props.get("model_alias") or props.get("model_path") or ""
                    )
                    name = Path(model).name
                    if name.endswith(".gguf"):
                        name = name[: -len(".gguf")]
                    candidate = replace(candidate, model=name)
            self._target = candidate
            self._model = candidate.model
            self._prev = None
            self._live_gen = self._live_prefill = None
            self._last_gen = self._last_prefill = None
            self._requests = 0
            self._history.clear()
            self._pending = None
            self._failures = 0
            self._state = "offline"
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
            history=tuple(self._history),
            ctx_total=ctx_total,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _failed(self, error: str) -> InferenceSnapshot:
        """Count a failed poll and drop a dead target after a few in a row."""
        self._failures += 1
        if self._failures >= DROP_AFTER_FAILURES:
            # Server restarted on another port / with another key.
            self._target = None
            self._failures = 0
            # Give it a moment before scanning /proc and probing again.
            self._next_discovery = time.monotonic() + 2.0
        return self._offline(error)

    def _poll_locked(self) -> InferenceSnapshot:
        now = time.monotonic()
        if self._target is None:
            # Rediscovery is /proc + a few HTTP probes; do it on an interval,
            # not on every poll, and only while nothing is answering.
            if now < self._next_discovery:
                return self._offline(
                    self.last_snapshot.error or "no inference server found"
                )
            self._next_discovery = now + REDISCOVER_S
            if self._select_target() is None:
                return self._offline("no inference server found")

        assert self._target is not None
        if self._target.engine == "strata":
            metrics, error = fetch_strata_metrics(self._target)
            if metrics is None:
                return self._failed(error or "server did not answer")
            self._failures = 0
            return self._strata_snapshot(metrics)

        slots, error = fetch_slots(self._target)
        if slots is None:
            return self._failed(error or "server did not answer")
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
        new_session = False
        if task_changed:
            # A new request took the slot: bank the finished one's numbers.
            self._commit_live()
            self._finish_pending()
            if processing and task is not None:
                self._requests += 1
                self._start_pending(now, prompt_total, prompt_cached, decoded)
                # First request of a session (first ever, or the first after
                # the server went quiet): consumers use this to reset their
                # session-scoped min/max.
                new_session = self._requests == 1
        elif phase_changed:
            self._commit_live()

        if processing and self._pending is None:
            # A request was already running when the app started.
            self._start_pending(now, prompt_total, prompt_cached, decoded)

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

        # After the rate block above: the running row wants this poll's rate,
        # not the previous poll's.
        if processing and self._pending is not None:
            self._update_pending(now, prompt_total, prompt_cached, decoded)

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
            self._finish_pending()
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
            history=tuple(self._history),
            new_session=new_session,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _strata_snapshot(self, metrics: dict) -> InferenceSnapshot:
        """Map one Strata ``/metrics`` document onto the tile fields.

        Strata reports its own rates, so nothing is diffed here. Its ``live``
        block is empty unless a request is running (``reading`` or
        ``generating``); the newest finished request fills the idle view.
        """
        live = metrics.get("live") if isinstance(metrics.get("live"), dict) else {}
        engine = (
            metrics.get("engine") if isinstance(metrics.get("engine"), dict) else {}
        )
        recent = [r for r in (metrics.get("requests") or []) if isinstance(r, dict)]

        model = str(engine.get("model") or "")
        if model:
            self._model = model

        raw = str(live.get("state") or "idle").lower()
        failure = engine.get("error")
        if raw == "error" or failure:
            message = (
                failure if isinstance(failure, str) and failure else "engine error"
            )
            return self._offline(message)
        state = {"reading": "prefill", "generating": "generate"}.get(raw, "idle")
        previous = self._state
        was_busy = previous in ("prefill", "generate")
        new_session = state != "idle" and not was_busy
        if new_session:
            self._requests += 1
        self._state = state
        new_session = new_session and self._requests == 1

        prompt_tokens = max(_as_int(live.get("prompt_tokens")), 0)
        generated = max(_as_int(live.get("generated")), 0)
        prompt_total = max(_as_int(live.get("prompt_total")), 0)
        prompt_processed = max(_as_int(live.get("prompt_read")), 0)
        ctx_used = prompt_tokens + generated

        if state == "generate":
            # /metrics keeps prompt_read/prompt_total only while reading.
            prompt_total = prompt_total or prompt_tokens
            prompt_processed = prompt_total
        elif state == "prefill":
            prompt_total = prompt_total or prompt_tokens
        elif recent:
            last = recent[0]  # newest first: Strata serves hist[::-1]
            prompt_tokens = max(_as_int(last.get("prompt_tokens")), 0)
            generated = max(_as_int(last.get("output_tokens")), 0)
            prompt_total = max(_as_int(last.get("prompt_total")), 0) or prompt_tokens
            prompt_processed = prompt_total
            ctx_used = prompt_tokens + generated

        gen_tps: float | None = None
        prefill_tps: float | None = None
        if state == "generate":
            gen_tps = _as_float(live.get("tok_s"))
            if gen_tps is None:
                gen_tps = _as_float(live.get("tok_s_mean"))
        elif state == "prefill":
            prefill_tps = _as_float(live.get("prefill_tok_s_mean"))
            elapsed = _as_float(live.get("elapsed_s"))
            if prefill_tps is None and prompt_processed and elapsed:
                prefill_tps = prompt_processed / elapsed

        # Bank the prefill rate when the prompt is done, the way the llama.cpp
        # path commits its live rate. A finished request's own prompt_ms also
        # covers the cache restore, so dividing by it understates a cached
        # prompt (42 new tokens over 1.5 s) and overstates a cold one.
        if state == "prefill":
            if prefill_tps is not None:
                self._live_prefill = prefill_tps
        elif previous == "prefill" and self._live_prefill is not None:
            self._last_prefill = self._live_prefill
            self._live_prefill = None

        if recent:
            last = recent[0]  # newest first: Strata serves hist[::-1]
            decode = _as_float(last.get("decode_tok_s"))
            if decode is None:
                output = _as_int(last.get("output_tokens"))
                decode_ms = _as_float(last.get("decode_ms"))
                decode = output / (decode_ms / 1000.0) if output and decode_ms else None
            self._last_gen = decode

        snapshot = InferenceSnapshot(
            ok=True,
            error=None,
            origin=self._target.origin,
            model=self._model or "strata",
            state=state,
            gen_tps=gen_tps,
            prefill_tps=prefill_tps,
            last_gen_tps=self._last_gen,
            last_prefill_tps=self._last_prefill,
            prompt_tokens=prompt_total + generated,
            prompt_processed=prompt_processed,
            prompt_cached=0,
            prompt_total=prompt_total,
            decoded_tokens=generated,
            ctx_used=ctx_used,
            ctx_total=max(
                _as_int(engine.get("max_context")), _as_int(engine.get("context"))
            ),
            requests=self._requests,
            history=_history_from_metrics(recent, state, live, prompt_tokens, generated),
            new_session=new_session,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _commit_live(self) -> None:
        if self._live_gen is not None:
            self._last_gen = self._live_gen
        if self._live_prefill is not None:
            self._last_prefill = self._live_prefill
        self._live_gen = self._live_prefill = None

    # -- request history (llama.cpp has no list of its own) -------------

    def _start_pending(
        self, now: float, prompt: int, reused: int, output: int
    ) -> None:
        self._pending = RequestRecord(
            at=_wall_time(),
            status="running",
            prompt_tokens=max(prompt, 0),
            reused_tokens=max(reused, 0),
            output_tokens=max(output, 0),
        )
        self._pending_started = now

    def _update_pending(
        self, now: float, prompt: int, reused: int, output: int
    ) -> None:
        pending = self._pending
        if pending is None:
            return
        pending.prompt_tokens = max(prompt, 0)
        pending.reused_tokens = max(reused, 0)
        pending.output_tokens = max(output, 0)
        pending.tok_s = self._live_gen
        pending.duration_s = round(now - self._pending_started, 1)

    def _finish_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        pending.status = "done" if pending.output_tokens else "cancelled"
        self._history.appendleft(pending)
        self._pending = None


def _history_from_metrics(
    recent: list[dict],
    state: str,
    live: dict,
    prompt_tokens: int,
    output: int,
) -> tuple[RequestRecord, ...]:
    """Map Strata's own request list onto the table's records.

    Strata serves that list newest first and writes the starting wall clock
    under ``time``, the engine's finish reason under ``finish``.
    """
    records: list[RequestRecord] = []
    for item in recent[:HISTORY_RECORDS]:
        records.append(
            RequestRecord(
                at=_as_float(item.get("time")) or 0.0,
                status=str(item.get("finish") or "done"),
                prompt_tokens=_as_int(item.get("prompt_tokens")),
                reused_tokens=_as_int(item.get("reused")),
                output_tokens=_as_int(item.get("output_tokens")),
                tok_s=_as_float(item.get("decode_tok_s")),
                duration_s=_as_float(item.get("duration_s")),
            )
        )
    if state != "idle":
        elapsed = _as_float(live.get("elapsed_s")) or 0.0
        records.insert(
            0,
            RequestRecord(
                at=_wall_time() - elapsed,
                status="running",
                prompt_tokens=prompt_tokens,
                reused_tokens=_as_int(live.get("reused")),
                output_tokens=output,
                tok_s=_as_float(live.get("tok_s")),
                duration_s=round(elapsed, 1) if elapsed else None,
            ),
        )
    return tuple(records)


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
