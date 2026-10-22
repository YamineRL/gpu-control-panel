"""Privileged sysfs writes, authenticated via an in-TUI sudo password prompt.

We never shell out to `sudo` and let it prompt on the raw terminal -- that
scribbles over the Textual screen. Instead the app collects the password
through a masked modal Input, and every privileged write feeds it to sudo
via `-A` (askpass), not by piping it over stdin with `-S`.

`-S` was tried first and dropped: on this system it intermittently fails
with `pam_unix(sudo:auth): auth could not identify password` / `conversation
failed` even for a correct password -- a known-fragile interaction between
sudo's stdin-conversation reader and the PAM stack (e.g. under `use_pty`).
`-A` sidesteps that entirely: sudo execs the askpass program and reads the
password from *its* stdout, never touching the piped-stdin conversation
path. The askpass "program" here is a tiny throwaway shell script that
prints the password, written to the per-user tmpfs runtime dir (mode 0600,
owner-only) for just the duration of the subprocess call and unlinked
immediately after.

`sudo_write` runs a single `sudo -A ... tee <path>` process rather than
authenticating and writing in two separate sudo invocations. An earlier
version split them (a leftover from the `-S` design, where the password and
the value both needed stdin and so had to be kept apart) -- but with `-A`
the password never touches this process's stdin at all, so nothing was
gained by splitting, and splitting was actively broken: the on-disk sudo
timestamp/ticket written by the first process wasn't reliably visible to
the second under Textual (no shared controlling tty), so the follow-up
`sudo -n` call kept failing with "a password is required" right after a
successful, correct-password authentication step. One process, one ticket,
no cross-process handoff needed.

The password itself is kept in memory only (on the App instance), for the
lifetime of the process, and is never written to a persistent disk location.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from dataclasses import dataclass


class SudoError(RuntimeError):
    """Raised when a privileged write fails (wrong password, or other error)."""


@dataclass
class SudoWriteResult:
    ok: bool
    message: str = ""


def _runtime_dir() -> str:
    return os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()


def _make_askpass_script(password: str) -> str:
    """Write a one-shot askpass helper that prints `password`, return its path.

    On a wrong password, `sudo` doesn't just fail once -- it re-invokes the
    askpass program up to its internal `passwd_tries` (default 3), each one
    a real PAM authentication attempt. This system's pam_faillock locks the
    account after 3 consecutive failures, so without a guard a single typo
    could burn all 3 strikes in one call and lock the account for 10 minutes.
    A marker file makes the script answer only sudo's *first* call and exit
    nonzero (which sudo treats as "no password available", giving up
    immediately) on any retry, so one typo costs at most one strike.
    """
    fd, path = tempfile.mkstemp(prefix="lact-tui-askpass-", dir=_runtime_dir())
    marker = f"{path}.used"
    try:
        script = (
            "#!/bin/sh\n"
            f"[ -e {shlex.quote(marker)} ] && exit 1\n"
            f": > {shlex.quote(marker)}\n"
            f"printf '%s\\n' {shlex.quote(password)}\n"
        )
        os.write(fd, script.encode())
    finally:
        os.close(fd)
    os.chmod(path, 0o700)
    return path


def _classify_sudo_error(stderr: bytes) -> str:
    err = stderr.decode(errors="replace").strip()
    lowered = err.lower()
    # "no password was provided" / "a password is required" both show up
    # when our askpass script refuses a retry (see _make_askpass_script)
    # after the real, first attempt already failed -- same meaning to the
    # user as "Incorrect password."
    if (
        "incorrect password" in lowered
        or "sorry, try again" in lowered
        or "no password was provided" in lowered
        or "a password is required" in lowered
    ):
        return "Incorrect password."
    return err or "Authentication failed."


async def verify_password(password: str) -> SudoWriteResult:
    """Check that `password` is a valid sudo password for the current user.

    `-k` drops any cached ticket first so this is a real check of the
    password rather than a reuse of an existing one. No command is run.
    """
    askpass = _make_askpass_script(password)
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-A", "-k", "-v",
            env={**os.environ, "SUDO_ASKPASS": askpass},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except FileNotFoundError:
        return SudoWriteResult(False, "sudo not found on this system.")
    except asyncio.TimeoutError:
        return SudoWriteResult(False, "Authentication timed out.")
    finally:
        for f in (askpass, f"{askpass}.used"):
            try:
                os.unlink(f)
            except OSError:
                pass

    if proc.returncode == 0:
        return SudoWriteResult(True)
    return SudoWriteResult(False, _classify_sudo_error(stderr))


async def sudo_write(path: str, value: str, password: str) -> SudoWriteResult:
    """Write `value` to `path` as root in a single authenticated sudo call."""
    askpass = _make_askpass_script(password)
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-A", "-k", "tee", path,
            env={**os.environ, "SUDO_ASKPASS": askpass},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(input=f"{value}\n".encode()), timeout=15
        )
    except FileNotFoundError:
        return SudoWriteResult(False, "sudo not found on this system.")
    except asyncio.TimeoutError:
        return SudoWriteResult(False, "Write timed out.")
    finally:
        for f in (askpass, f"{askpass}.used"):
            try:
                os.unlink(f)
            except OSError:
                pass

    if proc.returncode == 0:
        return SudoWriteResult(True)
    return SudoWriteResult(False, _classify_sudo_error(stderr))


# Root side of the fan-curve loop. One `sudo` (one PAM auth) starts it, and
# the app then feeds it one pwm value (0-255) per line on stdin. Anything
# that is not a plain number in range is dropped, so the pipe cannot be used
# to write anything else. When stdin closes (quit, crash, SIGKILL of the
# app: the kernel closes the pipe in every case) the EXIT trap puts the fan
# back on firmware auto, so the GPU is never left on a stale manual speed.
_FAN_HELPER_SCRIPT = r"""
en="$1"; pwm="$2"
trap 'echo 2 > "$en"' EXIT
trap 'exit 0' HUP INT TERM
echo 1 > "$en" || exit 1
echo ready
while IFS= read -r v; do
    case "$v" in ''|*[!0-9]*) continue ;; esac
    [ "$v" -le 255 ] || continue
    # Nobody drains stderr after startup, so keep it quiet here: a pipe
    # that fills up would block the loop.
    echo "$v" > "$pwm" 2>/dev/null
done
"""


class RootFanHelper:
    """Long-lived root process that writes pwm1 for the software fan curve."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def alive(self) -> bool:
        return self._proc.returncode is None

    @classmethod
    async def start(
        cls, enable_path: str, pwm_path: str, password: str
    ) -> tuple["RootFanHelper | None", SudoWriteResult]:
        """Authenticate once, switch the fan to manual, return the helper."""
        askpass = _make_askpass_script(password)
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-A", "-k", "sh", "-c", _FAN_HELPER_SCRIPT,
                "lact-tui-fan", enable_path, pwm_path,
                env={**os.environ, "SUDO_ASKPASS": askpass},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=15)
        except FileNotFoundError:
            return None, SudoWriteResult(False, "sudo not found on this system.")
        except asyncio.TimeoutError:
            proc.stdin.close()
            return None, SudoWriteResult(False, "Authentication timed out.")
        finally:
            for f in (askpass, f"{askpass}.used"):
                try:
                    os.unlink(f)
                except OSError:
                    pass

        if line.strip() == b"ready":
            return cls(proc), SudoWriteResult(True)
        _, stderr = await proc.communicate()
        return None, SudoWriteResult(False, _classify_sudo_error(stderr))

    async def write_pwm(self, value: int) -> bool:
        if not self.alive:
            return False
        try:
            self._proc.stdin.write(f"{value}\n".encode())
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    async def stop(self) -> None:
        """Close the pipe. The helper restores firmware auto and exits."""
        if self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
