"""Modal screens: sudo password prompt, typed fan-percentage entry, and the
fallback screen shown when no amdgpu hwmon device can be found."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from .privileged import verify_password


class PasswordModal(ModalScreen[str | None]):
    """Ask for the sudo password, verify it, and dismiss with it on success.

    Dismisses with `None` if the user cancels. The password returned has
    already been confirmed valid via `sudo -v`, so callers can use it
    immediately for a privileged write without re-checking.
    """

    DEFAULT_CSS = """
    PasswordModal {
        align: center middle;
    }

    #pw-dialog {
        width: 52;
        height: auto;
        border: round $warning;
        background: $panel;
        padding: 1 2;
    }

    #pw-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        padding-bottom: 1;
    }

    #pw-error {
        color: $error;
        height: 1;
        padding-top: 1;
    }

    #pw-buttons {
        padding-top: 1;
        align: right middle;
        height: auto;
    }

    #pw-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="pw-dialog"):
            yield Label("Sudo password required to change fan settings", id="pw-title")
            yield Input(placeholder="password", password=True, id="pw-input")
            yield Label("", id="pw-error")
            with Horizontal(id="pw-buttons"):
                yield Button("Cancel", id="pw-cancel")
                yield Button("OK", id="pw-ok", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#pw-input", Input).focus()

    @on(Input.Submitted, "#pw-input")
    async def _submitted(self, event: Input.Submitted) -> None:
        await self._try_submit()

    @on(Button.Pressed, "#pw-ok")
    async def _ok(self, event: Button.Pressed) -> None:
        await self._try_submit()

    @on(Button.Pressed, "#pw-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    async def _try_submit(self) -> None:
        pw_input = self.query_one("#pw-input", Input)
        error = self.query_one("#pw-error", Label)
        password = pw_input.value
        if not password:
            error.update("Password can't be empty.")
            return

        ok_btn = self.query_one("#pw-ok", Button)
        cancel_btn = self.query_one("#pw-cancel", Button)
        ok_btn.disabled = True
        cancel_btn.disabled = True
        error.update("Checking...")

        result = await verify_password(password)

        if result.ok:
            self.dismiss(password)
            return

        error.update(result.message or "Authentication failed.")
        pw_input.value = ""
        pw_input.focus()
        ok_btn.disabled = False
        cancel_btn.disabled = False


class SetSpeedModal(ModalScreen[int | None]):
    """Ask for an exact fan speed percentage (0-100)."""

    DEFAULT_CSS = """
    SetSpeedModal {
        align: center middle;
    }

    #speed-dialog {
        width: 44;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }

    #speed-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        padding-bottom: 1;
    }

    #speed-error {
        color: $error;
        height: 1;
        padding-top: 1;
    }

    #speed-buttons {
        padding-top: 1;
        align: right middle;
        height: auto;
    }

    #speed-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="speed-dialog"):
            yield Label("Set fan speed (0-100%)", id="speed-title")
            yield Input(placeholder="e.g. 65", id="speed-input", type="integer")
            yield Label("", id="speed-error")
            with Horizontal(id="speed-buttons"):
                yield Button("Cancel", id="speed-cancel")
                yield Button("Set", id="speed-ok", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#speed-input", Input).focus()

    @on(Input.Submitted, "#speed-input")
    def _submitted(self, event: Input.Submitted) -> None:
        self._try_submit()

    @on(Button.Pressed, "#speed-ok")
    def _ok(self, event: Button.Pressed) -> None:
        self._try_submit()

    @on(Button.Pressed, "#speed-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def _try_submit(self) -> None:
        raw = self.query_one("#speed-input", Input).value.strip()
        error = self.query_one("#speed-error", Label)
        try:
            pct = int(raw)
        except ValueError:
            error.update("Enter a whole number between 0 and 100.")
            return
        if not (0 <= pct <= 100):
            error.update("Must be between 0 and 100.")
            return
        self.dismiss(pct)


class HwmonMissingApp(App):
    """Minimal fallback app shown when no amdgpu hwmon device is found."""

    CSS = """
    Screen {
        align: center middle;
    }
    #box {
        width: 70;
        border: round $error;
        padding: 1 2;
    }
    #title {
        text-style: bold;
        color: $error;
        padding-bottom: 1;
    }
    """
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static("LACT-TUI: no amdgpu GPU found", id="title")
            yield Static(self._message)
            yield Static("\nPress q to quit.")
