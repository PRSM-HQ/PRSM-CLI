"""Resend confirmation modal — asks whether to revert file changes.

Shown when the user clicks a previous prompt to resend it.
The snapshot has already been auto-restored (session state), and this
dialog asks whether file changes made after that point should also
be reverted (Cursor-like behavior).

Three options:
- revert: Revert all file changes (restore working tree to snapshot state).
- keep: Keep current file state after conversation rewind.
- cancel: Dismiss this dialog and keep current file state.

Also offers a "remember" checkbox to persist the choice.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

# Map button IDs to result strings
_RESULT_MAP = {
    "btn-revert": "revert",
    "btn-keep": "keep",
    "btn-cancel": "cancel",
}

# Button ordering for arrow-key navigation
_BUTTON_ORDER = ["btn-revert", "btn-keep", "btn-cancel"]


class ResendConfirmScreen(ModalScreen[str | None]):
    """Modal dialog asking whether to revert file changes on resend."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("r", "select_revert", "Revert"),
        ("k", "select_keep", "Keep"),
    ]

    _MOUNT_GUARD_SECONDS = 0.3

    CSS = """
    ResendConfirmScreen {
        align: center middle;
    }
    ResendConfirmScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 24;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    ResendConfirmScreen Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    ResendConfirmScreen .info-text {
        width: 100%;
        margin-bottom: 1;
        color: $text-muted;
    }
    ResendConfirmScreen Button {
        width: 100%;
        margin-bottom: 1;
    }
    ResendConfirmScreen Button:focus {
        border: tall $accent;
    }
    """

    def __init__(
        self,
        file_count: int = 0,
        prompt_preview: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._file_count = file_count
        self._prompt_preview = prompt_preview[:60]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Resend prompt from earlier in conversation")
            if self._prompt_preview:
                safe_preview = self._prompt_preview.replace("[", "\\[")
                yield Static(
                    f'[dim italic]"{safe_preview}…"[/dim italic]',
                    classes="info-text",
                    markup=True,
                )
            if self._file_count > 0:
                yield Static(
                    f"[yellow]{self._file_count}[/yellow] file(s) were changed "
                    "after this point.",
                    classes="info-text",
                    markup=True,
                )
            else:
                yield Static(
                    "[dim]No tracked file changes after this point.[/dim]",
                    classes="info-text",
                    markup=True,
                )
            yield Button(
                "[r] Revert files to this point",
                id="btn-revert",
                variant="warning",
            )
            yield Button(
                "[k] Keep current files",
                id="btn-keep",
                variant="primary",
            )
            yield Button("[Esc] Dismiss", id="btn-cancel")

    def on_mount(self) -> None:
        import time

        self._mount_time = time.monotonic()
        try:
            self.query_one("#btn-revert", Button).focus()
        except Exception:
            pass

    def _is_guarded(self) -> bool:
        import time

        elapsed = time.monotonic() - getattr(self, "_mount_time", 0)
        return elapsed < self._MOUNT_GUARD_SECONDS

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._is_guarded():
            return
        self.dismiss(_RESULT_MAP.get(event.button.id))

    def key_up(self) -> None:
        self._move_focus(-1)

    def key_down(self) -> None:
        self._move_focus(1)

    def _move_focus(self, direction: int) -> None:
        focused = self.focused
        if focused is None or not hasattr(focused, "id"):
            try:
                self.query_one("#btn-revert", Button).focus()
            except Exception:
                pass
            return

        try:
            current_idx = _BUTTON_ORDER.index(focused.id)
        except ValueError:
            return
        new_idx = (current_idx + direction) % len(_BUTTON_ORDER)
        try:
            self.query_one(f"#{_BUTTON_ORDER[new_idx]}", Button).focus()
        except Exception:
            pass

    def action_cancel(self) -> None:
        if self._is_guarded():
            return
        self.dismiss("cancel")

    def action_select_revert(self) -> None:
        if self._is_guarded():
            return
        self.dismiss("revert")

    def action_select_keep(self) -> None:
        if self._is_guarded():
            return
        self.dismiss("keep")
