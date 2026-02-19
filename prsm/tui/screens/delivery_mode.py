"""Delivery mode selection modal for prompt injection.

Shows when the user sends a prompt while an agent is running.
Three modes:
- interrupt: Cancel current task and replace with this prompt.
- inject: Finish current tool call, then process this prompt.
- queue: Run this prompt after the current task completes.
"""
from __future__ import annotations

import time

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

# Map button IDs to delivery mode strings
_MODE_MAP = {
    "btn-interrupt": "interrupt",
    "btn-inject": "inject",
    "btn-queue": "queue",
    "btn-cancel": "cancel",
}

# Ordered list of button IDs for arrow-key navigation
_BUTTON_ORDER = ["btn-interrupt", "btn-inject", "btn-queue", "btn-cancel"]


class DeliveryModeScreen(ModalScreen[str | None]):
    """Modal dialog for selecting prompt delivery mode."""

    def __init__(self) -> None:
        super().__init__()
        # Start guard at creation time to cover key events that may arrive
        # before on_mount runs (same Enter key that opened this modal).
        self._guard_until = time.monotonic() + self._MOUNT_GUARD_SECONDS
        # Explicitly swallow the first Enter after opening. This avoids
        # accidental default selection from key carry-through/repeat.
        self._ignore_next_enter = True

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("i", "select_interrupt", "Interrupt"),
        ("j", "select_inject", "Inject"),
        ("q", "select_queue", "Queue"),
        ("up", "focus_prev", "Previous option"),
        ("down", "focus_next", "Next option"),
        ("left", "focus_prev", "Previous option"),
        ("right", "focus_next", "Next option"),
        ("enter", "submit_focused", "Submit"),
    ]

    # Guard period (seconds) to ignore input right after mount.
    # Prevents the Enter key that submitted the prompt from also
    # immediately pressing the focused button and dismissing the modal.
    _MOUNT_GUARD_SECONDS = 0.3

    CSS = """
    DeliveryModeScreen {
        align: center middle;
    }
    DeliveryModeScreen > Vertical {
        width: 60;
        height: auto;
        max-height: 20;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    DeliveryModeScreen Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    DeliveryModeScreen Button {
        width: 100%;
        margin-bottom: 1;
    }
    DeliveryModeScreen Button:focus {
        border: tall $accent;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Agent is busy. How should this prompt be delivered?")
            yield Button(
                "[i] Interrupt (cancel + replace)",
                id="btn-interrupt",
                variant="error",
            )
            yield Button(
                "[j] Inject after tool call",
                id="btn-inject",
                variant="warning",
            )
            yield Button(
                "[q] Queue after task",
                id="btn-queue",
                variant="primary",
            )
            yield Button("[Esc] Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        """Default focus to queue mode and re-arm post-mount input guard."""
        # Start (or refresh) the guard when the modal is actually mounted.
        # If mount/render is delayed, a guard that started only in __init__
        # may have already expired before the first visible frame.
        self._guard_until = max(
            getattr(self, "_guard_until", 0.0),
            time.monotonic() + self._MOUNT_GUARD_SECONDS,
        )
        try:
            self.query_one("#btn-queue", Button).focus()
        except Exception:
            pass

    def _is_guarded(self) -> bool:
        """Return True if still within the post-mount guard period.

        This prevents stray key events (e.g. the Enter that submitted
        the prompt) from immediately dismissing the modal.
        """
        return time.monotonic() < getattr(self, "_guard_until", 0.0)

    def on_key(self, event: events.Key) -> None:
        """Consume the first Enter keypress after modal open."""
        if event.key == "enter" and self._ignore_next_enter:
            self._ignore_next_enter = False
            event.stop()
            event.prevent_default()
            return
        if event.key != "enter":
            self._ignore_next_enter = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._is_guarded():
            return
        self.dismiss(_MODE_MAP.get(event.button.id))

    def action_focus_prev(self) -> None:
        """Move focus to previous button."""
        self._move_focus(-1)

    def action_focus_next(self) -> None:
        """Move focus to next button."""
        self._move_focus(1)

    def _move_focus(self, direction: int) -> None:
        """Move focus up or down in the button list."""
        focused = self.focused
        if focused is None or not hasattr(focused, "id"):
            # Default to queue when focus is lost.
            try:
                self.query_one("#btn-queue", Button).focus()
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
        """Escape key dismisses the modal."""
        if self._is_guarded():
            return
        self.dismiss("cancel")

    def action_select_interrupt(self) -> None:
        """Hotkey 'i' selects interrupt mode."""
        if self._is_guarded():
            return
        self.dismiss("interrupt")

    def action_select_inject(self) -> None:
        """Hotkey 'j' selects inject mode."""
        if self._is_guarded():
            return
        self.dismiss("inject")

    def action_select_queue(self) -> None:
        """Hotkey 'q' selects queue mode."""
        if self._is_guarded():
            return
        self.dismiss("queue")

    def action_submit_focused(self) -> None:
        """Enter submits the currently focused option."""
        if self._ignore_next_enter:
            self._ignore_next_enter = False
            return
        if self._is_guarded():
            return
        focused = self.focused
        if focused is None or not hasattr(focused, "id"):
            return
        self.dismiss(_MODE_MAP.get(focused.id))
