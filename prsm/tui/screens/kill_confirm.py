"""Kill agent confirmation modal — asks before removing an agent.

Shown when the user presses Delete on an agent in the tree or selects
"Kill Agent" from the context menu.  Prevents accidental removal of
agents that may still be doing useful work.

Returns "kill" if confirmed, None if cancelled.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from prsm.shared.models.agent import AgentNode


class KillConfirmScreen(ModalScreen[str | None]):
    """Modal confirmation dialog before killing/removing an agent."""

    CSS_PATH = "../styles/modal.tcss"

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("y", "confirm_kill", "Confirm"),
    ]

    _MOUNT_GUARD_SECONDS = 0.3

    def __init__(self, agent: AgentNode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent

    def compose(self) -> ComposeResult:
        safe_name = self.agent.name.replace("[", "\\[")
        short_id = self.agent.id[:8]
        state_label = self.agent.state.value if self.agent.state else "unknown"

        with Vertical(id="kill-confirm-dialog"):
            yield Label("Remove agent?")
            yield Static(
                f"[bold]{safe_name}[/bold] [dim]{short_id}[/dim]\n"
                f"State: [yellow]{state_label}[/yellow]",
                id="kill-confirm-details",
                markup=True,
            )
            yield Static(
                "[dim]This will remove the agent from the tree and session. "
                "If the agent is running, it will be killed.[/dim]",
                classes="info-text",
                markup=True,
            )
            yield Button(
                "[y] Confirm — Remove Agent",
                id="btn-kill-confirm",
                variant="error",
            )
            yield Button("[Esc] Cancel", id="btn-kill-cancel")

    def on_mount(self) -> None:
        import time

        self._mount_time = time.monotonic()
        try:
            self.query_one("#btn-kill-confirm", Button).focus()
        except Exception:
            pass

    def _is_guarded(self) -> bool:
        import time

        elapsed = time.monotonic() - getattr(self, "_mount_time", 0)
        return elapsed < self._MOUNT_GUARD_SECONDS

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._is_guarded():
            return
        if event.button.id == "btn-kill-confirm":
            self.dismiss("kill")
        else:
            self.dismiss(None)

    def key_up(self) -> None:
        self._move_focus(-1)

    def key_down(self) -> None:
        self._move_focus(1)

    _BUTTON_ORDER = ["btn-kill-confirm", "btn-kill-cancel"]

    def _move_focus(self, direction: int) -> None:
        focused = self.focused
        if focused is None or not hasattr(focused, "id"):
            try:
                self.query_one("#btn-kill-confirm", Button).focus()
            except Exception:
                pass
            return

        try:
            current_idx = self._BUTTON_ORDER.index(focused.id)
        except ValueError:
            return
        new_idx = (current_idx + direction) % len(self._BUTTON_ORDER)
        try:
            self.query_one(f"#{self._BUTTON_ORDER[new_idx]}", Button).focus()
        except Exception:
            pass

    def action_cancel(self) -> None:
        if self._is_guarded():
            return
        self.dismiss(None)

    def action_confirm_kill(self) -> None:
        if self._is_guarded():
            return
        self.dismiss("kill")
