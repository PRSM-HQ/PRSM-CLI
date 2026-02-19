"""Agent context menu — right-click popup for agent actions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from prsm.shared.models.agent import AgentNode


class AgentContextMenu(ModalScreen[str | None]):
    """Floating context menu for agent actions.

    Returns the chosen action string, or None if dismissed.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, agent: AgentNode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent

    def compose(self) -> ComposeResult:
        with Vertical(id="context-menu"):
            yield Static(
                f"[bold]{self.agent.name}[/bold] [dim]{self.agent.id[:8]}[/dim]",
                id="ctx-title",
            )
            yield Button("View Context", variant="primary", id="ctx-view")
            yield Button("Kill Agent", variant="error", id="ctx-kill")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id.removeprefix("ctx-")
        self.dismiss(action)

    def on_click(self, event) -> None:
        """Dismiss if clicking outside the menu."""
        try:
            menu = self.query_one("#context-menu")
            if not menu.region.contains(event.screen_x, event.screen_y):
                self.dismiss(None)
        except Exception:
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)
