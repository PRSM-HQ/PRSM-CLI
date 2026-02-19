"""File context modal — shows why an agent made a file change."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static


class FileContextScreen(ModalScreen[None]):
    """Modal dialog showing the context for a file change.

    Displays the file path, tool used, and the agent's rationale (from
    preceding messages/thinking content).
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(
        self,
        file_path: str,
        tool_name: str,
        agent_name: str,
        rationale: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.file_path = file_path
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.rationale = rationale or "No additional context available."

    def compose(self) -> ComposeResult:
        with Vertical(id="file-context-dialog"):
            yield Static(
                f"[bold $primary]File Change Context[/bold $primary]",
                id="file-context-title",
            )
            yield Static(
                f"Agent [bold]{self.agent_name}[/bold] modified "
                f"[cyan]{self.file_path}[/cyan] using [yellow]{self.tool_name}[/yellow]",
                id="file-context-header",
            )
            with VerticalScroll(id="file-context-content"):
                yield Markdown(self.rationale)
            yield Button("Close", variant="primary", id="btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.dismiss()
