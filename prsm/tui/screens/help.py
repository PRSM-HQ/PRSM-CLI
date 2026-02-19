"""Help modal showing mouse-first actions and key mappings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class HelpScreen(ModalScreen[None]):
    """Display usage instructions and keyboard shortcuts."""

    CSS_PATH = "../styles/modal.tcss"
    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+h", "close", "Close"),
        ("meta+h", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(
                "[bold $primary]PRSM Help[/bold $primary]",
                id="help-title",
                markup=True,
            )
            yield Static(
                "[bold]Mouse-first controls[/bold]\n"
                "- `+` in input bar: start a new session\n"
                "- `🔍` in input bar: search and load saved sessions\n"
                "- `⚙` in input bar: open settings\n"
                "- `🔧 Model`: select orchestration model\n"
                "- Click agents in tree: switch active conversation\n\n"
                "[bold]Keyboard shortcuts[/bold]\n"
                "- `Ctrl+H` / `Cmd+H`: open help\n"
                "- `Ctrl+N`: new session\n"
                "- `Ctrl+S`: save session\n"
                "- `Ctrl+T`: focus agent tree\n"
                "- `Ctrl+E`: focus prompt input\n"
                "- `Ctrl+Q`: quit\n"
                "- `F1`: toggle tool log\n"
                "- `F2`: open settings\n"
                "- `Esc`: cancel/blur\n\n"
                "[bold]Prompt tips[/bold]\n"
                "- Use `/help` to list slash commands\n"
                "- Type `@` in prompt to attach files/directories\n"
                "- Press `Enter` to send, `Shift+Enter` for newline",
                id="help-body",
                markup=True,
            )
            yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
