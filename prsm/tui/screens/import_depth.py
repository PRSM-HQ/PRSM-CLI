"""Import depth selection modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ImportDepthScreen(ModalScreen[str | None]):
    """Prompt user for transcript import depth."""

    CSS_PATH = "../styles/modal.tcss"
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="import-depth-dialog"):
            yield Static(
                "[bold $primary]Choose Import Depth[/bold $primary]",
                id="import-depth-title",
                markup=True,
            )
            yield Static(
                "How much transcript history should be imported?",
                id="import-depth-hint",
            )
            yield Button(
                "Recent 200 (Recommended)",
                id="import-depth-200",
                classes="import-depth-choice",
            )
            yield Button(
                "Recent 500",
                id="import-depth-500",
                classes="import-depth-choice",
            )
            yield Button(
                "Full Transcript",
                id="import-depth-full",
                classes="import-depth-choice",
            )
            with Horizontal(id="import-depth-actions"):
                yield Button("Cancel", id="import-depth-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "import-depth-cancel":
            self.dismiss(None)
            return
        if btn_id == "import-depth-200":
            self.dismiss("200")
            return
        if btn_id == "import-depth-500":
            self.dismiss("500")
            return
        if btn_id == "import-depth-full":
            self.dismiss("full")

    def action_cancel(self) -> None:
        self.dismiss(None)

