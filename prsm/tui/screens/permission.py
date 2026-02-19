"""Permission modal — asks user to approve/deny a tool call."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class PermissionScreen(ModalScreen[str]):
    """Modal dialog for tool permission requests.

    Returns one of: "allow", "allow_project", "allow_global",
    "deny", "deny_project", "view_agent"
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(
        self,
        tool_name: str,
        agent_name: str,
        arguments: str = "",
        agent_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.arguments = arguments
        self.agent_id = agent_id
        self.session_id = session_id

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Static(
                f"[bold $warning]Permission Request[/bold $warning]",
                id="permission-title",
            )
            yield Static(
                f"Agent [bold]{self.agent_name}[/bold] wants to use "
                f"[cyan]{self.tool_name}[/cyan]",
            )
            if self.arguments:
                yield Static(
                    self.arguments[:500],
                    id="permission-details",
                )
            with Horizontal(id="permission-buttons"):
                yield Button("Allow", variant="success", id="btn-allow")
                yield Button(
                    "Always (project)",
                    variant="warning",
                    id="btn-project",
                )
                yield Button(
                    "Always (all)",
                    variant="warning",
                    id="btn-global",
                )
                yield Button(
                    "Always reject",
                    variant="error",
                    id="btn-reject-project",
                )
                yield Button("Deny", variant="error", id="btn-deny")
                yield Button(
                    "View Agent", variant="default", id="btn-view-agent"
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result_map = {
            "btn-allow": "allow",
            "btn-project": "allow_project",
            "btn-global": "allow_global",
            "btn-reject-project": "deny_project",
            "btn-deny": "deny",
            "btn-view-agent": "view_agent",
        }
        self.dismiss(result_map.get(event.button.id, "deny"))
