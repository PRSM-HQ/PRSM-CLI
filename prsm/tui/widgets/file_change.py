"""File change widget — shows file modifications with diff info and context."""

from __future__ import annotations

from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static
from textual.widget import Widget


def _esc(text: str) -> str:
    """Escape Rich markup characters in dynamic content."""
    return text.replace("[", "\\[")


class FileChangeWidget(Widget):
    """A collapsible widget that shows file changes with diff info.

    Displays file path, change type (create/modify/delete), and +/- line counts.
    When expanded, shows colored diff. Includes a "View Context" button to show
    why the agent made this change.
    """

    class ViewContextRequested(Message):
        """Posted when the user clicks the 'View Context' button."""

        def __init__(
            self,
            agent_id: str,
            file_path: str,
            tool_name: str,
            tool_call_id: str,
            message_index: int,
        ) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.file_path = file_path
            self.tool_name = tool_name
            self.tool_call_id = tool_call_id
            self.message_index = message_index

    class ViewAgentRequested(Message):
        """Posted when the user clicks the 'View Agent' button."""

        def __init__(
            self,
            agent_id: str,
            tool_call_id: str | None = None,
            message_index: int | None = None,
        ) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.tool_call_id = tool_call_id
            self.message_index = message_index

    DEFAULT_CSS = """
    FileChangeWidget {
        layout: vertical;
        margin: 0 0 1 4;
        padding: 1 2;
        background: $surface-darken-1;
        border: round $accent;
        height: auto;
    }

    FileChangeWidget .change-header {
        layout: horizontal;
        height: 1;
        margin-bottom: 0;
    }

    FileChangeWidget .change-icon {
        width: 2;
        color: $text-muted;
    }

    FileChangeWidget .change-path {
        width: 1fr;
        color: $accent;
    }

    FileChangeWidget .change-stats {
        width: auto;
        margin-left: 2;
    }

    FileChangeWidget .change-agent-tool-info {
        width: auto;
        margin-left: 2;
        text-align: right;
        color: $text-muted;
    }

    FileChangeWidget .change-type-create {
        color: $success;
    }

    FileChangeWidget .change-type-modify {
        color: $warning;
    }

    FileChangeWidget .change-type-delete {
        color: $error;
    }

    FileChangeWidget .diff-content {
        margin-top: 1;
        padding: 1;
        background: $surface-darken-2;
        overflow-y: auto;
        max-height: 20;
    }

    FileChangeWidget .diff-line-added {
        color: $success;
    }

    FileChangeWidget .diff-line-removed {
        color: $error;
    }

    FileChangeWidget .diff-line-context {
        color: $text-muted;
    }

    FileChangeWidget .button-row {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }

    FileChangeWidget .button-row Button {
        margin: 0 1 0 0;
        min-width: 14;
    }

    FileChangeWidget.expanded {
        border: round $primary;
    }
    """

    def __init__(
        self,
        agent_id: str,
        file_path: str,
        change_type: str,
        tool_name: str,
        tool_call_id: str,
        message_index: int,
        old_content: str | None = None,
        added_ranges: list | None = None,
        removed_ranges: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._agent_id = agent_id
        self._file_path = file_path
        self._change_type = change_type  # "create", "modify", "delete"
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._message_index = message_index
        self._old_content = old_content
        self._added_ranges = added_ranges or []
        self._removed_ranges = removed_ranges or []
        self._expanded = False

    def compose(self):
        # Calculate stats
        added_lines = sum(
            r.get("endLine", 0) - r.get("startLine", 0) + 1
            for r in self._added_ranges
        )
        removed_lines = sum(
            r.get("endLine", 0) - r.get("startLine", 0) + 1
            for r in self._removed_ranges
        )

        # Icon and color based on change type
        if self._change_type == "create":
            icon = "[green]+[/green]"
            type_class = "change-type-create"
        elif self._change_type == "delete":
            icon = "[red]-[/red]"
            type_class = "change-type-delete"
        else:  # modify
            icon = "[yellow]~[/yellow]"
            type_class = "change-type-modify"

        # Build stats string
        stats_parts = []
        if added_lines > 0:
            stats_parts.append(f"[green]+{added_lines}[/green]")
        if removed_lines > 0:
            stats_parts.append(f"[red]-{removed_lines}[/red]")
        stats = " ".join(stats_parts) if stats_parts else "[dim]no changes[/dim]"

        # Header row
        with Horizontal(classes="change-header"):
            collapse_icon = "[dim]▼[/dim]" if self._expanded else "[dim]▶[/dim]"
            yield Static(f"{collapse_icon} {icon}", classes="change-icon", markup=True)
            safe_path = _esc(self._file_path)
            yield Static(safe_path, classes=f"change-path {type_class}", markup=True)
            yield Static(stats, classes="change-stats", markup=True)
            # Add agent and tool info
            yield Static(
                f"[dim]{self._agent_id[:8]} via {self._tool_name}[/dim]",
                classes="change-agent-tool-info",
                markup=True,
            )

        # Expanded diff content (shown only when expanded)
        if self._expanded:
            yield self._build_diff_view()

            # Button row
            with Horizontal(classes="button-row"):
                yield Button("View Context", variant="primary", id="btn-view-context")
                yield Button("View Agent", variant="default", id="btn-view-agent")
                yield Button("Collapse", variant="default", id="btn-collapse")

    def _build_diff_view(self) -> Static:
        """Build a colored diff view from the change data."""
        lines = []

        if self._change_type == "create":
            lines.append("[green]+ New file created[/green]")
        elif self._change_type == "delete":
            lines.append("[red]- File deleted[/red]")
        elif self._change_type == "modify":
            # Show a simplified diff based on ranges
            if self._removed_ranges:
                lines.append(f"[red]- Removed {len(self._removed_ranges)} section(s)[/red]")
            if self._added_ranges:
                lines.append(f"[green]+ Added {len(self._added_ranges)} section(s)[/green]")

            # If we have old content, show a preview
            if self._old_content:
                lines.append("")
                lines.append("[dim]Changes:[/dim]")
                # Show first few lines of old content as context
                old_lines = self._old_content.split("\n")[:5]
                for line in old_lines:
                    safe_line = _esc(line[:80])
                    lines.append(f"[dim]  {safe_line}[/dim]")
                if len(self._old_content.split("\n")) > 5:
                    lines.append("[dim]  ...[/dim]")

        diff_text = "\n".join(lines)
        return Static(diff_text, classes="diff-content", markup=True)

    def on_click(self) -> None:
        """Toggle between collapsed and expanded views."""
        # Don't toggle if clicking on buttons
        if self._expanded:
            return
        self._toggle_expand()

    def _toggle_expand(self) -> None:
        """Toggle the expanded state."""
        self._expanded = not self._expanded
        if self._expanded:
            self.add_class("expanded")
        else:
            self.remove_class("expanded")
        # Rebuild the widget
        self.remove_children()
        self.mount(*self.compose())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-view-context":
            self.post_message(
                self.ViewContextRequested(
                    agent_id=self._agent_id,
                    file_path=self._file_path,
                    tool_name=self._tool_name,
                    tool_call_id=self._tool_call_id,
                    message_index=self._message_index,
                )
            )
        elif event.button.id == "btn-view-agent":
            self.post_message(
                self.ViewAgentRequested(
                    agent_id=self._agent_id,
                    tool_call_id=self._tool_call_id,
                    message_index=self._message_index,
                )
            )
        elif event.button.id == "btn-collapse":
            self._toggle_expand()
