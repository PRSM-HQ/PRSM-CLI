"""Tool log — collapsible RichLog panel for tool calls and debug output."""

from __future__ import annotations

from textual.widgets import RichLog


class ToolLog(RichLog):
    """Collapsible log of tool calls and system events."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            auto_scroll=True,
            wrap=True,
            markup=True,
            highlight=True,
            max_lines=5000,
            **kwargs,
        )

    def log_tool_call(self, tool: str, args: str) -> None:
        self.write(f"[bold cyan]{tool}[/bold cyan]")
        self.write(f"  {args}")

    def log_tool_result(self, result: str, success: bool = True) -> None:
        color = "green" if success else "red"
        label = "Result" if success else "Error"
        self.write(f"  [{color}]{label}:[/{color}] {result}")

    def toggle(self) -> None:
        self.toggle_class("visible")
