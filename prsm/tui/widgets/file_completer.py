"""File completer — TUI autocomplete widget for @ references.

Re-exports shared utilities from prsm.shared.file_utils for convenience,
and provides the Textual FileCompleter widget.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# Re-export shared utilities so existing imports still work
from prsm.shared.file_utils import (
    FileEntry,
    FileAttachment,
    FileIndex,
    build_tree_outline,
    resolve_references,
    format_size,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult


# ── FileCompleter widget ─────────────────────────────────────

from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static


class FileCompleter(Widget):
    """Floating autocomplete dropdown for file/directory completion."""

    DEFAULT_CSS = """
    FileCompleter {
        display: none;
        layer: overlay;
        width: 100%;
        height: auto;
        max-height: 12;
        background: $surface-darken-1;
        border: tall $accent;
        padding: 0 1;
    }

    FileCompleter.visible {
        display: block;
    }

    FileCompleter .fc-entry {
        height: 1;
        padding: 0 1;
    }

    FileCompleter .fc-highlighted {
        background: $accent-darken-2;
        text-style: bold;
    }
    """

    class Selected(Message):
        """Fired when user picks a completion."""

        def __init__(self, path: str, is_dir: bool) -> None:
            self.path = path
            self.is_dir = is_dir
            super().__init__()

    class Dismissed(Message):
        """Fired when the completer is closed without selection."""

    def __init__(self, cwd: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._index = FileIndex(cwd)
        self._entries: list[FileEntry] = []
        self._highlight: int = 0

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="fc-container")

    def show(self, prefix: str = "") -> None:
        """Show the completer with entries matching prefix."""
        self._entries = self._index.search(prefix)[:10]
        self._highlight = 0
        self._render_entries()
        if self._entries:
            self.add_class("visible")
        else:
            self.remove_class("visible")

    def hide(self) -> None:
        """Hide the completer."""
        self.remove_class("visible")

    def update_filter(self, prefix: str) -> None:
        """Update the displayed entries as user types."""
        self._entries = self._index.search(prefix)[:10]
        self._highlight = 0
        self._render_entries()
        if not self._entries:
            self.remove_class("visible")

    def move_highlight(self, delta: int) -> None:
        """Move the highlight up or down."""
        if not self._entries:
            return
        self._highlight = max(
            0, min(len(self._entries) - 1, self._highlight + delta),
        )
        self._render_entries()

    def confirm(self) -> None:
        """Select the highlighted entry."""
        if 0 <= self._highlight < len(self._entries):
            entry = self._entries[self._highlight]
            self.post_message(self.Selected(entry.path, entry.is_dir))
            self.remove_class("visible")

    def _render_entries(self) -> None:
        """Rebuild the container with current entries and highlight."""
        try:
            container = self.query_one("#fc-container", VerticalScroll)
        except Exception:
            return
        container.remove_children()
        for i, entry in enumerate(self._entries):
            # Use distinct icons for file and directory
            if entry.is_dir:
                icon = "[bold blue]\U0001f4c1[/bold blue]"  # 📂
            else:
                icon = "[dim]\U0001f4c4[/dim]"  # 📄

            size_str = ""
            if entry.size is not None and not entry.is_dir: # Only show size for files
                size_str = f"[dim] ({format_size(entry.size)})[/dim]"
            
            classes = "fc-entry fc-highlighted" if i == self._highlight else "fc-entry"
            
            container.mount(Static(
                f"{icon} {entry.path}{size_str}", # Formatted as "ICON path (size)"
                classes=classes,
                markup=True,
            ))
