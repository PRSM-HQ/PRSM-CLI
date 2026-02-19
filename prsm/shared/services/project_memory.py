"""Project memory — persistent notes per working directory.

Like Claude Code's MEMORY.md, this stores user-editable project notes
that are auto-loaded when prsm launches in a directory.
"""
from __future__ import annotations

from pathlib import Path


class ProjectMemory:
    """Manages MEMORY.md for a project directory."""

    def __init__(self, memory_path: Path) -> None:
        self.path = memory_path

    def load(self) -> str:
        """Load project memory content, or empty string if none."""
        if self.path.exists():
            return self.path.read_text()
        return ""

    def save(self, content: str) -> None:
        """Write project memory content."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content)

    def exists(self) -> bool:
        """Whether a MEMORY.md file exists for this project."""
        return self.path.exists()
