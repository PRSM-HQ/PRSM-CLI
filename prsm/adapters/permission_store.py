"""Persistent storage for tool permission decisions.

Stores allowed tools at two levels:
- Global: ~/.prsm/allowed_tools.json (applies to all projects)
- Project: ~/.prsm/projects/{ID}/allowed_tools.json (per-project)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GLOBAL_DIR = Path.home() / ".prsm"
FILENAME = "allowed_tools.json"


class PermissionStore:
    """Load and save tool permission decisions."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._project_dir = project_dir
        self._global_path = GLOBAL_DIR / FILENAME
        self._project_path = (
            project_dir / FILENAME if project_dir else None
        )

    def load(self) -> set[str]:
        """Load all allowed tools (global + project merged)."""
        allowed: set[str] = set()
        allowed |= self._load_file(self._global_path)
        if self._project_path:
            allowed |= self._load_file(self._project_path)
        return allowed

    def add_project(self, tool_name: str) -> None:
        """Add a tool to the project-level allow list."""
        if not self._project_path:
            # No project context — fall back to global
            self.add_global(tool_name)
            return
        self._add_to_file(self._project_path, tool_name)

    def add_global(self, tool_name: str) -> None:
        """Add a tool to the global allow list."""
        self._add_to_file(self._global_path, tool_name)

    @staticmethod
    def _load_file(path: Path) -> set[str]:
        """Load a set of tool names from a JSON file."""
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return set(data)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load %s", path)
        return set()

    @staticmethod
    def _add_to_file(path: Path, tool_name: str) -> None:
        """Add a tool name to a JSON file (create if needed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = PermissionStore._load_file(path)
        existing.add(tool_name)
        try:
            path.write_text(json.dumps(sorted(existing), indent=2) + "\n")
        except OSError:
            logger.warning("Failed to write %s", path)
