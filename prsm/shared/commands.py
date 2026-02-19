"""Slash command parser and dispatch table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedCommand:
    """A parsed slash command."""

    name: str
    args: list[str]
    raw: str


def parse_command(text: str) -> ParsedCommand | None:
    """Parse a /command from input text.

    Returns None if text does not start with '/'.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    name = parts[0][1:]  # remove leading '/'
    args = parts[1:] if len(parts) > 1 else []
    return ParsedCommand(name=name, args=args, raw=stripped)


COMMAND_HELP: dict[str, str] = {
    "new": "Save current session and start a fresh new one",
    "session": "/session SESSION_UUID — save current + load by session UUID",
    "sessions": "List all saved sessions for this project",
    "fork": "/fork [NAME] — fork current session (optionally name it)",
    "save": "/save [NAME] — save session with optional name",
    "import": "/import list|preview|run|file — import transcripts from codex/claude/gemini sessions or .prsm files",
    "export": "/export [PATH] — export current session as a .prsm file",
    "model": "/model [list|NAME] — switch AI model or open model picker",
    "plugin": "/plugin add|add-json|list|remove|tag — manage MCP server plugins",
    "policy": "/policy [list|add-allow|add-block|remove] — manage bash command whitelist/blacklist",
    "settings": "Open the settings menu (also F2)",
    "worktree": "/worktree list|create|remove|switch — manage git worktrees",
    "help": "Show this help message",
}
