"""Per-agent structured conversation log.

Tracks thinking, text output, tool calls, tool results, and user
messages for each agent. Used by the master to review child history
via the get_child_history MCP tool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntryType(Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"


@dataclass
class ConversationEntry:
    entry_type: EntryType
    content: str = ""
    tool_name: str | None = None
    tool_id: str | None = None
    tool_args: str | None = None
    is_error: bool = False
    timestamp: float = field(default_factory=time.time)


class ConversationStore:
    """Per-agent structured conversation log.

    Thread-safe for single-event-loop usage. Each agent's entries
    are stored in insertion order.
    """

    def __init__(self) -> None:
        self._logs: dict[str, list[ConversationEntry]] = {}
        self._decision_reports: dict[str, list[dict[str, Any]]] = {}

    def append(self, agent_id: str, entry: ConversationEntry) -> None:
        """Add an entry to an agent's conversation log."""
        if agent_id not in self._logs:
            self._logs[agent_id] = []
        self._logs[agent_id].append(entry)

    def get_history(
        self,
        agent_id: str,
        detail_level: str = "full",
    ) -> list[dict[str, Any]]:
        """Get an agent's conversation history.

        Args:
            agent_id: The agent whose history to retrieve.
            detail_level: "full" returns everything, "summary" returns
                text + tool names only (skips thinking, args, results).
        """
        entries = self._logs.get(agent_id, [])
        result: list[dict[str, Any]] = []

        for entry in entries:
            if detail_level == "summary":
                # Skip thinking blocks in summary mode
                if entry.entry_type == EntryType.THINKING:
                    continue
                d: dict[str, Any] = {
                    "type": entry.entry_type.value,
                    "timestamp": entry.timestamp,
                }
                if entry.entry_type == EntryType.TEXT:
                    d["content"] = entry.content
                elif entry.entry_type == EntryType.TOOL_CALL:
                    d["tool_name"] = entry.tool_name
                elif entry.entry_type == EntryType.TOOL_RESULT:
                    d["tool_name"] = entry.tool_name
                    d["is_error"] = entry.is_error
                elif entry.entry_type == EntryType.USER_MESSAGE:
                    d["content"] = entry.content
                result.append(d)
            else:
                # Full detail
                d = {
                    "type": entry.entry_type.value,
                    "content": entry.content,
                    "timestamp": entry.timestamp,
                }
                if entry.tool_name:
                    d["tool_name"] = entry.tool_name
                if entry.tool_id:
                    d["tool_id"] = entry.tool_id
                if entry.tool_args:
                    d["tool_args"] = entry.tool_args
                if entry.is_error:
                    d["is_error"] = entry.is_error
                result.append(d)

        return result

    def has_history(self, agent_id: str) -> bool:
        """Check if an agent has any conversation history."""
        return bool(self._logs.get(agent_id))

    def resolve_agent_id(self, agent_id: str) -> str | None:
        """Resolve a (possibly truncated) agent ID to the full ID.

        Returns the exact match if found, or a unique prefix match,
        or None if no match / ambiguous.
        """
        if agent_id in self._logs:
            return agent_id
        # Prefix match
        matches = [k for k in self._logs if k.startswith(agent_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def append_decision_report(
        self,
        agent_id: str,
        report: dict[str, Any],
        linked_tool_id: str | None = None,
    ) -> None:
        """Store a structured decision report for an agent."""
        payload = dict(report)
        payload.setdefault("agent_id", agent_id)
        payload.setdefault("timestamp", time.time())
        if linked_tool_id:
            payload["linked_tool_id"] = linked_tool_id
        self._decision_reports.setdefault(agent_id, []).append(payload)

    def get_decision_reports(self, agent_id: str) -> list[dict[str, Any]]:
        """Return structured decision reports for an agent."""
        return [dict(report) for report in self._decision_reports.get(agent_id, [])]
