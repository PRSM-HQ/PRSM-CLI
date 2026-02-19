"""Session state — per-agent message history and active agent tracking."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import uuid
from typing import Any, Optional

from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import Message, MessageRole, ToolCall


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


FORKED_PREFIX = "(Forked) "


def format_forked_name(name: str) -> str:
    if name.startswith(FORKED_PREFIX):
        return name
    return f"{FORKED_PREFIX}{name}"


def is_default_session_name(name: str | None) -> bool:
    if not name:
        return True
    stripped_name = name.strip()
    base_name = stripped_name[len(FORKED_PREFIX):] if stripped_name.startswith(FORKED_PREFIX) else stripped_name
    return (
        re.match(r"^session \d+$", base_name, flags=re.IGNORECASE) is not None
        or re.match(r"^untitled session(?: \d+)?$", base_name, flags=re.IGNORECASE) is not None
    )


@dataclass
class WorktreeMetadata:
    """Metadata about the git worktree context when a session was created."""
    root: str  # Absolute path to worktree root (e.g., "/home/user/repos/prsm-cli")
    branch: Optional[str] = None  # Current branch (e.g., "feature/worktree-support")
    common_dir: Optional[str] = None  # Git common dir (e.g., "/home/user/repos/prsm-cli/.git")


@dataclass
class Session:
    """Holds all conversation state for a session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agents: dict[str, AgentNode] = field(default_factory=dict)
    messages: dict[str, list[Message]] = field(default_factory=dict)
    active_agent_id: str | None = None
    name: str | None = None
    created_at: datetime | None = field(default_factory=_utcnow)
    forked_from: str | None = None
    imported_from: dict[str, Any] | None = None
    worktree: WorktreeMetadata | None = None

    def add_agent(self, agent: AgentNode) -> None:
        self.agents[agent.id] = agent
        if agent.id not in self.messages:
            self.messages[agent.id] = []

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)
        self.messages.pop(agent_id, None)
        if self.active_agent_id == agent_id:
            self.active_agent_id = None

    def set_active(self, agent_id: str) -> None:
        self.active_agent_id = agent_id

    def get_active_agent(self) -> AgentNode | None:
        if self.active_agent_id:
            return self.agents.get(self.active_agent_id)
        return None

    def add_message(
        self,
        agent_id: str,
        role: MessageRole,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        snapshot_id: str | None = None,
    ) -> Message:
        msg = Message(
            role=role,
            content=content,
            agent_id=agent_id,
            snapshot_id=snapshot_id,
            timestamp=_utcnow(),
            tool_calls=tool_calls or [],
        )
        if agent_id not in self.messages:
            self.messages[agent_id] = []
        self.messages[agent_id].append(msg)
        return msg

    def get_messages(self, agent_id: str) -> list[Message]:
        return self.messages.get(agent_id, [])

    def get_active_messages(self) -> list[Message]:
        if self.active_agent_id:
            return self.get_messages(self.active_agent_id)
        return []

    def clear_messages(self, agent_id: str | None = None) -> None:
        if agent_id:
            self.messages[agent_id] = []
        else:
            for key in self.messages:
                self.messages[key] = []

    @property
    def message_count(self) -> int:
        return sum(len(msgs) for msgs in self.messages.values())

    def fork(self, new_name: str | None = None, new_worktree: WorktreeMetadata | None = None) -> Session:
        """Create a deep copy of this session for forking.

        Args:
            new_name: Optional name for the forked session
            new_worktree: Optional worktree metadata if forking to a different worktree
        """
        return Session(
            session_id=str(uuid.uuid4()),
            agents=copy.deepcopy(self.agents),
            messages=copy.deepcopy(self.messages),
            active_agent_id=self.active_agent_id,
            name=new_name,
            created_at=_utcnow(),
            forked_from=self.name,
            imported_from=copy.deepcopy(self.imported_from),
            worktree=new_worktree if new_worktree else copy.deepcopy(self.worktree),
        )
