"""Message and tool call models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    result: str | None = None
    success: bool = True


@dataclass
class Message:
    role: MessageRole
    content: str
    agent_id: str
    # Snapshot captured immediately before this user prompt was sent.
    # Used to restore file state when resending from history.
    snapshot_id: str | None = None
    id: str = field(default_factory=_gen_id)
    timestamp: datetime = field(default_factory=_utcnow)
    tool_calls: list[ToolCall] = field(default_factory=list)
    streaming: bool = False
