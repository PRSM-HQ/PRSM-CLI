"""Agent data models for the tree hierarchy.

Uses engine models as the single source of truth for AgentState and AgentRole.
No duplicate enums — all state/role types come from prsm.engine.models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# Single source of truth — engine models
from prsm.engine.models import AgentRole, AgentState

# Re-export so consumers can import from here
__all__ = ["AgentNode", "AgentState", "AgentRole"]


@dataclass
class AgentNode:
    """UI representation of an agent.

    Uses engine AgentState (9 states) and AgentRole (4 roles) directly.
    No information loss — the full engine state is preserved.

    AgentState values:
        PENDING, STARTING, RUNNING, WAITING_FOR_PARENT,
        WAITING_FOR_CHILD, WAITING_FOR_EXPERT, COMPLETED, FAILED, KILLED

    AgentRole values:
        MASTER, WORKER, EXPERT, REVIEWER
    """
    # Core fields
    id: str
    name: str
    state: AgentState = AgentState.PENDING
    role: AgentRole | None = None
    model: str = "claude-opus-4-6"
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    prompt_preview: str = ""

    # Extended fields (optional, for detail views and debugging)
    created_at: datetime | None = None
    completed_at: datetime | None = None
    last_active: datetime | None = None
    error: str | None = None
    tools: list[str] | None = None
    depth: int = 0
    cwd: str | None = None
    permission_mode: str | None = None
    provider: str = "claude"
