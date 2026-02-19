"""Agent lifecycle state machine.

Defines valid transitions and enforces them. Invalid transitions
raise ValueError rather than silently proceeding.

State Diagram:

    PENDING ──> STARTING ──> RUNNING ──┬──> COMPLETED
                                       │
                                       ├──> WAITING_FOR_PARENT ──> RUNNING
                                       │
                                       ├──> WAITING_FOR_CHILD ──> RUNNING
                                       │
                                       ├──> WAITING_FOR_EXPERT ──> RUNNING
                                       │
                                       └──> FAILED

    Any state ──> KILLED  (forced termination)
"""
from __future__ import annotations

from .models import AgentState

VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.PENDING: {
        AgentState.STARTING,
        AgentState.KILLED,
    },
    AgentState.STARTING: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.KILLED,
    },
    AgentState.RUNNING: {
        AgentState.WAITING_FOR_PARENT,
        AgentState.WAITING_FOR_CHILD,
        AgentState.WAITING_FOR_EXPERT,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.KILLED,
    },
    AgentState.WAITING_FOR_PARENT: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.KILLED,
    },
    AgentState.WAITING_FOR_CHILD: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.KILLED,
    },
    AgentState.WAITING_FOR_EXPERT: {
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.KILLED,
    },
    AgentState.COMPLETED: {
        AgentState.STARTING,  # restart by parent
        AgentState.KILLED,
    },
    AgentState.FAILED: {
        AgentState.STARTING,  # restart by parent
        AgentState.KILLED,
    },
    AgentState.KILLED: set(),
}


def validate_transition(current: AgentState, target: AgentState) -> None:
    """Validate a state transition. Raises ValueError if invalid."""
    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        allowed_str = ", ".join(s.value for s in allowed) or "none (terminal)"
        raise ValueError(
            f"Invalid state transition: {current.value} -> {target.value}. "
            f"Allowed from {current.value}: {allowed_str}"
        )
