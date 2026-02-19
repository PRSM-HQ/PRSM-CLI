"""Agent model adapter — UI-friendly transformations for engine models.

Provides display helpers (icons, labels, colors) for engine AgentState and
AgentRole values. No lossy mapping — all 9 engine states and 4 engine roles
are preserved with full fidelity.

Engine models are the single source of truth (prsm.engine.models).
This adapter only adds UI presentation logic.
"""

from datetime import datetime, timezone

from prsm.engine.models import (
    AgentDescriptor,
    AgentRole,
    AgentState,
)
from prsm.shared.models.agent import AgentNode
from prsm.adapters.orchestrator import _strip_prompt_prefix  # noqa: F401


# ── State display: icon + color for each of the 9 engine states ──

STATE_ICONS: dict[AgentState, tuple[str, str]] = {
    AgentState.PENDING: ("\u25cb", "dim"),
    AgentState.STARTING: ("\u25cb", "dim"),
    AgentState.RUNNING: ("\u25b6", "green"),
    AgentState.WAITING_FOR_PARENT: ("\u23f3", "yellow"),
    AgentState.WAITING_FOR_CHILD: ("\u25cf", "yellow"),
    AgentState.WAITING_FOR_EXPERT: ("\u23f3", "yellow"),
    AgentState.COMPLETED: ("\u2713", "dim green"),
    AgentState.FAILED: ("\u2717", "red"),
    AgentState.KILLED: ("\u2717", "red"),
}

STATE_DISPLAY: dict[AgentState, tuple[str, str]] = {
    AgentState.PENDING: ("\u25cb Pending", "dim"),
    AgentState.STARTING: ("\u25cb Starting", "dim"),
    AgentState.RUNNING: ("\u25b6 Running", "green"),
    AgentState.WAITING_FOR_PARENT: ("\u23f3 Waiting (parent)", "yellow"),
    AgentState.WAITING_FOR_CHILD: ("\u25cf Waiting (child)", "yellow"),
    AgentState.WAITING_FOR_EXPERT: ("\u23f3 Waiting (expert)", "yellow"),
    AgentState.COMPLETED: ("\u2713 Completed", "dim green"),
    AgentState.FAILED: ("\u2717 Failed", "red"),
    AgentState.KILLED: ("\u2717 Killed", "red"),
}


# ── Role display: label for each of the 4 engine roles ──

ROLE_DISPLAY: dict[AgentRole, str] = {
    AgentRole.MASTER: "\U0001f451 Orchestrator",
    AgentRole.WORKER: "\u2692 Worker",
    AgentRole.EXPERT: "\U0001f393 Expert",
    AgentRole.REVIEWER: "\U0001f50d Reviewer",
}


# ── Stale states: transient states that should be reset on session restore ──

STALE_STATES: frozenset[AgentState] = frozenset({
    AgentState.PENDING,
    AgentState.RUNNING,
    AgentState.WAITING_FOR_PARENT,
    AgentState.WAITING_FOR_CHILD,
    AgentState.WAITING_FOR_EXPERT,
    AgentState.STARTING,
})


# ── String → enum parsing (for persistence deserialization) ──

def parse_state(value: str) -> AgentState:
    """Parse a state string to AgentState enum.

    Handles both engine state values ("waiting_for_child") and
    legacy UI state values ("idle", "waiting", "error") for
    backward compatibility with persisted sessions.
    """
    # Try direct engine enum match first
    try:
        return AgentState(value)
    except ValueError:
        pass

    # Legacy UI state mapping for backward compatibility
    _LEGACY_MAP = {
        "idle": AgentState.PENDING,
        "waiting": AgentState.WAITING_FOR_CHILD,
        "error": AgentState.FAILED,
    }
    if value in _LEGACY_MAP:
        return _LEGACY_MAP[value]

    # Default fallback
    return AgentState.PENDING


def parse_role(value: str | None) -> AgentRole | None:
    """Parse a role string to AgentRole enum.

    Handles both engine role values ("master") and
    legacy UI role values ("orchestrator") for backward compatibility.
    """
    if value is None:
        return None

    # Try direct engine enum match first
    try:
        return AgentRole(value)
    except ValueError:
        pass

    # Legacy UI role mapping for backward compatibility
    _LEGACY_MAP = {
        "orchestrator": AgentRole.MASTER,
    }
    if value in _LEGACY_MAP:
        return _LEGACY_MAP[value]

    return AgentRole.WORKER


# ── AgentDescriptor → AgentNode conversion ──

class AgentAdapter:
    """Converts engine AgentDescriptor to UI AgentNode with full fidelity.

    All 9 engine states and 4 engine roles are preserved directly.
    No lossy mapping.
    """

    @staticmethod
    def to_ui_node(descriptor: AgentDescriptor) -> AgentNode:
        """Convert engine descriptor to UI node.

        Args:
            descriptor: Engine agent descriptor

        Returns:
            UI agent node with all information preserved
        """
        clean_prompt = _strip_prompt_prefix(descriptor.prompt)
        return AgentNode(
            id=descriptor.agent_id,
            name=AgentAdapter._derive_name(descriptor.role, descriptor.prompt),
            state=descriptor.state,
            role=descriptor.role,
            model=descriptor.model,
            parent_id=descriptor.parent_id,
            children_ids=descriptor.children.copy(),
            prompt_preview=clean_prompt[:50] + ("..." if len(clean_prompt) > 50 else ""),
            created_at=descriptor.created_at,
            completed_at=descriptor.completed_at,
            last_active=datetime.now(timezone.utc),
            error=descriptor.error,
            tools=descriptor.tools.copy() if descriptor.tools else None,
            depth=descriptor.depth,
            cwd=descriptor.cwd,
            permission_mode=descriptor.permission_mode.value,
            provider=descriptor.provider,
        )

    @staticmethod
    def _derive_name(role: AgentRole, prompt: str) -> str:
        """Derive display name from role and prompt."""
        prompt = _strip_prompt_prefix(prompt)
        if role == AgentRole.MASTER:
            return "Orchestrator"
        if role == AgentRole.EXPERT:
            name = prompt[:40].strip() if prompt else "Expert"
            if prompt and len(prompt) > 40:
                name += "..."
            return name
        if role == AgentRole.REVIEWER:
            name = prompt[:50].strip() if prompt else "Reviewer"
            if prompt and len(prompt) > 50:
                name += "..."
            return name
        # Worker
        name = prompt[:50].strip()
        if len(prompt) > 50:
            name += "..."
        return name or "Worker"
