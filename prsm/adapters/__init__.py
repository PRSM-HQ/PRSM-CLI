"""Adapters package - Bridge between engine and UI frontends.

This package contains the orchestrator bridge, event bus, and other
adapter components that connect the engine to TUI/VSCode frontends.
"""
from __future__ import annotations

__all__ = [
    "OrchestratorBridge",
    "EventBus",
    "generate_session_name",
    "AgentAdapter",
    "PermissionStore",
]

from prsm.adapters.agent_adapter import AgentAdapter
from prsm.adapters.orchestrator import OrchestratorBridge
from prsm.adapters.event_bus import EventBus
from prsm.adapters.permission_store import PermissionStore
from prsm.adapters.session_naming import generate_session_name
