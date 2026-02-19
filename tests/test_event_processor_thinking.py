from __future__ import annotations

from types import SimpleNamespace

from prsm.adapters.events import AgentStateChanged
from prsm.engine.models import AgentState
from prsm.tui.handlers.event_processor import EventProcessor


class _DummyBridge:
    def map_state(self, state_str: str) -> AgentState:
        mapping = {
            "running": AgentState.RUNNING,
            "waiting_for_child": AgentState.WAITING_FOR_CHILD,
        }
        return mapping.get(state_str, AgentState.PENDING)

    def cancel_agent_futures(self, _agent_id: str) -> None:
        return


class _DummyTree:
    def update_agent_state(self, _agent_id: str, _new_state: AgentState) -> None:
        return

    def sort_by_activity(self) -> None:
        return


def test_waiting_for_child_clears_thinking_indicator_state() -> None:
    session = SimpleNamespace(
        agents={
            "parent-1": SimpleNamespace(
                state=AgentState.RUNNING,
                last_active=None,
            )
        },
        active_agent_id="parent-1",
    )
    screen = SimpleNamespace(
        bridge=_DummyBridge(),
        session=session,
        _thinking_widget=None,
    )
    processor = EventProcessor(screen)
    processor._thinking_agents.add("parent-1")
    processor.sync_thinking = lambda: None
    tree = _DummyTree()
    sb = SimpleNamespace(status="streaming")
    tl = SimpleNamespace()

    processor._handle_agent_state_changed(
        AgentStateChanged(
            agent_id="parent-1",
            old_state="running",
            new_state="waiting_for_child",
        ),
        sb,
        tl,
        tree,
    )

    assert "parent-1" not in processor.thinking_agents
    assert session.agents["parent-1"].state == AgentState.WAITING_FOR_CHILD
    assert sb.status == "connected"
