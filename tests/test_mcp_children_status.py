from __future__ import annotations

import pytest

from prsm.engine.mcp_server.tools import OrchestrationTools
from prsm.engine.models import AgentDescriptor, AgentState


class _DummyManager:
    def __init__(self, descriptors: list[AgentDescriptor]) -> None:
        self._descriptors = descriptors

    def get_all_descriptors(self) -> list[AgentDescriptor]:
        return self._descriptors


def _mk_child(agent_id: str, state: AgentState, parent_id: str = "parent-1") -> AgentDescriptor:
    return AgentDescriptor(agent_id=agent_id, parent_id=parent_id, state=state, prompt="x")


@pytest.mark.asyncio
async def test_get_children_status_reports_waiting_and_terminal_states() -> None:
    tool = OrchestrationTools.__new__(OrchestrationTools)
    tool._agent_id = "parent-1"
    tool._manager = _DummyManager(
        [
            _mk_child("c-running", AgentState.RUNNING),
            _mk_child("c-waiting", AgentState.WAITING_FOR_CHILD),
            _mk_child("c-completed", AgentState.COMPLETED),
            _mk_child("c-failed", AgentState.FAILED),
            _mk_child("c-killed", AgentState.KILLED),
        ]
    )

    result = await OrchestrationTools._get_children_status_impl(tool)
    text = result["content"][0]["text"]

    assert "1 completed" in text
    assert "1 failed" in text
    assert "1 running" in text
    assert "1 waiting" in text
    assert "1 killed" in text
    assert "- c-waiting: state=waiting_for_child" in text


@pytest.mark.asyncio
async def test_get_children_status_filters_to_own_children() -> None:
    tool = OrchestrationTools.__new__(OrchestrationTools)
    tool._agent_id = "parent-1"
    tool._manager = _DummyManager(
        [
            _mk_child("mine", AgentState.COMPLETED, parent_id="parent-1"),
            _mk_child("other", AgentState.RUNNING, parent_id="parent-2"),
        ]
    )

    result = await OrchestrationTools._get_children_status_impl(tool)
    text = result["content"][0]["text"]

    assert "total: 1" in text
    assert "mine" in text
    assert "other" not in text
