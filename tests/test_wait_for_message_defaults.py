from __future__ import annotations

import pytest

from prsm.engine.mcp_server.tools import OrchestrationTools
from prsm.engine.models import AgentDescriptor, AgentRole, AgentState, MessageType, RoutedMessage


class _DummyRouter:
    def __init__(self) -> None:
        self.last_receive_timeout: float | None = None

    async def receive(
        self,
        agent_id: str,
        timeout: float = 120.0,
        message_type_filter=None,
        correlation_id=None,
    ) -> RoutedMessage:
        self.last_receive_timeout = timeout
        return RoutedMessage(
            message_type=MessageType.TASK_RESULT,
            source_agent_id="child-1",
            target_agent_id=agent_id,
            payload={"summary": "done"},
            correlation_id=correlation_id,
        )

    async def send(self, _message) -> None:
        return None


class _DummyManager:
    def __init__(self) -> None:
        self.master = AgentDescriptor(
            agent_id="master-1",
            role=AgentRole.MASTER,
            state=AgentState.RUNNING,
            prompt="master",
        )

    def get_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        if agent_id == self.master.agent_id:
            return self.master
        return None

    async def transition_agent_state(self, agent_id: str, state: AgentState) -> None:
        descriptor = self.get_descriptor(agent_id)
        if descriptor is not None:
            descriptor.state = state


@pytest.mark.asyncio
async def test_wait_for_message_default_timeout_is_disabled() -> None:
    manager = _DummyManager()
    router = _DummyRouter()
    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=router,  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.wait_for_message()
    text = result["content"][0]["text"]

    assert "Message received" in text
    assert "type: task_result" in text
    assert router.last_receive_timeout == 0.0
