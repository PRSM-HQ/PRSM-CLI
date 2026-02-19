from __future__ import annotations

import pytest

from prsm.engine.mcp_server.tools import OrchestrationTools
from prsm.engine.models import AgentDescriptor, AgentRole


class _DummyProvider:
    name = "codex"


class _DummyManager:
    def __init__(self) -> None:
        self._parent = AgentDescriptor(
            agent_id="master-1",
            role=AgentRole.MASTER,
            prompt="master",
        )

    def get_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        if agent_id == self._parent.agent_id:
            return self._parent
        return None

    async def spawn_agent(self, request) -> AgentDescriptor:
        child = AgentDescriptor(
            agent_id="child-1",
            parent_id=self._parent.agent_id,
            role=AgentRole.WORKER,
            model=request.model,
            prompt=request.prompt,
        )
        self._parent.children.append(child.agent_id)
        return child


class _DummyRouter:
    async def send(self, _message) -> None:
        return None


@pytest.mark.asyncio
async def test_spawn_child_response_prefers_alias_for_runtime_model_variant() -> None:
    tools = OrchestrationTools(
        agent_id="master-1",
        manager=_DummyManager(),  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        peer_models={
            "gpt-5-3-high": (_DummyProvider(), "gpt-5-3::reasoning_effort=high")
        },
    )

    result = await tools.spawn_child(
        prompt="Implement a change",
        model="gpt-5-3::reasoning_effort=high",
    )

    text = result["content"][0]["text"]
    assert "model: gpt-5-3-high" in text
