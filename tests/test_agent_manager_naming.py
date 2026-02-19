from __future__ import annotations

import pytest

from prsm.engine.agent_manager import AgentManager
from prsm.engine.config import EngineConfig
from prsm.engine.expert_registry import ExpertRegistry
from prsm.engine.message_router import MessageRouter
from prsm.engine.models import AgentRole


def _manager() -> AgentManager:
    return AgentManager(
        router=MessageRouter(),
        expert_registry=ExpertRegistry(),
        config=EngineConfig(),
    )


@pytest.mark.asyncio
async def test_derive_agent_name_uses_model_summary(monkeypatch):
    async def _fake_generate_session_name(user_message: str, model_id: str = "gpt-5-3-spark") -> str:
        assert "stabilize flaky auth tests" in user_message.lower()
        assert model_id == "gpt-5-3-spark"
        return "Stabilize flaky auth tests"

    monkeypatch.setattr(
        "prsm.shared.services.session_naming.generate_session_name",
        _fake_generate_session_name,
    )

    manager = _manager()
    name = await manager._derive_agent_name(AgentRole.WORKER, "Stabilize flaky auth tests in CI")
    assert name == "Stabilize flaky auth tests"


@pytest.mark.asyncio
async def test_derive_agent_name_falls_back_when_model_naming_fails(monkeypatch):
    async def _fake_generate_session_name(_user_message: str, model_id: str = "gpt-5-3-spark") -> str:
        raise RuntimeError("naming provider unavailable")

    monkeypatch.setattr(
        "prsm.shared.services.session_naming.generate_session_name",
        _fake_generate_session_name,
    )

    manager = _manager()
    name = await manager._derive_agent_name(
        AgentRole.WORKER,
        "Implement support for resilient websocket reconnect and retries",
    )
    assert name == "Implement support for resilient websocket reconnect and..."


@pytest.mark.asyncio
async def test_derive_agent_name_keeps_master_constant():
    manager = _manager()
    name = await manager._derive_agent_name(AgentRole.MASTER, "Any prompt")
    assert name == "Orchestrator"
