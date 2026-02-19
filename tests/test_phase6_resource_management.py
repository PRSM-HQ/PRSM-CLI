"""Phase 6 tests — expert lifecycle and resource management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from prsm.engine.agent_manager import AgentManager
from prsm.engine.config import EngineConfig
from prsm.engine.errors import AgentSpawnError
from prsm.engine.expert_registry import ExpertRegistry
from prsm.engine.message_router import MessageRouter
from prsm.engine.models import AgentResult, ExpertProfile, SpawnRequest
from prsm.engine.resource_manager import ResourceBudget, ResourceManager
from prsm.engine.yaml_config import load_yaml_config


def _make_expert(expert_id: str = "test-expert") -> ExpertProfile:
    return ExpertProfile(
        expert_id=expert_id,
        name="Test Expert",
        description="test",
        system_prompt="test prompt",
    )


def test_expert_registry_lifecycle_and_utility():
    registry = ExpertRegistry()
    expert = _make_expert()

    proposed_id = registry.propose(expert)
    assert proposed_id == "test-expert"
    assert registry.get("test-expert").lifecycle_state == "proposed"

    registry.approve_proposal("test-expert")
    assert registry.get("test-expert").lifecycle_state == "active"

    registry.record_consultation("test-expert", success=True, duration=2.0, confidence=0.9)
    registry.record_consultation("test-expert", success=False, duration=4.0, confidence=0.2)
    profile = registry.get("test-expert")
    assert profile.consultation_count == 2
    assert profile.success_count == 1
    assert profile.failure_count == 1
    assert profile.avg_duration_seconds == 3.0
    assert 0.0 <= profile.utility_score <= 1.0

    low = registry.get_low_utility_experts(threshold=1.0)
    assert any(p.expert_id == "test-expert" for p in low)

    registry.deprecate("test-expert", reason="unused")
    assert registry.get("test-expert").lifecycle_state == "deprecated"
    assert registry.get("test-expert").deprecation_reason == "unused"

    registry.archive("test-expert")
    assert registry.get("test-expert").lifecycle_state == "archived"


def test_expert_registry_persists_metrics(tmp_path):
    db_path = tmp_path / "expert-metrics.json"

    registry1 = ExpertRegistry()
    registry1.register(_make_expert("persisted"))
    registry1.set_metrics_db_path(db_path)
    registry1.record_consultation("persisted", success=True, duration=5.0, confidence=0.8)

    registry2 = ExpertRegistry()
    registry2.register(_make_expert("persisted"))
    registry2.set_metrics_db_path(db_path)
    loaded = registry2.get("persisted")
    assert loaded.consultation_count == 1
    assert loaded.success_count == 1
    assert loaded.avg_duration_seconds == 5.0
    assert loaded.avg_confidence == 0.8


def test_resource_manager_budget_and_circuit_breaker():
    manager = ResourceManager()
    manager.configure_budget(
        ResourceBudget(
            project_id="p1",
            max_total_tokens=10,
            max_concurrent_agents=1,
            max_agent_spawns_per_hour=1,
            max_tool_calls_per_hour=1,
        )
    )

    allowed, _ = manager.check_budget("p1", "spawn_agent")
    assert allowed
    manager.record_usage("p1", tokens=10, agent_spawn=True, tool_call=True)

    allowed, reason = manager.check_budget("p1", "spawn_agent")
    assert not allowed
    assert "token budget exceeded" in reason

    manager.record_failure("agent-1")
    manager.record_failure("agent-1")
    manager.record_failure("agent-1")
    allowed, _ = manager.check_circuit_breaker("agent-1")
    assert not allowed

    state = manager._circuit_breakers["agent-1"]  # noqa: SLF001
    state.tripped_at = datetime.now(timezone.utc) - timedelta(seconds=state.cooldown_seconds + 1)
    allowed, _ = manager.check_circuit_breaker("agent-1")
    assert allowed


@pytest.mark.asyncio
async def test_agent_manager_blocks_spawn_when_budget_exceeded(monkeypatch):
    class _FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self):
            await asyncio.sleep(0)
            return AgentResult(agent_id="fake", success=True, summary="ok")

        def terminate_subprocess(self):
            return None

    monkeypatch.setattr("prsm.engine.agent_manager.AgentSession", _FakeSession)

    resources = ResourceManager()
    resources.configure_budget(
        ResourceBudget(
            project_id="p1",
            max_concurrent_agents=0,
        )
    )
    config = EngineConfig(project_id="p1")
    manager = AgentManager(
        router=MessageRouter(),
        expert_registry=ExpertRegistry(),
        config=config,
        resource_manager=resources,
    )

    with pytest.raises(AgentSpawnError, match="Resource budget exceeded"):
        await manager.spawn_agent(
            SpawnRequest(
                parent_id=None,
                prompt="do work",
            )
        )


def test_yaml_config_parses_phase6_resources_and_lifecycle_fields(tmp_path):
    config_file = tmp_path / "prsm.yaml"
    config_file.write_text(
        """
engine: {}
resources:
  budgets:
    test-project:
      max_total_tokens: 123
      max_concurrent_agents: 4
experts:
  exp1:
    name: Expert One
    description: test
    system_prompt: hello
    lifecycle_state: deprecated
    deprecation_reason: stale
    evaluation_criteria: [accuracy]
    deprecation_policy: quarterly
    consultation_count: 3
    success_count: 2
    failure_count: 1
    avg_duration_seconds: 12.5
    avg_confidence: 0.75
    utility_score: 0.5
""".strip(),
        encoding="utf-8",
    )

    cfg = load_yaml_config(config_file)
    assert cfg.engine.resource_budgets["test-project"]["max_total_tokens"] == 123
    expert = cfg.experts[0]
    assert expert.lifecycle_state == "deprecated"
    assert expert.deprecation_reason == "stale"
    assert expert.evaluation_criteria == ["accuracy"]
    assert expert.consultation_count == 3
    assert expert.utility_score == 0.5
