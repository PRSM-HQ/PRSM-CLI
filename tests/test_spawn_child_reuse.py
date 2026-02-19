from __future__ import annotations

import pytest

from prsm.engine.mcp_server.tools import OrchestrationTools
from prsm.engine.models import AgentDescriptor, AgentRole, AgentState


class _DummyRouter:
    async def send(self, _message) -> None:
        return None


class _DummyTier:
    def __init__(self, value: str) -> None:
        self.value = value


class _DummyCapability:
    def __init__(
        self,
        tier: str,
        *,
        provider: str = "claude",
        available: bool = True,
    ) -> None:
        self.tier = _DummyTier(tier)
        self.provider = provider
        self.available = available


class _DummyRecommendation:
    def __init__(self, model_id: str, provider: str, tier: str) -> None:
        self.model_id = model_id
        self.provider = provider
        self.tier = _DummyTier(tier)


class _DummyModelRegistry:
    def __init__(
        self,
        caps: dict[str, _DummyCapability],
        *,
        recommendation: _DummyRecommendation | None = None,
    ) -> None:
        self._caps = caps
        self._recommendation = recommendation

    def resolve_alias(self, name: str) -> str:
        return name

    def recommend_model(self, _prompt: str, complexity: str = "medium"):
        return self._recommendation

    def get(self, model_id: str):
        return self._caps.get(model_id)

    def _infer_category(self, _prompt: str) -> str:
        return "coding"

    def get_ranked_for_task(self, _category: str, available_only: bool = True):
        out = []
        for model_id, cap in self._caps.items():
            if available_only and not cap.available:
                continue
            out.append((1.0, _DummyRecommendation(model_id, cap.provider, cap.tier.value)))
        return out

    def list_available(self):
        return []


class _DummyManager:
    def __init__(self) -> None:
        self.parent = AgentDescriptor(
            agent_id="master-1",
            role=AgentRole.MASTER,
            prompt="master",
        )
        self.children: dict[str, AgentDescriptor] = {}
        self.spawn_calls = 0
        self.restart_calls = 0
        self.last_spawn_model: str | None = None
        self.restart_noop_ids: set[str] = set()

    def get_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        if agent_id == self.parent.agent_id:
            return self.parent
        return self.children.get(agent_id)

    def get_all_descriptors(self) -> list[AgentDescriptor]:
        return [self.parent, *self.children.values()]

    async def spawn_agent(self, request) -> AgentDescriptor:
        self.spawn_calls += 1
        self.last_spawn_model = request.model
        child_id = f"spawned-{self.spawn_calls}"
        child = AgentDescriptor(
            agent_id=child_id,
            parent_id=self.parent.agent_id,
            role=AgentRole.WORKER,
            state=AgentState.RUNNING,
            model=request.model,
            provider=request.provider,
            prompt=request.prompt,
            tools=request.tools,
            cwd=request.cwd,
        )
        self.children[child_id] = child
        self.parent.children.append(child_id)
        return child

    async def restart_agent(self, agent_id: str, new_prompt: str) -> AgentDescriptor:
        self.restart_calls += 1
        child = self.children[agent_id]
        if agent_id in self.restart_noop_ids:
            return child
        child.prompt = new_prompt
        child.state = AgentState.RUNNING
        return child

    def get_completed_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        child = self.children.get(agent_id)
        if child is None:
            return None
        if child.state in (AgentState.COMPLETED, AgentState.FAILED, AgentState.KILLED):
            return child
        return None


@pytest.mark.asyncio
async def test_spawn_child_reuses_completed_similar_child_by_restart() -> None:
    manager = _DummyManager()
    manager.children["child-old"] = AgentDescriptor(
        agent_id="child-old",
        parent_id="master-1",
        role=AgentRole.WORKER,
        state=AgentState.COMPLETED,
        model="claude-sonnet-4-5-20250929",
        provider="claude",
        prompt="Implement parser cleanup",
    )
    manager.parent.children.append("child-old")

    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.spawn_child(prompt="Implement parser cleanup")
    text = result["content"][0]["text"]

    assert "Reused existing child agent via restart." in text
    assert manager.restart_calls == 1
    assert manager.spawn_calls == 0


@pytest.mark.asyncio
async def test_spawn_child_reuses_active_similar_child_without_new_spawn() -> None:
    manager = _DummyManager()
    manager.children["child-active"] = AgentDescriptor(
        agent_id="child-active",
        parent_id="master-1",
        role=AgentRole.WORKER,
        state=AgentState.RUNNING,
        model="claude-sonnet-4-5-20250929",
        provider="claude",
        prompt="Refactor router wiring",
    )
    manager.parent.children.append("child-active")

    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.spawn_child(prompt="Refactor router wiring")
    text = result["content"][0]["text"]

    assert "Reusing active child agent" in text
    assert manager.restart_calls == 0
    assert manager.spawn_calls == 0


@pytest.mark.asyncio
async def test_spawn_child_rejects_model_that_does_not_fit_complexity() -> None:
    manager = _DummyManager()
    registry = _DummyModelRegistry(
        {"fast-model": _DummyCapability("fast")},
    )
    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        model_registry=registry,
    )

    result = await tools.spawn_child(
        prompt="Deep system redesign",
        model="fast-model",
        complexity="complex",
    )

    assert result.get("is_error") is True
    assert "does not fit complexity 'complex'" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_spawn_child_does_not_reuse_when_complexity_requires_higher_tier() -> None:
    manager = _DummyManager()
    manager.children["child-fast"] = AgentDescriptor(
        agent_id="child-fast",
        parent_id="master-1",
        role=AgentRole.WORKER,
        state=AgentState.COMPLETED,
        model="fast-model",
        provider="claude",
        prompt="Investigate issue",
    )
    manager.parent.children.append("child-fast")
    registry = _DummyModelRegistry(
        {
            "fast-model": _DummyCapability("fast"),
            "strong-model": _DummyCapability("strong"),
        },
        recommendation=_DummyRecommendation("strong-model", "claude", "strong"),
    )

    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        model_registry=registry,
        default_model="strong-model",
    )

    result = await tools.spawn_child(
        prompt="Investigate issue",
        complexity="complex",
    )
    text = result["content"][0]["text"]

    assert "Child agent spawned in background." in text
    assert manager.restart_calls == 0
    assert manager.spawn_calls == 1
    assert manager.last_spawn_model == "strong-model"


@pytest.mark.asyncio
async def test_spawn_child_falls_back_to_new_spawn_when_restart_noops() -> None:
    manager = _DummyManager()
    manager.children["child-old"] = AgentDescriptor(
        agent_id="child-old",
        parent_id="master-1",
        role=AgentRole.WORKER,
        state=AgentState.COMPLETED,
        model="claude-sonnet-4-5-20250929",
        provider="claude",
        prompt="Implement parser cleanup",
    )
    manager.parent.children.append("child-old")
    manager.restart_noop_ids.add("child-old")

    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.spawn_child(prompt="Implement parser cleanup")
    text = result["content"][0]["text"]

    assert "Child agent spawned in background." in text
    assert manager.restart_calls == 1
    assert manager.spawn_calls == 1


@pytest.mark.asyncio
async def test_restart_child_errors_when_restart_keeps_terminal_state() -> None:
    manager = _DummyManager()
    manager.children["child-old"] = AgentDescriptor(
        agent_id="child-old",
        parent_id="master-1",
        role=AgentRole.WORKER,
        state=AgentState.COMPLETED,
        model="claude-sonnet-4-5-20250929",
        provider="claude",
        prompt="Old prompt",
    )
    manager.parent.children.append("child-old")
    manager.restart_noop_ids.add("child-old")

    tools = OrchestrationTools(
        agent_id="master-1",
        manager=manager,  # type: ignore[arg-type]
        router=_DummyRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.restart_child("child-old", "Run again")
    text = result["content"][0]["text"]

    assert result.get("is_error") is True
    assert "did not become active" in text
