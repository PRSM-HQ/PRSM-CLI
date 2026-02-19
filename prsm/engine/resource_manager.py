"""Per-project resource budgets and per-agent circuit breakers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ResourceBudget:
    project_id: str
    max_total_tokens: int = 0
    max_concurrent_agents: int = 10
    max_agent_spawns_per_hour: int = 50
    max_tool_calls_per_hour: int = 500
    current_token_usage: int = 0
    current_agent_count: int = 0
    spawns_this_hour: int = 0
    tool_calls_this_hour: int = 0
    last_reset: datetime = field(default_factory=_utcnow)


@dataclass
class CircuitBreakerState:
    agent_id: str
    consecutive_failures: int = 0
    tripped: bool = False
    tripped_at: datetime | None = None
    cooldown_seconds: float = 60.0
    max_failures: int = 3


class ResourceManager:
    """Per-project budget limits and agent-level circuit breakers."""

    def __init__(self) -> None:
        self._budgets: dict[str, ResourceBudget] = {}
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}

    def configure_budget(self, budget: ResourceBudget) -> None:
        self._budgets[budget.project_id] = budget

    def get_budget(self, project_id: str) -> ResourceBudget:
        budget = self._budgets.get(project_id)
        if budget is None:
            budget = ResourceBudget(project_id=project_id)
            self._budgets[project_id] = budget
        self._reset_hourly_if_needed(budget)
        return budget

    def check_budget(self, project_id: str, action: str) -> tuple[bool, str]:
        budget = self.get_budget(project_id)
        if budget.max_total_tokens > 0 and budget.current_token_usage >= budget.max_total_tokens:
            return (False, "token budget exceeded")
        if action == "spawn_agent":
            if budget.current_agent_count >= budget.max_concurrent_agents:
                return (False, "concurrent agent budget exceeded")
            if budget.spawns_this_hour >= budget.max_agent_spawns_per_hour:
                return (False, "agent spawns/hour budget exceeded")
        elif action == "tool_call":
            if budget.tool_calls_this_hour >= budget.max_tool_calls_per_hour:
                return (False, "tool calls/hour budget exceeded")
        return (True, "")

    def record_usage(
        self,
        project_id: str,
        tokens: int = 0,
        agent_spawn: bool = False,
        tool_call: bool = False,
    ) -> None:
        budget = self.get_budget(project_id)
        budget.current_token_usage += max(tokens, 0)
        if agent_spawn:
            budget.current_agent_count += 1
            budget.spawns_this_hour += 1
        if tool_call:
            budget.tool_calls_this_hour += 1

    def record_agent_completed(self, project_id: str) -> None:
        budget = self.get_budget(project_id)
        if budget.current_agent_count > 0:
            budget.current_agent_count -= 1

    def check_circuit_breaker(self, agent_id: str) -> tuple[bool, str]:
        state = self._circuit_breakers.get(agent_id)
        if state is None or not state.tripped:
            return (True, "")
        if state.tripped_at is None:
            return (False, "agent circuit breaker is open")
        elapsed = (_utcnow() - state.tripped_at).total_seconds()
        if elapsed >= state.cooldown_seconds:
            self.reset_circuit_breaker(agent_id)
            return (True, "")
        return (False, "agent circuit breaker is open")

    def record_failure(self, agent_id: str) -> None:
        state = self._circuit_breakers.get(agent_id)
        if state is None:
            state = CircuitBreakerState(agent_id=agent_id)
            self._circuit_breakers[agent_id] = state
        state.consecutive_failures += 1
        if state.consecutive_failures >= state.max_failures:
            state.tripped = True
            state.tripped_at = _utcnow()

    def record_success(self, agent_id: str) -> None:
        state = self._circuit_breakers.get(agent_id)
        if state is None:
            return
        state.consecutive_failures = 0
        state.tripped = False
        state.tripped_at = None

    def reset_circuit_breaker(self, agent_id: str) -> None:
        state = self._circuit_breakers.get(agent_id)
        if state is None:
            return
        state.consecutive_failures = 0
        state.tripped = False
        state.tripped_at = None

    def _reset_hourly_if_needed(self, budget: ResourceBudget) -> None:
        if _utcnow() - budget.last_reset < timedelta(hours=1):
            return
        budget.last_reset = _utcnow()
        budget.spawns_this_hour = 0
        budget.tool_calls_this_hour = 0
