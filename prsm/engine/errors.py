"""Exception hierarchy for the orchestration engine.

Specific exceptions for each failure mode. Never bare
`except Exception` without justification.
"""
from __future__ import annotations


class OrchestrationError(Exception):
    """Base exception for all orchestration errors."""


class AgentSpawnError(OrchestrationError):
    """Failed to create or start an agent session."""
    def __init__(self, agent_id: str, reason: str):
        self.agent_id = agent_id
        self.reason = reason
        super().__init__(f"Failed to spawn agent {agent_id}: {reason}")


class AgentTimeoutError(OrchestrationError):
    """Agent exceeded its time budget."""
    def __init__(self, agent_id: str, timeout_seconds: float):
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Agent {agent_id} timed out after {timeout_seconds}s"
        )


class MessageRoutingError(OrchestrationError):
    """Failed to deliver a message to the target agent."""
    def __init__(self, message_id: str, target_id: str, reason: str):
        self.message_id = message_id
        self.target_id = target_id
        super().__init__(
            f"Cannot route message {message_id} to {target_id}: {reason}"
        )


class DeadlockDetectedError(OrchestrationError):
    """Circular dependency detected in agent wait graph."""
    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__(
            f"Deadlock detected: {' -> '.join(cycle)}"
        )


class MaxDepthExceededError(OrchestrationError):
    """Agent hierarchy exceeded maximum nesting depth."""
    def __init__(self, agent_id: str, depth: int, max_depth: int):
        self.agent_id = agent_id
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(
            f"Agent {agent_id} at depth {depth} exceeds max {max_depth}"
        )


class ExpertNotFoundError(OrchestrationError):
    """Requested expert profile does not exist."""
    def __init__(self, expert_id: str):
        self.expert_id = expert_id
        super().__init__(f"Expert profile not found: {expert_id}")


class ToolCallTimeoutError(OrchestrationError):
    """A single tool call exceeded its time budget."""
    def __init__(
        self, agent_id: str, tool_name: str, timeout_seconds: float
    ):
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool call '{tool_name}' in agent {agent_id} "
            f"timed out after {timeout_seconds}s"
        )


class ProviderNotAvailableError(OrchestrationError):
    """Requested provider is not installed or not available."""
    def __init__(self, provider_name: str, available: list[str]):
        self.provider_name = provider_name
        self.available = available
        avail_str = ", ".join(available) if available else "none"
        super().__init__(
            f"Provider '{provider_name}' is not available "
            f"(CLI not installed or not on PATH). "
            f"Available providers: {avail_str}"
        )


class ModelNotAvailableError(OrchestrationError):
    """Requested model's provider is not available."""
    def __init__(self, model_id: str, provider_name: str, reason: str):
        self.model_id = model_id
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(
            f"Model '{model_id}' is not available: {reason}"
        )
