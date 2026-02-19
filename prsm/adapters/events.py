"""Event types emitted by the orchestrator engine.

Each event corresponds to an engine callback dict, parsed into
a typed dataclass for safe consumption by the TUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorEvent:
    """Base event from the orchestration engine."""
    event_type: str = ""
    project_id: str | None = None


@dataclass
class EngineStarted(OrchestratorEvent):
    event_type: str = "engine_started"
    task_definition: str = ""


@dataclass
class EngineFinished(OrchestratorEvent):
    event_type: str = "engine_finished"
    success: bool = True
    summary: str = ""
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class AgentSpawned(OrchestratorEvent):
    event_type: str = "agent_spawned"
    agent_id: str = ""
    parent_id: str | None = None
    role: str = ""
    model: str = ""
    depth: int = 0
    prompt: str = ""
    name: str = ""


@dataclass
class AgentStateChanged(OrchestratorEvent):
    event_type: str = "agent_state_changed"
    agent_id: str = ""
    old_state: str = ""
    new_state: str = ""


@dataclass
class AgentRestarted(OrchestratorEvent):
    event_type: str = "agent_restarted"
    agent_id: str = ""
    parent_id: str | None = None
    role: str = ""
    model: str = ""
    prompt: str = ""
    name: str = ""


@dataclass
class AgentKilled(OrchestratorEvent):
    event_type: str = "agent_killed"
    agent_id: str = ""


@dataclass
class StreamChunk(OrchestratorEvent):
    event_type: str = "stream_chunk"
    agent_id: str = ""
    text: str = ""


@dataclass
class ToolCallStarted(OrchestratorEvent):
    event_type: str = "tool_call_started"
    agent_id: str = ""
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""


@dataclass
class ToolCallCompleted(OrchestratorEvent):
    event_type: str = "tool_call_completed"
    agent_id: str = ""
    tool_id: str = ""
    result: str = ""
    is_error: bool = False


@dataclass
class ToolCallDelta(OrchestratorEvent):
    event_type: str = "tool_call_delta"
    agent_id: str = ""
    tool_id: str = ""
    delta: str = ""
    stream: str = "stdout"


@dataclass
class AgentResult(OrchestratorEvent):
    event_type: str = "agent_result"
    agent_id: str = ""
    result: str = ""
    is_error: bool = False


@dataclass
class PermissionRequest(OrchestratorEvent):
    event_type: str = "permission_request"
    agent_id: str = ""
    request_id: str = ""
    tool_name: str = ""
    agent_name: str = ""
    arguments: str = ""
    message_index: int = 0


@dataclass
class UserQuestionRequest(OrchestratorEvent):
    """An agent is asking the user a structured question with options."""
    event_type: str = "user_question_request"
    agent_id: str = ""
    request_id: str = ""
    agent_name: str = ""
    question: str = ""
    options: list = field(default_factory=list)


@dataclass
class Thinking(OrchestratorEvent):
    event_type: str = "thinking"
    agent_id: str = ""
    text: str = ""


@dataclass
class UserPrompt(OrchestratorEvent):
    event_type: str = "user_prompt"
    agent_id: str = ""
    text: str = ""


@dataclass
class ContextWindowUsage(OrchestratorEvent):
    """Per-turn token usage and context window utilization."""
    event_type: str = "context_window_usage"
    agent_id: str = ""
    model: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    max_context_tokens: int | None = None
    percent_used: float | None = None


@dataclass
class MessageRouted(OrchestratorEvent):
    event_type: str = "message_routed"
    source_agent_id: str = ""
    target_agent_id: str = ""
    message_type: str = ""


@dataclass
class TopicEvent(OrchestratorEvent):
    event_type: str = "topic_event"
    source_agent_id: str = ""
    target_agent_id: str = ""
    topic: str = ""
    scope: str | None = None
    urgency: str = "normal"
    message_id: str = ""
    policy_snapshot_id: str = ""


@dataclass
class DigestEvent(OrchestratorEvent):
    event_type: str = "digest_event"
    target_agent_id: str = ""
    message_count: int = 0


@dataclass
class LeaseStatus(OrchestratorEvent):
    """Capability lease granted, revoked, or expired."""
    event_type: str = "lease_status"
    agent_id: str = ""
    lease_id: str = ""
    action: str = ""
    tool_pattern: str = ""
    risk_tier_max: int = 0
    ttl_seconds: float = 0.0


@dataclass
class AuditEntry(OrchestratorEvent):
    """Policy decision audit record."""
    event_type: str = "audit_entry"
    agent_id: str = ""
    tool_name: str = ""
    decision: str = ""
    risk_tier: int = 0
    reason_text: str = ""
    policy_snapshot_id: str = ""


@dataclass
class ExpertStatus(OrchestratorEvent):
    """Expert lifecycle state change."""
    event_type: str = "expert_status"
    expert_id: str = ""
    lifecycle_state: str = ""
    utility_score: float = 0.0


@dataclass
class BudgetStatus(OrchestratorEvent):
    """Resource budget update."""
    event_type: str = "budget_status"
    current_token_usage: int = 0
    max_total_tokens: int = 0
    current_agent_count: int = 0
    spawns_this_hour: int = 0


@dataclass
class DecisionReport(OrchestratorEvent):
    """Ego decision report created."""
    event_type: str = "decision_report"
    decision_id: str = ""
    task_prompt: str = ""
    decision: str = ""
    rationale: str = ""


@dataclass
class FileChanged(OrchestratorEvent):
    """A file was created, modified, or deleted by an agent tool call."""
    event_type: str = "file_changed"
    agent_id: str = ""
    file_path: str = ""
    change_type: str = ""  # "create", "modify", "delete"
    tool_call_id: str = ""
    tool_name: str = ""
    message_index: int = 0
    old_content: str | None = None
    new_content: str | None = None  # For Edit: the new_string; for Write: the new file content
    pre_tool_content: str | None = None  # Full file content before the tool ran
    added_ranges: list = field(default_factory=list)  # [{startLine, endLine}]
    removed_ranges: list = field(default_factory=list)


@dataclass
class SnapshotCreated(OrchestratorEvent):
    event_type: str = "snapshot_created"
    session_id: str = ""
    snapshot_id: str = ""
    description: str = ""


@dataclass
class SnapshotRestored(OrchestratorEvent):
    event_type: str = "snapshot_restored"
    session_id: str = ""
    snapshot_id: str = ""


# Map of event type strings to dataclass constructors
_EVENT_MAP: dict[str, type[OrchestratorEvent]] = {
    "engine_started": EngineStarted,
    "engine_finished": EngineFinished,
    "agent_spawned": AgentSpawned,
    "agent_restarted": AgentRestarted,
    "agent_state_changed": AgentStateChanged,
    "agent_killed": AgentKilled,
    "stream_chunk": StreamChunk,
    "tool_call_started": ToolCallStarted,
    "tool_call_completed": ToolCallCompleted,
    "tool_call_delta": ToolCallDelta,
    "agent_result": AgentResult,
    "permission_request": PermissionRequest,
    "user_question_request": UserQuestionRequest,
    "thinking": Thinking,
    "user_prompt": UserPrompt,
    "context_window_usage": ContextWindowUsage,
    "message_routed": MessageRouted,
    "topic_event": TopicEvent,
    "digest_event": DigestEvent,
    "lease_status": LeaseStatus,
    "audit_entry": AuditEntry,
    "expert_status": ExpertStatus,
    "budget_status": BudgetStatus,
    "decision_report": DecisionReport,
    "file_changed": FileChanged,
    "snapshot_created": SnapshotCreated,
    "snapshot_restored": SnapshotRestored,
}


def event_to_dict(event: OrchestratorEvent) -> dict[str, Any]:
    """Convert a typed event dataclass to a plain dict for JSON serialization."""
    d: dict[str, Any] = {}
    for f in event.__dataclass_fields__:
        val = getattr(event, f)
        if val is not None:
            d[f] = val
    # Use "event" key instead of "event_type" for consistency with engine callbacks
    if "event_type" in d:
        d["event"] = d.pop("event_type")
    return d


def dict_to_event(data: dict[str, Any]) -> OrchestratorEvent:
    """Convert an engine callback dict to a typed event dataclass."""
    event_type = data.get("event", "")
    cls = _EVENT_MAP.get(event_type, OrchestratorEvent)
    # Filter dict keys to only those the dataclass accepts
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    # Map "event" key to "event_type" field
    if "event" in data and "event_type" not in filtered:
        filtered["event_type"] = data["event"]
    return cls(**filtered)
