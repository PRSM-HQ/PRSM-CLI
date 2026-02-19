"""Core data models for the orchestration engine.

All dataclasses, enums, and type aliases. Single source of truth
to avoid circular imports.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    """Agent lifecycle states. See lifecycle.py for transition rules."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    WAITING_FOR_PARENT = "waiting_for_parent"
    WAITING_FOR_CHILD = "waiting_for_child"
    WAITING_FOR_EXPERT = "waiting_for_expert"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class AgentRole(str, Enum):
    """Classification of agent purpose."""
    MASTER = "master"
    WORKER = "worker"
    EXPERT = "expert"
    REVIEWER = "reviewer"


class MessageType(str, Enum):
    """Types of messages routed between agents."""
    QUESTION = "question"
    ANSWER = "answer"
    PROGRESS_UPDATE = "progress_update"
    TASK_RESULT = "task_result"
    EXPERT_REQUEST = "expert_request"
    EXPERT_RESPONSE = "expert_response"
    SPAWN_REQUEST = "spawn_request"
    KILL_SIGNAL = "kill_signal"
    USER_PROMPT = "user_prompt"
    TOPIC_EVENT = "topic_event"
    DIGEST = "digest"
    SUPEREGO_REVIEW = "superego_review"
    SUPEREGO_VETO = "superego_veto"


class PermissionMode(str, Enum):
    """Maps to claude_agent_sdk permission modes."""
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"
    PLAN = "plan"
    DELEGATE = "delegate"


def _make_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ExpertProfile:
    """Defines an expert agent's capabilities and configuration.

    Users register these via engine.register_expert(). The library
    ships with no built-in profiles — all domain knowledge is
    provided by the consumer.
    """
    expert_id: str
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=lambda: [
        "Read", "Grep", "Glob", "Bash",
    ])
    model: str = "claude-opus-4-6"
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    max_concurrent_consultations: int = 3
    cwd: str | None = None
    provider: str = "claude"
    mcp_servers: dict[str, Any] | None = None
    lifecycle_state: str = "active"
    created_at: datetime | None = None
    deprecated_at: datetime | None = None
    deprecation_reason: str = ""
    evaluation_criteria: list[str] = field(default_factory=list)
    deprecation_policy: str = ""
    consultation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration_seconds: float = 0.0
    avg_confidence: float = 0.0
    utility_score: float = 0.0


@dataclass
class AgentDescriptor:
    """Complete description of an agent instance.

    Created by AgentManager, consumed by AgentSession.
    """
    agent_id: str = field(default_factory=_make_id)
    parent_id: str | None = None
    role: AgentRole = AgentRole.WORKER
    expert_profile: ExpertProfile | None = None
    state: AgentState = AgentState.PENDING
    prompt: str = ""
    tools: list[str] = field(default_factory=list)
    model: str = "claude-opus-4-6"
    permission_mode: PermissionMode = PermissionMode.ACCEPT_EDITS
    cwd: str = "."
    mcp_servers: dict[str, Any] | None = None
    exclude_plugins: list[str] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None
    result_summary: str | None = None
    result_artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    max_depth: int = 5
    provider: str = "claude"


@dataclass
class RoutedMessage:
    """A message in transit between agents."""
    message_id: str = field(default_factory=_make_id)
    message_type: MessageType = MessageType.PROGRESS_UPDATE
    source_agent_id: str = ""
    target_agent_id: str = ""
    payload: Any = None
    correlation_id: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    topic: str | None = None
    scope: str | None = None
    urgency: str = "normal"
    ttl: float | None = None
    permissions_tags: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)


@dataclass
class SpawnRequest:
    """Parameters for spawning a child agent."""
    parent_id: str | None
    prompt: str
    role: AgentRole = AgentRole.WORKER
    expert_id: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str = "claude-opus-4-6"
    permission_mode: PermissionMode = PermissionMode.ACCEPT_EDITS
    cwd: str | None = None
    mcp_servers: dict[str, Any] | None = None
    exclude_plugins: list[str] | None = None
    wait_for_result: bool = True
    provider: str = "claude"


@dataclass
class AgentResult:
    """Final result returned by a completed agent."""
    agent_id: str
    success: bool
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class ExpertOutput:
    """Structured output from an Id expert (worker/expert agent)."""
    agent_id: str
    summary: str
    steps: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    verification_hooks: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    rollback_plan: str = ""
    confidence: float = 0.0
    confidence_basis: str = ""


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_type: str
    command: str = ""
    passed: bool = False
    output_summary: str = ""
    artifact_id: str | None = None
    duration_seconds: float = 0.0


@dataclass
class EgoDecisionReport:
    """Master acceptance/rejection decision with evidence trail."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task_prompt: str = ""
    candidates: list[ExpertOutput] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    decision: str = ""
    rationale: str = ""
    policy_snapshot_id: str = ""
    artifact_ids: list[str] = field(default_factory=list)
    remaining_risks: list[str] = field(default_factory=list)
    rollback_plan: str = ""
    next_actions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
