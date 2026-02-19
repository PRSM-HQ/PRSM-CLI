"""Multi-project runtime registry primitives.

Phase 1 scope: foundational project runtime metadata + lazy engine creation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from .config import EngineConfig


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CrossProjectEvent:
    """Event envelope used by the cross-project broker."""

    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    topic: str = ""
    source_project_id: str = ""
    scope: str = "global"  # "global" or "project:<id>"
    urgency: str = "normal"  # "low", "normal", "high"
    ttl: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)


class CrossProjectEventBroker:
    """Routes cross-project events by topic filter."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[str]] = {}
        self._event_log: list[dict[str, Any]] = []

    async def publish(self, event: CrossProjectEvent) -> None:
        """Publish and record an event for subscribers."""
        recipients: set[str] = set()
        for topic_filter, project_ids in self._subscriptions.items():
            if self._topic_matches(topic_filter, event.topic):
                recipients.update(project_ids)

        self._event_log.append(
            {
                "event_id": event.event_id,
                "topic": event.topic,
                "source_project_id": event.source_project_id,
                "scope": event.scope,
                "urgency": event.urgency,
                "ttl": event.ttl,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
                "recipients": sorted(recipients),
            }
        )

    def subscribe(self, project_id: str, topic_filter: str) -> None:
        """Subscribe a project to a topic filter."""
        if topic_filter not in self._subscriptions:
            self._subscriptions[topic_filter] = set()
        self._subscriptions[topic_filter].add(project_id)

    def unsubscribe(self, project_id: str, topic_filter: str) -> None:
        """Remove a topic filter subscription."""
        project_ids = self._subscriptions.get(topic_filter)
        if not project_ids:
            return
        project_ids.discard(project_id)
        if not project_ids:
            self._subscriptions.pop(topic_filter, None)

    @property
    def event_log(self) -> list[dict[str, Any]]:
        return list(self._event_log)

    def get_event_log(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return recorded cross-project events, optionally capped to last N."""
        if limit is None or limit <= 0:
            return list(self._event_log)
        return list(self._event_log[-limit:])

    def clear_event_log(self) -> None:
        """Clear all recorded cross-project events."""
        self._event_log.clear()

    def get_subscriptions(self, project_id: str) -> list[str]:
        """Return sorted topic filters currently subscribed by a project."""
        topics = [
            topic_filter
            for topic_filter, project_ids in self._subscriptions.items()
            if project_id in project_ids
        ]
        return sorted(topics)

    @staticmethod
    def _topic_matches(topic_filter: str, topic: str) -> bool:
        if topic_filter == "*":
            return True
        if topic_filter.endswith(".*"):
            return topic.startswith(topic_filter[:-1])
        return topic_filter == topic


@dataclass
class ProjectRuntime:
    """Runtime container for a single project-scoped orchestration context."""

    project_id: str
    engine: Any | None
    bridge: Any | None
    config: EngineConfig
    policy_scope: str
    memory_scope: str
    status: Literal["idle", "active", "unloading"]
    created_at: datetime
    last_active: datetime


class ProjectRegistry:
    """Top-level coordinator for project-scoped runtime metadata."""

    def __init__(self, global_config: dict[str, Any] | None = None) -> None:
        self._global_config = global_config or {}
        self._projects: dict[str, ProjectRuntime] = {}
        self._default_project_id: str | None = None
        self._event_broker: CrossProjectEventBroker = CrossProjectEventBroker()

    def register_project(
        self,
        project_id: str,
        config: EngineConfig,
        *,
        bridge: Any | None = None,
        policy_scope: str | None = None,
        memory_scope: str = "",
    ) -> ProjectRuntime:
        """Register (or refresh) a project runtime without starting an engine."""
        now = _utc_now()
        runtime = self._projects.get(project_id)
        if runtime is None:
            runtime = ProjectRuntime(
                project_id=project_id,
                engine=None,
                bridge=bridge,
                config=config,
                policy_scope=policy_scope or f"project:{project_id}",
                memory_scope=memory_scope,
                status="idle",
                created_at=now,
                last_active=now,
            )
            self._projects[project_id] = runtime
        else:
            runtime.config = config
            runtime.bridge = bridge or runtime.bridge
            runtime.policy_scope = policy_scope or runtime.policy_scope
            runtime.memory_scope = memory_scope or runtime.memory_scope
            runtime.last_active = now

        if self._default_project_id is None:
            self._default_project_id = project_id

        return runtime

    def unregister_project(self, project_id: str) -> None:
        """Remove project runtime metadata from the registry."""
        self._projects.pop(project_id, None)
        if self._default_project_id == project_id:
            self._default_project_id = next(iter(self._projects.keys()), None)

    def get_project(self, project_id: str) -> ProjectRuntime:
        """Fetch a project runtime by ID."""
        if project_id not in self._projects:
            raise KeyError(f"Project not registered: {project_id}")
        return self._projects[project_id]

    def list_projects(self) -> list[ProjectRuntime]:
        """List all registered runtimes."""
        return list(self._projects.values())

    def get_or_create_engine(
        self,
        project_id: str,
        engine_factory: Any | None = None,
    ) -> Any:
        """Lazily create an engine for a project if one does not exist."""
        runtime = self.get_project(project_id)
        if runtime.engine is None:
            if engine_factory is None:
                from .engine import OrchestrationEngine

                runtime.engine = OrchestrationEngine(config=runtime.config)
            else:
                runtime.engine = engine_factory(runtime.config)
        runtime.status = "active"
        runtime.last_active = _utc_now()
        return runtime.engine

    @property
    def default_project(self) -> ProjectRuntime | None:
        """Return the default project for single-project compatibility."""
        if self._default_project_id is None:
            return None
        return self._projects.get(self._default_project_id)

    @property
    def event_broker(self) -> CrossProjectEventBroker:
        return self._event_broker
