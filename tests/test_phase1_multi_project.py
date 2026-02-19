from __future__ import annotations

import asyncio
from pathlib import Path

from prsm.adapters.events import dict_to_event
from prsm.engine.config import EngineConfig
from prsm.engine.engine import OrchestrationEngine
from prsm.engine.project_registry import ProjectRegistry
from prsm.engine.yaml_config import load_yaml_config
from prsm.shared.services.project import ProjectManager


def test_project_manager_phase1_paths(tmp_path: Path) -> None:
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()

    policy_dir = ProjectManager.get_policy_dir(project_dir)
    artifacts_dir = ProjectManager.get_artifacts_dir(project_dir)
    audit_log = ProjectManager.get_audit_log_path(project_dir)

    assert policy_dir == project_dir / "policy"
    assert artifacts_dir == project_dir / "artifacts"
    assert audit_log == project_dir / "audit" / "audit.db"
    assert policy_dir.exists()
    assert artifacts_dir.exists()
    assert audit_log.parent.exists()


def test_project_registry_register_and_default() -> None:
    registry = ProjectRegistry()
    cfg = EngineConfig(default_model="claude-opus-4-6")

    runtime = registry.register_project("proj-1", cfg, memory_scope="/tmp/mem")

    assert runtime.project_id == "proj-1"
    assert runtime.status == "idle"
    assert registry.default_project is not None
    assert registry.default_project.project_id == "proj-1"
    assert len(registry.list_projects()) == 1


def test_project_registry_lazy_engine_factory() -> None:
    registry = ProjectRegistry()
    cfg = EngineConfig(default_model="claude-opus-4-6")
    registry.register_project("proj-1", cfg)

    built = {"count": 0}

    def factory(local_cfg: EngineConfig):
        built["count"] += 1
        return {"project_id": local_cfg.project_id, "ok": True}

    cfg.project_id = "proj-1"
    engine_1 = registry.get_or_create_engine("proj-1", engine_factory=factory)
    engine_2 = registry.get_or_create_engine("proj-1", engine_factory=factory)

    assert built["count"] == 1
    assert engine_1 is engine_2
    assert registry.get_project("proj-1").status == "active"


def test_engine_event_callback_includes_project_id() -> None:
    captured: list[dict] = []

    async def callback(event: dict) -> None:
        captured.append(event)

    wrapped = OrchestrationEngine._build_scoped_event_callback(callback, "proj-1")
    assert wrapped is not None
    asyncio.run(wrapped({"event": "engine_started"}))  # type: ignore[misc]

    assert captured
    assert captured[0]["project_id"] == "proj-1"


def test_dict_to_event_keeps_project_id() -> None:
    event = dict_to_event(
        {"event": "engine_started", "task_definition": "x", "project_id": "proj-1"}
    )
    assert getattr(event, "project_id", None) == "proj-1"


def test_yaml_config_parses_projects_section(tmp_path: Path) -> None:
    config_path = tmp_path / "prsm.yaml"
    config_path.write_text(
        "projects:\n"
        "  a:\n"
        "    cwd: /tmp/a\n"
        "  b:\n"
        "    cwd: /tmp/b\n"
    )
    cfg = load_yaml_config(config_path)
    assert "a" in cfg.projects
    assert "b" in cfg.projects


def test_cross_project_broker_wildcards_and_recipients() -> None:
    registry = ProjectRegistry()
    cfg = EngineConfig(default_model="claude-opus-4-6")
    registry.register_project("proj-a", cfg)
    registry.register_project("proj-b", cfg)

    broker = registry.event_broker
    broker.subscribe("proj-a", "build.*")
    broker.subscribe("proj-b", "*")

    from prsm.engine.project_registry import CrossProjectEvent

    asyncio.run(
        broker.publish(
            CrossProjectEvent(
                topic="build.started",
                source_project_id="proj-a",
                payload={"step": 1},
            )
        )
    )
    log = broker.get_event_log(limit=1)
    assert len(log) == 1
    assert log[0]["topic"] == "build.started"
    assert sorted(log[0]["recipients"]) == ["proj-a", "proj-b"]


def test_cross_project_broker_unsubscribe_and_subscription_view() -> None:
    registry = ProjectRegistry()
    cfg = EngineConfig(default_model="claude-opus-4-6")
    registry.register_project("proj-a", cfg)
    broker = registry.event_broker

    broker.subscribe("proj-a", "alerts.*")
    assert broker.get_subscriptions("proj-a") == ["alerts.*"]

    broker.unsubscribe("proj-a", "alerts.*")
    assert broker.get_subscriptions("proj-a") == []
