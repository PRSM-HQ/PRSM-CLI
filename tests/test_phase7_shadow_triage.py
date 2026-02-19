"""Phase 7 tests — shadow triage and config plumbing."""

from __future__ import annotations

import pytest

from prsm.engine.config import EngineConfig
from prsm.engine.message_router import MessageRouter, TriageDecision
from prsm.engine.models import AgentDescriptor, AgentRole, MessageType, RoutedMessage
from prsm.engine.shadow_triage import ShadowTriageModel
from prsm.engine.yaml_config import load_yaml_config


def _agent(agent_id: str, role: AgentRole = AgentRole.WORKER) -> AgentDescriptor:
    return AgentDescriptor(agent_id=agent_id, role=role, prompt=f"agent:{agent_id}")


def test_compare_with_rules_logs_agreement_and_disagreement():
    model = ShadowTriageModel(model="test-model")
    agree = model.compare_with_rules(
        TriageDecision(decision="deliver_now"),
        TriageDecision(decision="deliver_now"),
    )
    disagree = model.compare_with_rules(
        TriageDecision(decision="deliver_now"),
        TriageDecision(decision="drop"),
    )

    assert agree["agreement"] is True
    assert disagree["agreement"] is False
    assert disagree["false_negative"] is True


def test_get_comparison_stats_computes_agreement_fp_fn_rates():
    model = ShadowTriageModel(model="test-model")
    model.compare_with_rules(
        TriageDecision(decision="deliver_now"),
        TriageDecision(decision="deliver_now"),
    )
    model.compare_with_rules(
        TriageDecision(decision="drop"),
        TriageDecision(decision="deliver_now"),
    )
    model.compare_with_rules(
        TriageDecision(decision="deliver_now"),
        TriageDecision(decision="deliver_digest"),
    )

    stats = model.get_comparison_stats()
    assert stats["total_comparisons"] == 3
    assert stats["agreement_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["false_positive_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["false_negative_rate"] == pytest.approx(1.0 / 3.0)


@pytest.mark.asyncio
async def test_shadow_decision_does_not_override_rules_enforcement():
    class _AlwaysDeliverShadow:
        async def shadow_evaluate(self, message, subscriber_profile):
            return TriageDecision(decision="deliver_now")

        def compare_with_rules(self, rules_decision, model_decision):
            return {"agreement": False, "false_positive": True, "false_negative": False}

    router = MessageRouter(
        triage_rules=[
            {"topic_pattern": "telemetry.*", "action": "drop"},
            {"topic_pattern": "*", "action": "deliver_now"},
        ],
        shadow_triage_model=_AlwaysDeliverShadow(),
    )
    router.register_agent(_agent("source"))
    router.register_agent(_agent("sink"))
    router.subscribe("sink", "*")

    decisions = await router.publish(
        RoutedMessage(
            source_agent_id="source",
            message_type=MessageType.TOPIC_EVENT,
            topic="telemetry.cpu",
            scope="global",
            urgency="low",
            payload={"value": 90},
        )
    )
    assert decisions[0].decision == "drop"
    with pytest.raises(TimeoutError):
        await router.receive("sink", timeout=0.05)


def test_yaml_config_parses_phase7_fields(tmp_path):
    config_file = tmp_path / "prsm.yaml"
    config_file.write_text(
        """
engine:
  triage_model_shadow_enabled: true
  triage_shadow_model: claude-haiku
  telemetry_db_path: ./tmp/telemetry.sqlite3
""".strip(),
        encoding="utf-8",
    )

    cfg = load_yaml_config(config_file)
    assert cfg.engine.triage_model_shadow_enabled is True
    assert cfg.engine.triage_shadow_model == "claude-haiku"
    assert cfg.engine.telemetry_db_path == "./tmp/telemetry.sqlite3"


def test_engine_config_from_env_phase7_fields(monkeypatch):
    monkeypatch.setenv("PRSM_TRIAGE_MODEL_SHADOW", "1")
    monkeypatch.setenv("PRSM_TRIAGE_SHADOW_MODEL", "my-shadow-model")
    monkeypatch.setenv("PRSM_TELEMETRY_DB_PATH", "/tmp/telemetry.sqlite3")

    cfg = EngineConfig.from_env()
    assert cfg.triage_model_shadow_enabled is True
    assert cfg.triage_shadow_model == "my-shadow-model"
    assert cfg.telemetry_db_path == "/tmp/telemetry.sqlite3"
