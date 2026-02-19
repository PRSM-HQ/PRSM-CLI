from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prsm.adapters.events import DigestEvent, TopicEvent, dict_to_event, event_to_dict
from prsm.engine.models import (
    AgentDescriptor,
    AgentRole,
    MessageType,
    RoutedMessage,
)
from prsm.engine.message_router import MessageRouter
from prsm.engine.telemetry import TelemetryCollector


def _agent(agent_id: str, role: AgentRole = AgentRole.WORKER) -> AgentDescriptor:
    return AgentDescriptor(agent_id=agent_id, role=role, prompt=f"agent:{agent_id}")


@pytest.mark.asyncio
async def test_direct_send_receive_behavior_unchanged() -> None:
    router = MessageRouter()
    router.register_agent(_agent("parent", AgentRole.MASTER))
    router.register_agent(_agent("child"))

    message = RoutedMessage(
        message_type=MessageType.QUESTION,
        source_agent_id="child",
        target_agent_id="parent",
        payload={"question": "status?"},
        correlation_id="corr-1",
    )
    await router.send(message)
    received = await router.receive("parent", timeout=0.2)

    assert received.message_type == MessageType.QUESTION
    assert received.source_agent_id == "child"
    assert received.correlation_id == "corr-1"


@pytest.mark.asyncio
async def test_publish_routes_by_topic_scope_and_urgency() -> None:
    router = MessageRouter(triage_rules=[{"topic_pattern": "*", "action": "deliver_now"}])
    router.register_agent(_agent("source"))
    router.register_agent(_agent("alpha"))
    router.register_agent(_agent("beta"))

    router.subscribe("alpha", "db.*", scope_filter="project:alpha", urgency_threshold="low")
    router.subscribe("beta", "db.*", scope_filter="global", urgency_threshold="low")

    decisions = await router.publish(
        RoutedMessage(
            message_type=MessageType.TOPIC_EVENT,
            source_agent_id="source",
            topic="db.migration",
            scope="project:alpha",
            urgency="normal",
            payload={"version": 3},
        )
    )

    received_alpha = await router.receive("alpha", timeout=0.2)
    assert received_alpha.message_type == MessageType.TOPIC_EVENT
    assert received_alpha.topic == "db.migration"
    assert len(decisions) == 1
    assert decisions[0].subscriber_id == "alpha"
    assert decisions[0].decision == "deliver_now"

    with pytest.raises(TimeoutError):
        await router.receive("beta", timeout=0.05)


@pytest.mark.asyncio
async def test_triage_rule_drop_prevents_delivery() -> None:
    router = MessageRouter(
        triage_rules=[
            {"topic_pattern": "telemetry.*", "action": "drop"},
            {"topic_pattern": "*", "action": "deliver_now"},
        ]
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
            payload={"value": 80},
        )
    )
    assert len(decisions) == 1
    assert decisions[0].decision == "drop"
    assert decisions[0].reason_code == "rule_0"

    with pytest.raises(TimeoutError):
        await router.receive("sink", timeout=0.05)


@pytest.mark.asyncio
async def test_dedup_blocks_identical_messages_within_window() -> None:
    router = MessageRouter(triage_rules=[{"topic_pattern": "*", "action": "deliver_now"}])
    router.register_agent(_agent("source"))
    router.register_agent(_agent("sink"))
    router.subscribe("sink", "alerts.*")

    payload = {"code": "A1", "detail": "same"}
    first = RoutedMessage(
        source_agent_id="source",
        message_type=MessageType.TOPIC_EVENT,
        topic="alerts.error",
        scope="global",
        urgency="high",
        payload=payload,
    )
    second = RoutedMessage(
        source_agent_id="source",
        message_type=MessageType.TOPIC_EVENT,
        topic="alerts.error",
        scope="global",
        urgency="high",
        payload=payload,
    )

    first_decisions = await router.publish(first)
    second_decisions = await router.publish(second)

    assert first_decisions[0].decision == "deliver_now"
    assert second_decisions[0].decision == "drop"
    assert second_decisions[0].reason_code == "dedup_window"

    _ = await router.receive("sink", timeout=0.2)
    with pytest.raises(TimeoutError):
        await router.receive("sink", timeout=0.05)


@pytest.mark.asyncio
async def test_rate_limiting_sends_digest_batch() -> None:
    router = MessageRouter(triage_rules=[{"topic_pattern": "*", "action": "deliver_now"}])
    router.register_agent(_agent("source"))
    router.register_agent(_agent("sink"))
    router.subscribe("sink", "*")
    router._rate_limit_per_minute = 1

    first = RoutedMessage(
        source_agent_id="source",
        message_type=MessageType.TOPIC_EVENT,
        topic="security.alert",
        scope="global",
        urgency="high",
        payload={"id": 1},
    )
    second = RoutedMessage(
        source_agent_id="source",
        message_type=MessageType.TOPIC_EVENT,
        topic="security.alert",
        scope="global",
        urgency="high",
        payload={"id": 2},
    )

    first_decision = (await router.publish(first))[0]
    second_decision = (await router.publish(second))[0]
    assert first_decision.decision == "deliver_now"
    assert second_decision.decision == "deliver_digest"
    assert second_decision.reason_code == "rate_limited"

    flush_count = await router.flush_digests()
    assert flush_count == 1

    first_msg = await router.receive("sink", timeout=0.2)
    digest_msg = await router.receive("sink", timeout=0.2)
    assert first_msg.message_type == MessageType.TOPIC_EVENT
    assert digest_msg.message_type == MessageType.DIGEST
    assert digest_msg.payload["count"] == 1


@pytest.mark.asyncio
async def test_ttl_expiry_drops_message() -> None:
    router = MessageRouter(triage_rules=[{"topic_pattern": "*", "action": "deliver_now"}])
    router.register_agent(_agent("source"))
    router.register_agent(_agent("sink"))
    router.subscribe("sink", "*")

    old = datetime.now(timezone.utc) - timedelta(seconds=120)
    decisions = await router.publish(
        RoutedMessage(
            source_agent_id="source",
            message_type=MessageType.TOPIC_EVENT,
            topic="alerts.expired",
            scope="global",
            urgency="high",
            ttl=30.0,
            timestamp=old,
            payload={"stale": True},
        )
    )
    assert decisions[0].decision == "drop"
    assert decisions[0].reason_code == "expired_ttl"


@pytest.mark.asyncio
async def test_publish_records_shadow_comparison_when_enabled(tmp_path) -> None:
    class _DisagreeingShadow:
        async def shadow_evaluate(self, message, subscriber_profile):
            return type("ShadowDecision", (), {"decision": "deliver_now"})()

        def compare_with_rules(self, rules_decision, model_decision):
            return {"agreement": False, "false_positive": True, "false_negative": False}

    telemetry = TelemetryCollector(tmp_path / "telemetry.sqlite3")
    router = MessageRouter(
        triage_rules=[{"topic_pattern": "telemetry.*", "action": "drop"}],
        telemetry_collector=telemetry,
        shadow_triage_model=_DisagreeingShadow(),
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
            payload={"value": 1},
        )
    )
    assert decisions[0].decision == "drop"
    with pytest.raises(TimeoutError):
        await router.receive("sink", timeout=0.05)

    dashboard = telemetry.get_dashboard_data("24h")
    assert dashboard["triage_false_positive_rate"] == 1.0


def test_topic_and_digest_event_mapping() -> None:
    topic = dict_to_event(
        {
            "event": "topic_event",
            "source_agent_id": "a",
            "target_agent_id": "b",
            "topic": "security.alert",
            "scope": "global",
            "urgency": "high",
            "message_id": "m1",
            "policy_snapshot_id": "rule:0",
        }
    )
    assert isinstance(topic, TopicEvent)
    assert topic.topic == "security.alert"

    digest = dict_to_event(
        {
            "event": "digest_event",
            "target_agent_id": "b",
            "message_count": 2,
        }
    )
    assert isinstance(digest, DigestEvent)
    data = event_to_dict(digest)
    assert data["event"] == "digest_event"
    assert data["message_count"] == 2
