"""Message router for inter-agent communication.

Routes messages between agents using asyncio.Queue per agent.
Supports blocking (ask_parent) and non-blocking (report_progress)
delivery patterns.

The router does NOT store message history — agents are responsible
for their own context. This keeps the router stateless and simple.
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .config import fire_event
from .models import AgentState, MessageType, RoutedMessage
from .errors import MessageRoutingError

if TYPE_CHECKING:
    from .models import AgentDescriptor

logger = logging.getLogger(__name__)

# Default cap on non-matching messages buffered during a filtered receive().
# Prevents unbounded memory growth when a queue has many unrelated messages.
DEFAULT_RECEIVE_BUFFER_LIMIT = 100


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_URGENCY_RANK: dict[str, int] = {
    "low": 0,
    "normal": 1,
    "high": 2,
}


@dataclass
class RouterMetrics:
    """Lightweight counters for observability."""

    messages_sent: int = 0
    messages_received: int = 0
    messages_requeued: int = 0
    messages_dropped_requeue: int = 0
    messages_dropped_send: int = 0
    receive_buffer_overflows: int = 0
    receive_timeouts: int = 0
    subscriptions_created: int = 0
    subscriptions_removed: int = 0
    topic_messages_published: int = 0
    triage_deliver_now: int = 0
    triage_deliver_digest: int = 0
    triage_drop: int = 0
    digests_flushed: int = 0

    def snapshot(self) -> dict[str, int]:
        """Return a dict copy of all counters."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_requeued": self.messages_requeued,
            "messages_dropped_requeue": self.messages_dropped_requeue,
            "messages_dropped_send": self.messages_dropped_send,
            "receive_buffer_overflows": self.receive_buffer_overflows,
            "receive_timeouts": self.receive_timeouts,
            "subscriptions_created": self.subscriptions_created,
            "subscriptions_removed": self.subscriptions_removed,
            "topic_messages_published": self.topic_messages_published,
            "triage_deliver_now": self.triage_deliver_now,
            "triage_deliver_digest": self.triage_deliver_digest,
            "triage_drop": self.triage_drop,
            "digests_flushed": self.digests_flushed,
        }


@dataclass
class TopicSubscription:
    """Topic subscription for a single receiving agent."""

    subscriber_id: str
    topic_pattern: str
    scope_filter: str | None = None
    urgency_threshold: str = "low"
    created_at: datetime = field(default_factory=_utc_now)


@dataclass
class TriageDecision:
    """Audit record for a publish triage decision."""

    decision: str
    reason_code: str = ""
    relevance_score: float = 0.0
    subscriber_id: str = ""
    message_id: str = ""
    policy_snapshot_id: str = ""


class MessageRouter:
    """Routes messages between agent sessions.

    Each agent gets an asyncio.Queue for incoming messages.
    Senders push to the target's queue; receivers await on their
    own queue.
    """

    def __init__(
        self,
        queue_maxsize: int = 1000,
        event_callback: object | None = None,
        receive_buffer_limit: int = DEFAULT_RECEIVE_BUFFER_LIMIT,
        triage_rules: list[dict] | None = None,
        telemetry_collector: object | None = None,
        shadow_triage_model: object | None = None,
    ) -> None:
        self._queues: dict[str, asyncio.Queue[RoutedMessage]] = {}
        self._agents: dict[str, AgentDescriptor] = {}
        self._queue_maxsize = queue_maxsize
        self._event_callback = event_callback
        self._receive_buffer_limit = receive_buffer_limit
        self.metrics = RouterMetrics()
        # Tracks who-waits-on-whom for deadlock detection
        self._waiting_on: dict[str, str] = {}
        self._subscriptions: dict[str, list[TopicSubscription]] = {}
        self._triage_rules: list[dict] = triage_rules or []
        self._digest_buffer: dict[str, list[RoutedMessage]] = {}
        self._triage_log: list[TriageDecision] = []
        self._telemetry_collector = telemetry_collector
        self._shadow_triage_model = shadow_triage_model
        self._dedup_cache: dict[str, datetime] = {}
        self._rate_window: dict[str, list[datetime]] = {}
        self._dedup_window_seconds: float = 60.0
        self._rate_limit_per_minute: int = 20

    def register_agent(self, descriptor: AgentDescriptor) -> None:
        """Register an agent and create its message queue."""
        if descriptor.agent_id in self._queues:
            logger.warning(
                "Agent already registered: %s", descriptor.agent_id
            )
            return
        self._queues[descriptor.agent_id] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        self._agents[descriptor.agent_id] = descriptor
        logger.info(
            "Agent registered: %s (role=%s)",
            descriptor.agent_id[:8],
            descriptor.role.value,
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's queue. Undelivered messages are dropped."""
        self._queues.pop(agent_id, None)
        self._agents.pop(agent_id, None)
        self._waiting_on.pop(agent_id, None)
        self._subscriptions.pop(agent_id, None)
        self._digest_buffer.pop(agent_id, None)
        self._rate_window.pop(agent_id, None)
        # Remove any agents waiting on this one
        to_remove = [
            k for k, v in self._waiting_on.items() if v == agent_id
        ]
        for k in to_remove:
            self._waiting_on.pop(k, None)
        logger.info("Agent unregistered: %s", agent_id[:8])

    async def send(self, message: RoutedMessage) -> None:
        """Send a message to the target agent's queue.

        Raises MessageRoutingError if target is not registered or
        queue is full.
        """
        queue = self._queues.get(message.target_agent_id)
        if queue is None:
            raise MessageRoutingError(
                message.message_id,
                message.target_agent_id,
                "target agent not registered",
            )
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            self.metrics.messages_dropped_send += 1
            logger.error(
                "Send failed — queue full for agent %s "
                "(msg=%s, type=%s, queue_size=%d)",
                message.target_agent_id[:8],
                message.message_id[:8],
                message.message_type.value,
                queue.qsize(),
            )
            raise MessageRoutingError(
                message.message_id,
                message.target_agent_id,
                "target agent message queue full",
            )
        self.metrics.messages_sent += 1
        logger.debug(
            "Message sent: %s -> %s (type=%s)",
            message.source_agent_id[:8],
            message.target_agent_id[:8],
            message.message_type.value,
        )
        await fire_event(self._event_callback, {
            "event": "message_routed",
            "source_agent_id": message.source_agent_id,
            "target_agent_id": message.target_agent_id,
            "message_type": message.message_type.value,
        })

    def subscribe(
        self,
        agent_id: str,
        topic_pattern: str,
        scope_filter: str | None = None,
        urgency_threshold: str = "low",
    ) -> None:
        """Subscribe an agent to topic-based messages."""
        if agent_id not in self._queues:
            raise MessageRoutingError("N/A", agent_id, "subscriber not registered")
        subscriptions = self._subscriptions.setdefault(agent_id, [])
        for sub in subscriptions:
            if (
                sub.topic_pattern == topic_pattern
                and sub.scope_filter == scope_filter
                and sub.urgency_threshold == urgency_threshold
            ):
                return
        subscriptions.append(
            TopicSubscription(
                subscriber_id=agent_id,
                topic_pattern=topic_pattern,
                scope_filter=scope_filter,
                urgency_threshold=urgency_threshold,
            )
        )
        self.metrics.subscriptions_created += 1

    def unsubscribe(self, agent_id: str, topic_pattern: str | None = None) -> None:
        """Remove subscriptions. None removes all for the agent."""
        subscriptions = self._subscriptions.get(agent_id)
        if not subscriptions:
            return
        if topic_pattern is None:
            removed = len(subscriptions)
            self._subscriptions.pop(agent_id, None)
            self.metrics.subscriptions_removed += removed
            return
        kept = [sub for sub in subscriptions if sub.topic_pattern != topic_pattern]
        self.metrics.subscriptions_removed += len(subscriptions) - len(kept)
        if kept:
            self._subscriptions[agent_id] = kept
        else:
            self._subscriptions.pop(agent_id, None)

    async def publish(self, message: RoutedMessage) -> list[TriageDecision]:
        """Publish a topic-based message to subscribers after triage."""
        if not message.topic:
            raise ValueError("publish() requires RoutedMessage.topic")
        self.metrics.topic_messages_published += 1
        self._prune_dedup_cache()
        decisions: list[TriageDecision] = []
        matched_subscriptions = self._match_subscribers(message)
        for sub in matched_subscriptions:
            decision = self._evaluate_triage(message, sub)
            decision.subscriber_id = sub.subscriber_id
            decision.message_id = message.message_id
            decisions.append(decision)
            self._record_metric(
                "triage_decision",
                1.0,
                tags={
                    "decision": decision.decision,
                    "reason_code": decision.reason_code,
                    "subscriber_id": sub.subscriber_id,
                    "topic": message.topic or "",
                    "scope": message.scope or "",
                    "urgency": message.urgency,
                },
            )
            if decision.decision == "deliver_now":
                self.metrics.triage_deliver_now += 1
                routed = replace(
                    message,
                    message_type=MessageType.TOPIC_EVENT,
                    target_agent_id=sub.subscriber_id,
                )
                await self.send(routed)
                await fire_event(self._event_callback, {
                    "event": "topic_event",
                    "source_agent_id": message.source_agent_id,
                    "target_agent_id": sub.subscriber_id,
                    "topic": message.topic,
                    "scope": message.scope,
                    "urgency": message.urgency,
                    "message_id": message.message_id,
                    "policy_snapshot_id": decision.policy_snapshot_id,
                })
            elif decision.decision == "deliver_digest":
                self.metrics.triage_deliver_digest += 1
                self._digest_buffer.setdefault(sub.subscriber_id, []).append(message)
            else:
                self.metrics.triage_drop += 1
            await self._run_shadow_triage_compare(
                message=message,
                subscription=sub,
                rules_decision=decision,
            )
        self._triage_log.extend(decisions)
        return decisions

    async def flush_digests(self) -> int:
        """Flush buffered digest messages to subscribers and return delivery count."""
        sent = 0
        for subscriber_id, messages in list(self._digest_buffer.items()):
            if not messages:
                continue
            digest_payload = {
                "count": len(messages),
                "messages": [self._message_summary(msg) for msg in messages],
            }
            digest_message = RoutedMessage(
                message_type=MessageType.DIGEST,
                source_agent_id="router",
                target_agent_id=subscriber_id,
                payload=digest_payload,
                topic="digest.batch",
                scope="global",
                urgency="low",
            )
            await self.send(digest_message)
            await fire_event(self._event_callback, {
                "event": "digest_event",
                "target_agent_id": subscriber_id,
                "message_count": len(messages),
            })
            sent += 1
            self._digest_buffer[subscriber_id] = []
        if sent:
            self.metrics.digests_flushed += sent
        return sent

    def get_triage_log(self, limit: int = 100) -> list[TriageDecision]:
        """Query triage decisions (most recent first)."""
        if limit <= 0:
            return []
        return list(self._triage_log[-limit:])

    def _match_subscribers(self, message: RoutedMessage) -> list[TopicSubscription]:
        matches: list[TopicSubscription] = []
        for subscriber_id, subscriptions in self._subscriptions.items():
            if subscriber_id not in self._queues:
                continue
            for sub in subscriptions:
                if self._subscription_matches(sub, message):
                    matches.append(sub)
                    break
        return matches

    def _subscription_matches(
        self,
        subscription: TopicSubscription,
        message: RoutedMessage,
    ) -> bool:
        if not message.topic or not fnmatch.fnmatch(message.topic, subscription.topic_pattern):
            return False
        if subscription.scope_filter and message.scope and subscription.scope_filter != message.scope:
            return False
        return _URGENCY_RANK.get(message.urgency, 1) >= _URGENCY_RANK.get(
            subscription.urgency_threshold, 0
        )

    def _evaluate_triage(
        self,
        message: RoutedMessage,
        subscription: TopicSubscription,
    ) -> TriageDecision:
        if self._is_ttl_expired(message):
            return TriageDecision(
                decision="drop",
                reason_code="expired_ttl",
                relevance_score=0.0,
                policy_snapshot_id="ttl",
            )
        signature = self._message_signature(message)
        now = _utc_now()
        seen_at = self._dedup_cache.get(signature)
        if seen_at and (now - seen_at).total_seconds() < self._dedup_window_seconds:
            return TriageDecision(
                decision="drop",
                reason_code="dedup_window",
                relevance_score=0.0,
                policy_snapshot_id="dedup",
            )
        self._dedup_cache[signature] = now

        if self._is_rate_limited(subscription.subscriber_id):
            return TriageDecision(
                decision="deliver_digest",
                reason_code="rate_limited",
                relevance_score=0.5,
                policy_snapshot_id="rate_limit",
            )

        for index, rule in enumerate(self._triage_rules):
            if not self._rule_matches(rule, message):
                continue
            action = str(rule.get("action", "deliver_now"))
            if action not in {"deliver_now", "deliver_digest", "drop"}:
                action = "deliver_now"
            return TriageDecision(
                decision=action,
                reason_code=f"rule_{index}",
                relevance_score=self._relevance_score(message),
                policy_snapshot_id=f"rule:{index}",
            )

        if _URGENCY_RANK.get(message.urgency, 1) <= _URGENCY_RANK["normal"]:
            return TriageDecision(
                decision="deliver_digest",
                reason_code="default_digest",
                relevance_score=self._relevance_score(message),
                policy_snapshot_id="default",
            )
        return TriageDecision(
            decision="deliver_now",
            reason_code="default_deliver",
            relevance_score=self._relevance_score(message),
            policy_snapshot_id="default",
        )

    def _rule_matches(self, rule: dict, message: RoutedMessage) -> bool:
        topic_pattern = str(rule.get("topic_pattern", "*"))
        if not fnmatch.fnmatch(message.topic or "", topic_pattern):
            return False
        msg_urgency = _URGENCY_RANK.get(message.urgency, 1)
        min_urgency = _URGENCY_RANK.get(str(rule.get("urgency_min", "low")), 0)
        max_urgency = _URGENCY_RANK.get(str(rule.get("urgency_max", "high")), 2)
        if msg_urgency < min_urgency or msg_urgency > max_urgency:
            return False
        return True

    def _is_ttl_expired(self, message: RoutedMessage) -> bool:
        if message.ttl is None:
            return False
        expiry = message.timestamp + timedelta(seconds=float(message.ttl))
        return _utc_now() > expiry

    def _message_signature(self, message: RoutedMessage) -> str:
        payload = message.payload
        try:
            payload_text = json.dumps(payload, sort_keys=True, default=str)
        except TypeError:
            payload_text = str(payload)
        base = f"{message.topic}|{message.scope}|{payload_text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _is_rate_limited(self, subscriber_id: str) -> bool:
        timestamps = self._rate_window.setdefault(subscriber_id, [])
        now = _utc_now()
        cutoff = now - timedelta(minutes=1)
        fresh = [ts for ts in timestamps if ts >= cutoff]
        self._rate_window[subscriber_id] = fresh
        if len(fresh) >= self._rate_limit_per_minute:
            return True
        fresh.append(now)
        return False

    def _prune_dedup_cache(self) -> None:
        cutoff = _utc_now() - timedelta(seconds=self._dedup_window_seconds)
        self._dedup_cache = {
            key: ts for key, ts in self._dedup_cache.items() if ts >= cutoff
        }

    def _relevance_score(self, message: RoutedMessage) -> float:
        urgency = _URGENCY_RANK.get(message.urgency, 1)
        return min(1.0, 0.4 + (0.3 * urgency))

    def _message_summary(self, message: RoutedMessage) -> dict[str, object]:
        return {
            "message_id": message.message_id,
            "source_agent_id": message.source_agent_id,
            "topic": message.topic,
            "scope": message.scope,
            "urgency": message.urgency,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
        }

    def _record_metric(
        self,
        metric_type: str,
        value: float,
        tags: dict[str, object] | None = None,
    ) -> None:
        collector = self._telemetry_collector
        if collector is None:
            return
        record_metric = getattr(collector, "record_metric", None)
        if record_metric is None:
            return
        try:
            record_metric(metric_type, value, tags)
        except Exception:
            logger.debug("Telemetry metric recording failed", exc_info=True)

    async def _run_shadow_triage_compare(
        self,
        message: RoutedMessage,
        subscription: TopicSubscription,
        rules_decision: TriageDecision,
    ) -> None:
        shadow = self._shadow_triage_model
        if shadow is None:
            return
        shadow_eval = getattr(shadow, "shadow_evaluate", None)
        shadow_compare = getattr(shadow, "compare_with_rules", None)
        if shadow_eval is None or shadow_compare is None:
            return

        try:
            model_decision = await shadow_eval(
                message,
                {
                    "subscriber_id": subscription.subscriber_id,
                    "topic_pattern": subscription.topic_pattern,
                    "scope_filter": subscription.scope_filter,
                    "urgency_threshold": subscription.urgency_threshold,
                },
            )
            comparison = shadow_compare(rules_decision, model_decision) or {}
        except Exception:
            logger.debug("Shadow triage evaluation failed", exc_info=True)
            return

        self._record_metric("triage_comparison", 1.0)
        if comparison.get("false_positive"):
            self._record_metric("triage_false_positive", 1.0)
        if comparison.get("false_negative"):
            self._record_metric("triage_false_negative", 1.0)

    async def receive(
        self,
        agent_id: str,
        timeout: float = 120.0,
        message_type_filter: MessageType | None = None,
        correlation_id: str | None = None,
    ) -> RoutedMessage:
        """Receive next message for an agent. Blocks until available.

        Supports filtering by message_type and/or correlation_id.
        Non-matching messages are buffered (up to ``receive_buffer_limit``)
        and requeued when the call completes.

        Raises asyncio.TimeoutError if timeout exceeded or if the
            requeue buffer is full (to prevent unbounded memory growth).
        Raises MessageRoutingError if agent not registered.
        """
        queue = self._queues.get(agent_id)
        if queue is None:
            raise MessageRoutingError(
                "N/A", agent_id, "receiving agent not registered"
            )

        timeout_disabled = timeout <= 0

        # No filters → fast path, no buffering needed
        if message_type_filter is None and correlation_id is None:
            try:
                if timeout_disabled:
                    msg = await queue.get()
                else:
                    msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self.metrics.receive_timeouts += 1
                raise
            self.metrics.messages_received += 1
            return msg

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout if not timeout_disabled else None
        requeue_buffer: list[RoutedMessage] = []

        try:
            while True:
                if timeout_disabled:
                    remaining = None
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                # Guard: bounded buffer to prevent memory exhaustion
                if len(requeue_buffer) >= self._receive_buffer_limit:
                    self.metrics.receive_buffer_overflows += 1
                    logger.error(
                        "Receive buffer overflow for agent %s: "
                        "buffered %d non-matching messages "
                        "(limit=%d, filter_type=%s, correlation=%s). "
                        "Returning TimeoutError to caller.",
                        agent_id[:8],
                        len(requeue_buffer),
                        self._receive_buffer_limit,
                        message_type_filter.value
                        if message_type_filter
                        else "none",
                        (correlation_id or "none")[:8],
                    )
                    raise asyncio.TimeoutError(
                        f"receive buffer limit ({self._receive_buffer_limit}) "
                        f"exceeded for agent {agent_id[:8]}"
                    )

                if timeout_disabled:
                    msg = await queue.get()
                else:
                    msg = await asyncio.wait_for(
                        queue.get(), timeout=remaining
                    )

                # Apply filters
                if (
                    message_type_filter is not None
                    and msg.message_type != message_type_filter
                ):
                    requeue_buffer.append(msg)
                    continue
                if (
                    correlation_id is not None
                    and msg.correlation_id != correlation_id
                ):
                    requeue_buffer.append(msg)
                    continue

                self.metrics.messages_received += 1
                return msg
        except asyncio.TimeoutError:
            # Single increment point for all timeout paths
            # (deadline expired, wait_for timeout, buffer overflow)
            self.metrics.receive_timeouts += 1
            raise
        finally:
            # Requeue non-matching messages (oldest first to preserve order)
            dropped = 0
            for buffered in requeue_buffer:
                try:
                    queue.put_nowait(buffered)
                    self.metrics.messages_requeued += 1
                except asyncio.QueueFull:
                    dropped += 1
                    self.metrics.messages_dropped_requeue += 1
                    logger.warning(
                        "Requeue dropped for agent %s: msg=%s type=%s "
                        "corr=%s (queue full)",
                        agent_id[:8],
                        buffered.message_id[:8],
                        buffered.message_type.value,
                        (buffered.correlation_id or "none")[:8],
                    )
            if dropped:
                logger.error(
                    "Dropped %d/%d buffered messages for agent %s "
                    "during requeue (queue full, queue_size=%d)",
                    dropped,
                    len(requeue_buffer),
                    agent_id[:8],
                    queue.qsize(),
                )

    def mark_waiting(self, waiter_id: str, target_id: str) -> None:
        """Record that waiter_id is blocked waiting on target_id."""
        self._waiting_on[waiter_id] = target_id

    def clear_waiting(self, waiter_id: str) -> None:
        """Clear the waiting record for waiter_id."""
        self._waiting_on.pop(waiter_id, None)

    def get_wait_graph(self) -> dict[str, str]:
        """Return a copy of the wait graph for deadlock detection."""
        return dict(self._waiting_on)

    @property
    def registered_agent_count(self) -> int:
        return len(self._queues)
