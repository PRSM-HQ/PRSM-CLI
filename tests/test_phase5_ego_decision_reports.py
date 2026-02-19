from __future__ import annotations

from datetime import timezone

import pytest

from prsm.engine.conversation_store import ConversationEntry, ConversationStore, EntryType
from prsm.engine.mcp_server.tools import OrchestrationTools
from prsm.engine.models import (
    AgentDescriptor,
    EgoDecisionReport,
    ExpertOutput,
    VerificationResult,
)
from prsm.engine.rationale_extractor import (
    extract_change_rationale,
    extract_structured_evidence,
)
from prsm.engine.verification import VerificationPipeline


class _FakeManager:
    def __init__(self, descriptor: AgentDescriptor) -> None:
        self._descriptor = descriptor

    def get_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        if agent_id == self._descriptor.agent_id:
            return self._descriptor
        return None


class _FakeRouter:
    async def send(self, message) -> None:
        return None


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict]] = []

    async def store_text(self, content: str, metadata: dict) -> str:
        self.saved.append((content, metadata))
        return "artifact-1"


def test_phase5_models_defaults() -> None:
    first = ExpertOutput(agent_id="a1", summary="done")
    second = ExpertOutput(agent_id="a2", summary="done")
    first.steps.append("x")
    assert second.steps == []

    vr = VerificationResult(check_type="unit_test")
    assert vr.passed is False
    assert vr.command == ""

    report = EgoDecisionReport()
    assert len(report.decision_id) == 12
    assert report.created_at.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_verification_pipeline_runs_checks_and_stores_artifacts() -> None:
    store = _FakeArtifactStore()
    pipeline = VerificationPipeline(artifact_store=store)
    hooks = [
        {"type": "unit_test", "command": "python -c \"print('ok')\""},
        {"type": "lint", "command": "python -c \"import sys; sys.exit(1)\""},
    ]
    results = await pipeline.run_checks(hooks, cwd=".")

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    assert all(result.duration_seconds >= 0 for result in results)
    assert all(result.artifact_id == "artifact-1" for result in results)
    assert len(store.saved) == 2


@pytest.mark.asyncio
async def test_task_complete_structured_fields_are_preserved() -> None:
    descriptor = AgentDescriptor(agent_id="agent-1", prompt="do work")
    tools = OrchestrationTools(
        agent_id=descriptor.agent_id,
        manager=_FakeManager(descriptor),  # type: ignore[arg-type]
        router=_FakeRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    await tools.task_complete(summary="done")
    assert descriptor.result_summary == "done"
    assert descriptor.result_artifacts == {}

    await tools.task_complete(
        summary="done with evidence",
        artifacts={"base": True},
        steps=["edit", "test"],
        assumptions=["env is stable"],
        risks=["edge case"],
        rollback_plan="revert patch",
        confidence=0.8,
        verification_results=[{"check_type": "unit_test", "passed": True}],
    )
    assert descriptor.result_artifacts["base"] is True
    assert descriptor.result_artifacts["steps"] == ["edit", "test"]
    assert descriptor.result_artifacts["assumptions"] == ["env is stable"]
    assert descriptor.result_artifacts["risks"] == ["edge case"]
    assert descriptor.result_artifacts["rollback_plan"] == "revert patch"
    assert descriptor.result_artifacts["confidence"] == 0.8
    assert descriptor.result_artifacts["verification_results"][0]["passed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["", "   ", "\n\t  "])
async def test_task_complete_requires_non_empty_summary(summary: str) -> None:
    descriptor = AgentDescriptor(agent_id="agent-1", prompt="do work")
    tools = OrchestrationTools(
        agent_id=descriptor.agent_id,
        manager=_FakeManager(descriptor),  # type: ignore[arg-type]
        router=_FakeRouter(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
    )

    result = await tools.task_complete(summary=summary)

    assert result.get("is_error") is True
    content = result.get("content", [])
    assert content and "non-empty summary" in str(content[0].get("text", ""))
    assert descriptor.result_summary is None
    assert descriptor.result_artifacts == {}


def test_conversation_store_decision_reports_round_trip() -> None:
    store = ConversationStore()
    store.append_decision_report(
        "agent-1",
        {"decision": "accept", "policy_snapshot_id": "snap-1", "artifact_ids": ["a1"]},
        linked_tool_id="tool-1",
    )
    reports = store.get_decision_reports("agent-1")
    assert len(reports) == 1
    assert reports[0]["decision"] == "accept"
    assert reports[0]["policy_snapshot_id"] == "snap-1"
    assert reports[0]["linked_tool_id"] == "tool-1"


def test_rationale_extractor_structured_evidence_and_legacy_behavior() -> None:
    store = ConversationStore()
    store.append(
        "agent-1",
        ConversationEntry(
            entry_type=EntryType.THINKING,
            content="Need to fix reliability so tests stop failing for users.",
        ),
    )
    store.append(
        "agent-1",
        ConversationEntry(
            entry_type=EntryType.TOOL_CALL,
            tool_name="task_complete",
            tool_id="tool-1",
            tool_args=(
                '{"steps":["patch"],"assumptions":["none"],"risks":["minor"],'
                '"verification_results":[{"check_type":"unit_test","passed":true}]}'
            ),
        ),
    )

    evidence = extract_structured_evidence("agent-1", "tool-1", store)
    assert evidence["steps"] == ["patch"]
    assert evidence["assumptions"] == ["none"]
    assert evidence["risks"] == ["minor"]

    rationale = extract_change_rationale("agent-1", "tool-1", store)
    assert "fix reliability" in rationale.lower()
