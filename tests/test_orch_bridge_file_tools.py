from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prsm.engine.mcp_server.orch_bridge import OrchBridge
from prsm.engine.models import AgentDescriptor


def _make_bridge(tmp_path):
    descriptor = AgentDescriptor(agent_id="master-1", prompt="test", cwd=str(tmp_path))
    manager = SimpleNamespace(
        get_descriptor=lambda _agent_id: descriptor,
        get_all_descriptors=lambda: [],
    )
    orch = SimpleNamespace(
        run_bash=AsyncMock(return_value={"content": [{"type": "text", "text": "ok"}]}),
        task_complete=AsyncMock(return_value={"content": [{"type": "text", "text": "ok"}]}),
    )
    registry = SimpleNamespace(list_profiles=lambda: [])
    bridge = OrchBridge(
        orch_tools=orch,  # type: ignore[arg-type]
        manager=manager,  # type: ignore[arg-type]
        expert_registry=registry,  # type: ignore[arg-type]
        agent_id=descriptor.agent_id,
    )
    return bridge, orch


@pytest.mark.asyncio
async def test_orch_bridge_file_tool_aliases(tmp_path) -> None:
    bridge, _ = _make_bridge(tmp_path)

    write_result = await bridge._dispatch("write_file", {
        "file_path": "notes.txt",
        "content": "hello world",
    })
    assert not write_result.get("is_error")
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello world"

    edit_result = await bridge._dispatch("edit_file", {
        "file_path": "notes.txt",
        "old_string": "world",
        "new_string": "there",
    })
    assert not edit_result.get("is_error")
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello there"

    read_result = await bridge._dispatch("read_file", {
        "file_path": "notes.txt",
    })
    assert not read_result.get("is_error")
    assert read_result["content"][0]["text"] == "hello there"


@pytest.mark.asyncio
async def test_orch_bridge_edit_ambiguity_requires_replace_all(tmp_path) -> None:
    bridge, _ = _make_bridge(tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("x + x", encoding="utf-8")

    ambiguous = await bridge._dispatch("Edit", {
        "file_path": "a.txt",
        "old_string": "x",
        "new_string": "y",
    })
    assert ambiguous.get("is_error") is True
    assert "ambiguous" in ambiguous["content"][0]["text"]
    assert path.read_text(encoding="utf-8") == "x + x"

    replaced_all = await bridge._dispatch("Edit", {
        "file_path": "a.txt",
        "old_string": "x",
        "new_string": "y",
        "replace_all": True,
    })
    assert not replaced_all.get("is_error")
    assert path.read_text(encoding="utf-8") == "y + y"


@pytest.mark.asyncio
async def test_orch_bridge_bash_alias_maps_to_run_bash(tmp_path) -> None:
    bridge, orch = _make_bridge(tmp_path)

    await bridge._dispatch("Bash", {"command": "echo hi"})

    orch.run_bash.assert_awaited_once_with(
        command="echo hi",
        timeout=None,
        cwd=None,
        tool_call_id=None,
    )


def test_orch_bridge_normalizes_run_bash_to_canonical_bash(tmp_path) -> None:
    bridge, _ = _make_bridge(tmp_path)
    assert bridge._normalize_method_name("run_bash") == "Bash"


@pytest.mark.asyncio
async def test_orch_bridge_task_complete_sets_bridge_result_on_success(tmp_path) -> None:
    bridge, orch = _make_bridge(tmp_path)
    orch.task_complete.return_value = {"content": [{"type": "text", "text": "ok"}]}

    result = await bridge._dispatch("task_complete", {"summary": "  done  "})

    assert result.get("is_error") is not True
    assert bridge.task_completed.is_set() is True
    assert bridge.task_result == "done"


@pytest.mark.asyncio
async def test_orch_bridge_task_complete_error_does_not_end_session(tmp_path) -> None:
    bridge, orch = _make_bridge(tmp_path)
    orch.task_complete.return_value = {
        "is_error": True,
        "content": [{"type": "text", "text": "summary required"}],
    }

    result = await bridge._dispatch("task_complete", {"summary": "   "})

    assert result.get("is_error") is True
    assert bridge.task_completed.is_set() is False
    assert bridge.task_result is None
