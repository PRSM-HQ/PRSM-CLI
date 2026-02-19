from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.engine.mcp_server.tools import OrchestrationTools


@pytest.fixture
def tools(tmp_path: Path) -> OrchestrationTools:
    return OrchestrationTools(
        agent_id="test_agent",
        manager=MagicMock(),
        router=MagicMock(),
        expert_registry=MagicMock(),
        event_callback=AsyncMock(),
        permission_callback=AsyncMock(),
        cwd=str(tmp_path),
    )


def test_extract_bash_output_targets_detects_redirect_and_tee(tools: OrchestrationTools) -> None:
    command = "rg foo src > .agent_session_inject.txt | tee .event_hits.txt"
    targets = tools._extract_bash_output_targets(command)
    assert ".agent_session_inject.txt" in targets
    assert ".event_hits.txt" in targets


def test_find_forbidden_redirect_targets_ignores_scratch_dir(tools: OrchestrationTools, tmp_path: Path) -> None:
    work_dir = str(tmp_path)
    scratch_dir = tools._scratch_dir_for_work_dir(work_dir)
    allowed = tools._find_forbidden_redirect_targets(
        f"echo ok > {scratch_dir / '.agent_session_inject.txt'}",
        work_dir,
        scratch_dir,
    )
    blocked = tools._find_forbidden_redirect_targets(
        "echo ok > .agent_session_inject.txt",
        work_dir,
        scratch_dir,
    )

    assert allowed == []
    assert blocked


def test_run_bash_blocks_workspace_scratch_redirect_before_exec(tools: OrchestrationTools) -> None:
    result = asyncio.run(
        tools.run_bash(
            "echo hi > .file_index.txt",
            tool_call_id="tool_redirect_block",
        )
    )
    assert result.get("is_error") is True
    text = result["content"][0]["text"]
    assert "Blocked bash command" in text
    assert "PRSM_SCRATCH_DIR" in text
