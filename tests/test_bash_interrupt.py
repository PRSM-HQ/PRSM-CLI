import asyncio
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from prsm.engine.agent_session import AgentSession
from prsm.engine.models import AgentDescriptor, AgentRole, PermissionMode
from prsm.engine.mcp_server.tools import OrchestrationTools, ToolTimeTracker

@pytest.fixture
def mock_agent_manager():
    manager = MagicMock()
    manager.get_descriptor.return_value = AgentDescriptor(
        agent_id="test_agent",
        role=AgentRole.MASTER,
        prompt="test prompt",
        model="test_model",
        permission_mode=PermissionMode.BYPASS,
        depth=0,
    )
    return manager

@pytest.fixture
def mock_router():
    return MagicMock()

@pytest.fixture
def mock_expert_registry():
    return MagicMock()

@pytest.fixture
def mock_event_callback():
    return AsyncMock()

@pytest.fixture
def mock_permission_callback():
    return AsyncMock()

@pytest.fixture
def mock_user_question_callback():
    return AsyncMock()

@pytest.fixture
def orchestration_tools_instance(
    mock_agent_manager,
    mock_router,
    mock_expert_registry,
    mock_event_callback,
    mock_permission_callback,
    mock_user_question_callback,
):
    return OrchestrationTools(
        agent_id="test_agent",
        manager=mock_agent_manager,
        router=mock_router,
        expert_registry=mock_expert_registry,
        event_callback=mock_event_callback,
        permission_callback=mock_permission_callback,
        user_question_callback=mock_user_question_callback,
    )

@pytest.fixture
def agent_session_instance(
    mock_agent_manager,
    mock_router,
    mock_expert_registry,
    mock_event_callback,
    orchestration_tools_instance, # Pass the real instance
):
    descriptor = AgentDescriptor(
        agent_id="test_agent",
        role=AgentRole.MASTER,
        prompt="test prompt",
        model="test_model",
        permission_mode=PermissionMode.BYPASS,
        depth=0,
    )
    session = AgentSession(
        descriptor=descriptor,
        manager=mock_agent_manager,
        router=mock_router,
        expert_registry=mock_expert_registry,
        event_callback=mock_event_callback,
        orchestration_tools=orchestration_tools_instance, # Pass the real instance
    )
    # Manually assign orch_tools to simulate AgentSession.run() behavior
    session._orchestration_tools = orchestration_tools_instance
    return session


@pytest.mark.asyncio
async def test_kill_bash_subprocess_via_agent_session(
    agent_session_instance, orchestration_tools_instance,
):
    command = "sleep 5"
    tool_call_id = "test_tool_call_id_123"

    # Start the bash command as a background task
    bash_task = asyncio.create_task(
        orchestration_tools_instance.run_bash(command, tool_call_id=tool_call_id)
    )

    # Give it a moment to start the subprocess
    await asyncio.sleep(0.1)

    # Ensure the process is active
    assert tool_call_id in orchestration_tools_instance._active_bash_processes
    proc = orchestration_tools_instance._active_bash_processes[tool_call_id]
    assert proc.returncode is None # Should still be running

    # Call kill_tool_call on the agent session
    agent_session_instance.kill_tool_call(tool_call_id)

    # Wait for the bash task to complete after cancellation
    result = await bash_task

    # Assertions
    assert result.get("is_error") is True
    assert "was cancelled" in result.get("content", [{}])[0].get("text", "")

    # Ensure the subprocess was terminated
    await proc.wait() # Wait for the process to truly terminate
    assert proc.returncode is not None # Should have a return code now, indicating termination
    assert proc.returncode != 0 # Should not be a clean exit

    # Ensure the process is removed from active_bash_processes
    assert tool_call_id not in orchestration_tools_instance._active_bash_processes


@pytest.mark.asyncio
async def test_cancel_sends_sigint_to_running_bash(
    agent_session_instance,
    orchestration_tools_instance,
    tmp_path: Path,
):
    tool_call_id = "tool_call_sigint"
    signal_file = tmp_path / "signal.txt"
    command = (
        "bash -lc '"
        f"trap \"echo INT > {signal_file}; exit 130\" INT; "
        "while true; do sleep 0.1; done'"
    )

    bash_task = asyncio.create_task(
        orchestration_tools_instance.run_bash(command, tool_call_id=tool_call_id)
    )
    await asyncio.sleep(0.2)

    cancelled = agent_session_instance.kill_tool_call(tool_call_id)
    assert cancelled is True

    result = await bash_task
    assert result.get("is_error") is True
    assert "was cancelled" in result.get("content", [{}])[0].get("text", "")

    for _ in range(10):
        if signal_file.exists():
            break
        await asyncio.sleep(0.05)

    assert signal_file.exists()
    assert signal_file.read_text(encoding="utf-8").strip() == "INT"
