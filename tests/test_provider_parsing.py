"""Unit tests for provider parsing and master command builders."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from prsm.engine.agent_session import AgentSession
from prsm.engine.models import AgentDescriptor, AgentState
from prsm.engine.providers.base import ProviderMessage
from prsm.engine.providers.gemini_provider import GeminiProvider
from prsm.engine.providers.minimax_provider import MiniMaxProvider


class _DummyRegistry:
    def list_ids(self) -> list[str]:
        return []


class _DummyProvider:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_master(self) -> bool:
        return True

    def build_master_cmd(
        self,
        prompt: str,
        bridge_port: int,
        *,
        system_prompt: str | None = None,
        model_id: str | None = None,
        cwd: str | None = None,
        plugin_mcp_servers: dict | None = None,
    ) -> tuple[list[str], dict[str, str] | None, str | None]:
        return ["fake-cli", prompt, str(bridge_port)], None, None


class _StreamingProvider:
    def __init__(self, name: str, messages: list[ProviderMessage]) -> None:
        self._name = name
        self._messages = messages

    @property
    def name(self) -> str:
        return self._name

    async def run_agent(self, **_kwargs):
        for msg in self._messages:
            yield msg


def _make_session(
    *,
    event_callback: AsyncMock | None = None,
    provider: object | None = None,
) -> AgentSession:
    manager = SimpleNamespace(
        has_active_children=lambda _: False,
        transition_agent_state=AsyncMock(),
    )
    return AgentSession(
        descriptor=AgentDescriptor(prompt="test"),
        manager=manager,  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=_DummyRegistry(),
        provider=provider,  # type: ignore[arg-type]
        event_callback=event_callback,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type,start_fields,expected_tool,expected_arg,complete_fields,expected_result",
    [
        (
            "command_execution",
            {"command": "ls -la"},
            "Bash",
            "ls -la",
            {"output": "ok"},
            "ok",
        ),
        (
            "file_edit",
            {"file_path": "notes.txt"},
            "Edit",
            "notes.txt",
            {"result": "done"},
            "done",
        ),
        (
            "file_write",
            {"path": "out.txt"},
            "Write",
            "out.txt",
            {"result": "written"},
            "written",
        ),
        (
            "file_read",
            {"file_path": "readme.md"},
            "Read",
            "readme.md",
            {"result": "contents"},
            "contents",
        ),
        (
            "mcp_tool_call",
            {"tool_name": "spawn_child", "arguments": {"prompt": "hi"}},
            "spawn_child",
            "{\"prompt\": \"hi\"}",
            {"result": {"ok": True}},
            "{'ok': True}",
        ),
        (
            "web_search",
            {"query": "search term"},
            "WebSearch",
            "search term",
            {"results": ["result"]},
            "['result']",
        ),
    ],
)
async def test_process_codex_jsonl_tool_events(
    item_type: str,
    start_fields: dict,
    expected_tool: str,
    expected_arg: str,
    complete_fields: dict,
    expected_result: str,
) -> None:
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback)
    active_tools: dict[str, str] = {}

    mock_fire_event = AsyncMock()
    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        start_line = json.dumps({
            "type": "item.started",
            "item": {"id": "item-1", "type": item_type, **start_fields},
        })
        display = await session._process_codex_jsonl(start_line, active_tools)

        assert display is None
        assert mock_fire_event.await_count == 1
        started_event = mock_fire_event.call_args_list[0].args[1]
        assert started_event["event"] == "tool_call_started"
        assert started_event["tool_name"] == expected_tool
        assert started_event["tool_id"] == "item-1"
        assert expected_arg in started_event["arguments"]

        complete_line = json.dumps({
            "type": "item.completed",
            "item": {"id": "item-1", "type": item_type, **complete_fields},
        })
        display = await session._process_codex_jsonl(complete_line, active_tools)

        assert display is None
        assert mock_fire_event.await_count == 2
        completed_event = mock_fire_event.call_args_list[1].args[1]
        assert completed_event["event"] == "tool_call_completed"
        assert completed_event["tool_id"] == "item-1"
        assert completed_event["result"] == expected_result
        assert completed_event["is_error"] is False
        assert active_tools == {}


@pytest.mark.asyncio
async def test_process_codex_jsonl_agent_message() -> None:
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback)
    mock_fire_event = AsyncMock()

    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        line = json.dumps({
            "type": "item.completed",
            "item": {"id": "msg-1", "type": "agent_message", "text": "Hello"},
        })
        display = await session._process_codex_jsonl(line, {})

    assert display == "Hello\n"
    assert mock_fire_event.await_count == 0


@pytest.mark.asyncio
async def test_process_gemini_stream_json_events() -> None:
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback)
    mock_fire_event = AsyncMock()

    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        init_line = json.dumps({"type": "init", "session_id": "s1"})
        assert await session._process_gemini_stream_json(init_line) is None

        user_line = json.dumps({
            "type": "message",
            "role": "user",
            "content": "hi",
        })
        assert await session._process_gemini_stream_json(user_line) is None

        assistant_line = json.dumps({
            "type": "message",
            "role": "assistant",
            "content": "Hello there",
        })
        assert await session._process_gemini_stream_json(assistant_line) == "Hello there"

        tool_use_line = json.dumps({
            "type": "tool_use",
            "tool_name": "run_shell_command",
            "tool_id": "tool-1",
            "parameters": {"command": "ls"},
        })
        assert await session._process_gemini_stream_json(tool_use_line) is None

        tool_result_line = json.dumps({
            "type": "tool_result",
            "tool_id": "tool-1",
            "status": "success",
            "output": "done",
        })
        assert await session._process_gemini_stream_json(tool_result_line) is None

        result_line = json.dumps({
            "type": "result",
            "stats": {"total_tokens": 10},
        })
        assert await session._process_gemini_stream_json(result_line) is None

    assert mock_fire_event.await_count == 3
    started_event = mock_fire_event.call_args_list[0].args[1]
    completed_event = mock_fire_event.call_args_list[1].args[1]
    usage_event = mock_fire_event.call_args_list[2].args[1]
    assert started_event["event"] == "tool_call_started"
    assert started_event["tool_name"] == "Bash"
    assert started_event["tool_id"] == "tool-1"
    assert json.loads(started_event["arguments"]) == {"command": "ls"}
    assert completed_event["event"] == "tool_call_completed"
    assert completed_event["tool_id"] == "tool-1"
    assert completed_event["result"] == "done"
    assert completed_event["is_error"] is False
    assert usage_event["event"] == "context_window_usage"


@pytest.mark.asyncio
async def test_process_codex_jsonl_spawn_children_parallel_keeps_large_args() -> None:
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback)
    active_tools: dict[str, str] = {}
    mock_fire_event = AsyncMock()

    long_children = [{"prompt": f"Task {i}: " + ("x" * 120)} for i in range(40)]
    assert len(json.dumps({"children": long_children})) > 2000

    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        start_line = json.dumps({
            "type": "item.started",
            "item": {
                "id": "item-long",
                "type": "mcp_tool_call",
                "tool_name": "spawn_children_parallel",
                "arguments": {"children": long_children},
            },
        })
        display = await session._process_codex_jsonl(start_line, active_tools)

    assert display is None
    started_event = mock_fire_event.call_args_list[0].args[1]
    assert started_event["event"] == "tool_call_started"
    assert started_event["tool_name"] == "spawn_children_parallel"
    # Should not be truncated down to the old 2k cap.
    assert len(started_event["arguments"]) > 2000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,expected",
    [
        ("read_file", "Read"),
        ("write_file", "Write"),
        ("edit_file", "Edit"),
        ("mcp__orchestrator__read_file", "Read"),
        ("mcp__orchestrator__write_file", "Write"),
        ("mcp__orchestrator__edit_file", "Edit"),
    ],
)
async def test_process_codex_jsonl_normalizes_mcp_file_tool_names(
    tool_name: str,
    expected: str,
) -> None:
    session = _make_session(event_callback=AsyncMock())
    active_tools: dict[str, str] = {}
    mock_fire_event = AsyncMock()

    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        start_line = json.dumps({
            "type": "item.started",
            "item": {
                "id": "item-file",
                "type": "mcp_tool_call",
                "tool_name": tool_name,
                "arguments": {"file_path": "a.txt"},
            },
        })
        display = await session._process_codex_jsonl(start_line, active_tools)

    assert display is None
    started_event = mock_fire_event.call_args_list[0].args[1]
    assert started_event["event"] == "tool_call_started"
    assert started_event["tool_name"] == expected


def test_minimax_build_master_cmd_includes_json_and_provider() -> None:
    provider = MiniMaxProvider(command="codex")
    cmd, _, _ = provider.build_master_cmd("hello", bridge_port=1234)

    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert "model_provider=minimax" in cmd


def test_sanitize_provider_stderr_filters_codex_rollout_noise() -> None:
    raw = (
        "[error] Codex ran out of room in the model's context window.\n"
        "2026-02-18T03:51:20Z ERROR codex_core::rollout::list: "
        "state db missing rollout path for thread abc\n"
    )

    cleaned = AgentSession._sanitize_provider_stderr("codex", raw)

    assert "state db missing rollout path for thread" not in cleaned.lower()
    assert "Codex ran out of room in the model's context window." in cleaned


def test_gemini_build_master_cmd_includes_stream_json(tmp_path) -> None:
    provider = GeminiProvider(command="gemini")
    cwd = str(tmp_path)

    cmd, env, _ = provider.build_master_cmd("hello", bridge_port=1234, cwd=cwd)

    assert "--yolo" in cmd
    # --output-format and --prompt use =value syntax to avoid yargs
    # positional/flag conflict
    assert any(arg.startswith("--output-format=") for arg in cmd)
    assert any("stream-json" in arg for arg in cmd)
    assert any(arg.startswith("--prompt=") for arg in cmd)

    # Verify .gemini/settings.json was created with orchestrator MCP config
    import os
    settings_path = os.path.join(cwd, ".gemini", "settings.json")
    assert os.path.exists(settings_path)
    with open(settings_path) as f:
        settings = json.load(f)
    assert "orchestrator" in settings.get("mcpServers", {})
    orch_config = settings["mcpServers"]["orchestrator"]
    assert "--port" in orch_config["args"]
    assert "1234" in orch_config["args"]

    # Test cleanup restores state
    provider.cleanup_master_settings()
    with open(settings_path) as f:
        settings_after = json.load(f)
    assert "orchestrator" not in settings_after.get("mcpServers", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["codex", "minimax"])
async def test_agent_session_detects_codex_providers(provider_name: str) -> None:
    event_callback = AsyncMock()
    provider = _DummyProvider(provider_name)
    session = _make_session(event_callback=event_callback, provider=provider)
    session._descriptor.state = AgentState.STARTING
    session._start_time = time.monotonic()

    mock_fire_event = AsyncMock()
    mock_codex = AsyncMock(return_value=None)
    mock_gemini = AsyncMock(return_value=None)

    import asyncio as _aio

    # Mock stdout with readuntil that returns a line then raises
    # IncompleteReadError (EOF) like a real StreamReader.
    stdout = AsyncMock()
    stdout.readuntil = AsyncMock(side_effect=[
        b"line\n",
        _aio.IncompleteReadError(b"", None),
    ])
    proc = AsyncMock()
    proc.stdout = stdout
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock()
    proc.returncode = 0

    import asyncio as _asyncio
    bridge = SimpleNamespace(
        start=AsyncMock(return_value=1234),
        stop=AsyncMock(),
        task_result=None,
        task_completed=_asyncio.Event(),  # never set → proc ends via EOF
    )

    with patch("prsm.engine.agent_session.fire_event", mock_fire_event), \
        patch("prsm.engine.mcp_server.server.build_agent_mcp_config",
              return_value=(None, object())), \
        patch("prsm.engine.mcp_server.orch_bridge.OrchBridge",
              return_value=bridge), \
        patch("asyncio.create_subprocess_exec", return_value=proc), \
        patch.object(session, "_process_codex_jsonl", mock_codex), \
        patch.object(session, "_process_gemini_stream_json", mock_gemini):
        await session._run_with_provider_mcp()

    assert mock_codex.await_count == 1
    assert mock_gemini.await_count == 0


@pytest.mark.asyncio
async def test_run_with_provider_codex_streams_child_history_events() -> None:
    codex_lines = [
        json.dumps({
            "type": "item.started",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "ls -la",
            },
        }),
        json.dumps({
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "output": "ok",
            },
        }),
        json.dumps({
            "type": "item.completed",
            "item": {
                "id": "msg-1",
                "type": "agent_message",
                "text": "Child done",
            },
        }),
    ]
    provider = _StreamingProvider(
        "codex",
        [ProviderMessage(text=line) for line in codex_lines]
        + [ProviderMessage(text="", is_result=True)],
    )
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback, provider=provider)
    session._descriptor.state = AgentState.STARTING
    session._start_time = time.monotonic()

    mock_fire_event = AsyncMock()
    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        result = await session._run_with_provider()

    events = [call.args[1] for call in mock_fire_event.call_args_list]
    event_types = [evt.get("event") for evt in events]
    assert "tool_call_started" in event_types
    assert "tool_call_completed" in event_types
    assert any(
        evt.get("event") == "stream_chunk" and "Child done" in str(evt.get("text", ""))
        for evt in events
    )
    assert result.success is True
    assert "Child done" in result.summary


@pytest.mark.asyncio
async def test_run_with_provider_gemini_streams_child_history_events() -> None:
    gemini_lines = [
        json.dumps({
            "type": "tool_use",
            "tool_name": "run_shell_command",
            "tool_id": "tool-1",
            "parameters": {"command": "pwd"},
        }),
        json.dumps({
            "type": "tool_result",
            "tool_id": "tool-1",
            "status": "success",
            "output": "/tmp",
        }),
        json.dumps({
            "type": "message",
            "role": "assistant",
            "content": "Done",
        }),
        json.dumps({
            "type": "result",
            "stats": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }),
    ]
    provider = _StreamingProvider(
        "gemini",
        [ProviderMessage(text=line) for line in gemini_lines]
        + [ProviderMessage(text="", is_result=True)],
    )
    event_callback = AsyncMock()
    session = _make_session(event_callback=event_callback, provider=provider)
    session._descriptor.state = AgentState.STARTING
    session._start_time = time.monotonic()

    mock_fire_event = AsyncMock()
    with patch("prsm.engine.agent_session.fire_event", mock_fire_event):
        result = await session._run_with_provider()

    events = [call.args[1] for call in mock_fire_event.call_args_list]
    event_types = [evt.get("event") for evt in events]
    assert "tool_call_started" in event_types
    assert "tool_call_completed" in event_types
    assert "context_window_usage" in event_types
    assert any(
        evt.get("event") == "stream_chunk" and "Done" in str(evt.get("text", ""))
        for evt in events
    )
    assert result.success is True
    assert "Done" in result.summary


# ── Bracket-block sanitizer tests ──


def test_sanitize_strips_bracket_tool_call_blocks() -> None:
    """_sanitize_agent_text strips [Tool call: ...] blocks echoed from history context."""
    text = (
        "Here is my plan:\n"
        "[Tool call: TodoWrite]\n"
        "[Tool args]\n"
        '{"todos": [{"content": "Do stuff", "status": "pending"}]}\n'
        "[Tool result: TodoWrite (success)]\n"
        "And now I'll continue.\n"
    )
    result = AgentSession._sanitize_agent_text(text)
    assert "[Tool call:" not in result
    assert "[Tool args]" not in result
    assert "[Tool result:" not in result
    assert "Here is my plan:" in result
    assert "And now I'll continue." in result


def test_sanitize_strips_partial_bracket_tool_block() -> None:
    """Even partial bracket blocks (e.g. just [Tool call: X]) should be stripped."""
    text = "Some text [Tool call: Read]\n more text"
    result = AgentSession._sanitize_agent_text(text)
    assert "[Tool call:" not in result
    assert "Some text" in result
    assert "more text" in result


def test_sanitize_preserves_normal_brackets() -> None:
    """Normal bracket usage (not tool-call markers) should be preserved."""
    text = "The array [1, 2, 3] is valid.\n"
    result = AgentSession._sanitize_agent_text(text)
    assert "[1, 2, 3]" in result
