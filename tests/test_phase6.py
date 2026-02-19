"""Phase 6 tests — CLI flags, slash commands, session management, plugins."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.shared.commands import COMMAND_HELP, ParsedCommand, parse_command
from prsm.shared.models.agent import AgentNode, AgentRole, AgentState
from prsm.shared.models.message import MessageRole, ToolCall
from prsm.shared.models.session import Session
from prsm.shared.services.persistence import SessionPersistence
from prsm.shared.services.plugins import PluginConfig, PluginManager

# Force demo mode in all headless TUI tests
_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


# ── Command Parser Tests ──


class TestCommandParser:
    def test_parse_simple_command(self):
        cmd = parse_command("/help")
        assert cmd is not None
        assert cmd.name == "help"
        assert cmd.args == []
        assert cmd.raw == "/help"

    def test_parse_command_with_args(self):
        cmd = parse_command("/plugin add myserver npx mcp-server")
        assert cmd is not None
        assert cmd.name == "plugin"
        assert cmd.args == ["add", "myserver", "npx", "mcp-server"]

    def test_parse_non_command(self):
        cmd = parse_command("hello world")
        assert cmd is None

    def test_parse_empty_slash(self):
        cmd = parse_command("/")
        assert cmd is not None
        assert cmd.name == ""
        assert cmd.args == []

    def test_parse_with_whitespace(self):
        cmd = parse_command("  /save my-session  ")
        assert cmd is not None
        assert cmd.name == "save"
        assert cmd.args == ["my-session"]

    def test_command_help_completeness(self):
        """All documented commands have help text."""
        expected = {"new", "session", "sessions", "fork", "save", "plugin", "policy", "settings", "help", "worktree", "model"}
        assert set(COMMAND_HELP.keys()) == expected


# ── Session Fork Tests ──


class TestSessionFork:
    def _make_session(self):
        session = Session(name="original")
        agent = AgentNode(
            id="root", name="Orchestrator",
            state=AgentState.RUNNING, role=AgentRole.MASTER,
        )
        session.add_agent(agent)
        session.add_message("root", MessageRole.USER, "test message")
        session.add_message("root", MessageRole.ASSISTANT, "test response")
        return session

    def test_fork_deep_copies_agents(self):
        session = self._make_session()
        forked = session.fork(new_name="forked")

        # Modify the forked agent
        forked.agents["root"].name = "Modified"
        assert session.agents["root"].name == "Orchestrator"

    def test_fork_deep_copies_messages(self):
        session = self._make_session()
        forked = session.fork()

        forked.add_message("root", MessageRole.USER, "new message")
        assert len(session.get_messages("root")) == 2
        assert len(forked.get_messages("root")) == 3

    def test_fork_sets_metadata(self):
        session = self._make_session()
        forked = session.fork(new_name="my-fork")

        assert forked.name == "my-fork"
        assert forked.forked_from == "original"
        assert forked.created_at is not None
        assert forked.created_at >= session.created_at

    def test_fork_preserves_active_agent(self):
        session = self._make_session()
        session.set_active("root")
        forked = session.fork()
        assert forked.active_agent_id == "root"


# ── Session Metadata Persistence Tests ──


class TestSessionMetadata:
    def test_new_session_has_created_at(self):
        session = Session()
        assert session.created_at is not None

    def test_persistence_roundtrip_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))
            session = Session(name="test-session", forked_from="parent-session")
            session.add_agent(AgentNode(
                id="root", name="Root",
                state=AgentState.PENDING, role=AgentRole.MASTER,
            ))
            session.add_message("root", MessageRole.SYSTEM, "init")

            persistence.save(session, "test-session")
            loaded = persistence.load("test-session")

            assert loaded.name == "test-session"
            assert loaded.forked_from == "parent-session"
            assert loaded.created_at is not None

    def test_backward_compat_v1(self):
        """v1.0 session files (no metadata) still load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a v1.0 format file manually
            v1_data = {
                "active_agent_id": "root",
                "agents": {
                    "root": {
                        "id": "root", "name": "Root",
                        "state": "idle", "role": "orchestrator",
                        "model": "opus-4.6", "parent_id": None,
                        "children_ids": [], "prompt_preview": "",
                    }
                },
                "messages": {
                    "root": [{
                        "role": "system", "content": "init",
                        "agent_id": "root",
                        "timestamp": datetime.now().isoformat(),
                        "streaming": False, "tool_calls": [],
                    }]
                },
                "saved_at": datetime.now().isoformat(),
                "version": "1.0",
            }
            path = Path(tmpdir) / "legacy.json"
            path.write_text(json.dumps(v1_data))

            persistence = SessionPersistence(base_dir=Path(tmpdir))
            loaded = persistence.load("legacy")

            assert loaded.name is None
            assert loaded.forked_from is None
            assert loaded.active_agent_id == "root"

    def test_list_sessions_detailed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            s1 = Session(name="first")
            s1.add_agent(AgentNode(id="r", name="R", state=AgentState.PENDING))
            s1.add_message("r", MessageRole.SYSTEM, "init")
            persistence.save(s1, "first")

            s2 = Session(name="second", forked_from="first")
            s2.add_agent(AgentNode(id="r", name="R", state=AgentState.PENDING))
            s2.add_message("r", MessageRole.SYSTEM, "init")
            s2.add_message("r", MessageRole.USER, "hello")
            persistence.save(s2, "second")

            details = persistence.list_sessions_detailed()
            assert len(details) == 2

            first = next(d for d in details if d["name"] == "first")
            assert first["agent_count"] == 1
            assert first["message_count"] == 1
            assert first["file_stem"] == "first"
            assert first["task_complete_count"] == 0
            assert first["latest_task_complete_summary"] == ""

            second = next(d for d in details if d["name"] == "second")
            assert second["forked_from"] == "first"
            assert second["message_count"] == 2
            assert second["file_stem"] == "second"
            assert second["task_complete_count"] == 0
            assert second["latest_task_complete_summary"] == ""

    def test_persists_task_complete_summary_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            session = Session(name="summary-session")
            session.add_agent(AgentNode(
                id="root",
                name="Root",
                state=AgentState.PENDING,
                role=AgentRole.MASTER,
            ))
            session.add_message(
                "root",
                MessageRole.TOOL,
                "",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="mcp__orchestrator__task_complete",
                        arguments=json.dumps(
                            {
                                "summary": "Implemented fix end-to-end",
                                "artifacts": {"tests": "passed"},
                            },
                        ),
                    ),
                ],
            )

            persistence.save(session, "summary-session")
            data = json.loads((Path(tmpdir) / "summary-session.json").read_text())

            assert data["task_complete_count"] == 1
            assert data["latest_task_complete_summary"] == "Implemented fix end-to-end"
            assert data["latest_task_complete_summary_preview"] == "Implemented fix end-to-end"
            assert len(data["task_complete_summaries"]) == 1
            first = data["task_complete_summaries"][0]
            assert first["summary"] == "Implemented fix end-to-end"
            assert first["artifacts"] == {"tests": "passed"}

    def test_list_sessions_detailed_uses_single_line_summary_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            session = Session(name="multiline-summary")
            session.add_agent(AgentNode(
                id="root",
                name="Root",
                state=AgentState.PENDING,
                role=AgentRole.MASTER,
            ))
            session.add_message(
                "root",
                MessageRole.TOOL,
                "",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="mcp__orchestrator__task_complete",
                        arguments=json.dumps(
                            {
                                "summary": "What I changed\n\n1. First item\n1. Second item",
                            },
                        ),
                    ),
                ],
            )
            persistence.save(session, "multiline-summary")

            details = persistence.list_sessions_detailed()
            entry = next(d for d in details if d["name"] == "multiline-summary")
            assert entry["latest_task_complete_summary"] == (
                "What I changed 1. First item 1. Second item"
            )


# ── Plugin Manager Tests ──


class TestPluginManager:
    def test_add_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add("myserver", "npx", ["-y", "my-mcp-server"])

            plugins = pm.list_plugins()
            assert len(plugins) == 1
            assert plugins[0].name == "myserver"
            assert plugins[0].command == "npx"
            assert plugins[0].args == ["-y", "my-mcp-server"]

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add("test", "echo", ["hello"])
            assert pm.remove("test") is True
            assert pm.remove("nonexistent") is False
            assert len(pm.list_plugins()) == 0

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm1 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm1.add("server1", "npx", ["arg1"])
            pm1.add("server2", "python", ["-m", "myserver"])

            # Create a new PluginManager with the same dir
            pm2 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            plugins = pm2.list_plugins()
            names = {p.name for p in plugins}
            assert names == {"server1", "server2"}

    def test_workspace_plugins_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            mcp_json = workspace / ".mcp.json"
            mcp_json.write_text(json.dumps({
                "mcpServers": {
                    "ws-server": {
                        "command": "node",
                        "args": ["server.js"],
                    }
                }
            }))

            pm = PluginManager(cwd=workspace)
            plugins = pm.list_plugins()
            assert len(plugins) == 1
            assert plugins[0].name == "ws-server"

    def test_workspace_overrides_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            # Project-level plugin
            plugins_json = project_dir / "plugins.json"
            plugins_json.write_text(json.dumps({
                "mcpServers": {
                    "shared": {"command": "project-cmd", "args": []},
                }
            }))

            # Workspace-level plugin with same name
            mcp_json = workspace / ".mcp.json"
            mcp_json.write_text(json.dumps({
                "mcpServers": {
                    "shared": {"command": "workspace-cmd", "args": ["--override"]},
                }
            }))

            pm = PluginManager(project_dir=project_dir, cwd=workspace)
            plugins = pm.list_plugins()
            assert len(plugins) == 1
            assert plugins[0].command == "workspace-cmd"

    def test_mcp_json_overrides_prsm_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            prsm_json = workspace / ".prsm.json"
            prsm_json.write_text(json.dumps({
                "mcpServers": {
                    "shared": {"command": "legacy-cmd", "args": []},
                }
            }))

            mcp_json = workspace / ".mcp.json"
            mcp_json.write_text(json.dumps({
                "mcpServers": {
                    "shared": {"command": "mcp-cmd", "args": ["--new"]},
                }
            }))

            pm = PluginManager(cwd=workspace)
            plugins = pm.list_plugins()
            assert len(plugins) == 1
            assert plugins[0].command == "mcp-cmd"

    def test_get_mcp_server_configs_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add("test", "npx", ["-y", "server"])

            configs = pm.get_mcp_server_configs()
            assert "test" in configs
            assert configs["test"]["type"] == "stdio"
            assert configs["test"]["command"] == "npx"
            assert configs["test"]["args"] == ["-y", "server"]


# ── Headless TUI Tests — Slash Commands ──


def _capture_tool_log_writes(tl):
    """Spy on ToolLog.write() to capture all logged messages."""
    captured = []
    original_write = tl.write

    def spy_write(content, *args, **kwargs):
        captured.append(str(content))
        return original_write(content, *args, **kwargs)

    tl.write = spy_write
    return captured


@pytest.mark.asyncio
async def test_help_command():
    """Typing /help shows commands in tool log."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert("/help")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "Available commands" in log_text or "/help" in log_text


@pytest.mark.asyncio
async def test_input_bar_quick_action_buttons_present():
    """Input bar shows + / search / settings quick action buttons."""
    from prsm.tui.app import PrsmApp
    from textual.widgets import Button

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            assert inp.query_one("#new-session-btn", Button).label == "+"
            assert str(inp.query_one("#search-session-btn", Button).label) == "🔍"
            assert str(inp.query_one("#settings-btn", Button).label) == "⚙"


@pytest.mark.asyncio
async def test_sessions_command():
    """Typing /sessions lists sessions in tool log."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert("/sessions")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "session" in log_text.lower() or "No saved" in log_text


@pytest.mark.asyncio
async def test_unknown_command():
    """Typing unknown /command shows error in tool log."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert("/bogus")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "Unknown command" in log_text or "bogus" in log_text


@pytest.mark.asyncio
async def test_plugin_list_command():
    """Typing /plugin list shows plugins in tool log."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert("/plugin list")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "plugin" in log_text.lower() or "No plugins" in log_text


# ── CLI Flag Tests ──


def test_list_flag():
    """--list exits with session listing (no TUI launched)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "prsm.app", "--list"],
        capture_output=True, text=True, timeout=10,
        cwd="/tmp",
    )
    # Should exit 0 and not crash
    assert result.returncode == 0


def test_new_flag_attribute():
    """PrsmApp accepts cli_args attribute."""
    from prsm.tui.app import PrsmApp
    import argparse

    app = PrsmApp()
    args = argparse.Namespace(new=True, resume=None, fork=None, list=False)
    app.cli_args = args
    assert app.cli_args.new is True


# Need InputBar import for headless tests
from prsm.tui.widgets.input_bar import InputBar
