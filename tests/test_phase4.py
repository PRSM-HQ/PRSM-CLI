"""Phase 4 tests — event system, orchestrator bridge, persistence, permissions."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.shared.models.agent import AgentNode, AgentRole, AgentState
from prsm.shared.models.message import Message, MessageRole, ToolCall
from prsm.shared.models.session import Session
from prsm.adapters.events import (
    AuditEntry,
    BudgetStatus,
    DecisionReport,
    AgentSpawned,
    EngineFinished,
    ExpertStatus,
    LeaseStatus,
    OrchestratorEvent,
    PermissionRequest,
    StreamChunk,
    ToolCallCompleted,
    ToolCallStarted,
    dict_to_event,
)
from prsm.adapters.event_bus import EventBus
from prsm.adapters.orchestrator import OrchestratorBridge
from prsm.shared.services.persistence import SessionPersistence


# ── Event System Tests ──


class TestDictToEvent:
    def test_stream_chunk(self):
        event = dict_to_event({
            "event": "stream_chunk",
            "agent_id": "abc123",
            "text": "Hello world",
        })
        assert isinstance(event, StreamChunk)
        assert event.agent_id == "abc123"
        assert event.text == "Hello world"
        assert event.event_type == "stream_chunk"

    def test_agent_spawned(self):
        event = dict_to_event({
            "event": "agent_spawned",
            "agent_id": "xyz",
            "parent_id": None,
            "role": "master",
            "model": "opus-4.6",
            "depth": 0,
            "prompt": "Do the thing",
            "name": "Plan architecture migration",
        })
        assert isinstance(event, AgentSpawned)
        assert event.role == "master"
        assert event.depth == 0
        assert event.name == "Plan architecture migration"

    def test_engine_finished(self):
        event = dict_to_event({
            "event": "engine_finished",
            "success": True,
            "summary": "All done",
            "duration_seconds": 42.5,
        })
        assert isinstance(event, EngineFinished)
        assert event.success is True
        assert event.duration_seconds == 42.5

    def test_unknown_event_type(self):
        event = dict_to_event({"event": "unknown_type", "foo": "bar"})
        assert isinstance(event, OrchestratorEvent)
        assert event.event_type == "unknown_type"

    def test_extra_keys_ignored(self):
        event = dict_to_event({
            "event": "stream_chunk",
            "agent_id": "a",
            "text": "t",
            "extra_field": "ignored",
        })
        assert isinstance(event, StreamChunk)
        assert event.text == "t"

    def test_phase8_event_types_registered(self):
        assert isinstance(dict_to_event({"event": "lease_status"}), LeaseStatus)
        assert isinstance(dict_to_event({"event": "audit_entry"}), AuditEntry)
        assert isinstance(dict_to_event({"event": "expert_status"}), ExpertStatus)
        assert isinstance(dict_to_event({"event": "budget_status"}), BudgetStatus)
        assert isinstance(dict_to_event({"event": "decision_report"}), DecisionReport)


# ── EventBus Tests ──


@pytest.mark.asyncio
async def test_event_bus_emit_consume():
    """Events emitted via callback round-trip through the bus."""
    bus = EventBus(maxsize=100)
    callback = bus.make_callback()

    # Emit via callback (simulates engine)
    await callback({"event": "stream_chunk", "agent_id": "a1", "text": "hi"})
    await callback({"event": "engine_finished", "success": True})

    events = []
    async for event in bus.consume():
        events.append(event)
        if isinstance(event, EngineFinished):
            break

    assert len(events) == 2
    assert isinstance(events[0], StreamChunk)
    assert events[0].text == "hi"
    assert isinstance(events[1], EngineFinished)

    bus.close()


@pytest.mark.asyncio
async def test_event_bus_close_stops_consumer():
    """Closing the bus causes consume() to exit."""
    bus = EventBus()

    async def consume_until_done():
        events = []
        async for event in bus.consume():
            events.append(event)
        return events

    task = asyncio.create_task(consume_until_done())
    await asyncio.sleep(0.1)
    bus.close()
    await asyncio.sleep(0.7)  # Wait for timeout cycle in consume()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result == []


# ── OrchestratorBridge Tests ──


class TestOrchestratorBridge:
    def test_map_agent_master(self):
        bridge = OrchestratorBridge()
        node = bridge.map_agent(
            agent_id="abc",
            parent_id=None,
            role="master",
            model="opus-4.6",
            prompt="Do things",
        )
        assert node.id == "abc"
        assert node.name == "Orchestrator"
        assert node.role == AgentRole.MASTER
        assert node.model == "opus-4.6"
        assert node.parent_id is None

    def test_map_agent_worker(self):
        bridge = OrchestratorBridge()
        node = bridge.map_agent(
            agent_id="w1",
            parent_id="abc",
            role="worker",
            model="opus-4.6",
            prompt="Explore the codebase and find all auth-related files",
        )
        assert node.role == AgentRole.WORKER
        assert node.parent_id == "abc"
        assert "Explore the codebase" in node.name

    def test_map_agent_uses_event_name_when_provided(self):
        bridge = OrchestratorBridge()
        node = bridge.map_agent(
            agent_id="w2",
            parent_id="abc",
            role="worker",
            model="opus-4.6",
            prompt="Long prompt that should not become the display name",
            name="Fix flaky auth integration tests",
        )
        assert node.name == "Fix flaky auth integration tests"

    def test_map_agent_expert(self):
        bridge = OrchestratorBridge()
        node = bridge.map_agent(
            agent_id="e1",
            parent_id="abc",
            role="expert",
            model="opus-4.6",
            prompt="Rust systems expert",
        )
        assert node.role == AgentRole.EXPERT
        assert node.name == "Rust systems expert"

    def test_map_state(self):
        bridge = OrchestratorBridge()
        assert bridge.map_state("running") == AgentState.RUNNING
        assert bridge.map_state("completed") == AgentState.COMPLETED
        assert bridge.map_state("waiting_for_child") == AgentState.WAITING_FOR_CHILD
        assert bridge.map_state("failed") == AgentState.FAILED
        assert bridge.map_state("killed") == AgentState.KILLED
        assert bridge.map_state("pending") == AgentState.PENDING
        assert bridge.map_state("unknown") == AgentState.PENDING

    def test_agent_map_tracking(self):
        bridge = OrchestratorBridge()
        bridge.map_agent("a1", None, "master", "opus-4.6", "Go")
        bridge.map_agent("a2", "a1", "worker", "opus-4.6", "Work")
        assert len(bridge.agent_map) == 2
        assert "a1" in bridge.agent_map
        assert "a2" in bridge.agent_map


# ── Session Persistence Tests ──


class TestSessionPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            # Build a session
            session = Session()
            agent = AgentNode(
                id="root",
                name="Orchestrator",
                state=AgentState.RUNNING,
                role=AgentRole.MASTER,
                model="opus-4.6",
            )
            session.add_agent(agent)
            session.set_active("root")
            session.add_message("root", MessageRole.SYSTEM, "Started")
            session.add_message("root", MessageRole.USER, "Do something")
            session.add_message(
                "root", MessageRole.TOOL, "",
                tool_calls=[ToolCall(
                    id="tc-1", name="read_file",
                    arguments="test.py", result="OK",
                )],
            )

            # Save
            path = persistence.save(session, "test-session")
            assert path.exists()

            # Load
            loaded = persistence.load("test-session")
            assert loaded.active_agent_id == "root"
            assert len(loaded.agents) == 1
            assert loaded.agents["root"].name == "Orchestrator"
            # State should be reset from RUNNING to COMPLETED on load
            assert loaded.agents["root"].state == AgentState.COMPLETED
            assert loaded.agents["root"].role == AgentRole.MASTER

            msgs = loaded.get_messages("root")
            assert len(msgs) == 3
            assert msgs[0].role == MessageRole.SYSTEM
            assert msgs[0].content == "Started"
            assert msgs[1].role == MessageRole.USER
            assert msgs[2].role == MessageRole.TOOL
            assert len(msgs[2].tool_calls) == 1
            assert msgs[2].tool_calls[0].name == "read_file"

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))
            session = Session()

            persistence.save(session, "alpha")
            persistence.save(session, "beta")

            names = persistence.list_sessions()
            assert names == ["alpha", "beta"]

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))
            session = Session()

            persistence.save(session, "to-delete")
            assert persistence.delete("to-delete") is True
            assert persistence.list_sessions() == []
            assert persistence.delete("nonexistent") is False


# ── Permission Screen Tests ──


@pytest.mark.asyncio
async def test_permission_screen_allow():
    """PermissionScreen returns 'allow' when Allow is clicked."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.permission import PermissionScreen

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        result_holder = []

        def on_dismiss(result):
            result_holder.append(result)

        screen = PermissionScreen(
            tool_name="Bash",
            agent_name="Orchestrator",
            arguments="ls -la",
        )
        app.push_screen(screen, callback=on_dismiss)
        await pilot.pause()

        # Click the Allow button
        btn = screen.query_one("#btn-allow")
        await pilot.click(btn)
        await pilot.pause()

        assert result_holder == ["allow"]


@pytest.mark.asyncio
async def test_permission_screen_deny():
    """PermissionScreen returns 'deny' when Deny is clicked."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.permission import PermissionScreen

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        result_holder = []

        screen = PermissionScreen(
            tool_name="Write",
            agent_name="Code Writer",
            arguments="path=/etc/passwd",
        )
        app.push_screen(screen, callback=lambda r: result_holder.append(r))
        await pilot.pause()

        btn = screen.query_one("#btn-deny")
        await pilot.click(btn)
        await pilot.pause()

        assert result_holder == ["deny"]


# ── Fix 1: Permission Request Event Parsing ──


class TestPermissionRequestEvent:
    def test_dict_to_event_permission_request(self):
        event = dict_to_event({
            "event": "permission_request",
            "agent_id": "agent-1",
            "request_id": "req-abc",
            "tool_name": "Bash",
            "agent_name": "Orchestrator",
            "arguments": "ls -la",
        })
        assert isinstance(event, PermissionRequest)
        assert event.agent_id == "agent-1"
        assert event.request_id == "req-abc"
        assert event.tool_name == "Bash"
        assert event.agent_name == "Orchestrator"
        assert event.arguments == "ls -la"

    def test_tool_call_completed_parsing(self):
        event = dict_to_event({
            "event": "tool_call_completed",
            "agent_id": "a1",
            "tool_id": "tool-123",
            "result": "file contents here",
            "is_error": False,
        })
        assert isinstance(event, ToolCallCompleted)
        assert event.tool_id == "tool-123"
        assert event.result == "file contents here"
        assert event.is_error is False

    def test_tool_call_completed_error(self):
        event = dict_to_event({
            "event": "tool_call_completed",
            "agent_id": "a1",
            "tool_id": "tool-456",
            "result": "Permission denied",
            "is_error": True,
        })
        assert isinstance(event, ToolCallCompleted)
        assert event.is_error is True


# ── Fix 3: EventBus Reset ──


@pytest.mark.asyncio
async def test_event_bus_reset():
    """EventBus.reset() drains leftover events and re-opens for consumption."""
    bus = EventBus(maxsize=100)
    callback = bus.make_callback()

    # Emit some events, then close
    await callback({"event": "stream_chunk", "agent_id": "a1", "text": "old"})
    await callback({"event": "stream_chunk", "agent_id": "a1", "text": "stale"})
    bus.close()

    # Reset should drain the queue and re-open
    bus.reset()

    # Verify the bus is open again — emit new events
    await callback({"event": "stream_chunk", "agent_id": "a2", "text": "fresh"})

    events = []
    async for event in bus.consume():
        events.append(event)
        # Only one event should be there (the "fresh" one)
        break

    assert len(events) == 1
    assert isinstance(events[0], StreamChunk)
    assert events[0].text == "fresh"
    assert events[0].agent_id == "a2"
    bus.close()


@pytest.mark.asyncio
async def test_event_bus_reset_drains_queue():
    """EventBus.reset() removes all queued events."""
    bus = EventBus(maxsize=100)
    callback = bus.make_callback()

    # Fill with events
    for i in range(10):
        await callback({"event": "stream_chunk", "agent_id": "a1", "text": f"chunk-{i}"})

    # Reset should drain everything
    bus.reset()

    # Queue should be empty after reset
    assert bus._queue.empty()
    bus.close()


# ── Fix 1 continued: OrchestratorBridge Permission Resolution ──


class TestBridgePermissions:
    def test_resolve_permission(self):
        """resolve_permission resolves a pending Future."""
        bridge = OrchestratorBridge()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            bridge._permission_futures["req-1"] = future

            bridge.resolve_permission("req-1", "allow")
            assert future.done()
            assert future.result() == "allow"
        finally:
            loop.close()

    def test_resolve_permission_allow_always(self):
        bridge = OrchestratorBridge()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            bridge._permission_futures["req-2"] = future

            bridge.resolve_permission("req-2", "allow_always")
            assert future.result() == "allow_always"
        finally:
            loop.close()

    def test_resolve_permission_nonexistent(self):
        """Resolving a nonexistent request is a no-op."""
        bridge = OrchestratorBridge()
        # Should not raise
        bridge.resolve_permission("nonexistent", "deny")

    def test_resolve_permission_already_done(self):
        """Resolving an already-resolved future is a no-op."""
        bridge = OrchestratorBridge()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            future.set_result("allow")
            bridge._permission_futures["req-3"] = future

            # Should not raise
            bridge.resolve_permission("req-3", "deny")
            assert future.result() == "allow"  # Original result unchanged
        finally:
            loop.close()

    def test_running_property_default_false(self):
        bridge = OrchestratorBridge()
        assert bridge.running is False


# ── Fix 4: ConversationView Tool Result Updates ──


@pytest.mark.asyncio
async def test_conversation_update_tool_result():
    """update_tool_result patches the ToolCall and re-renders the widget."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.conversation import ConversationView

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Access the conversation view via current screen
        conv = app.screen.query_one(ConversationView)

        # Set up an active agent
        agent = AgentNode(
            id="test-agent", name="Test", state=AgentState.RUNNING,
            role=AgentRole.WORKER, model="opus-4.6",
        )
        conv.session.add_agent(agent)
        conv.show_agent("test-agent")
        await pilot.pause()

        # Add a tool call with no result (pending)
        msg = conv.add_tool_call(
            agent_id="test-agent",
            tool_name="Read",
            args="test.py",
            result="",
            tool_id="tc-1",
        )
        await pilot.pause()

        # Verify it's tracked as pending
        assert "tc-1" in conv._pending_tool_calls

        # Update with result
        conv.update_tool_result("tc-1", "file contents here", is_error=False)
        await pilot.pause()

        # Pending should be cleared
        assert "tc-1" not in conv._pending_tool_calls

        # The ToolCall in the message should be updated
        assert msg.tool_calls[0].result == "file contents here"
        assert msg.tool_calls[0].success is True


@pytest.mark.asyncio
async def test_conversation_update_tool_result_error():
    """update_tool_result handles error results."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.conversation import ConversationView

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        conv = app.screen.query_one(ConversationView)

        agent = AgentNode(
            id="err-agent", name="Test", state=AgentState.RUNNING,
            role=AgentRole.WORKER, model="opus-4.6",
        )
        conv.session.add_agent(agent)
        conv.show_agent("err-agent")
        await pilot.pause()

        msg = conv.add_tool_call(
            agent_id="err-agent",
            tool_name="Bash",
            args="rm -rf /",
            result="",
            tool_id="tc-err",
        )
        await pilot.pause()

        conv.update_tool_result("tc-err", "Permission denied", is_error=True)
        await pilot.pause()

        assert msg.tool_calls[0].result == "Permission denied"
        assert msg.tool_calls[0].success is False


@pytest.mark.asyncio
async def test_conversation_update_tool_result_nonexistent():
    """update_tool_result is a no-op for unknown tool_id."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.conversation import ConversationView

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        conv = app.screen.query_one(ConversationView)

        # Should not raise
        conv.update_tool_result("nonexistent-id", "result", is_error=False)


# ── Fix 6: Permission Screen Allow Always ──


@pytest.mark.asyncio
async def test_permission_screen_allow_project():
    """PermissionScreen returns 'allow_project' when Always (project) is clicked."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.permission import PermissionScreen

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        result_holder = []

        screen = PermissionScreen(
            tool_name="Read",
            agent_name="Explorer",
            arguments="src/main.py",
        )
        app.push_screen(screen, callback=lambda r: result_holder.append(r))
        await pilot.pause()

        btn = screen.query_one("#btn-project")
        await pilot.click(btn)
        await pilot.pause()

        assert result_holder == ["allow_project"]


@pytest.mark.asyncio
async def test_permission_screen_deny_project():
    """PermissionScreen returns 'deny_project' for Always reject."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.permission import PermissionScreen

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        result_holder = []

        screen = PermissionScreen(
            tool_name="Bash",
            agent_name="Worker",
            arguments="rm -rf build",
        )
        app.push_screen(screen, callback=lambda r: result_holder.append(r))
        await pilot.pause()

        btn = screen.query_one("#btn-reject-project")
        await pilot.click(btn)
        await pilot.pause()

        assert result_holder == ["deny_project"]


# ── Stream Buffering Tests ──


@pytest.mark.asyncio
async def test_conversation_stream_buffering():
    """Stream chunks for non-active agents are buffered and replayed on switch."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.conversation import ConversationView

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        conv = app.screen.query_one(ConversationView)

        # Set up two agents
        a1 = AgentNode(
            id="agent-a", name="Agent A", state=AgentState.RUNNING,
            role=AgentRole.MASTER, model="opus-4.6",
        )
        a2 = AgentNode(
            id="agent-b", name="Agent B", state=AgentState.RUNNING,
            role=AgentRole.WORKER, model="opus-4.6",
        )
        conv.session.add_agent(a1)
        conv.session.add_agent(a2)

        # Show agent A
        conv.show_agent("agent-a")
        await pilot.pause()

        # Buffer chunks for agent B (not currently shown)
        conv.buffer_stream_chunk("agent-b", "Hello ")
        conv.buffer_stream_chunk("agent-b", "world")

        assert "agent-b" in conv._stream_buffers
        assert conv._stream_buffers["agent-b"] == ["Hello ", "world"]

        # Flush to session
        conv.flush_stream_buffer("agent-b")

        # Buffer should be cleared
        assert "agent-b" not in conv._stream_buffers

        # Message should be in session
        msgs = conv.session.get_messages("agent-b")
        assert len(msgs) == 1
        assert msgs[0].content == "Hello world"
        assert msgs[0].role == MessageRole.ASSISTANT


# ── Directory-Aware Project Manager Tests ──


class TestProjectManager:
    def test_get_project_dir_creates_path(self):
        from prsm.shared.services.project import ProjectManager
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch home to use tmpdir
            import os
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir

            try:
                cwd = Path("/home/user/myproject")
                project_dir = ProjectManager.get_project_dir(cwd)
                assert project_dir.name == "home-user-myproject"
                assert project_dir.parent.name == "projects"
                assert project_dir.parent.parent.name == ".prsm"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)

    def test_get_project_dir_nested_path(self):
        from prsm.shared.services.project import ProjectManager
        cwd = Path("/home/user/Documents/my-app")
        project_dir = ProjectManager.get_project_dir(cwd)
        assert project_dir.name == "home-user-Documents-my-app"

    def test_get_memory_path(self):
        from prsm.shared.services.project import ProjectManager
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test-project"
            project_dir.mkdir()
            memory_path = ProjectManager.get_memory_path(project_dir)
            assert memory_path.name == "MEMORY.md"
            assert memory_path.parent.name == "memory"
            assert memory_path.parent.exists()

    def test_get_sessions_dir(self):
        from prsm.shared.services.project import ProjectManager
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test-project"
            project_dir.mkdir()
            sessions_dir = ProjectManager.get_sessions_dir(project_dir)
            assert sessions_dir.name == "sessions"
            assert sessions_dir.exists()


# ── Project Memory Tests ──


class TestProjectMemory:
    def test_load_nonexistent(self):
        from prsm.shared.services.project_memory import ProjectMemory
        memory = ProjectMemory(Path("/nonexistent/MEMORY.md"))
        assert memory.load() == ""
        assert memory.exists() is False

    def test_save_and_load(self):
        from prsm.shared.services.project_memory import ProjectMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory" / "MEMORY.md"
            memory = ProjectMemory(memory_path)

            content = "# My Project\n\nSome notes about the project."
            memory.save(content)

            assert memory.exists() is True
            assert memory.load() == content


# ── Directory-Aware Persistence Tests ──


class TestDirectoryAwarePersistence:
    def test_cwd_mode_saves_with_marker(self):
        """Sessions saved in cwd mode write .active_session marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir) / "fake-project"
            cwd.mkdir()

            # Override home to use tmpdir for .prsm/
            import os
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir

            try:
                persistence = SessionPersistence(cwd=cwd)

                session = Session()
                agent = AgentNode(
                    id="root", name="Orchestrator",
                    state=AgentState.RUNNING,
                    role=AgentRole.MASTER, model="opus-4.6",
                )
                session.add_agent(agent)
                session.add_message("root", MessageRole.USER, "Hello")

                path = persistence.save(session, "test-save")
                assert path.exists()

                # Check .active_session marker was written
                marker = persistence.project_dir / ".active_session"
                assert marker.exists()
                assert marker.read_text() == "test-save"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)

    def test_auto_resume_loads_last_session(self):
        """auto_resume loads the session referenced by .active_session marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir) / "my-project"
            cwd.mkdir()

            import os
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir

            try:
                persistence = SessionPersistence(cwd=cwd)

                # Save a session
                session = Session()
                agent = AgentNode(
                    id="a1", name="Worker",
                    state=AgentState.COMPLETED,
                    role=AgentRole.WORKER, model="opus-4.6",
                )
                session.add_agent(agent)
                session.add_message("a1", MessageRole.ASSISTANT, "Done!")
                persistence.save(session, "my-session")

                # Create a fresh persistence and auto_resume
                persistence2 = SessionPersistence(cwd=cwd)
                resumed = persistence2.auto_resume()

                assert resumed is not None
                assert "a1" in resumed.agents
                assert resumed.agents["a1"].name == "Worker"
                msgs = resumed.get_messages("a1")
                assert len(msgs) == 1
                assert msgs[0].content == "Done!"
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)

    def test_auto_resume_returns_none_without_marker(self):
        """auto_resume returns None when no .active_session exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir) / "empty-project"
            cwd.mkdir()

            import os
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmpdir

            try:
                persistence = SessionPersistence(cwd=cwd)
                assert persistence.auto_resume() is None
            finally:
                if old_home:
                    os.environ["HOME"] = old_home
                else:
                    os.environ.pop("HOME", None)

    def test_auto_resume_returns_none_for_legacy_mode(self):
        """auto_resume returns None in legacy (base_dir) mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))
            assert persistence.auto_resume() is None

    def test_backward_compat_base_dir(self):
        """Legacy base_dir mode still works unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            session = Session()
            session.add_message("root", MessageRole.USER, "test")
            path = persistence.save(session, "legacy")

            assert path.exists()
            assert path.parent == Path(tmpdir)

            loaded = persistence.load("legacy")
            msgs = loaded.get_messages("root")
            assert len(msgs) == 1
            assert msgs[0].content == "test"

    def test_save_failure_keeps_previous_session_file(self):
        """Failed save should not corrupt the last durable on-disk session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = SessionPersistence(base_dir=Path(tmpdir))

            original = Session()
            original.add_message("root", MessageRole.USER, "old")
            persistence.save(original, "stable")

            updated = Session()
            updated.add_message("root", MessageRole.USER, "new")

            with patch(
                "prsm.shared.services.persistence.atomic_write_text",
                side_effect=OSError("disk full"),
            ):
                with pytest.raises(OSError):
                    persistence.save(updated, "stable")

            loaded = persistence.load("stable")
            msgs = loaded.get_messages("root")
            assert len(msgs) == 1
            assert msgs[0].content == "old"
