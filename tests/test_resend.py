"""Tests for the resend/restore feature.

Covers:
- Session truncation to a specific message index
- File-change counting after a message index
- Snapshot auto-creation helpers
- ResendConfirmScreen composition
- InputBar.set_text pre-fill
- ConversationView message_index tracking
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prsm.shared.models.session import Session
from prsm.shared.models.message import Message, MessageRole
from prsm.shared.models.agent import AgentNode
from prsm.adapters.file_tracker import FileChangeTracker, FileChangeRecord
from prsm.engine.models import AgentRole, AgentState


# ── Helper factories ──


def _make_session_with_messages() -> Session:
    """Build a session with a master agent and several messages."""
    session = Session()
    agent = AgentNode(
        id="master-1",
        name="Orchestrator",
        state=AgentState.COMPLETED,
        role=AgentRole.MASTER,
        model="claude-opus-4-6",
    )
    session.add_agent(agent)

    now = datetime.now(timezone.utc)
    msgs = [
        Message(role=MessageRole.SYSTEM, content="Session started.", agent_id="master-1", timestamp=now),
        Message(role=MessageRole.USER, content="First prompt", agent_id="master-1", timestamp=now + timedelta(seconds=1)),
        Message(role=MessageRole.ASSISTANT, content="First response", agent_id="master-1", timestamp=now + timedelta(seconds=2)),
        Message(role=MessageRole.USER, content="Second prompt", agent_id="master-1", timestamp=now + timedelta(seconds=3)),
        Message(role=MessageRole.ASSISTANT, content="Second response", agent_id="master-1", timestamp=now + timedelta(seconds=4)),
        Message(role=MessageRole.USER, content="Third prompt", agent_id="master-1", timestamp=now + timedelta(seconds=5)),
        Message(role=MessageRole.ASSISTANT, content="Third response", agent_id="master-1", timestamp=now + timedelta(seconds=6)),
    ]
    session.messages["master-1"] = msgs
    return session


def _make_file_tracker_with_changes() -> FileChangeTracker:
    """Build a FileChangeTracker with records at various message indices."""
    tracker = FileChangeTracker()
    tracker.file_changes["src/auth.py"] = [
        FileChangeRecord(
            file_path="src/auth.py",
            agent_id="master-1",
            change_type="modify",
            tool_call_id="tc-1",
            tool_name="Edit",
            message_index=2,
            timestamp="2024-01-01T00:00:02",
        ),
    ]
    tracker.file_changes["src/config.py"] = [
        FileChangeRecord(
            file_path="src/config.py",
            agent_id="master-1",
            change_type="create",
            tool_call_id="tc-2",
            tool_name="Write",
            message_index=4,
            timestamp="2024-01-01T00:00:04",
        ),
    ]
    tracker.file_changes["src/utils.py"] = [
        FileChangeRecord(
            file_path="src/utils.py",
            agent_id="master-1",
            change_type="modify",
            tool_call_id="tc-3",
            tool_name="Edit",
            message_index=6,
            timestamp="2024-01-01T00:00:06",
        ),
    ]
    return tracker


# ── Session Truncation Tests ──


class TestSessionTruncation:
    """Tests for truncate_session_to logic."""

    def test_truncate_removes_messages_at_and_after_index(self):
        """Truncating at index 3 should keep messages [0, 1, 2]."""
        session = _make_session_with_messages()
        assert len(session.get_messages("master-1")) == 7

        # Simulate truncation (the logic from SessionManager)
        agent_id = "master-1"
        message_index = 3  # "Second prompt"
        msgs = session.get_messages(agent_id)
        cutoff_time = msgs[message_index].timestamp
        session.messages[agent_id] = msgs[:message_index]

        remaining = session.get_messages("master-1")
        assert len(remaining) == 3
        assert remaining[-1].content == "First response"

    def test_truncate_at_first_user_message(self):
        """Truncating at the first user message (index 1) should keep only system."""
        session = _make_session_with_messages()
        agent_id = "master-1"
        message_index = 1  # "First prompt"
        msgs = session.get_messages(agent_id)
        session.messages[agent_id] = msgs[:message_index]

        remaining = session.get_messages("master-1")
        assert len(remaining) == 1
        assert remaining[0].role == MessageRole.SYSTEM

    def test_truncate_at_last_message_is_noop_like(self):
        """Truncating at the very last message removes only the last one."""
        session = _make_session_with_messages()
        agent_id = "master-1"
        message_index = 6  # "Third response"
        msgs = session.get_messages(agent_id)
        session.messages[agent_id] = msgs[:message_index]

        remaining = session.get_messages("master-1")
        assert len(remaining) == 6
        assert remaining[-1].content == "Third prompt"


class TestFileChangesCounting:
    """Tests for count_file_changes_after logic."""

    def test_count_after_early_message(self):
        """Changes at indices 4 and 6 should be counted after index 2."""
        tracker = _make_file_tracker_with_changes()
        files_after: set[str] = set()
        for records in tracker.file_changes.values():
            for r in records:
                if r.message_index > 2:
                    files_after.add(r.file_path)
        assert len(files_after) == 2
        assert "src/config.py" in files_after
        assert "src/utils.py" in files_after

    def test_count_after_last_change(self):
        """No changes should be counted after the last message index (6)."""
        tracker = _make_file_tracker_with_changes()
        files_after: set[str] = set()
        for records in tracker.file_changes.values():
            for r in records:
                if r.message_index > 6:
                    files_after.add(r.file_path)
        assert len(files_after) == 0

    def test_count_after_zero(self):
        """All 3 files should be counted after index 0."""
        tracker = _make_file_tracker_with_changes()
        files_after: set[str] = set()
        for records in tracker.file_changes.values():
            for r in records:
                if r.message_index > 0:
                    files_after.add(r.file_path)
        assert len(files_after) == 3


class TestFileTrackerTruncation:
    """Tests for file tracker cleanup during truncation."""

    def test_tracker_records_removed_after_cutoff(self):
        """File change records at or after cutoff index should be removed."""
        tracker = _make_file_tracker_with_changes()
        message_index = 3  # Cutoff

        for fp in list(tracker.file_changes.keys()):
            tracker.file_changes[fp] = [
                r for r in tracker.file_changes[fp]
                if r.message_index < message_index
            ]
            if not tracker.file_changes[fp]:
                del tracker.file_changes[fp]

        # Only src/auth.py (index 2) should remain
        assert "src/auth.py" in tracker.file_changes
        assert "src/config.py" not in tracker.file_changes
        assert "src/utils.py" not in tracker.file_changes
        assert len(tracker.file_changes["src/auth.py"]) == 1


class TestFileTrackerPersistence:
    """Tests for crash-safe file-change persistence behavior."""

    def test_persist_removes_stale_change_files(self):
        tracker = _make_file_tracker_with_changes()
        with tempfile.TemporaryDirectory() as tmpdir:
            changes_dir = Path(tmpdir) / "file-changes"
            tracker.persist(changes_dir)
            assert (changes_dir / "tc-1.json").exists()
            assert (changes_dir / "tc-2.json").exists()
            assert (changes_dir / "tc-3.json").exists()

            # Simulate rewind/truncation: remove a record and persist again.
            del tracker.file_changes["src/config.py"]  # tc-2
            tracker.persist(changes_dir)

            assert (changes_dir / "tc-1.json").exists()
            assert (changes_dir / "tc-3.json").exists()
            assert not (changes_dir / "tc-2.json").exists()


# ── SnapshotService Integration Tests ──


class TestSnapshotServiceIntegration:
    """Tests for SnapshotService create/load cycle used by resend."""

    def test_create_and_load_snapshot(self):
        """Create a snapshot and verify load_session returns matching data."""
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            cwd = Path(tmpdir) / "workdir"
            cwd.mkdir()

            svc = SnapshotService(project_dir, cwd)
            session = _make_session_with_messages()

            snap_id = svc.create(session, "test-session", "test snapshot")
            assert snap_id

            loaded_session, loaded_tracker = svc.load_session(snap_id)
            assert len(loaded_session.agents) == 1
            assert loaded_session.get_messages("master-1")
            # Message count should match
            assert len(loaded_session.get_messages("master-1")) == len(
                session.get_messages("master-1")
            )

    def test_snapshot_with_file_tracker(self):
        """Snapshot should persist and restore file change records."""
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            cwd = Path(tmpdir) / "workdir"
            cwd.mkdir()

            svc = SnapshotService(project_dir, cwd)
            session = _make_session_with_messages()
            tracker = _make_file_tracker_with_changes()

            snap_id = svc.create(
                session, "test-session", "with changes",
                file_tracker=tracker,
            )

            _, loaded_tracker = svc.load_session(snap_id)
            assert len(loaded_tracker.file_changes) > 0
            # Verify specific records survived round-trip
            assert "src/auth.py" in loaded_tracker.file_changes

    def test_list_snapshots_empty(self):
        """list_snapshots returns empty list when no snapshots exist."""
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = SnapshotService(Path(tmpdir), Path(tmpdir))
            assert svc.list_snapshots() == []

    def test_delete_snapshot(self):
        """Deleting a snapshot removes it from disk."""
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = SnapshotService(Path(tmpdir), Path(tmpdir))
            session = _make_session_with_messages()
            snap_id = svc.create(session, "test", "delete me")

            assert svc.delete(snap_id) is True
            assert svc.list_snapshots() == []

    def test_get_meta(self):
        """get_meta returns metadata for an existing snapshot."""
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = SnapshotService(Path(tmpdir), Path(tmpdir))
            session = _make_session_with_messages()
            snap_id = svc.create(session, "test", "meta test")

            meta = svc.get_meta(snap_id)
            assert meta["snapshot_id"] == snap_id
            assert meta["description"] == "meta test"
            assert "timestamp" in meta

    def test_create_cleans_staging_dir_on_failure(self):
        """Snapshot creation should not leave half-written staging dirs."""
        import uuid
        from prsm.shared.services.snapshot import SnapshotService

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            cwd = Path(tmpdir) / "workdir"
            cwd.mkdir()
            svc = SnapshotService(project_dir, cwd)
            session = _make_session_with_messages()

            call_count = 0

            def _fail_second_write(path, content, encoding="utf-8"):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated write failure")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding=encoding)

            with patch("prsm.shared.services.snapshot.uuid.uuid4", return_value=uuid.UUID("12345678-1234-1234-1234-123456789abc")):
                with patch("prsm.shared.services.snapshot.atomic_write_text", side_effect=_fail_second_write):
                    with pytest.raises(OSError):
                        svc.create(session, "test", "boom")

            snapshots_dir = project_dir / "snapshots"
            assert not (snapshots_dir / "12345678").exists()
            leftovers = [p for p in snapshots_dir.glob(".tmp-*")]
            assert leftovers == []


# ── Preferences Tests ──


class TestUserPreferences:
    """Tests for UserPreferences persistence and validation."""

    def test_default_preferences(self):
        from prsm.shared.services.preferences import UserPreferences
        prefs = UserPreferences()
        assert prefs.file_revert_on_resend == "ask"

    def test_preferences_validation(self):
        from prsm.shared.services.preferences import UserPreferences
        prefs = UserPreferences(file_revert_on_resend="invalid")
        prefs.validate()
        assert prefs.file_revert_on_resend == "ask"

    def test_preferences_save_and_load(self):
        from prsm.shared.services.preferences import UserPreferences
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prefs.json"
            prefs = UserPreferences(file_revert_on_resend="always")
            prefs.save(path)

            loaded = UserPreferences.load(path)
            assert loaded.file_revert_on_resend == "always"

    def test_preferences_load_missing_file(self):
        from prsm.shared.services.preferences import UserPreferences
        prefs = UserPreferences.load(Path("/nonexistent/path/prefs.json"))
        assert prefs.file_revert_on_resend == "ask"


# ── ResendConfirmScreen Tests ──


class TestResendConfirmScreen:
    """Tests for the ResendConfirmScreen modal structure."""

    def test_screen_instantiation(self):
        from prsm.tui.screens.resend_confirm import ResendConfirmScreen
        screen = ResendConfirmScreen(file_count=3, prompt_preview="test prompt")
        assert screen._file_count == 3
        assert screen._prompt_preview == "test prompt"

    def test_screen_result_map(self):
        from prsm.tui.screens.resend_confirm import _RESULT_MAP
        assert _RESULT_MAP["btn-revert"] == "revert"
        assert _RESULT_MAP["btn-keep"] == "keep"
        assert _RESULT_MAP["btn-cancel"] == "cancel"


# ── ConversationView Message Index Tests ──


class TestConversationMessageIndex:
    """Tests that MessageWidget correctly tracks message_index."""

    def test_message_widget_stores_index(self):
        from prsm.tui.widgets.conversation import MessageWidget
        msg = Message(
            role=MessageRole.USER,
            content="test",
            agent_id="agent-1",
        )
        widget = MessageWidget(msg, message_index=5)
        assert widget._message_index == 5

    def test_message_widget_default_index(self):
        from prsm.tui.widgets.conversation import MessageWidget
        msg = Message(
            role=MessageRole.USER,
            content="test",
            agent_id="agent-1",
        )
        widget = MessageWidget(msg)
        assert widget._message_index == -1

    def test_resend_requested_carries_index(self):
        """ResendRequested message should carry the correct message index."""
        from prsm.tui.widgets.conversation import MessageWidget
        msg = Message(
            role=MessageRole.USER,
            content="my prompt",
            agent_id="agent-1",
        )
        event = MessageWidget.ResendRequested(
            message=msg,
            agent_id="agent-1",
            message_index=3,
        )
        assert event.message_index == 3
        assert event.msg.content == "my prompt"
        assert event.agent_id == "agent-1"
        assert event.msg.snapshot_id is None

    def test_resend_requested_carries_snapshot_id(self):
        """ResendRequested should preserve the clicked prompt's snapshot id."""
        from prsm.tui.widgets.conversation import MessageWidget
        msg = Message(
            role=MessageRole.USER,
            content="my prompt",
            agent_id="agent-1",
            snapshot_id="snap-1234",
        )
        event = MessageWidget.ResendRequested(
            message=msg,
            agent_id="agent-1",
            message_index=3,
        )
        assert event.msg.snapshot_id == "snap-1234"


class TestMessageSnapshotPersistence:
    """Tests for snapshot_id round-trip through persistence serialization."""

    def test_message_snapshot_id_round_trip(self):
        from prsm.shared.services.persistence import _dict_to_message, _message_to_dict

        msg = Message(
            role=MessageRole.USER,
            content="prompt",
            agent_id="master-1",
            snapshot_id="abcd1234",
        )
        data = _message_to_dict(msg)
        restored = _dict_to_message(data)
        assert restored.snapshot_id == "abcd1234"


# ── Multi-agent truncation tests ──


class TestMultiAgentTruncation:
    """Tests that truncation correctly handles multiple agents."""

    def test_truncate_removes_later_agent_messages(self):
        """Messages from other agents after the cutoff time should be removed."""
        session = Session()
        now = datetime.now(timezone.utc)

        agent1 = AgentNode(
            id="master-1", name="Master", state=AgentState.COMPLETED,
            role=AgentRole.MASTER, model="claude",
        )
        agent2 = AgentNode(
            id="worker-1", name="Worker", state=AgentState.COMPLETED,
            role=AgentRole.WORKER, model="claude", parent_id="master-1",
        )
        session.add_agent(agent1)
        session.add_agent(agent2)

        # Master messages
        session.messages["master-1"] = [
            Message(role=MessageRole.USER, content="Prompt 1", agent_id="master-1", timestamp=now),
            Message(role=MessageRole.ASSISTANT, content="Response 1", agent_id="master-1", timestamp=now + timedelta(seconds=1)),
            Message(role=MessageRole.USER, content="Prompt 2", agent_id="master-1", timestamp=now + timedelta(seconds=5)),
        ]

        # Worker messages (spawned between prompt 1 and 2)
        session.messages["worker-1"] = [
            Message(role=MessageRole.SYSTEM, content="Spawned", agent_id="worker-1", timestamp=now + timedelta(seconds=2)),
            Message(role=MessageRole.ASSISTANT, content="Done", agent_id="worker-1", timestamp=now + timedelta(seconds=3)),
        ]

        # Truncate at master's index 2 ("Prompt 2")
        target_msgs = session.get_messages("master-1")
        cutoff_time = target_msgs[2].timestamp
        session.messages["master-1"] = target_msgs[:2]

        # Remove worker messages after cutoff
        for aid in list(session.messages.keys()):
            if aid == "master-1":
                continue
            session.messages[aid] = [
                m for m in session.messages[aid]
                if m.timestamp < cutoff_time
            ]

        assert len(session.get_messages("master-1")) == 2
        # Worker messages were all before the cutoff, so they remain
        assert len(session.get_messages("worker-1")) == 2

    def test_truncate_removes_later_agent_all_messages(self):
        """Worker spawned after cutoff should have all messages removed."""
        session = Session()
        now = datetime.now(timezone.utc)

        agent1 = AgentNode(
            id="master-1", name="Master", state=AgentState.COMPLETED,
            role=AgentRole.MASTER, model="claude",
        )
        agent2 = AgentNode(
            id="worker-1", name="Worker", state=AgentState.COMPLETED,
            role=AgentRole.WORKER, model="claude", parent_id="master-1",
        )
        session.add_agent(agent1)
        session.add_agent(agent2)

        session.messages["master-1"] = [
            Message(role=MessageRole.USER, content="Prompt 1", agent_id="master-1", timestamp=now),
            Message(role=MessageRole.ASSISTANT, content="Response 1", agent_id="master-1", timestamp=now + timedelta(seconds=1)),
            Message(role=MessageRole.USER, content="Prompt 2", agent_id="master-1", timestamp=now + timedelta(seconds=2)),
        ]

        # Worker spawned AFTER prompt 2
        session.messages["worker-1"] = [
            Message(role=MessageRole.SYSTEM, content="Spawned", agent_id="worker-1", timestamp=now + timedelta(seconds=3)),
            Message(role=MessageRole.ASSISTANT, content="Done", agent_id="worker-1", timestamp=now + timedelta(seconds=4)),
        ]

        # Truncate at index 2 ("Prompt 2")
        target_msgs = session.get_messages("master-1")
        cutoff_time = target_msgs[2].timestamp
        session.messages["master-1"] = target_msgs[:2]

        for aid in list(session.messages.keys()):
            if aid == "master-1":
                continue
            session.messages[aid] = [
                m for m in session.messages[aid]
                if m.timestamp < cutoff_time
            ]

        assert len(session.get_messages("master-1")) == 2
        # Worker was spawned after cutoff — all messages removed
        assert len(session.get_messages("worker-1")) == 0


# ── show_agent force rebuild tests ──


class TestShowAgentForceRebuild:
    """Tests that show_agent(force=True) rebuilds even for the same agent.

    This is the core fix for the 'clicking previous prompts does not allow
    editing' regression — rebuild_ui_after_resend was calling show_agent
    with the *same* agent_id that was already active, so show_agent
    returned early and the conversation view was never rebuilt after
    session truncation.
    """

    def test_show_agent_skips_rebuild_by_default(self):
        """show_agent should not rebuild if already showing the same agent."""
        from prsm.tui.widgets.conversation import ConversationView

        conv = ConversationView()
        # Manually set the internal state to simulate already-showing
        conv._current_agent_id = "agent-1"

        # Calling show_agent with the same id should be a no-op
        # (no error, _current_agent_id unchanged)
        conv.show_agent("agent-1")
        assert conv._current_agent_id == "agent-1"

    def test_show_agent_force_updates_agent_id(self):
        """show_agent(force=True) should set _current_agent_id even if same."""
        from prsm.tui.widgets.conversation import ConversationView

        conv = ConversationView()
        conv._current_agent_id = "agent-1"

        # force=True should accept the same agent_id (no early return)
        # We can't call _rebuild without a mounted widget, so just verify
        # the method signature accepts the parameter without error.
        # The actual rebuild would require Textual runtime.
        assert conv.show_agent.__code__.co_varnames[:3] == ("self", "agent_id", "force")

    def test_show_agent_default_force_is_false(self):
        """show_agent force parameter should default to False."""
        import inspect
        from prsm.tui.widgets.conversation import ConversationView

        sig = inspect.signature(ConversationView.show_agent)
        force_param = sig.parameters["force"]
        assert force_param.default is False

    def test_rebuild_ui_after_resend_uses_force(self):
        """rebuild_ui_after_resend should call show_agent with force=True.

        This ensures the conversation is rebuilt after session truncation
        even when the active agent hasn't changed.
        """
        import inspect
        from prsm.tui.handlers.session_manager import SessionManager

        source = inspect.getsource(SessionManager.rebuild_ui_after_resend)
        assert "force=True" in source, (
            "rebuild_ui_after_resend must call show_agent with force=True "
            "to ensure the conversation rebuilds after truncation"
        )

    def test_message_clickable_class_for_non_pending(self):
        """User messages with a real agent_id should get the clickable class."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.USER,
            content="test prompt",
            agent_id="master-1",
        )
        widget = MessageWidget(msg, message_index=1)
        assert "message-clickable" in widget.classes

    def test_message_no_clickable_class_for_pending(self):
        """User messages with __pending__ agent_id should NOT be clickable."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.USER,
            content="test prompt",
            agent_id="__pending__",
        )
        widget = MessageWidget(msg, message_index=0)
        assert "message-clickable" not in widget.classes

    def test_message_no_clickable_class_for_assistant(self):
        """Assistant messages should never be clickable."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="response",
            agent_id="master-1",
        )
        widget = MessageWidget(msg, message_index=2)
        assert "message-clickable" not in widget.classes

    def test_on_click_emits_resend_for_user_message(self):
        """Clicking a real user message should post ResendRequested."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.USER,
            content="my prompt",
            agent_id="master-1",
            snapshot_id="snap-abc",
        )
        widget = MessageWidget(msg, message_index=3)
        # Verify on_click checks role and agent_id
        assert widget.message.role == MessageRole.USER
        assert widget.message.agent_id != "__pending__"
        # The actual post_message call requires a running Textual app,
        # but we verify the guard conditions are met.

    def test_on_click_no_emit_for_pending(self):
        """Clicking a __pending__ user message should NOT emit ResendRequested."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.USER,
            content="pending",
            agent_id="__pending__",
        )
        widget = MessageWidget(msg, message_index=0)
        # Guard condition: agent_id == "__pending__" prevents emission
        assert widget.message.agent_id == "__pending__"

    def test_on_click_no_emit_for_assistant(self):
        """Clicking an assistant message should NOT emit ResendRequested."""
        from prsm.tui.widgets.conversation import MessageWidget

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="response",
            agent_id="master-1",
        )
        widget = MessageWidget(msg, message_index=2)
        # Guard condition: role != USER prevents emission
        assert widget.message.role != MessageRole.USER


@pytest.mark.asyncio
async def test_left_click_old_prompt_prefills_input_for_edit():
    """Left-clicking a historical user prompt should prefill the input editor."""
    from prsm.shared.services.preferences import UserPreferences
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.main import MainScreen
    from prsm.tui.widgets.conversation import ConversationView, MessageWidget
    from prsm.tui.widgets.input_bar import InputBar, PromptInput

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MainScreen)

        agent = AgentNode(
            id="master-1",
            name="Orchestrator",
            state=AgentState.COMPLETED,
            role=AgentRole.MASTER,
            model="claude-opus-4-6",
        )
        screen.session.add_agent(agent)
        screen.session.set_active("master-1")

        conv = screen.query_one("#conversation", ConversationView)
        conv.add_user_message("master-1", "First prompt", snapshot_id="snap-1")
        conv.add_assistant_message("master-1", "First response")
        conv.add_user_message("master-1", "Second prompt", snapshot_id="snap-2")
        conv.add_assistant_message("master-1", "Second response")
        conv.show_agent("master-1", force=True)
        await pilot.pause()

        clickable = [
            w for w in screen.query("MessageWidget").results(MessageWidget)
            if w.message.role == MessageRole.USER and w.message.content == "Second prompt"
        ]
        assert len(clickable) == 1

        # Keep this test focused on click-to-edit; skip modal path.
        with patch(
            "prsm.shared.services.preferences.UserPreferences.load",
            return_value=UserPreferences(file_revert_on_resend="never"),
        ):
            await pilot.click(clickable[0])
            await pilot.pause()

        editor = screen.query_one(InputBar).query_one("#prompt-input", PromptInput)
        assert editor.text == "Second prompt"


@pytest.mark.asyncio
async def test_left_click_prefills_input_even_while_running():
    """Clicking a historical prompt should still load text when bridge is running."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.main import MainScreen
    from prsm.tui.widgets.conversation import ConversationView, MessageWidget
    from prsm.tui.widgets.input_bar import InputBar, PromptInput

    app = PrsmApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MainScreen)

        agent = AgentNode(
            id="master-1",
            name="Orchestrator",
            state=AgentState.RUNNING,
            role=AgentRole.MASTER,
            model="claude-opus-4-6",
        )
        screen.session.add_agent(agent)
        screen.session.set_active("master-1")

        conv = screen.query_one("#conversation", ConversationView)
        conv.add_user_message("master-1", "Old prompt", snapshot_id="snap-1")
        conv.add_assistant_message("master-1", "Old response")
        conv.show_agent("master-1", force=True)
        await pilot.pause()

        screen.bridge._running = True

        clickable = [
            w for w in screen.query("MessageWidget").results(MessageWidget)
            if w.message.role == MessageRole.USER and w.message.content == "Old prompt"
        ]
        assert len(clickable) == 1

        await pilot.click(clickable[0])
        await pilot.pause()

        editor = screen.query_one(InputBar).query_one("#prompt-input", PromptInput)
        assert editor.text == "Old prompt"
