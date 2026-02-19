from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from prsm.adapters.file_tracker import FileChangeRecord
from prsm.shared.models.message import MessageRole, ToolCall
from prsm.shared.models.session import Session
from prsm.shared.models.session import WorktreeMetadata
from prsm.vscode.server import PrsmServer, SessionState


def _build_server(tmp_path: Path) -> PrsmServer:
    with patch("prsm.vscode.server.SessionPersistence") as persistence_cls:
        persistence = persistence_cls.return_value
        persistence.project_dir = tmp_path / ".prsm_project"
        persistence.list_sessions_by_mtime.return_value = []
        persistence.list_sessions_detailed.return_value = []
        return PrsmServer(cwd=str(tmp_path), model="claude-opus-4-6")


def _build_state(workspace_root: Path) -> SessionState:
    session = Session(
        agents={},
        messages={},
        active_agent_id=None,
        name="Fallback Test",
        created_at=None,
        forked_from=None,
        worktree=None,
    )
    state = SessionState(
        session_id="session-1",
        name="Fallback Test",
        summary=None,
        project_id="project-1",
        bridge=MagicMock(),
        session=session,
    )
    state._workspace_root = str(workspace_root)
    return state


def test_parse_working_tree_patch_extracts_changes_and_ranges(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    patch_text = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+line2b\n"
        "+line3\n"
        "diff --git a/src/new.txt b/src/new.txt\n"
        "--- /dev/null\n"
        "+++ b/src/new.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+hello\n"
        "+world\n"
        "diff --git a/src/old.txt b/src/old.txt\n"
        "--- a/src/old.txt\n"
        "+++ /dev/null\n"
        "@@ -3,2 +0,0 @@\n"
        "-gone\n"
        "-gone2\n"
    )

    parsed = server._parse_working_tree_patch(patch_text)
    assert len(parsed) == 3

    assert parsed[0]["file_path"] == "src/a.py"
    assert parsed[0]["change_type"] == "modify"
    assert parsed[0]["removed_ranges"] == [{"startLine": 0, "endLine": 1}]
    assert parsed[0]["added_ranges"] == [{"startLine": 0, "endLine": 2}]

    assert parsed[1]["file_path"] == "src/new.txt"
    assert parsed[1]["change_type"] == "create"
    assert parsed[1]["added_ranges"] == [{"startLine": 0, "endLine": 1}]

    assert parsed[2]["file_path"] == "src/old.txt"
    assert parsed[2]["change_type"] == "delete"
    assert parsed[2]["removed_ranges"] == [{"startLine": 2, "endLine": 3}]


def test_handle_accept_change_is_noop_on_disk(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_target = workspace_root / "src/accept-endpoint.py"
    workspace_target.parent.mkdir(parents=True, exist_ok=True)
    workspace_target.write_text("print('changed')\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    server._sessions[state.session_id] = state

    record = FileChangeRecord(
        file_path=str(workspace_target.resolve()),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-accept-endpoint",
        tool_name="Write",
        message_index=0,
        old_content="print('old')\n",
        pre_tool_content="print('old')\n",
        new_content="print('changed')\n",
        status="pending",
    )
    state.file_tracker.file_changes[str(workspace_target.resolve())] = [record]

    request = MagicMock()
    request.match_info = {
        "id": state.session_id,
        "tool_call_id": "tool-accept-endpoint",
    }

    response = asyncio.run(server._handle_accept_change(request))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["status"] == "accepted"
    # Accept is a no-op on disk — file content stays as-is
    assert workspace_target.read_text(encoding="utf-8") == "print('changed')\n"
    # Record kept with status="accepted" (prevents patch-fallback regeneration)
    found = server._find_file_change_record(state, "tool-accept-endpoint")
    assert found is not None
    assert found[2].status == "accepted"


def test_display_file_path_for_session_id_maps_legacy_tmp_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    server = _build_server(tmp_path)
    session_id = "session-legacy"

    server._known_projects["project-legacy"] = {
        "project_id": "project-legacy",
        "label": "Legacy",
        "cwd": str(workspace_root),
    }
    server._session_index[session_id] = {
        "session_id": session_id,
        "project_id": "project-legacy",
    }

    mapped = server._display_file_path_for_session_id(
        session_id,
        f"/tmp/{session_id}/src/mapped.py",
    )
    assert mapped == str((workspace_root / "src/mapped.py").resolve())


def test_load_file_changes_normalizes_tmp_paths(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    session_id = "session-load-1"
    worktree_root = Path("/tmp") / session_id

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state.session_id = session_id
    state.session.worktree = WorktreeMetadata(root=str(worktree_root))

    changes_dir = tmp_path / "sessions" / session_id / "file-changes"
    changes_dir.mkdir(parents=True, exist_ok=True)
    (changes_dir / "tool-1.json").write_text(
        json.dumps(
            {
                "file_path": str(worktree_root / "src/from-load.py"),
                "agent_id": "agent-1",
                "change_type": "modify",
                "tool_call_id": "tool-1",
                "tool_name": "Write",
                "message_index": 0,
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )

    with patch.object(server, "_file_changes_dir", return_value=changes_dir):
        server._load_file_changes(state)

    expected = str((workspace_root / "src/from-load.py").resolve())
    assert expected in state.file_tracker.file_changes
    assert state.file_tracker.file_changes[expected][0].tool_call_id == "tool-1"


def test_get_file_changes_cold_session_maps_tmp_paths(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    session_id = "session-cold-1"

    server = _build_server(tmp_path)
    server._known_projects["project-cold"] = {
        "project_id": "project-cold",
        "label": "Cold",
        "cwd": str(workspace_root),
    }
    server._session_index[session_id] = {
        "session_id": session_id,
        "project_id": "project-cold",
    }

    changes_dir = tmp_path / ".prsm_project" / "sessions" / session_id / "file-changes"
    changes_dir.mkdir(parents=True, exist_ok=True)
    (changes_dir / "tool-1.json").write_text(
        json.dumps(
            {
                "file_path": f"/tmp/{session_id}/src/from-cold.py",
                "agent_id": "agent-1",
                "change_type": "modify",
                "tool_call_id": "tool-1",
                "tool_name": "Write",
                "message_index": 0,
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )

    request = MagicMock()
    request.match_info = {"id": session_id}
    response = asyncio.run(server._handle_get_file_changes(request))
    payload = json.loads(response.body.decode("utf-8"))
    expected = str((workspace_root / "src/from-cold.py").resolve())
    assert expected in payload["file_changes"]
    assert payload["file_changes"][expected][0]["file_path"] == expected
    persisted = json.loads((changes_dir / "tool-1.json").read_text(encoding="utf-8"))
    assert persisted["file_path"] == expected


def test_get_file_changes_does_not_mutate_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_target = workspace_root / "src/fallback.py"
    workspace_target.parent.mkdir(parents=True, exist_ok=True)
    workspace_target.write_text("print('current')\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    server._sessions[state.session_id] = state

    # Add a tracked change so GET has something to return
    record = FileChangeRecord(
        file_path=str(workspace_target.resolve()),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-no-mutate",
        tool_name="Write",
        message_index=0,
        old_content="print('old')\n",
        new_content="print('current')\n",
        pre_tool_content="print('old')\n",
        status="pending",
    )
    state.file_tracker.file_changes[str(workspace_target.resolve())] = [record]

    request = MagicMock()
    request.match_info = {"id": state.session_id}
    asyncio.run(server._handle_get_file_changes(request))

    # File content should not be mutated by reading file changes
    assert workspace_target.read_text(encoding="utf-8") == "print('current')\n"


# ── Per-session worktree tests ──


def _git_init(path: Path) -> None:
    """Initialize a git repo in *path* with an initial commit."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(path), capture_output=True, check=True)
    (path / ".gitkeep").write_text("")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), capture_output=True, check=True)


def test_worktree_to_workspace_path_maps_correctly(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path / "workspace")
    state._worktree_path = "/tmp/prsm-wt-test-session"

    result = server._worktree_to_workspace_path(state, "/tmp/prsm-wt-test-session/src/file.py")
    expected = str((tmp_path / "workspace" / "src/file.py").resolve())
    assert result == expected


def test_worktree_to_workspace_path_noop_without_worktree(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path / "workspace")
    # No worktree set

    path = "/some/absolute/path.py"
    assert server._worktree_to_workspace_path(state, path) == path


def test_workspace_to_worktree_path_maps_correctly(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    state = _build_state(workspace_root)
    state._worktree_path = "/tmp/prsm-wt-test-session"

    ws_path = str((workspace_root / "src/file.py").resolve())
    result = server._workspace_to_worktree_path(state, ws_path)
    assert result == str(Path("/tmp/prsm-wt-test-session/src/file.py").resolve())


def test_agent_cwd_for_state_returns_worktree_when_set(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path / "workspace")
    state._worktree_path = "/tmp/prsm-wt-test-session"

    assert server._agent_cwd_for_state(state) == Path("/tmp/prsm-wt-test-session")


def test_agent_cwd_for_state_returns_workspace_without_worktree(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    state = _build_state(workspace_root)

    assert server._agent_cwd_for_state(state) == workspace_root.resolve()


def test_setup_session_worktree_creates_worktree(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)

    server._setup_session_worktree(state)

    assert state._worktree_path is not None
    assert Path(state._worktree_path).exists()
    # Verify it's a valid git worktree
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=state._worktree_path,
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    # Cleanup
    server._cleanup_session_worktree(state)
    assert state._worktree_path is None


def test_setup_session_worktree_auto_git_init(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    # No git init — should auto-init

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)

    server._setup_session_worktree(state)

    # After auto-init + worktree creation, the workspace should be a git repo
    from prsm.shared.services.project import ProjectManager
    assert ProjectManager.is_git_repo(workspace_root)

    # But worktree creation may fail (no commits yet) — that's OK
    # The important thing is git init happened
    if state._worktree_path:
        server._cleanup_session_worktree(state)


def test_sync_file_to_workspace_copies_on_accept(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    # Create a file in the worktree
    wt_file = worktree_root / "src" / "hello.py"
    wt_file.parent.mkdir(parents=True)
    wt_file.write_text("print('hello from worktree')\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)

    record = FileChangeRecord(
        file_path=str(wt_file),
        agent_id="agent-1",
        change_type="create",
        tool_call_id="tool-sync-1",
        tool_name="Write",
        message_index=0,
        old_content=None,
        new_content="print('hello from worktree')\n",
        pre_tool_content=None,
        status="pending",
    )

    # Sync to workspace
    server._sync_file_to_workspace(state, record)

    ws_file = workspace_root / "src" / "hello.py"
    assert ws_file.exists()
    assert ws_file.read_text(encoding="utf-8") == "print('hello from worktree')\n"


def test_sync_file_to_workspace_noop_without_worktree(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target = workspace_root / "existing.txt"
    target.write_text("original", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    # No worktree set

    record = FileChangeRecord(
        file_path=str(target),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-noop-1",
        tool_name="Write",
        message_index=0,
        old_content="old",
        new_content="original",
        pre_tool_content="old",
        status="pending",
    )

    # Should be a no-op
    server._sync_file_to_workspace(state, record)
    assert target.read_text(encoding="utf-8") == "original"


def test_accept_change_syncs_worktree_to_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    # File exists in worktree (agent wrote it)
    wt_file = worktree_root / "synced.txt"
    wt_file.write_text("worktree content\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)
    server._sessions[state.session_id] = state

    record = FileChangeRecord(
        file_path=str(wt_file),
        agent_id="agent-1",
        change_type="create",
        tool_call_id="tool-accept-wt",
        tool_name="Write",
        message_index=0,
        old_content=None,
        new_content="worktree content\n",
        pre_tool_content=None,
        status="pending",
    )
    state.file_tracker.file_changes[str(wt_file)] = [record]

    request = MagicMock()
    request.match_info = {"id": state.session_id, "tool_call_id": "tool-accept-wt"}

    response = asyncio.run(server._handle_accept_change(request))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["status"] == "accepted"
    assert record.status == "accepted"
    # File should now exist in workspace
    ws_file = workspace_root / "synced.txt"
    assert ws_file.exists()
    assert ws_file.read_text(encoding="utf-8") == "worktree content\n"


def test_reject_change_reverts_in_worktree_not_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    # Original file in worktree was modified by agent
    wt_file = worktree_root / "reverted.txt"
    wt_file.write_text("agent modified\n", encoding="utf-8")

    # Workspace should have no file (or its own version)
    ws_file = workspace_root / "reverted.txt"
    ws_file.write_text("workspace version\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)
    server._sessions[state.session_id] = state

    record = FileChangeRecord(
        file_path=str(wt_file),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-reject-wt",
        tool_name="Write",
        message_index=0,
        old_content="original in worktree\n",
        new_content="agent modified\n",
        pre_tool_content="original in worktree\n",
        status="pending",
    )
    state.file_tracker.file_changes[str(wt_file)] = [record]

    request = MagicMock()
    request.match_info = {"id": state.session_id, "tool_call_id": "tool-reject-wt"}

    response = asyncio.run(server._handle_reject_change(request))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["status"] == "rejected"
    assert record.status == "rejected"
    # Worktree file should be reverted
    assert wt_file.read_text(encoding="utf-8") == "original in worktree\n"
    # Workspace file should be UNTOUCHED
    assert ws_file.read_text(encoding="utf-8") == "workspace version\n"


def test_display_file_path_maps_worktree_to_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = "/tmp/prsm-wt-test-display"

    display = server._display_file_path_for_state(
        state, "/tmp/prsm-wt-test-display/src/mapped.py"
    )
    expected = str((workspace_root / "src/mapped.py").resolve())
    assert display == expected


# ── Lazy worktree lifecycle tests ──


def test_has_pending_file_changes_true_when_pending(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path)
    record = FileChangeRecord(
        file_path=str(tmp_path / "a.txt"),
        agent_id="a1", change_type="modify", tool_call_id="tc1",
        tool_name="Write", message_index=0,
        old_content="old", new_content="new", pre_tool_content="old",
        status="pending",
    )
    state.file_tracker.file_changes[str(tmp_path / "a.txt")] = [record]
    assert server._has_pending_file_changes(state) is True


def test_has_pending_file_changes_false_when_all_accepted(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path)
    record = FileChangeRecord(
        file_path=str(tmp_path / "a.txt"),
        agent_id="a1", change_type="modify", tool_call_id="tc1",
        tool_name="Write", message_index=0,
        old_content="old", new_content="new", pre_tool_content="old",
        status="accepted",
    )
    state.file_tracker.file_changes[str(tmp_path / "a.txt")] = [record]
    assert server._has_pending_file_changes(state) is False


def test_has_pending_file_changes_false_when_empty(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path)
    assert server._has_pending_file_changes(state) is False


def test_maybe_cleanup_removes_worktree_when_no_pending(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)

    # All records are accepted — no pending
    record = FileChangeRecord(
        file_path=str(worktree_root / "a.txt"),
        agent_id="a1", change_type="modify", tool_call_id="tc1",
        tool_name="Write", message_index=0,
        old_content="old", new_content="new", pre_tool_content="old",
        status="accepted",
    )
    state.file_tracker.file_changes[str(worktree_root / "a.txt")] = [record]

    with patch.object(server, "_cleanup_session_worktree") as mock_cleanup:
        server._maybe_cleanup_empty_worktree(state)
        mock_cleanup.assert_called_once_with(state)


def test_maybe_cleanup_keeps_worktree_when_pending(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)

    record = FileChangeRecord(
        file_path=str(worktree_root / "a.txt"),
        agent_id="a1", change_type="modify", tool_call_id="tc1",
        tool_name="Write", message_index=0,
        old_content="old", new_content="new", pre_tool_content="old",
        status="pending",
    )
    state.file_tracker.file_changes[str(worktree_root / "a.txt")] = [record]

    with patch.object(server, "_cleanup_session_worktree") as mock_cleanup:
        server._maybe_cleanup_empty_worktree(state)
        mock_cleanup.assert_not_called()


def test_maybe_cleanup_noop_without_worktree(tmp_path: Path) -> None:
    server = _build_server(tmp_path)
    state = _build_state(tmp_path)
    # No worktree set — should be a no-op
    with patch.object(server, "_cleanup_session_worktree") as mock_cleanup:
        server._maybe_cleanup_empty_worktree(state)
        mock_cleanup.assert_not_called()


def test_ensure_session_worktree_creates_when_missing(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = None

    with patch.object(server, "_setup_session_worktree") as mock_setup, \
         patch.object(server, "_configure_bridge") as mock_configure:
        # Simulate _setup_session_worktree setting _worktree_path
        def set_wt(s: SessionState) -> None:
            s._worktree_path = "/tmp/prsm-wt-test"
        mock_setup.side_effect = set_wt

        server._ensure_session_worktree(state)

        mock_setup.assert_called_once_with(state)
        mock_configure.assert_called_once()


def test_ensure_session_worktree_noop_when_exists(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = "/tmp/prsm-wt-already-exists"

    with patch.object(server, "_setup_session_worktree") as mock_setup:
        server._ensure_session_worktree(state)
        mock_setup.assert_not_called()


def test_accept_all_cleans_up_worktree(tmp_path: Path) -> None:
    """After accepting all changes, the worktree should be cleaned up."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    wt_file = worktree_root / "cleaned.txt"
    wt_file.write_text("content\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)
    server._sessions[state.session_id] = state

    record = FileChangeRecord(
        file_path=str(wt_file),
        agent_id="agent-1", change_type="create", tool_call_id="tool-cleanup",
        tool_name="Write", message_index=0,
        old_content=None, new_content="content\n", pre_tool_content=None,
        status="pending",
    )
    state.file_tracker.file_changes[str(wt_file)] = [record]

    request = MagicMock()
    request.match_info = {"id": state.session_id}

    with patch.object(server, "_cleanup_session_worktree") as mock_cleanup:
        asyncio.run(server._handle_accept_all_changes(request))
        mock_cleanup.assert_called_once_with(state)


# ── Worktree file change fallback tests ──


def test_worktree_fallback_detects_edit_via_git_head(tmp_path: Path) -> None:
    """When snapshot-based tracking misses a change (event timing race),
    the worktree fallback detects it by comparing against git HEAD."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    # Create a file in the repo and commit it.
    src_file = workspace_root / "src" / "app.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text("original content\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_root), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add app"], cwd=str(workspace_root), capture_output=True, check=True)

    # Create a real worktree
    wt_path = str(tmp_path / "worktree")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    # Simulate agent editing the file in the worktree.
    wt_file = Path(wt_path) / "src" / "app.py"
    wt_file.write_text("modified content\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = wt_path
    server._sessions[state.session_id] = state

    # Add a tool call to session messages (as _on_tool_call_started would).
    import json as _json
    state.session.add_message(
        "agent-1",
        MessageRole.TOOL,
        "",
        tool_calls=[ToolCall(
            id="tc-edit-1",
            name="Edit",
            arguments=_json.dumps({
                "file_path": str(wt_file),
                "old_string": "original content\n",
                "new_string": "modified content\n",
            }),
        )],
    )

    records = server._worktree_file_change_fallback(
        state, "tc-edit-1", "agent-1", 0,
    )

    assert len(records) == 1
    assert records[0].change_type == "modify"
    assert records[0].old_content == "original content\n"
    assert records[0].new_content == "modified content\n"
    assert records[0].tool_call_id == "tc-edit-1"

    # Cleanup worktree
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt_path],
        cwd=str(workspace_root), capture_output=True,
    )


def test_worktree_fallback_skips_already_tracked(tmp_path: Path) -> None:
    """Fallback doesn't create duplicate records for already-tracked files."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    wt_file = worktree_root / "tracked.txt"
    wt_file.write_text("new", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)

    # File already has a record.
    existing = FileChangeRecord(
        file_path=str(wt_file), agent_id="a1", change_type="modify",
        tool_call_id="tc-edit-1", tool_name="Edit", message_index=0,
        old_content="old", new_content="new", pre_tool_content="old",
        status="pending",
    )
    state.file_tracker.file_changes[str(wt_file)] = [existing]

    # Add tool call to session.
    import json as _json
    state.session.add_message(
        "a1", MessageRole.TOOL, "",
        tool_calls=[ToolCall(
            id="tc-edit-1",
            name="Edit",
            arguments=_json.dumps({"file_path": str(wt_file)}),
        )],
    )

    records = server._worktree_file_change_fallback(
        state, "tc-edit-1", "a1", 0,
    )

    assert len(records) == 0


def test_worktree_fallback_noop_without_worktree(tmp_path: Path) -> None:
    """Fallback returns empty when not in worktree mode."""
    server = _build_server(tmp_path)
    state = _build_state(tmp_path)
    # No worktree set.
    records = server._worktree_file_change_fallback(state, "tc-1", "a1", 0)
    assert records == []


def test_worktree_diff_fallback_detects_bash_changes(tmp_path: Path) -> None:
    """The diff fallback picks up changes from Bash commands in the worktree."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    # Commit a file.
    (workspace_root / "data.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_root), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add data"], cwd=str(workspace_root), capture_output=True, check=True)

    # Create worktree.
    wt_path = str(tmp_path / "worktree")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    # Simulate Bash modifying a file in the worktree.
    (Path(wt_path) / "data.txt").write_text("v2\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = wt_path

    records = server._worktree_diff_fallback(state, "tc-bash-1", "a1", 0, "Bash")

    assert len(records) == 1
    assert records[0].change_type == "modify"
    assert records[0].old_content == "v1\n"
    assert records[0].new_content == "v2\n"

    # Cleanup
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt_path],
        cwd=str(workspace_root), capture_output=True,
    )


def test_engine_finished_prunes_empty_worktree(tmp_path: Path) -> None:
    """When a run finishes with no pending changes and no queued prompts,
    the worktree should be cleaned up automatically."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    wt_path = str(tmp_path / "worktree")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = wt_path
    # No pending file changes — worktree should be pruned.

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server._on_engine_finished(state))
    finally:
        loop.close()

    # Worktree should have been removed.
    assert state._worktree_path is None


def test_engine_finished_keeps_worktree_with_pending_changes(tmp_path: Path) -> None:
    """When a run finishes but there are pending file changes,
    the worktree should NOT be cleaned up."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    wt_path = str(tmp_path / "worktree")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = wt_path

    # Add a pending file change record.
    record = FileChangeRecord(
        file_path=str(Path(wt_path) / "test.txt"),
        agent_id="agent-1",
        change_type="create",
        tool_call_id="tc-1",
        tool_name="Write",
        message_index=0,
        old_content=None,
        new_content="hello",
        pre_tool_content=None,
        added_ranges=[],
        removed_ranges=[],
        timestamp="2025-01-01T00:00:00",
    )
    state.file_tracker.file_changes.setdefault(record.file_path, []).append(record)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server._on_engine_finished(state))
    finally:
        loop.close()

    # Worktree should still exist because of pending changes.
    assert state._worktree_path == wt_path

    # Cleanup
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt_path],
        cwd=str(workspace_root), capture_output=True,
    )


def test_reconnect_prunes_empty_worktree(tmp_path: Path) -> None:
    """When reconnecting to a worktree that has no changes (clean diff),
    it should be pruned immediately rather than kept around."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    wt_path = str(tmp_path / f"prsm-wt-session-1")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state.session.worktree = WorktreeMetadata(root=wt_path)
    # _worktree_path is None — simulates a server restart.

    server._reconnect_session_worktree(state)

    # The worktree had no changes (clean diff), so it should be pruned.
    assert state._worktree_path is None


def test_reconnect_keeps_worktree_with_changes(tmp_path: Path) -> None:
    """When reconnecting to a worktree that has actual changes,
    it should be kept and changes reconciled."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _git_init(workspace_root)

    # Commit a file.
    (workspace_root / "file.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_root), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add file"], cwd=str(workspace_root), capture_output=True, check=True)

    wt_path = str(tmp_path / f"prsm-wt-session-1")
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt_path],
        cwd=str(workspace_root), capture_output=True, check=True,
    )

    # Modify file in worktree — creates a diff.
    (Path(wt_path) / "file.txt").write_text("modified\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state.session.worktree = WorktreeMetadata(root=wt_path)

    server._reconnect_session_worktree(state)

    # Worktree has changes — should be kept.
    assert state._worktree_path == wt_path
    # Should have reconciled the change.
    all_records = [r for records in state.file_tracker.file_changes.values() for r in records]
    assert len(all_records) >= 1

    # Cleanup
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt_path],
        cwd=str(workspace_root), capture_output=True,
    )


def test_prune_orphaned_worktrees(tmp_path: Path) -> None:
    """Startup pruning removes worktree dirs that don't belong to any known session."""
    server = _build_server(tmp_path)

    # Create some fake worktree dirs in /tmp.
    orphan_dir = tmp_path / "prsm-wt-orphan-session-xyz"
    orphan_dir.mkdir()
    (orphan_dir / "marker.txt").write_text("stale")

    # Monkey-patch glob to look in tmp_path instead of /tmp.
    import glob as glob_mod
    original_glob = glob_mod.glob

    def patched_glob(pattern):
        if pattern == "/tmp/prsm-wt-*":
            return [str(orphan_dir)]
        return original_glob(pattern)

    with patch("glob.glob", side_effect=patched_glob):
        server._prune_orphaned_worktrees()

    # The orphan should be gone.
    assert not orphan_dir.exists()


# ---------------------------------------------------------------------------
# Tests: dual-path handling (record stores workspace path, worktree active)
# ---------------------------------------------------------------------------


def test_sync_file_to_workspace_with_workspace_path_record(tmp_path: Path) -> None:
    """Accept works even when record.file_path is a workspace path (not worktree)."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    # Agent wrote in worktree
    wt_file = worktree_root / "src" / "app.py"
    wt_file.parent.mkdir(parents=True, exist_ok=True)
    wt_file.write_text("new content from agent\n", encoding="utf-8")

    # Workspace has old content
    ws_file = workspace_root / "src" / "app.py"
    ws_file.parent.mkdir(parents=True, exist_ok=True)
    ws_file.write_text("old workspace content\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)

    # Record stores a WORKSPACE path (from snapshot-based tracking)
    record = FileChangeRecord(
        file_path=str(ws_file),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-ws-path",
        tool_name="Write",
        message_index=0,
        old_content="old workspace content\n",
        new_content="new content from agent\n",
        status="accepted",
    )

    server._sync_file_to_workspace(state, record)

    # Should have copied from worktree to workspace
    assert ws_file.read_text(encoding="utf-8") == "new content from agent\n"


def test_reject_with_workspace_path_record_reverts_worktree(tmp_path: Path) -> None:
    """Reject reverts the worktree file even when record.file_path is a workspace path."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()

    # Agent wrote in worktree
    wt_file = worktree_root / "reverted.txt"
    wt_file.write_text("agent modified\n", encoding="utf-8")

    # Workspace has its own version — should NOT be touched
    ws_file = workspace_root / "reverted.txt"
    ws_file.write_text("workspace version\n", encoding="utf-8")

    server = _build_server(tmp_path)
    state = _build_state(workspace_root)
    state._worktree_path = str(worktree_root)
    server._sessions[state.session_id] = state

    # Record stores a WORKSPACE path (from snapshot-based tracking)
    record = FileChangeRecord(
        file_path=str(ws_file),
        agent_id="agent-1",
        change_type="modify",
        tool_call_id="tool-reject-ws-path",
        tool_name="Write",
        message_index=0,
        old_content="original content\n",
        new_content="agent modified\n",
        pre_tool_content="original content\n",
        status="pending",
    )
    state.file_tracker.file_changes[str(ws_file)] = [record]

    request = MagicMock()
    request.match_info = {"id": state.session_id, "tool_call_id": "tool-reject-ws-path"}

    response = asyncio.run(server._handle_reject_change(request))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["status"] == "rejected"
    assert record.status == "rejected"
    # Worktree file should be reverted (mapped from workspace path)
    assert wt_file.read_text(encoding="utf-8") == "original content\n"
    # Workspace file should be UNTOUCHED
    assert ws_file.read_text(encoding="utf-8") == "workspace version\n"
