from __future__ import annotations

from prsm.adapters.file_tracker import FileChangeTracker, normalize_tool_name


def test_normalize_tool_name_aliases() -> None:
    assert normalize_tool_name("write_file") == "Write"
    assert normalize_tool_name("edit_file") == "Edit"
    assert normalize_tool_name("run_bash") == "Bash"
    assert normalize_tool_name("mcp__orchestrator__file_read") == "Read"


def test_capture_pre_tool_accepts_write_alias_and_tracks_change(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "created.txt"

    tracker.capture_pre_tool(
        "tool-1",
        "write_file",
        {"file_path": str(path), "content": "hello"},
    )

    path.write_text("hello", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-1")

    assert record is not None
    assert record.tool_name == "Write"
    assert record.change_type == "create"
    assert record.file_path == str(path)


def test_capture_pre_tool_accepts_prefixed_edit_alias_and_tracks_change(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "edit.txt"
    path.write_text("hello world", encoding="utf-8")

    tracker.capture_pre_tool(
        "tool-2",
        "mcp__orchestrator__edit_file",
        {
            "file_path": str(path),
            "old_string": "hello",
            "new_string": "goodbye",
        },
    )

    path.write_text("goodbye world", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-2")

    assert record is not None
    assert record.tool_name == "Edit"
    assert record.change_type == "modify"
    assert record.old_content == "hello"
    assert record.new_content == "goodbye"


def test_non_file_tool_alias_is_ignored(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "unchanged.txt"
    path.write_text("data", encoding="utf-8")

    tracker.capture_pre_tool(
        "tool-3",
        "run_bash",
        {"command": "cat unchanged.txt"},
    )
    assert tracker.track_change("agent-1", "tool-3") is None


def test_bash_raw_payload_never_becomes_fake_file_path(tmp_path) -> None:
    tracker = FileChangeTracker()
    raw = "python3 - <<'PY'\nprint('hello')\nPY"

    tracker.capture_pre_tool("tool-3b", "run_bash", raw, cwd=tmp_path)
    assert tracker.track_change("agent-1", "tool-3b") is None


def test_read_tool_never_tracks_file_changes(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "read-only.txt"
    path.write_text("data\n", encoding="utf-8")

    tracker.capture_pre_tool(
        "tool-read-1",
        "read_file",
        {"file_path": str(path)},
        cwd=tmp_path,
    )
    assert tracker.track_change("agent-1", "tool-read-1") is None
    assert tracker.track_changes("agent-1", "tool-read-1") == []


def test_capture_pre_tool_bash_sed_inplace_tracks_edit(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "edit.sh.txt"
    path.write_text("hello world", encoding="utf-8")

    tracker.capture_pre_tool(
        "tool-4",
        "run_bash",
        {"command": f"sed -i 's/hello/goodbye/g' {path}"},
    )

    path.write_text("goodbye world", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-4")

    assert record is not None
    assert record.tool_name == "Edit"
    assert record.file_path == str(path)


def test_capture_pre_tool_bash_redirect_tracks_write(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "redirect.txt"

    tracker.capture_pre_tool(
        "tool-5",
        "run_bash",
        {"command": f"echo hello > {path}"},
    )

    path.write_text("hello\n", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-5")

    assert record is not None
    assert record.tool_name == "Write"
    assert record.change_type == "create"
    assert record.file_path == str(path)


def test_capture_pre_tool_bash_raw_string_command_tracks_write(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "raw.txt"
    raw_command = f"echo hello > {path}"

    tracker.capture_pre_tool("tool-6", "run_bash", raw_command)
    path.write_text("hello\n", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-6")

    assert record is not None
    assert record.tool_name == "Write"
    assert record.file_path == str(path)


def test_capture_pre_tool_bash_cd_chain_tracks_edit(tmp_path) -> None:
    tracker = FileChangeTracker()
    sub = tmp_path / "sub"
    sub.mkdir()
    path = sub / "c.txt"
    path.write_text("hello world", encoding="utf-8")
    command = f"cd {sub} && sed -i 's/hello/goodbye/g' c.txt"

    tracker.capture_pre_tool("tool-7", "run_bash", {"command": command})
    path.write_text("goodbye world", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-7")

    assert record is not None
    assert record.tool_name == "Edit"
    assert record.file_path == str(path)


def test_capture_pre_tool_accepts_plain_path_string_for_write(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "plain-path.txt"

    tracker.capture_pre_tool("tool-8", "Write", str(path))
    path.write_text("hello", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-8")

    assert record is not None
    assert record.tool_name == "Write"
    assert record.file_path == str(path)
    assert record.change_type == "create"


def test_capture_pre_tool_accepts_json_string_path_for_edit(tmp_path) -> None:
    tracker = FileChangeTracker()
    path = tmp_path / "json-path.txt"
    path.write_text("hello world", encoding="utf-8")

    tracker.capture_pre_tool("tool-9", "Edit", f"\"{path}\"")
    path.write_text("goodbye world", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-9")

    assert record is not None
    assert record.tool_name == "Edit"
    assert record.file_path == str(path)


def test_capture_pre_tool_resolves_relative_path_with_cwd(tmp_path) -> None:
    tracker = FileChangeTracker()
    sub = tmp_path / "subdir"
    sub.mkdir()
    path = sub / "rel.txt"

    tracker.capture_pre_tool(
        "tool-10",
        "Write",
        {"file_path": "rel.txt", "cwd": str(sub)},
    )
    path.write_text("content", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-10")

    assert record is not None
    assert record.file_path == str(path)


def test_capture_pre_tool_resolves_relative_path_with_default_cwd_param(tmp_path) -> None:
    tracker = FileChangeTracker()
    sub = tmp_path / "session-worktree"
    sub.mkdir()
    path = sub / "from-default-cwd.txt"

    tracker.capture_pre_tool(
        "tool-11",
        "Write",
        {"file_path": "from-default-cwd.txt"},
        cwd=sub,
    )
    path.write_text("content", encoding="utf-8")
    record = tracker.track_change("agent-1", "tool-11")

    assert record is not None
    assert record.file_path == str(path)


def test_track_changes_detects_multiple_files_for_single_tool(tmp_path) -> None:
    tracker = FileChangeTracker()
    (tmp_path / "a.txt").write_text("old-a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("old-b\n", encoding="utf-8")

    tracker.capture_pre_tool(
        "tool-12",
        "run_bash",
        {"command": "echo change"},
        cwd=tmp_path,
    )

    (tmp_path / "a.txt").write_text("new-a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("new-b\n", encoding="utf-8")

    records = tracker.track_changes("agent-1", "tool-12")
    assert len(records) >= 2
    changed_paths = {r.file_path for r in records}
    assert str((tmp_path / "a.txt").resolve()) in changed_paths
    assert str((tmp_path / "b.txt").resolve()) in changed_paths


def test_track_changes_ignores_workspace_artifact_files(tmp_path) -> None:
    tracker = FileChangeTracker()
    artifact = tmp_path / ".grep_agent_manager.txt"

    tracker.capture_pre_tool(
        "tool-13",
        "Write",
        {"file_path": str(artifact)},
        cwd=tmp_path,
    )
    artifact.write_text("noise\n", encoding="utf-8")

    records = tracker.track_changes("agent-1", "tool-13")
    assert records == []
