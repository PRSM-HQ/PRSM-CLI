from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prsm.adapters.file_tracker import FileChangeRecord
from prsm.shared.models.session import Session
from prsm.vscode.server import PrsmServer, SessionState


@dataclass
class _Request:
    match_info: dict[str, str]
    query: dict[str, str] = field(default_factory=dict)
    body: dict | None = None

    @property
    def can_read_body(self) -> bool:
        return self.body is not None

    async def json(self) -> dict:
        return self.body or {}


def _json_payload(resp) -> dict:
    return json.loads(resp.text)


def _build_server(tmpdir: str) -> tuple[PrsmServer, SessionState]:
    project_dir = Path(tmpdir) / ".prsm_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    with patch("prsm.vscode.server.SessionPersistence") as persistence_cls:
        persistence = persistence_cls.return_value
        persistence.project_dir = project_dir
        persistence.list_sessions_by_mtime.return_value = []
        persistence.list_sessions_detailed.return_value = []
        server = PrsmServer(cwd=tmpdir, model="claude-opus-4-6")
    session = Session()
    bridge = SimpleNamespace(_engine=None, running=False, current_model="claude-opus-4-6")
    state = SessionState(
        session_id="sess-phase8",
        name="Phase8",
        project_id=server._default_project_id,
        bridge=bridge,
        session=session,
    )
    server._sessions[state.session_id] = state
    server._session_projects[state.session_id] = state.project_id
    return server, state


@pytest.mark.asyncio
async def test_command_policy_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        req = _Request(match_info={"id": state.session_id})
        with patch("prsm.vscode.server.ProjectManager.get_project_dir", return_value=Path(tmpdir) / ".prsm_project"):
            get_before = await server._handle_get_command_policy(req)
            before = _json_payload(get_before)
            assert before["whitelist"] == []

            update_req = _Request(
                match_info={"id": state.session_id},
                body={"whitelist": [r"^pytest"], "blacklist": [r"rm\\s+-rf"]},
            )
            update_resp = await server._handle_update_command_policy(update_req)
            updated = _json_payload(update_resp)
            assert updated["ok"] is True

            get_after = await server._handle_get_command_policy(req)
            after = _json_payload(get_after)
            assert after["whitelist"] == [r"^pytest"]
            assert after["blacklist"] == [r"rm\\s+-rf"]


@pytest.mark.asyncio
async def test_project_memory_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        with patch("prsm.vscode.server.ProjectManager.get_project_dir", return_value=Path(tmpdir) / ".prsm_project"):
            put_req = _Request(
                match_info={"id": state.session_id},
                body={"content": "# MEMORY\nphase 8 notes"},
            )
            put_resp = await server._handle_update_project_memory(put_req)
            assert _json_payload(put_resp)["ok"] is True

            get_req = _Request(match_info={"id": state.session_id})
            get_resp = await server._handle_get_project_memory(get_req)
            payload = _json_payload(get_resp)
            assert payload["exists"] is True
            assert "phase 8 notes" in payload["content"]


@pytest.mark.asyncio
async def test_preferences_get_put() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefs_path = Path(tmpdir) / "preferences.json"
        with patch("prsm.vscode.server.UserPreferences.load") as load_mock, patch(
            "prsm.vscode.server.UserPreferences.save"
        ) as save_mock:
            from prsm.shared.services.preferences import UserPreferences

            prefs = UserPreferences()
            load_mock.return_value = prefs
            server, state = _build_server(tmpdir)
            _ = state

            get_resp = await server._handle_get_preferences(_Request(match_info={}))
            assert _json_payload(get_resp)["preferences"]["file_revert_on_resend"] == "ask"

            put_resp = await server._handle_update_preferences(
                _Request(
                    match_info={},
                    body={"preferences": {"file_revert_on_resend": "never"}},
                )
            )
            assert _json_payload(put_resp)["preferences"]["file_revert_on_resend"] == "never"
            save_mock.assert_called()
        assert prefs_path.parent.exists()


@pytest.mark.asyncio
async def test_governance_endpoint_shapes_and_memory_post() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        sid = state.session_id

        policy = await server._handle_get_policy(_Request(match_info={"id": sid}))
        assert "triage_rules" in _json_payload(policy)

        leases = await server._handle_get_leases(_Request(match_info={"id": sid}))
        assert _json_payload(leases)["leases"] == []

        audit = await server._handle_get_audit(_Request(match_info={"id": sid}))
        assert "entries" in _json_payload(audit)

        add_memory = await server._handle_add_memory(
            _Request(match_info={"id": sid}, body={"content": "remember this"})
        )
        entry = _json_payload(add_memory)["entry"]
        assert entry["content"] == "remember this"

        get_memory = await server._handle_get_memory(_Request(match_info={"id": sid}))
        assert any(e["id"] == entry["id"] for e in _json_payload(get_memory)["entries"])

        experts = await server._handle_get_expert_stats(_Request(match_info={"id": sid}))
        assert "experts" in _json_payload(experts)

        budget = await server._handle_get_budget(_Request(match_info={"id": sid}))
        assert "project_id" in _json_payload(budget)

        decisions = await server._handle_get_decisions(_Request(match_info={"id": sid}))
        assert _json_payload(decisions)["reports"] == []


@pytest.mark.asyncio
async def test_telemetry_export_disabled_returns_409() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        resp = await server._handle_export_telemetry(
            _Request(
                match_info={"id": state.session_id},
                body={"metric_type": "triage_decision"},
            )
        )
        assert resp.status == 409
        assert "Telemetry is not enabled" in _json_payload(resp)["error"]


@pytest.mark.asyncio
async def test_reject_change_reverts_modified_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "sample.txt"
        file_path.write_text("new content", encoding="utf-8")

        record = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="modify",
            tool_call_id="tc-modify",
            tool_name="Write",
            message_index=0,
            old_content="old content",
            new_content="new content",
            pre_tool_content="old content",
            status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [record]

        resp = await server._handle_reject_change(
            _Request(match_info={"id": state.session_id, "tool_call_id": "tc-modify"})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "rejected"
        assert file_path.read_text(encoding="utf-8") == "old content"
        assert record.status == "rejected"


@pytest.mark.asyncio
async def test_reject_change_reverts_created_file_by_deleting() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "created.txt"
        file_path.write_text("created by agent", encoding="utf-8")

        record = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="create",
            tool_call_id="tc-create",
            tool_name="Write",
            message_index=0,
            old_content=None,
            new_content="created by agent",
            pre_tool_content=None,
            status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [record]

        resp = await server._handle_reject_change(
            _Request(match_info={"id": state.session_id, "tool_call_id": "tc-create"})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "rejected"
        assert not file_path.exists()
        assert record.status == "rejected"


@pytest.mark.asyncio
async def test_reject_change_without_restore_data_marks_rejected() -> None:
    """When no pre_tool_content or old_content exists, reject still marks
    the record as rejected but cannot revert the file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "missing-restore.txt"
        file_path.write_text("still changed", encoding="utf-8")

        record = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="modify",
            tool_call_id="tc-bad",
            tool_name="Write",
            message_index=0,
            old_content=None,
            new_content="still changed",
            pre_tool_content=None,
            status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [record]

        resp = await server._handle_reject_change(
            _Request(match_info={"id": state.session_id, "tool_call_id": "tc-bad"})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "rejected"
        # File unchanged on disk (no restore data available)
        assert file_path.read_text(encoding="utf-8") == "still changed"
        # Record marked as rejected (kept for dedup)
        assert record.status == "rejected"


@pytest.mark.asyncio
async def test_reject_all_reverts_to_earliest_pre_tool_content() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "stacked.txt"
        file_path.write_text("v2", encoding="utf-8")
        created_path = Path(tmpdir) / "created-2.txt"
        created_path.write_text("created", encoding="utf-8")

        first = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="modify",
            tool_call_id="tc-1",
            tool_name="Write",
            message_index=0,
            old_content="v0",
            new_content="v1",
            pre_tool_content="v0",
            status="pending",
        )
        second = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="modify",
            tool_call_id="tc-2",
            tool_name="Write",
            message_index=1,
            old_content="v1",
            new_content="v2",
            pre_tool_content="v1",
            status="pending",
        )
        created = FileChangeRecord(
            file_path=str(created_path),
            agent_id="agent-1",
            change_type="create",
            tool_call_id="tc-3",
            tool_name="Write",
            message_index=2,
            old_content=None,
            new_content="created",
            pre_tool_content=None,
            status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [first, second]
        state.file_tracker.file_changes[str(created_path)] = [created]

        resp = await server._handle_reject_all_changes(
            _Request(match_info={"id": state.session_id})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "rejected"
        assert payload["count"] == 3
        assert file_path.read_text(encoding="utf-8") == "v0"
        assert not created_path.exists()
        assert first.status == "rejected"
        assert second.status == "rejected"
        assert created.status == "rejected"


@pytest.mark.asyncio
async def test_reject_change_cascades_later_records() -> None:
    """Rejecting a change cascade-rejects all later pending records for the same file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "cascade.txt"
        file_path.write_text("v3", encoding="utf-8")

        r0 = FileChangeRecord(
            file_path=str(file_path), agent_id="a1", change_type="modify",
            tool_call_id="tc-0", tool_name="Write", message_index=0,
            old_content="v0", new_content="v1", pre_tool_content="v0", status="pending",
        )
        r1 = FileChangeRecord(
            file_path=str(file_path), agent_id="a1", change_type="modify",
            tool_call_id="tc-1", tool_name="Write", message_index=1,
            old_content="v1", new_content="v2", pre_tool_content="v1", status="pending",
        )
        r2 = FileChangeRecord(
            file_path=str(file_path), agent_id="a1", change_type="modify",
            tool_call_id="tc-2", tool_name="Write", message_index=2,
            old_content="v2", new_content="v3", pre_tool_content="v2", status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [r0, r1, r2]

        # Reject r0 → should cascade-reject r1 and r2
        resp = await server._handle_reject_change(
            _Request(match_info={"id": state.session_id, "tool_call_id": "tc-0"})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "rejected"
        assert set(payload["cascade_rejected"]) == {"tc-1", "tc-2"}
        assert file_path.read_text(encoding="utf-8") == "v0"
        assert r0.status == "rejected"
        assert r1.status == "rejected"
        assert r2.status == "rejected"


@pytest.mark.asyncio
async def test_accept_change_is_noop_on_disk() -> None:
    """Accept marks the record as accepted; the file is untouched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server, state = _build_server(tmpdir)
        file_path = Path(tmpdir) / "accepted.txt"
        file_path.write_text("new content", encoding="utf-8")

        record = FileChangeRecord(
            file_path=str(file_path),
            agent_id="agent-1",
            change_type="modify",
            tool_call_id="tc-accept",
            tool_name="Write",
            message_index=0,
            old_content="old content",
            new_content="new content",
            pre_tool_content="old content",
            status="pending",
        )
        state.file_tracker.file_changes[str(file_path)] = [record]

        resp = await server._handle_accept_change(
            _Request(match_info={"id": state.session_id, "tool_call_id": "tc-accept"})
        )
        payload = _json_payload(resp)

        assert resp.status == 200
        assert payload["status"] == "accepted"
        # File content unchanged — accept is a no-op on disk
        assert file_path.read_text(encoding="utf-8") == "new content"
        # Record kept with status="accepted" (prevents fallback regeneration)
        assert record.status == "accepted"
