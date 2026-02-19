from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prsm.adapters.events import AgentResult
from prsm.shared.models.message import MessageRole, ToolCall
from prsm.shared.models.session import Session
from prsm.vscode.server import PrsmServer, SessionState


class _Bridge:
    def __init__(self) -> None:
        self.running = False
        self.current_model = "claude-opus-4-6"
        self.current_model_display_name = "Claude Opus 4.6"


def _json_payload(resp) -> dict:
    return json.loads(resp.text)


def _build_server(tmpdir: str) -> PrsmServer:
    with patch("prsm.vscode.server.SessionPersistence") as persistence_cls:
        persistence = persistence_cls.return_value
        persistence.project_dir = Path(tmpdir) / ".prsm_project"
        persistence.list_sessions_by_mtime.return_value = []
        persistence.list_sessions_detailed.return_value = []
        return PrsmServer(cwd=tmpdir, model="claude-opus-4-6")


def _build_loaded_state(server: PrsmServer, session_id: str = "sess-1") -> SessionState:
    session = Session(name="Loaded Session")
    session.add_message(
        "root",
        MessageRole.TOOL,
        "",
        tool_calls=[
            ToolCall(
                id="tc-1",
                name="mcp__orchestrator__task_complete",
                arguments=json.dumps({"summary": "Loaded summary text\n\n1. Step A"}),
            ),
        ],
    )
    state = SessionState(
        session_id=session_id,
        name="Loaded Session",
        project_id=server._default_project_id,
        bridge=_Bridge(),
        session=session,
    )
    server._sessions[session_id] = state
    server._session_projects[session_id] = state.project_id
    return state


@pytest.mark.asyncio
async def test_list_sessions_exposes_task_complete_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        _build_loaded_state(server, session_id="loaded-1")
        server._session_index["cold-1"] = {
            "name": "Cold Session",
            "project_id": server._default_project_id,
            "forked_from": None,
            "agent_count": 2,
            "message_count": 10,
            "created_at": None,
            "last_activity": None,
            "current_model": "claude-opus-4-6",
            "current_model_display": "Claude Opus 4.6",
            "task_complete_count": 3,
            "latest_task_complete_summary": "Cold summary text\n\n1. Step B",
        }

        resp = await server._handle_list_sessions(SimpleNamespace())
        payload = _json_payload(resp)
        assert len(payload["sessions"]) == 2

        by_id = {s["sessionId"]: s for s in payload["sessions"]}
        assert by_id["loaded-1"]["taskCompleteCount"] == 1
        assert by_id["loaded-1"]["latestTaskCompleteSummary"] == "Loaded summary text 1. Step A"
        assert by_id["cold-1"]["taskCompleteCount"] == 3
        assert by_id["cold-1"]["latestTaskCompleteSummary"] == "Cold summary text 1. Step B"


@pytest.mark.asyncio
async def test_direct_restart_engine_finished_summary_is_not_truncated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        state = _build_loaded_state(server, session_id="loaded-2")
        state._directly_restarted_agents.add("agent-1")

        emitted: list[dict] = []
        server._broadcast_sse = lambda event, data: emitted.append({"event": event, "data": data})
        server._save_session = lambda _state: None

        long_summary = "A" * 1200
        await server._on_agent_result(
            state,
            AgentResult(agent_id="agent-1", result=long_summary, is_error=False),
        )

        assert emitted, "Expected engine_finished SSE event"
        assert emitted[-1]["event"] == "engine_finished"
        assert emitted[-1]["data"]["summary"] == long_summary


@pytest.mark.asyncio
async def test_list_sessions_refreshes_idle_loaded_name_from_disk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        state = _build_loaded_state(server, session_id="loaded-3")
        state.name = "Stale In-Memory Name"
        state.session.name = "Stale In-Memory Name"

        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        persisted = sessions_dir / "loaded-3.json"
        persisted.write_text(
            json.dumps(
                {
                    "session_id": "loaded-3",
                    "name": "Fresh Persisted Name",
                    "agents": {},
                    "messages": {},
                    "saved_at": "2026-02-17T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        server._persistence._dir = sessions_dir
        server._persistence.list_sessions_by_mtime.return_value = ["loaded-3"]

        resp = await server._handle_list_sessions(SimpleNamespace())
        payload = _json_payload(resp)
        by_id = {s["sessionId"]: s for s in payload["sessions"]}

        assert by_id["loaded-3"]["name"] == "Fresh Persisted Name"
        assert state.name == "Fresh Persisted Name"
        assert state.session.name == "Fresh Persisted Name"
