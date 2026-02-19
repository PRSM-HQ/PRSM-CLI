from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.shared.models.session import Session
from prsm.vscode.server import PrsmServer, SessionState


class _Bridge:
    def __init__(self) -> None:
        self.running = False
        self.shutdown_calls = 0
        self.current_model = "claude-opus-4-6"
        self.current_model_display_name = "Claude Opus 4.6"

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def _build_server(tmpdir: str) -> PrsmServer:
    with patch("prsm.vscode.server.SessionPersistence") as persistence_cls:
        persistence = persistence_cls.return_value
        persistence.project_dir = Path(tmpdir) / ".prsm_project"
        persistence.list_sessions_by_mtime.return_value = []
        persistence.list_sessions_detailed.return_value = []
        return PrsmServer(cwd=tmpdir, model="claude-opus-4-6")


def _build_state(session_id: str) -> SessionState:
    return SessionState(
        session_id=session_id,
        name="Unload Test",
        project_id="proj",
        bridge=_Bridge(),
        session=Session(),
    )


@pytest.mark.asyncio
async def test_inactivity_unload_shuts_down_and_removes_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        state = _build_state(session_id="sess-idle")
        state._last_touched_at = time.time() - 3600
        server._sessions[state.session_id] = state
        server._session_projects[state.session_id] = state.project_id
        server._session_inactivity_seconds = 1

        await server._unload_inactive_sessions()

        assert state.session_id not in server._sessions
        assert state.bridge.shutdown_calls == 1


@pytest.mark.asyncio
async def test_inactivity_unload_skips_running_sessions() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        state = _build_state(session_id="sess-running")
        state._last_touched_at = time.time() - 3600
        state.bridge.running = True
        server._sessions[state.session_id] = state
        server._session_projects[state.session_id] = state.project_id
        server._session_inactivity_seconds = 1

        await server._unload_inactive_sessions()

        # Running session should not be unloaded
        assert state.session_id in server._sessions
        assert state.bridge.shutdown_calls == 0


@pytest.mark.asyncio
async def test_inactivity_unload_skips_recently_active_sessions() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = _build_server(tmpdir)
        state = _build_state(session_id="sess-active")
        state._last_touched_at = time.time()  # just now
        server._sessions[state.session_id] = state
        server._session_projects[state.session_id] = state.project_id
        server._session_inactivity_seconds = 3600

        await server._unload_inactive_sessions()

        assert state.session_id in server._sessions
        assert state.bridge.shutdown_calls == 0
