from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import AioHTTPTestCase

from prsm.vscode.server import PrsmServer


class TestPhase2MultiProjectServer(AioHTTPTestCase):
    async def get_application(self):
        self.tmpdir = tempfile.mkdtemp()
        root = Path(self.tmpdir)
        (root / "alpha").mkdir(parents=True, exist_ok=True)
        (root / "beta").mkdir(parents=True, exist_ok=True)

        config_path = root / "prsm.yaml"
        config_path.write_text(
            "projects:\n"
            f"  alpha:\n    cwd: {root / 'alpha'}\n"
            f"  beta:\n    cwd: {root / 'beta'}\n",
            encoding="utf-8",
        )

        self._env_patch = patch.dict(
            os.environ,
            {"PRSM_MULTI_PROJECT": "1", "HOME": self.tmpdir},
            clear=False,
        )
        self._env_patch.start()
        self._project_dir_patch = patch(
            "prsm.shared.services.project.ProjectManager.get_project_dir",
            return_value=root / ".prsm_project",
        )
        self._project_dir_patch.start()
        self._base_dir_patch = patch(
            "prsm.shared.services.persistence.BASE_DIR",
            root / ".prsm_sessions",
        )
        self._base_dir_patch.start()

        self.prsm_server = PrsmServer(
            cwd=self.tmpdir,
            model="claude-opus-4-6",
            config_path=str(config_path),
        )
        return self.prsm_server._app

    async def asyncTearDown(self):
        self._base_dir_patch.stop()
        self._project_dir_patch.stop()
        self._env_patch.stop()
        await super().asyncTearDown()

    async def test_create_session_with_explicit_project(self):
        resp = await self.client.post(
            "/sessions",
            json={"name": "Beta Session", "project_id": "beta"},
        )
        assert resp.status == 201
        data = await resp.json()

        assert data["project_id"] == "beta"
        session_id = data["session_id"]
        assert self.prsm_server._session_projects[session_id] == "beta"
        assert self.prsm_server._sessions[session_id].project_id == "beta"

    async def test_create_session_rejects_unknown_project(self):
        resp = await self.client.post(
            "/sessions",
            json={"name": "Bad", "project_id": "missing-project"},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "Unknown project_id" in data["error"]

    async def test_sessions_and_projects_include_project_metadata(self):
        default_resp = await self.client.post("/sessions", json={"name": "Default Session"})
        assert default_resp.status == 201
        default_data = await default_resp.json()

        beta_resp = await self.client.post(
            "/sessions",
            json={"name": "Beta Session", "project_id": "beta"},
        )
        assert beta_resp.status == 201
        beta_data = await beta_resp.json()

        sessions_resp = await self.client.get("/sessions")
        assert sessions_resp.status == 200
        sessions = (await sessions_resp.json())["sessions"]
        by_id = {s["sessionId"]: s for s in sessions}

        assert by_id[default_data["session_id"]]["projectId"] == self.prsm_server._default_project_id
        assert by_id[beta_data["session_id"]]["projectId"] == "beta"

        projects_resp = await self.client.get("/projects")
        assert projects_resp.status == 200
        projects = (await projects_resp.json())["projects"]
        by_project = {p["project_id"]: p for p in projects}

        assert "beta" in by_project
        assert by_project["beta"]["session_count"] == 1
        assert self.prsm_server._default_project_id in by_project
        assert by_project[self.prsm_server._default_project_id]["session_count"] >= 1
