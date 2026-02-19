from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import AioHTTPTestCase

from prsm.vscode.server import PrsmServer


class TestPhase3MultiProjectBrokerServer(AioHTTPTestCase):
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

    async def test_subscribe_unsubscribe_endpoints(self):
        sub = await self.client.post(
            "/projects/beta/subscriptions",
            json={"topic_filter": "build.*"},
        )
        assert sub.status == 200
        sub_data = await sub.json()
        assert sub_data["status"] == "subscribed"
        assert sub_data["subscriptions"] == ["build.*"]

        unsub = await self.client.delete(
            "/projects/beta/subscriptions",
            json={"topic_filter": "build.*"},
        )
        assert unsub.status == 200
        unsub_data = await unsub.json()
        assert unsub_data["status"] == "unsubscribed"
        assert unsub_data["subscriptions"] == []

    async def test_publish_and_list_events_with_limit(self):
        await self.client.post(
            "/projects/beta/subscriptions",
            json={"topic_filter": "build.*"},
        )

        publish = await self.client.post(
            "/projects/alpha/events",
            json={"topic": "build.started", "payload": {"ok": True}},
        )
        assert publish.status == 202
        payload = await publish.json()
        assert payload["status"] == "accepted"
        event = payload["event"]
        assert event["topic"] == "build.started"
        assert event["source_project_id"] == "alpha"
        assert "beta" in event["recipients"]

        listed = await self.client.get("/projects/events?limit=1")
        assert listed.status == 200
        events = (await listed.json())["events"]
        assert len(events) == 1
        assert events[0]["topic"] == "build.started"

    async def test_unknown_project_returns_400(self):
        resp = await self.client.post(
            "/projects/unknown/events",
            json={"topic": "x"},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "Unknown project_id" in data["error"]


class TestPhase3MultiProjectBrokerDisabled(AioHTTPTestCase):
    async def get_application(self):
        self.tmpdir = tempfile.mkdtemp()
        self._env_patch = patch.dict(
            os.environ,
            {"PRSM_MULTI_PROJECT": "0", "HOME": self.tmpdir},
            clear=False,
        )
        self._env_patch.start()
        self._project_dir_patch = patch(
            "prsm.shared.services.project.ProjectManager.get_project_dir",
            return_value=Path(self.tmpdir) / ".prsm_project",
        )
        self._project_dir_patch.start()
        self._base_dir_patch = patch(
            "prsm.shared.services.persistence.BASE_DIR",
            Path(self.tmpdir) / ".prsm_sessions",
        )
        self._base_dir_patch.start()
        self.prsm_server = PrsmServer(cwd=self.tmpdir, model="claude-opus-4-6")
        return self.prsm_server._app

    async def asyncTearDown(self):
        self._base_dir_patch.stop()
        self._project_dir_patch.stop()
        self._env_patch.stop()
        await super().asyncTearDown()

    async def test_broker_endpoints_return_409_when_disabled(self):
        resp = await self.client.get("/projects/events")
        assert resp.status == 409
        data = await resp.json()
        assert "disabled" in data["error"].lower()

        resp = await self.client.post(
            f"/projects/{self.prsm_server._default_project_id}/events",
            json={"topic": "build.started"},
        )
        assert resp.status == 409
