from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prsm.shared.models.session import Session
from prsm.shared.services.persistence import SessionPersistence


def test_save_persists_summary_and_project_id_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        persistence = SessionPersistence(base_dir=Path(tmpdir))
        session = Session()

        path = persistence.save(
            session,
            "meta-test",
            session_id="session-meta",
            summary="Restore missing session metadata",
            project_id="project-alpha",
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["summary"] == "Restore missing session metadata"
        assert payload["project_id"] == "project-alpha"

        _, meta = persistence.load_with_meta("session-meta")
        assert meta["summary"] == "Restore missing session metadata"
        assert meta["project_id"] == "project-alpha"


def test_save_persists_imported_from_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        persistence = SessionPersistence(base_dir=Path(tmpdir))
        session = Session()
        session.imported_from = {
            "provider": "codex",
            "source_session_id": "019c0000-0000-7000-8000-000000000001",
            "source_path": "/home/user/.codex/sessions/.../rollout.jsonl",
            "imported_at": "2026-02-18T12:00:00+00:00",
        }

        path = persistence.save(
            session,
            "imported-meta-test",
            session_id="session-imported-meta",
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["imported_from"]["provider"] == "codex"
        assert payload["imported_from"]["source_session_id"] == (
            "019c0000-0000-7000-8000-000000000001"
        )

        loaded, meta = persistence.load_with_meta("session-imported-meta")
        assert loaded.imported_from is not None
        assert loaded.imported_from["provider"] == "codex"
        assert meta["imported_from"]["provider"] == "codex"
