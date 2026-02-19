"""Snapshot service — capture and restore working tree + session state.

Snapshots store session JSON + git working tree diffs as patch files,
allowing users to revert both code changes and conversation history
without polluting the git commit history.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from prsm.adapters.file_tracker import FileChangeTracker
from prsm.shared.models.session import Session
from prsm.shared.services.persistence import (
    _agent_to_dict,
    _dict_to_agent,
    _message_to_dict,
    _dict_to_message,
)
from prsm.shared.services.durable_write import _fsync_dir, atomic_write_text

logger = logging.getLogger(__name__)


class SnapshotService:
    """Create and restore snapshots of session state + working tree."""

    def __init__(self, project_dir: Path, cwd: Path) -> None:
        self._snapshots_dir = project_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._cwd = cwd

    def create(
        self,
        session: Session,
        session_name: str,
        description: str = "",
        file_tracker: FileChangeTracker | None = None,
        session_id: str | None = None,
        parent_snapshot_id: str | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
        parent_agent_id: str | None = None,
    ) -> str:
        """Create a snapshot of the current session and working tree.

        Args:
            session: The session to snapshot.
            session_name: Display name for the session.
            description: Human-readable description of the snapshot.
            file_tracker: If provided, file change records are persisted
                         into the snapshot so they survive restore.
            session_id: Optional session ID for grouping snapshots by session.

        Returns the snapshot ID.
        """
        effective_session_id = session_id or session.session_id
        snapshot_id = str(uuid.uuid4())[:8]
        snap_dir = self._snapshots_dir / snapshot_id
        staging_dir = self._snapshots_dir / f".tmp-{snapshot_id}-{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        try:
            # 1. Write metadata
            meta = {
                "snapshot_id": snapshot_id,
                "session_name": session_name,
                "session_id": effective_session_id,
                "parent_snapshot_id": parent_snapshot_id,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "parent_agent_id": parent_agent_id,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "git_branch": self._get_git_branch(),
            }
            atomic_write_text(staging_dir / "meta.json", json.dumps(meta, indent=2))

            # 2. Serialize session state
            session_data = {
                "session_id": session.session_id,
                "active_agent_id": session.active_agent_id,
                "agents": {
                    aid: _agent_to_dict(agent)
                    for aid, agent in session.agents.items()
                },
                "messages": {
                    aid: [_message_to_dict(m) for m in msgs]
                    for aid, msgs in session.messages.items()
                },
                "name": session.name,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "forked_from": session.forked_from,
                "imported_from": session.imported_from,
            }
            atomic_write_text(staging_dir / "session.json", json.dumps(session_data, indent=2))

            # 3. Capture git diff (tracked changes vs HEAD)
            try:
                diff = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(self._cwd),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if diff.returncode == 0 and diff.stdout:
                    atomic_write_text(staging_dir / "working_tree.patch", diff.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug("Failed to capture git diff for snapshot %s", snapshot_id)

            # 4. Copy untracked files
            try:
                untracked = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard"],
                    cwd=str(self._cwd),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if untracked.returncode == 0 and untracked.stdout.strip():
                    untracked_dir = staging_dir / "untracked"
                    for rel_path in untracked.stdout.strip().split("\n"):
                        rel_path = rel_path.strip()
                        if not rel_path:
                            continue
                        src = self._cwd / rel_path
                        dst = untracked_dir / rel_path
                        if src.is_file():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(src), str(dst))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug("Failed to copy untracked files for snapshot %s", snapshot_id)

            # 5. Persist file change records (if tracker provided)
            if file_tracker and file_tracker.file_changes:
                file_tracker.persist(staging_dir / "file-changes")

            # Publish only once snapshot directory is fully populated.
            os.replace(staging_dir, snap_dir)
            _fsync_dir(self._snapshots_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        logger.info("Created snapshot %s: %s", snapshot_id, description or "no description")
        return snapshot_id

    def restore(self, snapshot_id: str) -> tuple[Session, FileChangeTracker]:
        """Restore a snapshot: revert working tree and return saved session.

        Returns (session, file_tracker) — the tracker is populated with any
        file change records that were captured when the snapshot was created.

        WARNING: This resets tracked files to HEAD then applies the saved patch.
        """
        snap_dir = self._snapshots_dir / snapshot_id
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")

        # 1. Reset tracked files to HEAD
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=str(self._cwd),
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Failed to reset tracked files during restore")

        # 2. Clean untracked files that were created after the snapshot
        # (We only clean files that exist in the working tree but NOT in the snapshot)
        try:
            current_untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if current_untracked.returncode == 0 and current_untracked.stdout.strip():
                snapshot_untracked = set()
                untracked_dir = snap_dir / "untracked"
                if untracked_dir.exists():
                    for p in untracked_dir.rglob("*"):
                        if p.is_file():
                            snapshot_untracked.add(
                                str(p.relative_to(untracked_dir))
                            )

                for rel_path in current_untracked.stdout.strip().split("\n"):
                    rel_path = rel_path.strip()
                    if rel_path and rel_path not in snapshot_untracked:
                        target = self._cwd / rel_path
                        if target.is_file():
                            target.unlink()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # 3. Apply saved patch
        patch_file = snap_dir / "working_tree.patch"
        if patch_file.exists() and patch_file.stat().st_size > 0:
            try:
                subprocess.run(
                    ["git", "apply", str(patch_file)],
                    cwd=str(self._cwd),
                    capture_output=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Failed to apply patch for snapshot %s", snapshot_id)

        # 4. Restore untracked files
        untracked_dir = snap_dir / "untracked"
        if untracked_dir.exists():
            for src in untracked_dir.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(untracked_dir)
                    dst = self._cwd / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))

        # 5. Deserialize session + file changes
        session, file_tracker = self._load_snapshot_state(snap_dir)
        logger.info("Restored snapshot %s", snapshot_id)
        return session, file_tracker

    def load_session(self, snapshot_id: str) -> tuple[Session, FileChangeTracker]:
        """Load a snapshot's session + file-change state without restoring files."""
        snap_dir = self._snapshots_dir / snapshot_id
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        return self._load_snapshot_state(snap_dir)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all snapshots with metadata."""
        result = []
        if not self._snapshots_dir.exists():
            return result

        for snap_dir in sorted(self._snapshots_dir.iterdir()):
            meta_file = snap_dir / "meta.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    result.append(meta)
                except (json.JSONDecodeError, OSError):
                    result.append({
                        "snapshot_id": snap_dir.name,
                        "timestamp": "corrupt",
                    })
        return result

    def list_snapshots_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """List all snapshots for a specific session.

        Args:
            session_id: The session ID to filter by.

        Returns:
            List of snapshot metadata dictionaries for the given session.
        """
        all_snapshots = self.list_snapshots()
        return [
            snap for snap in all_snapshots
            if snap.get("session_id") == session_id
        ]

    def group_snapshots_by_session(self) -> dict[str, list[dict[str, Any]]]:
        """Group all snapshots by session ID.

        Returns:
            Dictionary mapping session_id -> list of snapshot metadata.
            Snapshots without a session_id are grouped under the key None.
        """
        grouped: dict[str | None, list[dict[str, Any]]] = {}
        for snap in self.list_snapshots():
            session_id = snap.get("session_id")
            if session_id not in grouped:
                grouped[session_id] = []
            grouped[session_id].append(snap)
        return grouped

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        snap_dir = self._snapshots_dir / snapshot_id
        if snap_dir.exists():
            shutil.rmtree(str(snap_dir))
            return True
        return False

    def get_meta(self, snapshot_id: str) -> dict[str, Any]:
        """Get snapshot metadata."""
        meta_file = self._snapshots_dir / snapshot_id / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        return json.loads(meta_file.read_text())

    def _load_snapshot_state(self, snap_dir: Path) -> tuple[Session, FileChangeTracker]:
        """Load session + file changes from a snapshot directory."""
        session_data = json.loads(
            (snap_dir / "session.json").read_text()
        )

        session = Session()
        session.session_id = str(session_data.get("session_id") or session.session_id)
        session.active_agent_id = session_data.get("active_agent_id")
        session.name = session_data.get("name")
        session.forked_from = session_data.get("forked_from")
        imported_from = session_data.get("imported_from")
        if isinstance(imported_from, dict):
            session.imported_from = imported_from
        created_at_str = session_data.get("created_at")
        if created_at_str:
            session.created_at = datetime.fromisoformat(created_at_str)

        for aid, agent_data in session_data.get("agents", {}).items():
            session.agents[aid] = _dict_to_agent(agent_data)

        for aid, msgs_data in session_data.get("messages", {}).items():
            session.messages[aid] = [_dict_to_message(m) for m in msgs_data]

        file_tracker = FileChangeTracker()
        file_changes_dir = snap_dir / "file-changes"
        if file_changes_dir.exists():
            file_tracker.load(file_changes_dir)

        return session, file_tracker

    def _get_git_branch(self) -> str | None:
        """Get the current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
