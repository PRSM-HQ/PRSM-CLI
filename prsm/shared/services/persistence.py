"""Session persistence — save and load sessions to disk.

Storage layout:
    ~/.prsm/sessions/{repo_identity}/{session_id}.json

Where {repo_identity} is based on git-common-dir for git repos (stable across
worktrees) or directory name for non-git directories. Falls back to legacy
flat storage at ~/.prsm/sessions/ when no cwd is given.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
import uuid

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import Message, MessageRole, ToolCall
from prsm.shared.models.session import Session, WorktreeMetadata
from prsm.adapters.agent_adapter import STALE_STATES, parse_role, parse_state
from prsm.shared.services.durable_write import _fsync_dir, atomic_write_text

logger = logging.getLogger(__name__)

_IMPORTED_BASE_DIR = Path.home() / ".prsm" / "sessions"
BASE_DIR = _IMPORTED_BASE_DIR


def _resolve_base_dir() -> Path:
    """Resolve base sessions dir at runtime.

    If tests monkeypatch BASE_DIR, respect it.
    Otherwise, re-evaluate from current HOME so patched HOME environments
    don't keep writing to the import-time path.
    """
    base_dir = Path(BASE_DIR)
    if base_dir != _IMPORTED_BASE_DIR:
        return base_dir
    return Path.home() / ".prsm" / "sessions"


class SessionPersistence:
    """Save and load sessions to JSON files with git worktree awareness.

    When constructed with `cwd`, uses repo-identity-based storage so all
    worktrees of the same repository share the same session directory.
    When constructed with `base_dir` (or no args), uses legacy flat storage.
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else _resolve_base_dir()
        if cwd is not None:
            from prsm.shared.services.project import ProjectManager

            # Get repository context for worktree awareness
            self._repo_context = ProjectManager.get_repository_context(cwd)

            # Use repo identity (git-common-dir based) for storage path
            repo_id = self._repo_context.repo_identity
            self._dir = self._base_dir / repo_id
            self._workspace = repo_id

            # Keep project_dir for snapshots and other project-level data
            self._project_dir = Path.home() / ".prsm" / "projects" / repo_id
            self._project_dir.mkdir(parents=True, exist_ok=True)
            self._dir.mkdir(parents=True, exist_ok=True)

            # Store current worktree metadata for tagging new sessions
            if self._repo_context.is_git_repo and self._repo_context.worktree_root:
                self._worktree_metadata = WorktreeMetadata(
                    root=str(self._repo_context.worktree_root),
                    branch=self._repo_context.branch,
                    common_dir=str(self._repo_context.common_dir) if self._repo_context.common_dir else None,
                )
            else:
                self._worktree_metadata = None

            # Migrate sessions from legacy layouts
            self._migrate_legacy(cwd)
            self._migrate_basename_layout(cwd)
        else:
            self._dir = self._base_dir
            self._workspace = None
            self._project_dir = None
            self._repo_context = None
            self._worktree_metadata = None
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_dir(self) -> Path | None:
        """The project directory, or None if using legacy mode."""
        return self._project_dir

    @property
    def repo_context(self):
        """Repository context resolved at init time, if cwd mode is enabled."""
        return self._repo_context

    @property
    def workspace(self) -> str | None:
        """The workspace name, or None if using legacy mode."""
        return self._workspace

    def save(
        self,
        session: Session,
        name: str,
        session_id: str | None = None,
        *,
        summary: str | None = None,
        project_id: str | None = None,
    ) -> Path:
        """Serialize session to a JSON file.

        Args:
            session: The session to save.
            name: Display name for the session.
            session_id: If provided, used as the filename.
            summary: Optional one-line task/session summary metadata.
            project_id: Optional project identifier for multi-project sessions.
        """
        # Tag session with current worktree metadata if not already set
        if session.worktree is None and self._worktree_metadata is not None:
            session.worktree = self._worktree_metadata
        if not session.session_id:
            session.session_id = str(uuid.uuid4())
        effective_session_id = session_id or name

        data = {
            "active_agent_id": session.active_agent_id,
            "agents": {
                aid: _agent_to_dict(agent)
                for aid, agent in session.agents.items()
            },
            "messages": {
                aid: [_message_to_dict(m) for m in msgs]
                for aid, msgs in session.messages.items()
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.3",  # Bumped for worktree support
            "name": name,
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "forked_from": session.forked_from,
        }
        if session.imported_from is not None:
            data["imported_from"] = session.imported_from

        if self._workspace:
            data["workspace"] = self._workspace
        if summary is not None:
            data["summary"] = summary
        if project_id is not None:
            data["project_id"] = project_id

        # Add worktree metadata if present
        if session.worktree:
            data["worktree"] = {
                "root": session.worktree.root,
                "branch": session.worktree.branch,
                "common_dir": session.worktree.common_dir,
            }

        filename = effective_session_id
        path = self._dir / f"{filename}.json"
        atomic_write_text(path, json.dumps(data, indent=2))
        logger.info("Session saved to %s", path)

        # Write .active_session marker in project_dir (if in cwd mode)
        if self._project_dir is not None:
            marker = self._project_dir / ".active_session"
            atomic_write_text(marker, effective_session_id)

        return path

    def _resolve_session_stem(self, session_ref: str) -> str:
        """Resolve a user/session reference to an on-disk file stem.

        Prefers direct stem lookup for fast path. Falls back to scanning
        metadata for legacy files whose stem does not match session_id.
        """
        direct = self._dir / f"{session_ref}.json"
        if direct.exists():
            return session_ref

        for stem in self.list_sessions_by_mtime():
            path = self._dir / f"{stem}.json"
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if str(data.get("session_id") or "") == session_ref:
                return stem
        raise FileNotFoundError(session_ref)

    def load(self, session_ref: str) -> Session:
        """Deserialize session by UUID (or legacy file stem)."""
        stem = self._resolve_session_stem(session_ref)
        path = self._dir / f"{stem}.json"
        data = json.loads(path.read_text())

        session = Session()
        session.session_id = str(data.get("session_id") or stem)
        session.active_agent_id = data.get("active_agent_id")
        session.name = data.get("name")
        session.forked_from = data.get("forked_from")
        imported_from = data.get("imported_from")
        if isinstance(imported_from, dict):
            session.imported_from = imported_from
        created_at_str = data.get("created_at")
        if created_at_str:
            dt = datetime.fromisoformat(created_at_str)
            session.created_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # Load worktree metadata if present
        worktree_data = data.get("worktree")
        if worktree_data:
            session.worktree = WorktreeMetadata(
                root=worktree_data["root"],
                branch=worktree_data.get("branch"),
                common_dir=worktree_data.get("common_dir"),
            )

            # Warn if session's worktree doesn't match current context
            if self._worktree_metadata and session.worktree.root != self._worktree_metadata.root:
                logger.warning(
                    "Session '%s' was created in worktree %s but loading from %s",
                    session_ref,
                    session.worktree.root,
                    self._worktree_metadata.root,
                )

        for aid, agent_data in data.get("agents", {}).items():
            session.agents[aid] = _dict_to_agent(agent_data)

        for aid, msgs_data in data.get("messages", {}).items():
            session.messages[aid] = [_dict_to_message(m) for m in msgs_data]

        self._retroactively_link_snapshots(session)
        self._reset_stale_states(session)
        return session

    def load_with_meta(self, session_ref: str) -> tuple[Session, dict]:
        """Load session and return (Session, metadata_dict)."""
        stem = self._resolve_session_stem(session_ref)
        path = self._dir / f"{stem}.json"
        data = json.loads(path.read_text())

        session = Session()
        session.session_id = str(data.get("session_id") or stem)
        session.active_agent_id = data.get("active_agent_id")
        session.name = data.get("name")
        session.forked_from = data.get("forked_from")
        imported_from = data.get("imported_from")
        if isinstance(imported_from, dict):
            session.imported_from = imported_from
        created_at_str = data.get("created_at")
        if created_at_str:
            dt = datetime.fromisoformat(created_at_str)
            session.created_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        for aid, agent_data in data.get("agents", {}).items():
            session.agents[aid] = _dict_to_agent(agent_data)

        for aid, msgs_data in data.get("messages", {}).items():
            session.messages[aid] = [_dict_to_message(m) for m in msgs_data]

        self._retroactively_link_snapshots(session)
        self._reset_stale_states(session)
        meta = {
            "saved_at": data.get("saved_at"),
            "session_id": data.get("session_id") or stem,
            "name": data.get("name") or stem,
            "summary": data.get("summary"),
            "project_id": data.get("project_id"),
            "imported_from": data.get("imported_from"),
        }
        return session, meta

    def _retroactively_link_snapshots(self, session: Session) -> None:
        """Link messages to existing snapshots based on timestamps and content.
        
        Searches GLOBALLY across all projects to find snapshots for this session.
        Uses a scoring system and saves the session back to disk if changed.
        """
        # 1. Collect all snapshots for this session from ALL project directories
        all_snapshots = []
        projects_root = Path.home() / ".prsm" / "projects"
        if projects_root.exists():
            for proj_dir in projects_root.iterdir():
                snaps_dir = proj_dir / "snapshots"
                if not snaps_dir.exists():
                    continue
                for snap_dir in snaps_dir.iterdir():
                    meta_path = snap_dir / "meta.json"
                    if not meta_path.exists():
                        continue
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        if str(meta.get("session_id")) == session.session_id:
                            # Add snapshot metadata for matching
                            all_snapshots.append(meta)
                    except Exception:
                        continue
        
        if not all_snapshots:
            return
        
        # Sort snapshots by timestamp (oldest first)
        all_snapshots.sort(key=lambda s: s.get("timestamp", ""))
        
        # 2. Collect all user messages that need linking
        user_msgs = []
        for aid, msgs in session.messages.items():
            for m in msgs:
                if m.role == MessageRole.USER and not m.snapshot_id:
                    user_msgs.append(m)
        
        if not user_msgs:
            return
            
        user_msgs.sort(key=lambda m: m.timestamp)
        
        used_snapshot_ids = set()
        any_changes = False
        
        for msg in user_msgs:
            msg_ts = msg.timestamp.timestamp()
            msg_content_lower = (msg.content or "").strip().lower()
            
            best_snap = None
            best_score = -1.0
            
            for snap in all_snapshots:
                snap_id = snap["snapshot_id"]
                if snap_id in used_snapshot_ids:
                    continue
                
                snap_ts_str = snap.get("timestamp")
                if not snap_ts_str:
                    continue
                
                try:
                    # Snapshots use naive local isoformat() or UTC
                    snap_dt = datetime.fromisoformat(snap_ts_str)
                    if snap_dt.tzinfo is None:
                        snap_dt = snap_dt.replace(tzinfo=timezone.utc)
                    snap_ts = snap_dt.timestamp()
                    
                    # Scoring logic
                    score = 0.0
                    
                    # 1. Content matching (highest priority)
                    desc = snap.get("description", "").lower()
                    import re
                    match = re.search(r"'(.*?)'", desc)
                    if match:
                        snippet = match.group(1).strip()
                        if snippet and msg_content_lower.startswith(snippet):
                            score += 1000.0
                    elif msg_content_lower[:20] in desc:
                        score += 500.0
                        
                    # 2. Agent ID matching
                    if snap.get("agent_id") == msg.agent_id:
                        score += 100.0
                        
                    # 3. Time matching (timezone-adjusted)
                    diff = abs(msg_ts - snap_ts)
                    # Account for hour-level offsets (TZ drift)
                    tz_adj_diff = min(diff % 3600, 3600 - (diff % 3600))
                    
                    if tz_adj_diff < 300.0: # 5 minute window after TZ adj
                        score += (300.0 - tz_adj_diff)
                    
                    # Absolute proximity bonus (minor tie-breaker)
                    score += (1.0 - (min(diff, 86400) / 86400))
                    
                    if score > best_score and score > 1.0:
                        best_score = score
                        best_snap = snap
                except Exception:
                    continue
            
            if best_snap:
                msg.snapshot_id = best_snap["snapshot_id"]
                used_snapshot_ids.add(best_snap["snapshot_id"])
                any_changes = True
                logger.info(
                    "Retro-linked historical msg in session %s to snap %s (score=%.1f)",
                    session.session_id[:8],
                    msg.snapshot_id,
                    best_score
                )
        
        # 3. Persist the links if we found any, so we don't have to do this again
        if any_changes:
            try:
                # We need to find the right path to save back.
                # Since we are inside SessionPersistence, we can use save().
                # But we need the display name.
                self.save(session, session.name or session.session_id, session_id=session.session_id)
                logger.info("Permanently persisted retroactive snapshot links for session %s", session.session_id[:8])
            except Exception as e:
                logger.warning("Failed to persist retroactive links for session %s: %s", session.session_id[:8], e)

    def _reset_stale_states(self, session: Session) -> None:
        """Reset stale running/waiting states on restored sessions.

        No orchestration is running after a restore, so agents left in
        transient states (pending, running, waiting, starting) are moved
        to completed.  Also persists the corrected states back to disk
        so the JSON file reflects reality.
        """
        any_reset = False
        for agent in session.agents.values():
            if agent.state in STALE_STATES:
                logger.info(
                    "Resetting stale agent %s from %s to completed",
                    getattr(agent, "id", "?"),
                    agent.state,
                )
                agent.state = AgentState.COMPLETED
                any_reset = True

        if any_reset:
            try:
                self.save(
                    session,
                    session.name or session.session_id,
                    session_id=session.session_id,
                )
                logger.info(
                    "Persisted stale-state corrections for session %s",
                    session.session_id[:8] if session.session_id else "?",
                )
            except Exception:
                logger.warning(
                    "Failed to persist stale-state corrections",
                    exc_info=True,
                )

    def auto_resume(self) -> Session | None:
        """Load the most recently saved session."""
        sessions = self.list_sessions_by_mtime()
        if not sessions:
            return None
        try:
            return self.load(sessions[0])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to auto-resume session '%s': %s", sessions[0], exc)
            return None

    def list_sessions(self) -> list[str]:
        """List saved session names."""
        return sorted(
            p.stem for p in self._dir.glob("*.json")
        )

    def list_sessions_by_mtime(self) -> list[str]:
        """List saved session names, most recently modified first."""
        paths = list(self._dir.glob("*.json"))
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in paths]

    def list_sessions_detailed(self, all_worktrees: bool = False) -> list[dict]:
        """List sessions with metadata, most recent first.

        Args:
            all_worktrees: If False (default), only show sessions from current worktree.
                          If True, show sessions from all worktrees of this repository.
        """
        result = []
        paths = list(self._dir.glob("*.json"))
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        current_worktree_root = self._worktree_metadata.root if self._worktree_metadata else None

        for path in paths:
            try:
                data = json.loads(path.read_text())

                # Filter by worktree if requested and metadata is available
                if not all_worktrees and current_worktree_root and "worktree" in data:
                    session_worktree_root = data["worktree"].get("root")
                    if session_worktree_root and session_worktree_root != current_worktree_root:
                        continue  # Skip sessions from other worktrees

                worktree_info = data.get("worktree", {})
                result.append({
                    "name": data.get("name", path.stem),
                    "session_id": data.get("session_id", path.stem),
                    "file_stem": path.stem,
                    "saved_at": data.get("saved_at", "unknown"),
                    "forked_from": data.get("forked_from"),
                    "agent_count": len(data.get("agents", {})),
                    "message_count": sum(
                        len(msgs) for msgs in data.get("messages", {}).values()
                    ),
                    "worktree_root": worktree_info.get("root"),
                    "branch": worktree_info.get("branch"),
                })
            except Exception:
                result.append({
                    "name": path.stem,
                    "session_id": path.stem,
                    "file_stem": path.stem,
                    "saved_at": "corrupt",
                    "worktree_root": None,
                    "branch": None,
                })
        return result

    def delete(self, session_ref: str) -> bool:
        """Delete a saved session by UUID (or legacy file stem)."""
        try:
            stem = self._resolve_session_stem(session_ref)
        except FileNotFoundError:
            return False
        path = self._dir / f"{stem}.json"
        if path.exists():
            path.unlink()
            _fsync_dir(self._dir)
            return True
        return False

    def _migrate_legacy(self, cwd: Path) -> None:
        """Copy sessions from old ~/.prsm/projects/{ID}/sessions/ to new layout.

        Only migrates sessions for the given cwd. Copies (doesn't move) so
        old files remain as a backup. Skips files that already exist at dest.
        """
        from prsm.shared.services.project import ProjectManager
        old_sessions_dir = ProjectManager.get_project_dir(cwd) / "sessions"
        if not old_sessions_dir.exists():
            return

        migrated = 0
        for json_file in old_sessions_dir.glob("*.json"):
            dest = self._dir / json_file.name
            if not dest.exists():
                try:
                    shutil.copy2(json_file, dest)
                    migrated += 1
                except Exception:
                    logger.debug("Failed to migrate %s", json_file)

        if migrated:
            logger.info("Migrated %d sessions from legacy storage to %s", migrated, self._dir)

    def _migrate_basename_layout(self, cwd: Path) -> None:
        """Migrate from basename-based (~/.prsm/sessions/prsm-cli/)
        to repo-identity-based layout (~/.prsm/sessions/{repo-id}/).

        This migration is needed when moving from path-based to git-common-dir-based
        storage for repositories with worktrees.
        """
        old_dir = self._base_dir / cwd.resolve().name
        if old_dir.exists() and old_dir != self._dir:
            migrated = 0
            for json_file in old_dir.glob("*.json"):
                dest = self._dir / json_file.name
                if not dest.exists():
                    try:
                        # Copy and tag with current worktree metadata
                        data = json.loads(json_file.read_text())
                        if "worktree" not in data and self._worktree_metadata:
                            data["worktree"] = {
                                "root": self._worktree_metadata.root,
                                "branch": self._worktree_metadata.branch,
                                "common_dir": self._worktree_metadata.common_dir,
                            }
                            data["version"] = "1.3"
                        atomic_write_text(dest, json.dumps(data, indent=2))
                        migrated += 1
                    except Exception as e:
                        logger.debug("Failed to migrate %s: %s", json_file, e)

            if migrated:
                logger.info("Migrated %d sessions from basename layout %s to repo layout %s",
                           migrated, old_dir, self._dir)


def _agent_to_dict(agent: AgentNode) -> dict:
    d = {
        "id": agent.id,
        "name": agent.name,
        "state": agent.state.value,
        "role": agent.role.value if agent.role else None,
        "model": agent.model,
        "parent_id": agent.parent_id,
        "children_ids": agent.children_ids,
        "prompt_preview": agent.prompt_preview,
    }
    if agent.created_at is not None:
        d["created_at"] = agent.created_at.isoformat()
    if agent.completed_at is not None:
        d["completed_at"] = agent.completed_at.isoformat()
    if agent.last_active is not None:
        d["last_active"] = agent.last_active.isoformat()
    return d


def _ensure_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp string, ensuring timezone awareness."""
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return _ensure_aware(dt)


def _dict_to_agent(data: dict) -> AgentNode:
    """Convert a dict to an AgentNode."""
    kwargs: dict = dict(
        id=data["id"],
        name=data["name"],
        state=parse_state(data["state"]),
        role=parse_role(data.get("role")),
        model=data.get("model", "claude-opus-4-6"),
        parent_id=data.get("parent_id"),
        children_ids=data.get("children_ids", []),
        prompt_preview=data.get("prompt_preview", ""),
    )

    last_active = _parse_timestamp(data.get("last_active"))
    if last_active:
        kwargs["last_active"] = last_active

    created_at = _parse_timestamp(data.get("created_at"))
    if created_at:
        kwargs["created_at"] = created_at

    completed_at = _parse_timestamp(data.get("completed_at"))
    if completed_at:
        kwargs["completed_at"] = completed_at

    return AgentNode(**kwargs)


def _message_to_dict(msg: Message) -> dict:
    return {
        "id": msg.id,
        "role": msg.role.value,
        "content": msg.content,
        "agent_id": msg.agent_id,
        "snapshot_id": msg.snapshot_id,
        "timestamp": msg.timestamp.isoformat(),
        "streaming": msg.streaming,
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "success": tc.success,
            }
            for tc in msg.tool_calls
        ],
    }


def _dict_to_message(data: dict) -> Message:
    return Message(
        role=MessageRole(data["role"]),
        content=data["content"],
        agent_id=data["agent_id"],
        snapshot_id=data.get("snapshot_id"),
        id=data.get("id", ""),
        timestamp=_ensure_aware(datetime.fromisoformat(data["timestamp"])),
        streaming=data.get("streaming", False),
        tool_calls=[
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc["arguments"],
                result=tc.get("result"),
                success=tc.get("success", True),
            )
            for tc in data.get("tool_calls", [])
        ],
    )
