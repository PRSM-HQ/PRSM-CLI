"""Session lifecycle manager extracted from MainScreen.

Handles session creation, resume, fork, save, restore, and resend —
keeping MainScreen focused on layout, event consumption, and
orchestrator lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from prsm.shared.models.session import Session

if TYPE_CHECKING:
    from prsm.tui.screens.main import MainScreen
    from prsm.tui.widgets.tool_log import ToolLog

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session lifecycle on behalf of MainScreen.

    Keeps a reference to the screen so it can access widgets, bridge,
    persistence, and plugin state without duplicating any of them.
    """

    def __init__(self, screen: MainScreen) -> None:
        self._screen = screen
        self._yaml_config = self._auto_discover_yaml_config()

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def _tl(self) -> ToolLog:
        from prsm.tui.widgets.tool_log import ToolLog

        return self._screen.query_one("#tool-log", ToolLog)

    def _auto_discover_yaml_config(self):
        """Auto-discover and load .prism/prsm.yaml or prsm.yaml.

        Mirrors the config discovery logic used by the server and app.py.
        Returns the parsed OrchestrationConfig or None.
        """
        cli_args = getattr(self._screen.app, "cli_args", None)
        config_path = getattr(cli_args, "config", None) if cli_args else None

        if not config_path:
            cwd = self._screen._cwd
            prism_yaml = cwd / ".prism" / "prsm.yaml"
            legacy_yaml = cwd / "prsm.yaml"
            if prism_yaml.exists():
                config_path = str(prism_yaml)
            elif legacy_yaml.exists():
                config_path = str(legacy_yaml)

        if config_path:
            try:
                from prsm.engine.yaml_config import load_yaml_config
                yaml_config = load_yaml_config(config_path)
                logger.info(
                    "TUI auto-discovered YAML config: %s "
                    "(providers=%d, models=%d, experts=%d)",
                    config_path,
                    len(yaml_config.providers),
                    len(yaml_config.models),
                    len(yaml_config.experts),
                )
                return yaml_config
            except FileNotFoundError:
                logger.warning("Config file not found: %s", config_path)
            except Exception:
                logger.exception(
                    "Failed to load YAML config from %s", config_path
                )
        return None

    # ── public API ──────────────────────────────────────────────────

    def handle_cli_args(self, cli_args) -> None:
        """Dispatch session setup based on CLI arguments.

        Called from MainScreen.on_mount() after plugins/memory are loaded.
        """
        tl = self._tl
        s = self._screen

        if cli_args and getattr(cli_args, "fork_snapshot", None):
            self.fork_from_snapshot_cli(cli_args.fork_snapshot)
        elif cli_args and getattr(cli_args, "fork", None):
            self.fork_from_cli(cli_args.fork)
        elif cli_args and getattr(cli_args, "resume", None):
            self.resume_named(cli_args.resume)
        elif cli_args and getattr(cli_args, "new", False):
            self.start_fresh_session()
        else:
            # Default: auto-resume or fresh
            resumed = s._persistence.auto_resume()
            if resumed and resumed.agents:
                s.session = resumed
                self.restore_session(resumed)
                tl.write(
                    f"[green]Resumed[/green] session "
                    f"({len(resumed.agents)} agents, "
                    f"{resumed.message_count} messages)"
                )
            else:
                self.start_fresh_session()

    def start_fresh_session(self) -> None:
        """Start a fresh session (new or default startup)."""
        s = self._screen
        tl = self._tl

        plugin_configs = (
            s._plugin_manager.get_mcp_server_configs()
            if s._plugin_manager
            else {}
        )
        if s.bridge.available:
            configure_kwargs: dict = {
                "plugin_mcp_servers": plugin_configs,
                "plugin_manager": s._plugin_manager,
                "project_dir": s._persistence.project_dir,
            }
            if self._yaml_config is not None:
                configure_kwargs["yaml_config"] = self._yaml_config
            s._live_mode = s.bridge.configure(**configure_kwargs)

        if s._live_mode:
            s._setup_live_mode()
        else:
            s._setup_demo_mode()
            s._populate_demo_agents()
            s._populate_demo_conversations()
        
        # Select the root agent after setup
        s._select_agent("root")

        # Log session start info
        repo_context = s._persistence.repo_context
        branch_info = ""
        if repo_context and repo_context.is_git_repo and repo_context.branch:
            branch_info = f" on branch '{repo_context.branch}'"
        
        tl.write(f"[green]Starting new session{branch_info}[/green]")
        if s._live_mode:
            tl.write("[dim]Connected to orchestrator[/dim]")
        else:
            tl.write("[dim]Demo mode (simulated responses)[/dim]")

    def resume_named(self, name: str) -> None:
        """Resume a specific named session from CLI --resume flag."""
        s = self._screen
        tl = self._tl

        try:
            session = s._persistence.load(name)
            s.session = session
            self.restore_session(session)
            tl.write(
                f"[green]Resumed[/green] session '{name}' "
                f"({len(session.agents)} agents, "
                f"{session.message_count} messages)"
            )
        except FileNotFoundError:
            tl.write(f"[red]Session '{name}' not found[/red]")
            self.start_fresh_session()

    def fork_from_cli(self, name: str) -> None:
        """Fork an existing session from CLI --fork flag."""
        s = self._screen
        tl = self._tl

        from prsm.shared.models.session import format_forked_name

        try:
            original = s._persistence.load(name)
            base_name = original.name or name
            original.name = base_name
            forked = original.fork(new_name=format_forked_name(base_name))
            s.session = forked
            self.restore_session(forked)
            tl.write(
                f"[green]Forked[/green] session '{name}' "
                f"({len(forked.agents)} agents, "
                f"{forked.message_count} messages)"
            )
        except FileNotFoundError:
            tl.write(
                f"[red]Session '{name}' not found, cannot fork[/red]"
            )
            self.start_fresh_session()

    def fork_from_snapshot_cli(self, snapshot_id: str) -> None:
        """Fork a session from snapshot metadata without restoring files."""
        s = self._screen
        tl = self._tl

        from prsm.shared.models.session import format_forked_name
        from prsm.shared.services.snapshot import SnapshotService

        project_dir = s._persistence.project_dir
        if project_dir is None:
            tl.write("[red]Snapshots unavailable outside a project workspace[/red]")
            self.start_fresh_session()
            return

        snapshot_service = SnapshotService(project_dir, Path.cwd())
        try:
            meta = snapshot_service.get_meta(snapshot_id)
            snapshot_session, file_tracker = snapshot_service.load_session(snapshot_id)
            base_name = meta.get("session_name") or snapshot_session.name or snapshot_id
            snapshot_session.name = base_name
            forked = snapshot_session.fork(new_name=format_forked_name(base_name))
            forked.forked_from = f"snapshot:{snapshot_id}"
            s.session = forked
            self.restore_session(forked)
            s.event_processor.file_tracker = file_tracker
            tl.write(
                f"[green]Forked[/green] snapshot '{snapshot_id}' "
                f"({len(forked.agents)} agents, {forked.message_count} messages)"
            )
        except FileNotFoundError:
            tl.write(f"[red]Snapshot '{snapshot_id}' not found[/red]")
            self.start_fresh_session()

    def restore_session(self, session: Session) -> None:
        """Rebuild the UI from a loaded session."""
        from prsm.engine.models import AgentState
        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView
        from prsm.tui.widgets.status_bar import StatusBar

        s = self._screen
        tree = s.query_one("#agent-tree", AgentTree)
        conv = s.query_one("#conversation", ConversationView)
        sb = s.query_one("#status-bar", StatusBar)

        # Update conversation's session reference
        conv.session = session

        # Add agents to tree
        for agent in session.agents.values():
            parent_id = agent.parent_id
            tree.add_agent(parent_id, agent)

        # Sort agents by last activity (most recent first)
        tree.sort_by_activity()

        # Restore persisted file change records
        s.event_processor.load_file_changes()

        # Try to configure live mode for future prompts
        if s.bridge.available:
            plugin_configs = (
                s._plugin_manager.get_mcp_server_configs()
                if s._plugin_manager
                else {}
            )
            configure_kwargs: dict = {
                "plugin_mcp_servers": plugin_configs,
                "plugin_manager": s._plugin_manager,
                "project_dir": s._persistence.project_dir,
            }
            if self._yaml_config is not None:
                configure_kwargs["yaml_config"] = self._yaml_config
            s._live_mode = s.bridge.configure(**configure_kwargs)
            sb.status = "connected"
            sb.mode = "live"
            s._sync_model_indicators(s.bridge.current_model_display_name)
        else:
            sb.status = "connected"
            sb.mode = "demo"
            s.app.sub_title = "\u26a0 DEMO MODE \u2014 Resumed Session"

        # Select the previously active agent, or root
        active_id = session.active_agent_id
        if active_id and active_id in session.agents:
            s._select_agent(active_id)
        elif session.agents:
            first_id = next(iter(session.agents))
            s._select_agent(first_id)

    def save_session(self) -> Path | None:
        """Save the current session. Returns the path, or None on error."""
        s = self._screen
        if s.session.message_count == 0:
            return None
        try:
            from datetime import datetime

            name = s.session.name or datetime.now().strftime(
                "session_%Y%m%d_%H%M%S"
            )
            path = s._persistence.save(s.session, name, session_id=s.session.session_id)
            # Also persist file changes alongside the session
            s.event_processor.persist_file_changes()
            return path
        except Exception:
            logger.exception("Failed to save session")
            return None

    async def auto_name_session(self, prompt: str) -> None:
        """Generate a descriptive session name from the user prompt."""
        try:
            from prsm.shared.services.session_naming import (
                generate_session_name,
            )
            from prsm.shared.models.session import format_forked_name
            from prsm.tui.widgets.agent_tree import AgentTree

            s = self._screen
            name = await generate_session_name(prompt)
            if name:
                if s.session.forked_from or (s.session.name or "").startswith("(Forked) "):
                    name = format_forked_name(name)
                s.session.name = name
                tree = s.query_one("#agent-tree", AgentTree)
                tree.root.set_label(name)
                tree.root.refresh()
        except Exception:
            pass  # Non-critical -- keep default name

    # ── Snapshot helpers ───────────────────────────────────────────

    def _get_snapshot_service(self):
        """Create a SnapshotService for the current project."""
        from prsm.shared.services.snapshot import SnapshotService

        s = self._screen
        project_dir = s._persistence.project_dir
        if project_dir is None:
            return None
        return SnapshotService(project_dir, s._cwd)

    def create_auto_snapshot(self, description: str = "") -> str | None:
        """Create a snapshot of the current session + working tree.

        Called automatically before each prompt to enable resend/restore.
        Returns the snapshot ID, or None if snapshots are unavailable.
        """
        s = self._screen
        svc = self._get_snapshot_service()
        if svc is None:
            return None

        try:
            session_name = s.session.name or "unnamed"
            snapshot_id = svc.create(
                s.session,
                session_name,
                description,
                file_tracker=s.event_processor.file_tracker,
                session_id=s.session.session_id,
            )
            logger.info("Auto-snapshot created: %s (%s)", snapshot_id, description)
            return snapshot_id
        except Exception:
            logger.debug("Auto-snapshot failed", exc_info=True)
            return None

    # ── Resend from message ────────────────────────────────────────

    def count_file_changes_after(self, agent_id: str, message_index: int) -> int:
        """Count unique files changed after the given message index.

        Examines file change records from the tracker and counts those
        that were recorded at a message_index > the given one.
        """
        s = self._screen
        tracker = s.event_processor.file_tracker
        files_after: set[str] = set()
        for records in tracker.file_changes.values():
            for r in records:
                if r.message_index > message_index:
                    files_after.add(r.file_path)
        return len(files_after)

    def truncate_session_to(self, agent_id: str, message_index: int) -> None:
        """Truncate the session's message history to the given message index.

        Keeps messages [0..message_index) — i.e. everything before the
        selected user message (excluding it, since it will be re-sent).
        Messages for ALL agents are truncated to remove anything that
        happened after the selected message's timestamp.
        """
        s = self._screen
        session = s.session

        # Get the timestamp of the target message as the cutoff
        target_msgs = session.get_messages(agent_id)
        if message_index < 0 or message_index >= len(target_msgs):
            return
        cutoff_time = target_msgs[message_index].timestamp

        # Truncate the target agent's messages to before the selected one
        session.messages[agent_id] = target_msgs[:message_index]

        # For other agents, remove messages that came after the cutoff
        for aid in list(session.messages.keys()):
            if aid == agent_id:
                continue
            session.messages[aid] = [
                m for m in session.messages[aid]
                if m.timestamp < cutoff_time
            ]

        # Clean up file tracker records after the cutoff
        tracker = s.event_processor.file_tracker
        for fp in list(tracker.file_changes.keys()):
            tracker.file_changes[fp] = [
                r for r in tracker.file_changes[fp]
                if r.message_index < message_index
            ]
            if not tracker.file_changes[fp]:
                del tracker.file_changes[fp]

    def restore_snapshot_files(self, snapshot_id: str) -> bool:
        """Restore the working tree from a snapshot (files only).

        Uses the SnapshotService to reset tracked files to HEAD,
        then apply the saved patch and untracked files.
        Returns True if successful.
        """
        svc = self._get_snapshot_service()
        if svc is None:
            return False

        try:
            # We use the full restore which handles git checkout, patch, and
            # untracked files — but we discard the returned session since we
            # manage the session state ourselves via truncation.
            svc.restore(snapshot_id)
            return True
        except Exception:
            logger.debug("Failed to restore snapshot files %s", snapshot_id, exc_info=True)
            return False

    def rebuild_ui_after_resend(self) -> None:
        """Rebuild the UI widgets after truncating the session.

        Clears and re-renders the agent tree and conversation from
        the (now truncated) session state.
        """
        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView

        s = self._screen
        session = s.session
        tree = s.query_one("#agent-tree", AgentTree)
        conv = s.query_one("#conversation", ConversationView)

        # Clear and rebuild agent tree
        tree.clear_agents()
        for agent in session.agents.values():
            tree.add_agent(agent.parent_id, agent)
        tree.sort_by_activity()

        # Rebuild conversation view — force=True because the active agent
        # hasn't changed but its message list was truncated by resend.
        conv.session = session
        if session.active_agent_id and session.active_agent_id in session.agents:
            conv.show_agent(session.active_agent_id, force=True)
        elif session.agents:
            first_id = next(iter(session.agents))
            conv.show_agent(first_id, force=True)
