"""Event processor extracted from MainScreen.

Consumes orchestrator events from the EventBus and updates the TUI:
agent tree, conversation view, tool log, status bar, thinking indicators,
and streaming state.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Markdown

from prsm.engine.models import AgentState
from prsm.shared.models.message import MessageRole
from prsm.adapters.events import (
    AgentKilled,
    AgentRestarted,
    AgentResult,
    AgentSpawned,
    AgentStateChanged,
    ContextWindowUsage,
    EngineStarted,
    EngineFinished,
    FileChanged,
    MessageRouted,
    PermissionRequest,
    SnapshotCreated,
    SnapshotRestored,
    StreamChunk,
    Thinking,
    ToolCallCompleted,
    ToolCallDelta,
    ToolCallStarted,
    UserQuestionRequest,
)
from prsm.adapters.file_tracker import FileChangeTracker, normalize_tool_name

if TYPE_CHECKING:
    from prsm.tui.screens.main import MainScreen

logger = logging.getLogger(__name__)

# Map agent states to status-bar strings (shared by select_agent and event handler)
STATE_TO_STATUS: dict[AgentState, str] = {
    AgentState.PENDING: "connected",
    AgentState.STARTING: "connected",
    AgentState.RUNNING: "streaming",
    AgentState.WAITING_FOR_PARENT: "connected",
    AgentState.WAITING_FOR_CHILD: "connected",
    AgentState.WAITING_FOR_EXPERT: "connected",
    AgentState.COMPLETED: "connected",
    AgentState.FAILED: "error",
    AgentState.KILLED: "error",
}

_TERMINAL_STATES = frozenset({AgentState.COMPLETED, AgentState.FAILED, AgentState.KILLED})


class EventProcessor:
    """Processes orchestrator events on behalf of *MainScreen*.

    Owns the event-consumption loop, per-agent thinking state, and the
    stream buffer / handle bookkeeping.  Keeps a back-reference to the
    screen so it can query widgets without duplicating them.
    """

    def __init__(self, screen: MainScreen) -> None:
        self._screen = screen

        # Per-agent thinking tracking
        self._thinking_agents: set[str] = set()

        # Stream state — one active Markdown widget + handle per agent
        self._active_streams: dict[str, Markdown] = {}
        self._stream_handles: dict[str, object] = {}

        # File change tracking — captures pre-tool content and computes diffs
        self._file_tracker = FileChangeTracker()

        # Liveness watchdog — tracks when the last engine event was received
        self._last_event_at: float = 0.0
        self._liveness_check_interval: float = 30.0
        self._engine_start_time: float = 0.0

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def file_tracker(self) -> FileChangeTracker:
        return self._file_tracker

    @file_tracker.setter
    def file_tracker(self, tracker: FileChangeTracker) -> None:
        self._file_tracker = tracker

    def _file_changes_dir(self) -> Path | None:
        """Return the on-disk directory for current session's file-change records."""
        s = self._screen
        project_dir = s._persistence.project_dir
        if project_dir is None:
            return None
        session_key = self._session_storage_key()
        return project_dir / "sessions" / session_key / "file-changes"

    def _session_storage_key(self) -> str:
        """Resolve the stable UUID storage key for the current session."""
        s = self._screen
        return s.session.session_id

    def _persist_session_for_recovery(self) -> None:
        """Persist session metadata eagerly with file changes for crash recovery."""
        s = self._screen
        session_key = self._session_storage_key()
        display_name = s.session.name or session_key
        s._persistence.save(s.session, display_name, session_id=s.session.session_id)

    def persist_file_changes(self) -> None:
        """Persist file change records to disk for recovery across restarts."""
        changes_dir = self._file_changes_dir()
        if not changes_dir:
            return
        try:
            self._file_tracker.persist(changes_dir)
            self._persist_session_for_recovery()
        except Exception:
            logger.debug("Failed to eagerly persist session/file changes", exc_info=True)

    def load_file_changes(self) -> None:
        """Load persisted file change records from disk."""
        changes_dir = self._file_changes_dir()
        if changes_dir:
            self._file_tracker.load(changes_dir)

    @property
    def thinking_agents(self) -> set[str]:
        return self._thinking_agents

    def _touch_agent(self, agent_id: str, tree=None) -> None:
        """Update an agent's last_active timestamp and re-sort the tree."""
        s = self._screen
        agent = s.session.agents.get(agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)
        if tree is not None:
            tree.sort_by_activity()

    def _display_file_path(self, file_path: str) -> str:
        """Map file paths into workspace alias space and render them relative to workspace root."""
        s = self._screen
        workspace_root_raw = getattr(s, "_cwd", None) or "."
        workspace_root = Path(str(workspace_root_raw))
        file_candidate = Path(file_path)
        abs_path = file_candidate if file_candidate.is_absolute() else (workspace_root / file_candidate)

        worktree_root = s.session.worktree.root if s.session.worktree else None
        if worktree_root:
            worktree_path = Path(worktree_root)
            try:
                rel = abs_path.relative_to(worktree_path)
                abs_path = workspace_root / rel
            except ValueError:
                pass

        try:
            return str(abs_path.relative_to(workspace_root))
        except ValueError:
            return str(abs_path)

    # ── Main event loop ─────────────────────────────────────────────

    async def consume_events(self) -> None:
        """Consume orchestrator events and update the TUI.

        Runs until an ``EngineFinished`` event arrives or the task is
        cancelled externally.
        """
        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView
        from prsm.tui.widgets.status_bar import StatusBar
        from prsm.tui.widgets.tool_log import ToolLog

        s = self._screen
        conv = s.query_one("#conversation", ConversationView)
        sb = s.query_one("#status-bar", StatusBar)
        tl = s.query_one("#tool-log", ToolLog)
        tree = s.query_one("#agent-tree", AgentTree)

        self._last_event_at = time.monotonic()

        async for event in s.bridge.event_bus.consume():
            now = time.monotonic()
            gap = now - self._last_event_at
            if gap > self._liveness_check_interval:
                gap_s = int(gap)
                warning = f"No engine events received for {gap_s}s \u2014 engine may be unresponsive"
                tl.write(f"[yellow]\u26a0 {warning}[/yellow]")
                logger.warning(warning)
                # Show warning in the ThinkingIndicator if one is mounted
                if s._thinking_widget is not None and s._thinking_widget.is_mounted:
                    try:
                        s._thinking_widget._status_text = f"\u26a0 No events for {gap_s}s"
                        s._thinking_widget._render_text()
                    except Exception:
                        pass
            self._last_event_at = now

            try:
                if isinstance(event, AgentSpawned):
                    self._handle_agent_spawned(event, conv, sb, tl, tree)
                elif isinstance(event, AgentRestarted):
                    self._handle_agent_restarted(event, tl, tree)
                elif isinstance(event, AgentStateChanged):
                    self._handle_agent_state_changed(event, sb, tl, tree)
                elif isinstance(event, StreamChunk):
                    await self._handle_stream_chunk(event, conv)
                elif isinstance(event, ToolCallStarted):
                    await self._handle_tool_call_started(event, conv, tl)
                elif isinstance(event, ToolCallCompleted):
                    await self._handle_tool_call_completed(event, conv, tl, tree)
                elif isinstance(event, ToolCallDelta):
                    self._handle_tool_call_delta(event, conv)
                elif isinstance(event, AgentResult):
                    await self._handle_agent_result(event, conv)
                elif isinstance(event, AgentKilled):
                    self._handle_agent_killed(event, tl, tree)
                elif isinstance(event, FileChanged):
                    s._show_file_change(event)
                elif isinstance(event, UserQuestionRequest):
                    s._show_user_question(event)
                elif isinstance(event, PermissionRequest):
                    s._show_permission_modal(event)
                elif isinstance(event, ContextWindowUsage):
                    self._handle_context_window_usage(event, sb)
                elif isinstance(event, MessageRouted):
                    self._handle_message_routed(event, tl)
                elif isinstance(event, Thinking):
                    self._handle_thinking_text(event, tl)
                elif isinstance(event, EngineStarted):
                    self._handle_engine_started(sb)
                elif isinstance(event, SnapshotCreated):
                    self._handle_snapshot_created(event, tl)
                elif isinstance(event, SnapshotRestored):
                    self._handle_snapshot_restored(event, tl)
                elif isinstance(event, EngineFinished):
                    self._handle_engine_finished(event, sb, tl)
                    break
            except Exception:
                logger.exception("Error processing event: %s", event.event_type)

    # ── Individual event handlers ───────────────────────────────────

    def _handle_agent_spawned(self, event: AgentSpawned, conv, sb, tl, tree) -> None:
        s = self._screen
        node = s.bridge.map_agent(
            event.agent_id,
            event.parent_id,
            event.role,
            event.model,
            event.prompt,
            event.name,
        )
        s.session.add_agent(node)
        s.session.add_message(
            event.agent_id, MessageRole.SYSTEM,
            f"Agent spawned (role={event.role}, depth={event.depth})",
        )

        if event.parent_id is None:
            # Remove temp widgets (temporary user message and thinking indicator)
            # This must happen BEFORE adding the new agent to the tree,
            # otherwise the temp user message might get overwritten if
            # the active agent changes before the "real" user message is added.
            s._remove_temp_widgets()

            # Add master as top-level child (preserves previous runs)
            tree.add_agent(None, node)
            tree.root.expand()

            # Select the new master agent and show its conversation
            s._select_agent(event.agent_id)

            # Now add the actual user prompt (if any) to the conversation
            if s._pending_prompt:
                s.session.add_message(
                    event.agent_id,
                    MessageRole.USER,
                    s._pending_prompt,
                    snapshot_id=s._pending_prompt_snapshot_id,
                )
                # Display the user message widget
                conv.add_user_message(event.agent_id, s._pending_prompt, snapshot_id=s._pending_prompt_snapshot_id)
                s._pending_prompt = None
                s._pending_prompt_snapshot_id = None
            
            # Add a clear system message to mark the new orchestration start
            s.session.add_message(
                event.agent_id,
                MessageRole.SYSTEM,
                f"Orchestration started with {node.name} ({event.agent_id[:8]})",
            )
            conv.add_system_message(event.agent_id, f"Orchestration started with {node.name} ({event.agent_id[:8]})")

            # Re-show thinking after rebuild
            self.set_agent_thinking(event.agent_id, True)
        else:
            tree.add_agent(event.parent_id, node)
            # Child agents start thinking when spawned
            self.set_agent_thinking(event.agent_id, True)

        self._touch_agent(event.agent_id, tree)

        tl.write(
            f"[cyan]Spawned[/cyan] {node.name} "
            f"[dim]({event.agent_id[:8]})[/dim]"
        )

    def _handle_agent_restarted(self, event: AgentRestarted, tl, tree) -> None:
        s = self._screen
        agent = s.session.agents.get(event.agent_id)
        if agent:
            agent.state = AgentState.RUNNING
            tree.update_agent_state(event.agent_id, AgentState.RUNNING)
        else:
            node = s.bridge.map_agent(
                event.agent_id,
                event.parent_id,
                event.role,
                event.model,
                event.prompt,
                event.name,
            )
            s.session.add_agent(node)
            tree.add_agent(event.parent_id, node)

        is_root = event.parent_id is None

        if is_root:
            # Root agent follow-up: the user message was already added
            # by on_input_bar_submitted. Just consume the pending prompt
            # and make sure we're viewing this agent (should already be).
            if s._pending_prompt:
                s._pending_prompt = None
            # Remove any temp user widget (thinking indicator was already mounted
            # by on_input_bar_submitted, keep it)
            if s._temp_user_widget:
                try:
                    if s._temp_user_widget.is_mounted:
                        s._temp_user_widget.remove()
                except Exception:
                    pass
                s._temp_user_widget = None
            # Ensure we're viewing the restarted master
            if s.session.active_agent_id != event.agent_id:
                s._select_agent(event.agent_id)

            # Add a clear system message to mark the continuation
            agent_name = agent.name if agent else event.agent_id[:8]
            s.session.add_message(
                event.agent_id,
                MessageRole.SYSTEM,
                f"Continuing orchestration with {agent_name} ({event.agent_id[:8]})",
            )
            s.query_one("#conversation").add_system_message(event.agent_id, f"Continuing orchestration with {agent_name} ({event.agent_id[:8]})")

        self.set_agent_thinking(event.agent_id, True)
        self._touch_agent(event.agent_id, tree)
        tl.write(
            f"[cyan]Restarted[/cyan] {event.agent_id[:8]} "
            f"[dim](continuing conversation)[/dim]"
        )

    def _handle_agent_state_changed(self, event: AgentStateChanged, sb, tl, tree) -> None:
        s = self._screen
        new_state = s.bridge.map_state(event.new_state)
        tree.update_agent_state(event.agent_id, new_state)
        # Clean up futures if agent failed or was killed
        if event.new_state in ("failed", "killed"):
            s.bridge.cancel_agent_futures(event.agent_id)
        agent = s.session.agents.get(event.agent_id)
        if agent:
            agent.state = new_state
        self._touch_agent(event.agent_id, tree)
        # Stop thinking for terminal states and parent wait-on-child state;
        # start thinking when entering RUNNING.
        if new_state in _TERMINAL_STATES or new_state == AgentState.WAITING_FOR_CHILD:
            self.set_agent_thinking(event.agent_id, False)
        elif new_state == AgentState.RUNNING:
            self.set_agent_thinking(event.agent_id, True)
        # Update status bar if this is the active agent
        if event.agent_id == s.session.active_agent_id:
            sb.status = STATE_TO_STATUS.get(new_state, "connected")

    async def _handle_stream_chunk(self, event: StreamChunk, conv) -> None:
        s = self._screen
        self._touch_agent(event.agent_id)  # No tree re-sort for high-frequency events
        self.set_agent_thinking(event.agent_id, False)
        if event.agent_id == s.session.active_agent_id:
            await self._write_stream_chunk(event.agent_id, event.text)
        else:
            # Buffer for non-active agents
            conv.buffer_stream_chunk(event.agent_id, event.text)

    async def _handle_tool_call_started(self, event: ToolCallStarted, conv, tl) -> None:
        self._touch_agent(event.agent_id)  # No tree re-sort for tool starts
        self.set_agent_thinking(event.agent_id, False)
        canonical_tool_name = normalize_tool_name(event.tool_name)
        safe_args = event.arguments[:80].replace("[", "\\[")
        tl.write(
            f"[dim]▤[/dim] [cyan]{canonical_tool_name}[/cyan] ({safe_args})"
        )
        # Capture pre-tool file content for canonical Write/Edit tools.
        self._file_tracker.capture_pre_tool(event.tool_id, canonical_tool_name, event.arguments)
        # Finalize any active stream before showing tool call
        await self.finalize_stream(event.agent_id)
        conv.add_tool_call(
            event.agent_id,
            canonical_tool_name,
            event.arguments,  # Full args — engine already truncates (2000/50000)
            "",  # Result comes later
            tool_id=event.tool_id,
        )

    async def _handle_tool_call_completed(self, event: ToolCallCompleted, conv, tl, tree) -> None:
        self._touch_agent(event.agent_id, tree)
        result_preview = event.result[:100] if event.result else ""
        safe_result = result_preview.replace("[", "\\[")
        color = "red" if event.is_error else "green"
        result_icon = "✘" if event.is_error else "✔"
        tl.write(f"[dim]  └─>[/dim] [{color}]{result_icon} {safe_result}[/{color}]")

        # Peek at the pending tool call to detect task_complete before it's removed
        task_complete_summary = self._extract_task_complete_summary(conv, event)

        # Update the pending tool call message
        conv.update_tool_result(
            event.tool_id,
            event.result,
            is_error=event.is_error,
        )

        # Inject task_complete summary as a visible assistant message in the conversation
        if task_complete_summary and not event.is_error:
            conv.add_assistant_message(event.agent_id, task_complete_summary)

        # Track file changes from Write/Edit tools
        if not event.is_error:
            records = self._file_tracker.track_changes(event.agent_id, event.tool_id)
            for record in records:
                file_changed = FileChanged(
                    agent_id=event.agent_id,
                    file_path=self._display_file_path(record.file_path),
                    change_type=record.change_type,
                    tool_call_id=record.tool_call_id,
                    tool_name=record.tool_name,
                    message_index=record.message_index,
                    old_content=record.old_content,
                    new_content=record.new_content,
                    pre_tool_content=record.pre_tool_content,
                    added_ranges=record.added_ranges,
                    removed_ranges=record.removed_ranges,
                )
                # Emit directly to the screen handler
                self._screen._show_file_change(file_changed)
            if records:
                # Persist to disk for recovery across restarts
                self.persist_file_changes()

        # Check for pending inject prompt on this agent
        s = self._screen
        inject_prompts = s._inject_prompts_by_agent.get(event.agent_id, [])
        if inject_prompts:
            inject_prompt = inject_prompts.pop(0)
            if not inject_prompts:
                s._inject_prompts_by_agent.pop(event.agent_id, None)
            tl.write("[yellow]Injecting[/yellow] queued prompt after tool call")
            # Trigger interrupt-style restart with the inject prompt.
            s._do_interrupt(inject_prompt)
            return

        # Agent goes back to thinking after tool completes
        self.set_agent_thinking(event.agent_id, True)

    def _extract_task_complete_summary(self, conv, event: ToolCallCompleted) -> str | None:
        """Extract the summary from a task_complete tool call if present.

        Returns the summary string if the completing tool is task_complete and
        has a non-empty summary argument, otherwise returns None.

        Note: conv.add_tool_call stores the already-normalized tool name
        (via normalize_tool_name), so tc.name will be "task_complete" not
        the full "mcp__orchestrator__task_complete" prefix.
        """
        from prsm.shared.formatters.tool_call import parse_args

        entry = conv._pending_tool_calls.get(event.tool_id)
        if entry is None:
            return None
        msg, _ = entry
        if not msg.tool_calls:
            return None
        tc = msg.tool_calls[0]
        bare_name = str(tc.name).strip()
        # Handle both prefixed and already-normalized names
        if bare_name.startswith("mcp__") and "__" in bare_name:
            bare_name = bare_name.split("__", 2)[-1]
        if bare_name != "task_complete":
            return None
        args = parse_args(tc.arguments)
        summary = str(args.get("summary", "")).strip()
        return summary if summary else None

    def _handle_tool_call_delta(self, event: ToolCallDelta, conv) -> None:
        self._touch_agent(event.agent_id)  # No tree re-sort for high-frequency events
        conv.update_bash_live_output(
            agent_id=event.agent_id,
            tool_id=event.tool_id,
            delta=event.delta,
            stream=event.stream,
        )

    async def _handle_agent_result(self, event: AgentResult, conv) -> None:
        # Finalize stream and flush buffer for this agent
        await self.finalize_stream(event.agent_id)
        conv.flush_stream_buffer(event.agent_id)

    def _handle_agent_killed(self, event: AgentKilled, tl, tree) -> None:
        s = self._screen
        tree.update_agent_state(event.agent_id, AgentState.KILLED)
        self.set_agent_thinking(event.agent_id, False)
        self._touch_agent(event.agent_id, tree)
        # Clean up any pending futures for this agent
        s.bridge.cancel_agent_futures(event.agent_id)
        tl.write(
            f"[red]Killed[/red] agent "
            f"[dim]({event.agent_id[:8]})[/dim]"
        )

    def _handle_engine_finished(self, event: EngineFinished, sb, tl) -> None:
        self.clear_all_thinking()
        sb.status = "connected"

        # Prefer the authoritative duration from the engine event;
        # fall back to locally computed wall-clock duration.
        if event.duration_seconds:
            duration = event.duration_seconds
            duration_str = f" in {duration:.1f}s"
        elif self._engine_start_time:
            duration = time.monotonic() - self._engine_start_time
            duration_str = f" in {duration:.1f}s"
        else:
            duration_str = ""
        self._engine_start_time = 0.0

        # During an interrupt, suppress the error display — the
        # engine was intentionally stopped and a continuation is about
        # to start immediately.
        if getattr(self._screen, '_interrupt_in_progress', False):
            self._screen._interrupt_in_progress = False
            tl.write("[yellow]Interrupted[/yellow] \u2014 continuing with new prompt")
            return

        if event.success:
            tl.write(f"[green]Engine finished{duration_str}[/green]")
        else:
            error_msg = event.error or "unknown error"
            safe_err = error_msg.replace("[", "\\[")
            tl.write(f"[red]Engine failed{duration_str}:[/red] {safe_err}")
            self._screen._show_error_in_conversation(error_msg)

    def _handle_engine_started(self, sb) -> None:
        self._engine_start_time = time.monotonic()
        sb.status = "streaming"

    def _handle_context_window_usage(self, event: ContextWindowUsage, sb) -> None:
        if event.agent_id == self._screen.session.active_agent_id:
            sb.tokens_used = int(event.total_tokens or 0)

    def _handle_message_routed(self, event: MessageRouted, tl) -> None:
        msg_type = (event.message_type or "").replace("[", "\\[")
        tl.write(
            f"[dim]↔ routed[/dim] {event.source_agent_id[:8]} -> "
            f"{event.target_agent_id[:8]} [dim]({msg_type})[/dim]"
        )

    def _handle_thinking_text(self, event: Thinking, tl) -> None:
        text = (event.text or "").strip()
        if not text:
            return
        safe_text = text[:160].replace("[", "\\[")
        tl.write(f"[dim]… {safe_text}[/dim]")

    def _handle_snapshot_created(self, event: SnapshotCreated, tl) -> None:
        tl.write(
            f"[dim]snapshot created[/dim] "
            f"{event.snapshot_id[:12]} [dim]({event.description[:48]})[/dim]"
        )

    def _handle_snapshot_restored(self, event: SnapshotRestored, tl) -> None:
        tl.write(
            f"[yellow]snapshot restored[/yellow] "
            f"{event.snapshot_id[:12]}"
        )

    # ── Thinking indicator management ───────────────────────────────

    def set_agent_thinking(self, agent_id: str, thinking: bool) -> None:
        """Update an agent's thinking state and sync the display."""
        if thinking:
            self._thinking_agents.add(agent_id)
        else:
            self._thinking_agents.discard(agent_id)
        self.sync_thinking()

    def sync_thinking(self) -> None:
        """Mount or remove the ThinkingIndicator based on active agent's state."""
        from prsm.tui.widgets.thinking import ThinkingIndicator
        from prsm.tui.widgets.conversation import ConversationView

        s = self._screen
        active = s.session.active_agent_id
        should_show = active is not None and active in self._thinking_agents
        agent = s.session.agents.get(active) if active else None
        agent_name = agent.name if agent else None
        status_text = (
            "Waiting for child agent..."
            if agent and agent.state == AgentState.WAITING_FOR_CHILD
            else None
        )

        if should_show:
            conv = s.query_one("#conversation", ConversationView)
            needs_remount = (
                s._thinking_widget is None
                or s._thinking_widget.agent_name != agent_name
                or s._thinking_widget.status_text != status_text
            )
            if needs_remount:
                if s._thinking_widget is not None:
                    try:
                        if s._thinking_widget.is_mounted:
                            s._thinking_widget.remove()
                    except Exception:
                        pass
                s._thinking_widget = ThinkingIndicator(
                    agent_name=agent_name,
                    status_text=status_text,
                )
                conv.mount_before_queued(s._thinking_widget)
                conv._smart_scroll()
        elif not should_show and s._thinking_widget is not None:
            try:
                if s._thinking_widget.is_mounted:
                    s._thinking_widget.remove()
            except Exception:
                pass
            s._thinking_widget = None

    def clear_all_thinking(self) -> None:
        """Clear all thinking state (engine finished/error)."""
        s = self._screen
        self._thinking_agents.clear()
        if s._thinking_widget is not None:
            try:
                if s._thinking_widget.is_mounted:
                    s._thinking_widget.remove()
            except Exception:
                pass
            s._thinking_widget = None

    # ── Stream management ───────────────────────────────────────────

    async def _write_stream_chunk(self, agent_id: str, text: str) -> None:
        """Write a streaming chunk to the active agent's conversation."""
        from prsm.tui.widgets.conversation import ConversationView

        s = self._screen
        conv = s.query_one("#conversation", ConversationView)
        container = conv.query_one("#message-container")

        if agent_id not in self._active_streams:
            # Start a new streaming Markdown widget, inserting before
            # any queued prompt widgets to keep them pinned to the bottom
            md = Markdown("", classes="message-assistant")
            conv.mount_before_queued(md)
            # Only anchor (auto-follow) if user is already near the bottom
            if conv.is_near_bottom():
                container.anchor()
            stream = Markdown.get_stream(md)
            self._active_streams[agent_id] = md
            self._stream_handles[agent_id] = stream

        stream = self._stream_handles.get(agent_id)
        if stream:
            await stream.write(text)

    async def finalize_stream(self, agent_id: str) -> None:
        """Finalize a streaming response for an agent."""
        stream = self._stream_handles.pop(agent_id, None)
        _md = self._active_streams.pop(agent_id, None)
        if stream:
            await stream.stop()

    async def finalize_all_streams(self) -> None:
        """Finalize all active streams."""
        for agent_id in list(self._active_streams.keys()):
            await self.finalize_stream(agent_id)
