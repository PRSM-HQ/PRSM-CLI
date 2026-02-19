"""Main screen — primary split-pane workspace."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Tree

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import MessageRole, ToolCall
from prsm.shared.models.session import Session
from prsm.adapters.events import (
    FileChanged,
    PermissionRequest,
    UserQuestionRequest,
)
from prsm.adapters.orchestrator import OrchestratorBridge
from prsm.shared.services.persistence import SessionPersistence
from prsm.shared.services.plugins import PluginManager
from prsm.shared.services.project import ProjectManager
from prsm.shared.services.project_memory import ProjectMemory
from prsm.tui.widgets.agent_context_panel import AgentContextPanel
from prsm.tui.widgets.agent_tree import AgentTree
from prsm.tui.widgets.conversation import (
    ConversationView,
    MessageWidget,
    QueuedPromptWidget,
    RunStoppedIndicator,
)
from prsm.tui.widgets.file_change import FileChangeWidget
from prsm.tui.widgets.input_bar import InputBar
from prsm.tui.widgets.question import QuestionWidget
from prsm.tui.widgets.status_bar import StatusBar
from prsm.tui.widgets.tool_log import ToolLog
from prsm.tui.handlers.command_handler import CommandHandler
from prsm.tui.handlers.event_processor import EventProcessor, STATE_TO_STATUS
from prsm.tui.handlers.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MainScreen(Screen):
    """Primary workspace with agent tree, conversation, and input."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = Session()
        self.bridge = OrchestratorBridge()
        self._live_mode = False
        self._event_consumer_task: asyncio.Task | None = None
        # Directory-aware persistence
        self._cwd = Path.cwd()
        self._persistence = SessionPersistence(cwd=self._cwd)
        self._plugin_manager: PluginManager | None = None
        self._project_memory: str = ""
        self._project_instructions: str = ""
        # Thinking indicator state for live mode
        self._pending_prompt: str | None = None
        self._thinking_widget = None  # Currently mounted ThinkingIndicator
        self._temp_user_widget = None
        # Delivery mode state (inject / queue)
        self._pending_inject_text: str | None = None
        self._inject_prompts_by_agent: dict[str, list[str]] = {}
        self._delivery_queued_prompts: list[tuple[int, str]] = []
        self._queued_prompt_widgets: dict[int, QueuedPromptWidget] = {}
        self._plan_index_counter: int = 0  # Counts up from 1 for each queued plan
        self._interrupt_in_progress: bool = False
        self._user_stop_requested: bool = False
        self.command_handler = CommandHandler(self)
        self.event_processor = EventProcessor(self)
        self.session_manager = SessionManager(self)
        # Last auto-snapshot ID for resend/restore
        self._last_snapshot_id: str | None = None
        # Snapshot taken immediately before the currently pending prompt.
        self._pending_prompt_snapshot_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            yield AgentTree(id="agent-tree")
            with Vertical(id="main-pane"):
                yield ConversationView(session=self.session, id="conversation")
                yield ToolLog(id="tool-log")
            yield AgentContextPanel(id="agent-context-panel")
        yield InputBar(cwd=self._cwd, id="input-bar")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        tl = self.query_one("#tool-log", ToolLog)

        # Load project memory (MEMORY.md in ~/.prsm/projects/{ID}/memory/)
        if self._persistence.project_dir:
            memory_path = ProjectManager.get_memory_path(
                self._persistence.project_dir
            )
            memory = ProjectMemory(memory_path)
            if memory.exists():
                self._project_memory = memory.load()
                tl.write(f"[dim]Loaded project memory ({len(self._project_memory)} chars)[/dim]")

        # Load PRSM.md project instructions from cwd (like CLAUDE.md)
        prsm_md = self._cwd / "PRSM.md"
        if prsm_md.exists():
            self._project_instructions = prsm_md.read_text()
            tl.write("[dim]Loaded PRSM.md project instructions[/dim]")

        # Initialize plugin manager
        self._plugin_manager = PluginManager(
            project_dir=self._persistence.project_dir,
            cwd=self._cwd,
        )
        loaded_plugins = self._plugin_manager.list_plugins()
        if loaded_plugins:
            tl.write(f"[dim]Loaded {len(loaded_plugins)} plugin(s)[/dim]")

        # CLI flag handling — delegated to SessionManager
        cli_args = getattr(self.app, "cli_args", None)
        self.session_manager.handle_cli_args(cli_args)

        self.query_one(InputBar).focus_input()

        # Dynamically set placeholder text based on Git context
        repo_context = (
            self._persistence.repo_context
            or ProjectManager.get_repository_context(self._cwd)
        )
        placeholder_text = "Type your prompt or /command..."
        if repo_context.is_git_repo and repo_context.branch:
            placeholder_text = f"Prompt on `{repo_context.branch}` branch or /command..."
        self.query_one(InputBar).set_placeholder(placeholder_text)

    def _setup_live_mode(self) -> None:
        """Set up the UI for live orchestrator mode."""
        tl = self.query_one("#tool-log", ToolLog)
        tl.write("[green]Connected[/green] to Claude orchestrator")
        sb = self.query_one("#status-bar", StatusBar)
        sb.status = "connected"
        sb.mode = "live"
        sb.agent_name = "Ready"
        self._refresh_model_display(show_demo_banner=False)

    def _sync_model_indicators(self, model: str) -> None:
        """Keep model display consistent across status, header, and input."""
        sb = self.query_one("#status-bar", StatusBar)
        sb.model = model
        input_bar = self.query_one(InputBar)
        input_bar.update_model_label(model)
        self.app.sub_title = f"Agent Orchestrator · {model}"

    def request_stop_current_run(self) -> None:
        """Stop the active run immediately."""
        if not self.bridge.running:
            return
        self._user_stop_requested = True
        self._stop_current_run()

    @work(name="stop-orchestration")
    async def _stop_current_run(self) -> None:
        """Hard-stop the active run and let the run worker finalize UI."""
        tl = self.query_one("#tool-log", ToolLog)
        self.event_processor.clear_all_thinking()
        tl.write("[yellow]Stopping run...[/yellow]")
        try:
            await self.bridge.shutdown()
        except Exception as exc:
            safe_err = str(exc).replace("[", "\\[")
            tl.write(f"[red]Stop failed:[/red] {safe_err}")

    def _setup_demo_mode(self) -> None:
        """Set up the UI for demo mode with prominent indicators."""
        sb = self.query_one("#status-bar", StatusBar)
        sb.mode = "demo"
        sb.status = "disconnected"

        tl = self.query_one("#tool-log", ToolLog)
        tl.write("[bold yellow]\u26a0 DEMO MODE[/bold yellow] — "
                 "No orchestrator connected. Responses are simulated.")
        tl.write("[dim]Install Claude Code CLI and run "
                 "`claude auth login` for live mode.[/dim]")

        self._refresh_model_display(show_demo_banner=True)

        # Add a prominent banner to the conversation area
        from textual.widgets import Static
        container = self.query_one("#conversation #message-container")
        banner = Static(
            "\u26a0  DEMO MODE  \u26a0\n"
            "Responses are simulated. No API calls are being made.\n"
            "Install Claude Code CLI and run `claude auth login` for live orchestration.",
            classes="demo-banner",
        )
        container.mount(banner)

        # Update input placeholder
        self.query_one(InputBar).set_placeholder(
            "Demo mode \u2014 type anything to see a simulated response"
        )

    def _refresh_model_display(self, *, show_demo_banner: bool) -> None:
        """Sync default/current model into header, status bar, and wrench button."""
        display_model = self.bridge.current_model_display
        sb = self.query_one("#status-bar", StatusBar)
        sb.model = display_model
        self.query_one(InputBar).update_model_label(display_model)
        if show_demo_banner:
            self.app.sub_title = f"\u26a0 DEMO MODE \u2014 Default model: {display_model}"
        else:
            self.app.sub_title = f"Agent Orchestrator \u2014 Default model: {display_model}"

    def _start_fresh_session(self, tl: ToolLog | None = None) -> None:
        """Start a fresh session — delegates to SessionManager."""
        self.session_manager.start_fresh_session()

    def _restore_session(self, session: Session) -> None:
        """Rebuild the UI from a loaded session — delegates to SessionManager."""
        self.session_manager.restore_session(session)

    def save_session(self) -> Path | None:
        """Save the current session — delegates to SessionManager."""
        return self.session_manager.save_session()

    # ── Event Handlers ──

    def _select_agent(self, agent_id: str) -> None:
        """Switch the active agent: update session, conversation, and status bar."""
        self.session.set_active(agent_id)
        conv = self.query_one("#conversation", ConversationView)
        conv.show_agent(agent_id)

        agent = self.session.agents.get(agent_id)
        if agent:
            sb = self.query_one("#status-bar", StatusBar)
            sb.agent_name = agent.name
            sb.model = agent.model
            sb.status = STATE_TO_STATUS.get(agent.state, "connected")
            sb.tokens_used = len(self.session.get_messages(agent_id)) * 150

        # Show/hide thinking indicator based on new agent's state
        self.event_processor.sync_thinking()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        agent = event.node.data
        if agent:
            self._select_agent(agent.id)

    def on_agent_tree_context_menu_requested(
        self, event: AgentTree.ContextMenuRequested,
    ) -> None:
        """Handle right-click on an agent — show context menu."""
        from prsm.tui.screens.agent_context_menu import AgentContextMenu

        agent = event.agent

        def _on_menu_result(action: str | None) -> None:
            if action == "kill":
                self._confirm_kill_agent(agent)
            elif action == "view":
                self._open_agent_context_panel(agent.id)

        self.app.push_screen(AgentContextMenu(agent), callback=_on_menu_result)

    def on_agent_tree_view_context_requested(
        self, event: AgentTree.ViewContextRequested,
    ) -> None:
        """Handle 'i' key on an agent in the tree — open context panel."""
        self._open_agent_context_panel(event.agent.id)

    def on_agent_tree_kill_requested(self, event: AgentTree.KillRequested) -> None:
        """Handle Delete key on an agent in the tree — show confirmation first."""
        self._confirm_kill_agent(event.agent)

    def _confirm_kill_agent(self, agent) -> None:
        """Show a confirmation dialog before killing/removing an agent."""
        from prsm.tui.screens.kill_confirm import KillConfirmScreen

        def _on_confirm(result: str | None) -> None:
            if result == "kill":
                self._do_kill_agent(agent)

        self.app.push_screen(KillConfirmScreen(agent), callback=_on_confirm)

    def _do_kill_agent(self, agent) -> None:
        """Kill an agent while preserving it in the tree/history."""
        tl = self.query_one("#tool-log", ToolLog)
        tree = self.query_one("#agent-tree", AgentTree)

        if not self._live_mode:
            agent.state = AgentState.KILLED
            tree.update_agent_state(agent.id, AgentState.KILLED)
            tl.write(f"[red]Killed[/red] {agent.name} [dim](demo)[/dim]")
            return

        if not self.bridge.running:
            agent.state = AgentState.KILLED
            tree.update_agent_state(agent.id, AgentState.KILLED)
            tl.write(f"[red]Killed[/red] {agent.name}")
            return

        self._kill_agent(agent.id, agent.name)

    def _is_model_busy(self) -> bool:
        """Return True when orchestration is running or agents are still active."""
        if self.bridge.running:
            return True

        active_states = {
            AgentState.PENDING,
            AgentState.STARTING,
            AgentState.RUNNING,
            AgentState.WAITING_FOR_PARENT,
            AgentState.WAITING_FOR_CHILD,
            AgentState.WAITING_FOR_EXPERT,
        }
        return any(agent.state in active_states for agent in self.session.agents.values())

    @work(name="kill-agent")
    async def _kill_agent(self, agent_id: str, agent_name: str) -> None:
        """Background worker: kill an agent via the orchestrator."""
        tl = self.query_one("#tool-log", ToolLog)
        try:
            if self.bridge._engine:
                await self.bridge._engine.agent_manager.kill_agent(agent_id)
            tl.write(f"[red]Killed[/red] {agent_name} [dim]({agent_id[:8]})[/dim]")
        except Exception as exc:
            safe_err = str(exc).replace("[", "\\[")
            tl.write(f"[red]Failed to kill {agent_name}:[/red] {safe_err}")

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        if self._live_mode:
            if self._is_model_busy():
                # Show delivery mode dialog
                from prsm.tui.screens.delivery_mode import DeliveryModeScreen
                self._pending_inject_text = event.text
                self.app.push_screen(
                    DeliveryModeScreen(),
                    self._handle_delivery_mode,
                )
                return

            # Check if this is a follow-up to a completed master agent
            continuation_master_id = self.bridge.last_master_id
            is_continuation = (
                continuation_master_id is not None
                and continuation_master_id in self.session.agents
            )

            from prsm.tui.widgets.thinking import ThinkingIndicator

            conv = self.query_one("#conversation", ConversationView)
            container = conv.query_one("#message-container")
            display_prompt = event.text[:50].replace("[", "\\[")
            snap_id = self.session_manager.create_auto_snapshot(
                f"Auto: before prompt '{display_prompt}'"
            )
            self._pending_prompt_snapshot_id = snap_id
            if snap_id:
                self._last_snapshot_id = snap_id

            if is_continuation:
                # Follow-up: add user message directly to the existing
                # master agent's conversation — no temp widgets needed.
                # Make sure we're viewing the master agent's conversation.
                if conv._current_agent_id != continuation_master_id:
                    self._select_agent(continuation_master_id)
                    container = conv.query_one("#message-container")
                conv.add_user_message(
                    continuation_master_id,
                    event.text,
                    snapshot_id=snap_id,
                )
                self._pending_prompt_snapshot_id = None
                # Show thinking indicator (will be cleaned up by event processor)
                master_agent = self.session.agents.get(continuation_master_id)
                master_agent_name = master_agent.name if master_agent else "Orchestrator"
                self._thinking_widget = ThinkingIndicator(agent_name=master_agent_name)
                conv.mount_before_queued(self._thinking_widget)
                container.scroll_end(animate=False)
                # Store as pending so the restarted event handler knows
                self._pending_prompt = event.text
            else:
                # First prompt: show temp user message + thinking
                from prsm.tui.widgets.conversation import MessageWidget
                from prsm.shared.models.message import Message

                self._temp_user_widget = MessageWidget(Message(
                    role=MessageRole.USER,
                    content=event.text,
                    agent_id="__pending__",
                ))
                conv.mount_before_queued(self._temp_user_widget)

                # For the first prompt, we don't know the exact agent name yet,
                # so use a generic one. The EventProcessor will handle the real one.
                self._thinking_widget = ThinkingIndicator(agent_name="Orchestrator")
                conv.mount_before_queued(self._thinking_widget)
                container.scroll_end(animate=False)

                self._pending_prompt = event.text

            self._run_orchestration(event.text)
        else:
            # Demo mode: simulate response on active agent
            agent_id = self.session.active_agent_id
            if not agent_id:
                return
            conv = self.query_one("#conversation", ConversationView)
            conv.add_user_message(agent_id, event.text)
            tl = self.query_one("#tool-log", ToolLog)
            tl.write(f"[yellow]\u26a0 Demo[/yellow] [dim]User \u2192 {agent_id}:[/dim] {event.text[:80]}")
            self._stream_response(agent_id, event.text)

    def on_input_bar_command_submitted(self, event: InputBar.CommandSubmitted) -> None:
        """Handle slash commands from the input bar."""
        self.command_handler.handle_command(event.name, event.args)

    def on_input_bar_new_session_requested(self) -> None:
        """Handle '+' quick action from the input bar."""
        self.command_handler.handle_command("new", [])

    def on_input_bar_settings_requested(self) -> None:
        """Handle settings quick action from the input bar."""
        from prsm.tui.screens.settings import SettingsScreen

        self.app.push_screen(SettingsScreen(cwd=self._cwd))

    def on_input_bar_search_sessions_requested(self) -> None:
        """Handle saved-session search quick action from the input bar."""
        from prsm.tui.screens.session_search import SessionSearchScreen

        sessions = self._persistence.list_sessions_detailed(all_worktrees=True)
        if not sessions:
            self.query_one("#tool-log", ToolLog).write(
                "[dim]No saved sessions found[/dim]"
            )
            return

        def _on_select(session_id: str | None) -> None:
            if not session_id:
                return
            self.command_handler.handle_command("session", [session_id])

        self.app.push_screen(
            SessionSearchScreen(sessions=sessions),
            callback=_on_select,
        )

    # ── Delivery Mode Handling ──

    def _handle_delivery_mode(self, mode: str | None) -> None:
        """Callback from DeliveryModeScreen."""
        prompt = getattr(self, '_pending_inject_text', None)
        self._pending_inject_text = None
        if not prompt:
            return

        if mode in (None, "cancel"):
            return

        tl = self.query_one("#tool-log", ToolLog)

        if mode == "interrupt":
            tl.write(f"[red]Interrupting[/red] current task, replacing with new prompt")
            self._do_interrupt(prompt)
        elif mode == "inject":
            tl.write(f"[yellow]Injecting[/yellow] prompt after current tool call")
            self._do_inject(prompt)
        elif mode == "queue":
            plan_num = self._plan_index_counter + 1  # Preview next index
            tl.write(f"[yellow]Queued[/yellow] Plan {plan_num} for after current task completes")
            self._do_queue(prompt)

    def on_queued_prompt_widget_cancelled(self, event: QueuedPromptWidget.Cancelled) -> None:
        """Handle cancellation of a queued prompt."""
        tl = self.query_one("#tool-log", ToolLog)

        # Remove by stable plan index to avoid duplicate-text mismatches.
        self._delivery_queued_prompts = [
            (plan_index, prompt)
            for (plan_index, prompt) in self._delivery_queued_prompts
            if plan_index != event.plan_index
        ]

        # Remove the widget from conversation and index.
        widget = self._queued_prompt_widgets.pop(event.plan_index, None)
        if widget and widget.is_mounted:
            widget.remove()

        tl.write(f"[yellow]Cancelled[/yellow] Plan {event.plan_index}")

    @work(name="interrupt")
    async def _do_interrupt(self, prompt: str) -> None:
        """Interrupt the current orchestration and continue with a new prompt.

        Uses the engine's interrupt mechanism to kill child agents while
        keeping the master agent restartable. The new prompt is then sent
        as a continuation of the interrupted master, preserving the full
        conversation history so the chat is not cleared.

        If no master agent is available for continuation (edge case),
        falls back to a full shutdown + fresh orchestration.
        """
        # Suppress "engine failed" error display during interrupt
        self._interrupt_in_progress = True
        # Try interrupt (keeps master restartable) instead of full shutdown
        master_id = await self.bridge.interrupt()

        if master_id and master_id in self.session.agents:
            # Continue on the interrupted master — conversation is preserved.
            # The user message will be appended to the master's conversation
            # by on_input_bar_submitted's continuation path via _run_orchestration.
            from prsm.tui.widgets.conversation import ConversationView
            from prsm.tui.widgets.thinking import ThinkingIndicator

            conv = self.query_one("#conversation", ConversationView)

            # Make sure we're viewing the master agent's conversation
            if conv._current_agent_id != master_id:
                self._select_agent(master_id)

            container = conv.query_one("#message-container")

            # Add the user message to the master's conversation
            display_prompt = prompt[:50].replace("[", "\\[")
            snap_id = self.session_manager.create_auto_snapshot(
                f"Auto: before prompt '{display_prompt}'"
            )
            self._pending_prompt_snapshot_id = snap_id
            if snap_id:
                self._last_snapshot_id = snap_id
            conv.add_user_message(master_id, prompt, snapshot_id=snap_id)
            self._pending_prompt_snapshot_id = None

            # Show thinking indicator
            master_agent = self.session.agents.get(master_id)
            master_agent_name = master_agent.name if master_agent else "Orchestrator"
            self._thinking_widget = ThinkingIndicator(agent_name=master_agent_name)
            conv.mount_before_queued(self._thinking_widget)
            container.scroll_end(animate=False)

            self._pending_prompt = prompt
            self._run_orchestration(prompt)
        else:
            # Fallback: no master to continue — do a full shutdown + fresh start
            await self.bridge.shutdown()
            if self.bridge._engine:
                self.bridge._engine._last_master_id = None

            from prsm.tui.widgets.conversation import ConversationView, MessageWidget
            from prsm.tui.widgets.thinking import ThinkingIndicator
            from prsm.shared.models.message import Message

            conv = self.query_one("#conversation", ConversationView)
            container = conv.query_one("#message-container")

            self._temp_user_widget = MessageWidget(Message(
                role=MessageRole.USER,
                content=prompt,
                agent_id="__pending__",
            ))
            conv.mount_before_queued(self._temp_user_widget)

            # For the fallback case, use a generic "Orchestrator" name
            self._thinking_widget = ThinkingIndicator(agent_name="Orchestrator")
            conv.mount_before_queued(self._thinking_widget)
            container.scroll_end(animate=False)

            self._pending_prompt = prompt
            self._run_orchestration(prompt)

    @work(name="inject")
    async def _do_inject(self, prompt: str) -> None:
        """Inject prompt after the current tool call completes.

        Stores the inject prompt and agent ID. When the next
        ToolCallCompleted event fires for the master agent, the event
        processor will shut down the current run and start a new
        orchestration with the injected prompt.
        """
        tl = self.query_one("#tool-log", ToolLog)
        if not self.bridge._engine:
            tl.write("[red]Inject failed:[/red] engine unavailable")
            return

        target_agent_id: str | None = None
        target_session = None
        for desc in reversed(self.bridge._engine.agent_manager.get_all_descriptors()):
            if desc.parent_id is None:
                candidate = self.bridge._engine.agent_manager.get_session(desc.agent_id)
                if candidate:
                    target_agent_id = desc.agent_id
                    target_session = candidate
                    break

        if target_agent_id is None or target_session is None:
            tl.write("[red]Inject failed:[/red] no active root session found")
            return

        target_session.inject_after_tool(prompt)
        self._inject_prompts_by_agent.setdefault(target_agent_id, []).append(prompt)
        tl.write(f"[yellow]Inject armed[/yellow] for {target_agent_id[:8]}")

    def _do_queue(self, prompt: str) -> None:
        """Queue prompt to run after the current task finishes.

        The prompt is stored on the screen. When the orchestration
        completes (engine_finished), _run_orchestration's finally block
        checks for queued prompts and starts the next one.

        A QueuedPromptWidget is mounted at the bottom of the conversation
        to visually indicate the queued plan with a yellow style and a
        cancel button.
        """
        # Assign a plan index
        self._plan_index_counter += 1
        plan_index = self._plan_index_counter
        self._delivery_queued_prompts.append((plan_index, prompt))

        # Show the queued prompt widget in the conversation (pinned to bottom)
        conv = self.query_one("#conversation", ConversationView)
        container = conv.query_one("#message-container")
        widget = QueuedPromptWidget(plan_index=plan_index, prompt=prompt)
        container.mount(widget)
        container.scroll_end(animate=False)
        self._queued_prompt_widgets[plan_index] = widget

        # Also set on the engine session for consistency
        if self.bridge._engine:
            for desc in reversed(self.bridge._engine.agent_manager.get_all_descriptors()):
                if desc.parent_id is None:
                    session = self.bridge._engine.agent_manager.get_session(desc.agent_id)
                    if session:
                        session.queue_prompt(prompt)
                        break

    # ── Live Orchestrator Mode ──

    @work(exclusive=True, name="orchestration")
    async def _run_orchestration(self, prompt: str) -> None:
        """Background worker: run a real orchestrator session."""
        sb = self.query_one("#status-bar", StatusBar)
        tl = self.query_one("#tool-log", ToolLog)

        sb.status = "streaming"

        # Auto-snapshot before each prompt (enables resend/restore).
        # If caller already captured one, reuse it.
        if self._pending_prompt_snapshot_id is None:
            display_prompt = prompt[:50].replace("[", "\\[")
            snap_id = self.session_manager.create_auto_snapshot(
                f"Auto: before prompt '{display_prompt}'"
            )
            self._pending_prompt_snapshot_id = snap_id
            if snap_id:
                self._last_snapshot_id = snap_id

        # Auto-name session if it doesn't have a stable name yet
        from prsm.shared.models.session import is_default_session_name

        if is_default_session_name(self.session.name):
            asyncio.create_task(self._auto_name_session(prompt))

        # Detect follow-up: is there a completed master we can continue?
        continuation_master_id = self.bridge.last_master_id
        is_continuation = (
            continuation_master_id is not None
            and continuation_master_id in self.session.agents
        )

        if is_continuation:
            tl.write(f"[yellow]Continuing[/yellow] conversation...")
        else:
            tl.write(f"[yellow]Starting[/yellow] orchestration...")

        # Reset the event bus for a fresh run
        self.bridge.event_bus.reset()

        # Start event consumer (delegated to EventProcessor)
        self._event_consumer_task = asyncio.create_task(
            self.event_processor.consume_events(),
            name="event-consumer",
        )

        try:
            if is_continuation:
                result = await self.bridge.run_continuation(
                    prompt, continuation_master_id,
                )
            else:
                result = await self.bridge.run(prompt)
            tl.write(f"[green]Orchestration complete[/green]")
            orchestration_successful = True # Added to track success
        except Exception as exc:
            error_msg = str(exc)
            if self._user_stop_requested:
                tl.write("[yellow]Run stopped by user[/yellow]")
            else:
                safe_err = error_msg.replace("[", "\\[")
                tl.write(f"[red]Orchestration error:[/red] {safe_err}")
                logger.exception("Orchestration failed")
                # Show error visibly in the conversation area
                self._show_error_in_conversation(error_msg)
            orchestration_successful = False # Added to track failure
        finally:
            # If pending prompt was never consumed (no AgentSpawned arrived),
            # preserve the user message as a permanent widget
            if self._pending_prompt and self._temp_user_widget:
                # Keep the user message visible — just detach from temp tracking
                self._temp_user_widget = None
                self._pending_prompt = None
                self._pending_prompt_snapshot_id = None
            # Always clean up thinking indicator and remaining temp widgets
            self._remove_temp_widgets()
            # Wait for the event consumer to process all queued events
            # (AgentSpawned, StreamChunk, etc.) before cancelling it.
            # This replaces a fragile sleep(0.5) with a proper drain.
            tl.write("[dim]Draining remaining events...[/dim]")
            await self.bridge.event_bus.drain(timeout=5.0)
            if self._event_consumer_task:
                self._event_consumer_task.cancel()
                try:
                    await self._event_consumer_task
                except asyncio.CancelledError:
                    pass
                self._event_consumer_task = None
            # Finalize any open streams
            await self.event_processor.finalize_all_streams()
            sb.status = "connected"

            if self._user_stop_requested:
                conv = self.query_one("#conversation", ConversationView)
                container = conv.query_one("#message-container")
                conv.mount_before_queued(RunStoppedIndicator(reason="Stopped by user"))
                container.scroll_end(animate=False)
                self._user_stop_requested = False

                queued = getattr(self, "_delivery_queued_prompts", [])
                if queued:
                    next_plan_index, next_prompt = queued.pop(0)
                    widget = self._queued_prompt_widgets.pop(next_plan_index, None)
                    if widget and widget.is_mounted:
                        widget.remove()
                    tl.write("[cyan]Stopped current run; starting queued prompt[/cyan]")
                    self._run_orchestration(next_prompt)
                return

            # Check for queued prompts — run the next one if available
            queued = getattr(self, '_delivery_queued_prompts', [])
            if queued:
                next_plan_index, next_prompt = queued.pop(0)
                widget = self._queued_prompt_widgets.pop(next_plan_index, None)
                if widget and widget.is_mounted:
                    widget.remove()
                tl.write("[cyan]Running queued prompt[/cyan]")
                self._run_orchestration(next_prompt)

    async def _auto_name_session(self, prompt: str) -> None:
        """Generate a descriptive session name — delegates to SessionManager."""
        await self.session_manager.auto_name_session(prompt)

    def _remove_temp_widgets(self) -> None:
        """Remove temporary user message and thinking indicator."""
        if self._temp_user_widget is not None:
            try:
                if self._temp_user_widget.is_mounted:
                    self._temp_user_widget.remove()
            except Exception:
                pass
            self._temp_user_widget = None
        self.event_processor.clear_all_thinking()

    def _show_error_in_conversation(self, error_msg: str) -> None:
        """Show an error message visibly in the conversation area."""
        from textual.widgets import Static

        safe_err = error_msg.replace("[", "\\[")
        conv = self.query_one("#conversation", ConversationView)
        container = conv.query_one("#message-container")
        error_widget = Static(
            f"[bold red]Error:[/bold red] {safe_err}\n"
            "[dim]Press F1 to open the tool log for details.[/dim]",
            classes="message-system",
            markup=True,
        )
        conv.mount_before_queued(error_widget)
        container.scroll_end(animate=False)

        # Auto-show the tool log on errors
        tl = self.query_one("#tool-log", ToolLog)
        if not tl.has_class("visible"):
            tl.toggle()

    def _show_permission_modal(self, event: PermissionRequest) -> None:
        """Show the permission modal for a tool call request."""
        from prsm.tui.screens.permission import PermissionScreen

        def on_dismiss(result: str) -> None:
            if result == "view_agent":
                # Open the agent context panel so the user can review full
                # conversation history, then re-show the permission dialog.
                if event.agent_id:
                    # Get the current message index for this agent to highlight
                    # the message that triggered this permission request
                    message_index = None
                    if event.agent_id in self.session.agents:
                        messages = self.session.get_messages(event.agent_id)
                        # The permission request happens during tool execution,
                        # so highlight the last message (current tool call)
                        if messages:
                            message_index = len(messages) - 1

                    self._open_agent_context_panel(
                        event.agent_id,
                        highlight_message_index=message_index,
                    )
                # Re-push the permission modal after a short delay so
                # Textual finishes dismissing the current one first.
                self.set_timer(0.1, lambda: self._show_permission_modal(event))
                return
            self.bridge.resolve_permission(event.request_id, result)

        screen = PermissionScreen(
            tool_name=event.tool_name,
            agent_name=event.agent_name,
            arguments=event.arguments,
            agent_id=event.agent_id,
        )
        self.app.push_screen(screen, callback=on_dismiss)

    def _show_user_question(self, event: UserQuestionRequest) -> None:
        """Show a user question widget with clickable options."""
        from prsm.tui.widgets.question import QuestionWidget

        # Hide thinking while question is shown
        self.event_processor.set_agent_thinking(event.agent_id, False)

        # Switch to the asking agent's conversation if not already active
        if event.agent_id != self.session.active_agent_id:
            self._select_agent(event.agent_id)

        conv = self.query_one("#conversation", ConversationView)
        container = conv.query_one("#message-container")

        widget = QuestionWidget(
            request_id=event.request_id,
            agent_name=event.agent_name,
            question=event.question,
            options=event.options,
        )
        conv.mount_before_queued(widget)
        conv._smart_scroll()

    def _show_file_change(self, event: FileChanged) -> None:
        """Show a file change widget in the conversation."""
        # Get agent name for display
        agent = self.session.agents.get(event.agent_id)
        agent_name = agent.name if agent else event.agent_id[:8]

        # Log to tool log
        tl = self.query_one("#tool-log", ToolLog)
        change_icon = {"create": "+", "modify": "~", "delete": "-"}.get(
            event.change_type, "?"
        )
        safe_path = event.file_path.replace("[", "\\[")
        tl.write(f"  [dim]{change_icon}[/dim] {safe_path}")

        # Add widget to conversation (only for active agent or switch to that agent)
        conv = self.query_one("#conversation", ConversationView)

        # If this is the active agent, show the widget
        if event.agent_id == self.session.active_agent_id:
            container = conv.query_one("#message-container")
            widget = FileChangeWidget(
                agent_id=event.agent_id,
                file_path=event.file_path,
                change_type=event.change_type,
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                message_index=event.message_index,
                old_content=event.old_content,
                added_ranges=event.added_ranges,
                removed_ranges=event.removed_ranges,
            )
            conv.mount_before_queued(widget)
            conv._smart_scroll()

    def on_file_change_widget_view_context_requested(
        self, event: FileChangeWidget.ViewContextRequested,
    ) -> None:
        """Handle the user clicking 'View Context' on a file change widget."""
        from prsm.tui.screens.file_context import FileContextScreen
        from prsm.engine.rationale_extractor import extract_change_rationale

        # Get agent name
        agent = self.session.agents.get(event.agent_id)
        agent_name = agent.name if agent else event.agent_id[:8]

        # Extract rationale using the proper rationale extractor
        rationale = "Loading rationale..."
        if self.bridge._engine and self.bridge._engine.conversation_store:
            try:
                rationale = extract_change_rationale(
                    agent_id=event.agent_id,
                    tool_call_id=event.tool_call_id,
                    conversation_store=self.bridge._engine.conversation_store,
                    max_sentences=3,
                )
                if not rationale:
                    rationale = "No rationale found for this change."
            except Exception as e:
                rationale = f"Could not extract rationale: {str(e)}"
        else:
            rationale = "Engine not available to extract rationale."

        screen = FileContextScreen(
            file_path=event.file_path,
            tool_name=event.tool_name,
            agent_name=agent_name,
            rationale=rationale,
        )
        self.app.push_screen(screen)

    def _open_agent_context_panel(
        self,
        agent_id: str,
        highlight_message_index: int | None = None,
        highlight_tool_call_id: str | None = None,
    ) -> None:
        """Open the agent context side panel for the given agent.

        Args:
            agent_id: The agent to display
            highlight_message_index: Optional message index to highlight
            highlight_tool_call_id: Optional tool call ID to highlight
        """
        if agent_id not in self.session.agents:
            return
        panel = self.query_one("#agent-context-panel", AgentContextPanel)
        panel.show_agent(
            agent_id,
            self.session,
            highlight_message_index=highlight_message_index,
            highlight_tool_call_id=highlight_tool_call_id,
        )

    def _close_agent_context_panel(self) -> None:
        """Close the agent context side panel."""
        panel = self.query_one("#agent-context-panel", AgentContextPanel)
        panel.hide()

    def on_agent_context_panel_close_requested(
        self, event: AgentContextPanel.CloseRequested,
    ) -> None:
        """Handle close button on the agent context panel."""
        pass  # Panel hides itself; nothing else to do

    def on_agent_context_panel_navigate_to_agent(
        self, event: AgentContextPanel.NavigateToAgent,
    ) -> None:
        """Handle 'Switch To' button — navigate to the agent and close panel."""
        if event.agent_id in self.session.agents:
            self._select_agent(event.agent_id)
            self._close_agent_context_panel()

    def on_file_change_widget_view_agent_requested(
        self, event: FileChangeWidget.ViewAgentRequested,
    ) -> None:
        """Handle the user clicking 'View Agent' on a file change widget.

        This navigates to the agent in the tree view and opens the agent
        context panel to show the full conversation history. The message
        containing the tool call that made this change will be highlighted.
        """
        if event.agent_id in self.session.agents:
            # Navigate to agent and select them in the tree
            self._select_agent(event.agent_id)

            # Open the context panel with the relevant message highlighted
            self._open_agent_context_panel(
                event.agent_id,
                highlight_message_index=event.message_index,
                highlight_tool_call_id=event.tool_call_id,
            )

    def on_question_widget_answered(
        self, event: QuestionWidget.Answered,
    ) -> None:
        """Handle the user clicking an option on a question widget."""
        # Resolve @file/@directory references in the answer (matches server behavior)
        from prsm.shared.file_utils import resolve_references

        resolved_text, attachments = resolve_references(event.answer, self._cwd)
        if attachments:
            context_parts = []
            for att in attachments:
                tag = "directory" if att.is_directory else "file"
                warning = " (truncated)" if att.truncated else ""
                context_parts.append(
                    f"<{tag} path=\"{att.path}\"{warning}>\n"
                    f"{att.content}\n"
                    f"</{tag}>"
                )
            answer = resolved_text + "\n\n" + "\n\n".join(context_parts)
        else:
            answer = event.answer
        self.bridge.resolve_user_question(event.request_id, answer)
        tl = self.query_one("#tool-log", ToolLog)
        safe = event.answer.replace("[", "\\[")
        tl.write(f"[green]Answered:[/green] {safe}")

    # ── Resend (click previous prompt to edit and re-submit) ──

    def on_message_widget_resend_requested(
        self, event: MessageWidget.ResendRequested,
    ) -> None:
        """Handle click on a previous user message to resend it.

        Flow:
        1. Create a snapshot of the current state (safety net).
        2. Immediately truncate the session to the selected message.
        3. Pre-fill the input bar with the message text for editing.
        4. Ask whether file changes should be reverted to that snapshot.
        """
        agent_id = event.agent_id
        message_index = event.message_index
        prompt_text = event.msg.content
        target_snapshot_id = event.msg.snapshot_id
        tl = self.query_one("#tool-log", ToolLog)

        # Always let left-click load the prompt into the editor immediately.
        input_bar = self.query_one(InputBar)
        input_bar.set_text(prompt_text)
        input_bar.focus_input()

        # Don't allow session rewind while orchestration is running.
        if self.bridge.running:
            tl.write(
                "[yellow]Cannot resend while agent is running; prompt loaded for editing[/yellow]"
            )
            return

        # Safety: snapshot current state before rewinding (for manual recovery)
        safety_snap = self.session_manager.create_auto_snapshot(
            f"Auto: before resend from message {message_index}"
        )
        if safety_snap:
            self._last_snapshot_id = safety_snap

        # Immediately rewind conversation to the selected point and prefill input.
        self._execute_resend(agent_id, message_index, prompt_text)

        # Count file changes after the selected message
        file_count = self.session_manager.count_file_changes_after(
            agent_id, message_index,
        )

        # Check user preference
        from prsm.shared.services.preferences import UserPreferences
        prefs = UserPreferences.load()

        if prefs.file_revert_on_resend == "always":
            self._revert_files_from_snapshot(target_snapshot_id)
        elif prefs.file_revert_on_resend == "never":
            return
        else:
            # "ask" — show confirmation dialog
            from prsm.tui.screens.resend_confirm import ResendConfirmScreen

            def _on_resend_result(result: str | None) -> None:
                if result == "revert":
                    self._revert_files_from_snapshot(target_snapshot_id)
                elif result not in ("keep", None):
                    tl.write("[dim]Resend continues with current files[/dim]")

            self.app.push_screen(
                ResendConfirmScreen(
                    file_count=file_count,
                    prompt_preview=prompt_text,
                ),
                callback=_on_resend_result,
            )

    def _execute_resend(
        self,
        agent_id: str,
        message_index: int,
        prompt_text: str,
    ) -> None:
        """Execute the resend: truncate session and pre-fill input.

        Args:
            agent_id: The agent whose conversation is being rewound.
            message_index: Index of the user message to resend from.
            prompt_text: The original prompt text to pre-fill.
        """
        tl = self.query_one("#tool-log", ToolLog)

        # Truncate session to the selected message
        self.session_manager.truncate_session_to(agent_id, message_index)

        # Reset the bridge's last_master_id so the next prompt starts fresh
        # (the master agent's conversation was truncated, it can't continue)
        if self.bridge._engine:
            self.bridge._engine._last_master_id = None

        # Rebuild the UI from truncated session
        self.session_manager.rebuild_ui_after_resend()

        # Pre-fill the input bar with the original prompt for editing
        input_bar = self.query_one(InputBar)
        input_bar.set_text(prompt_text)
        input_bar.focus_input()

        safe_preview = prompt_text[:40].replace("[", "\\[")
        tl.write(
            f"[cyan]Resend[/cyan] ready — edit and submit: "
            f"[dim]\"{safe_preview}…\"[/dim]"
        )

    def _revert_files_from_snapshot(self, snapshot_id: str | None) -> None:
        """Restore files to the snapshot associated with the selected prompt."""
        tl = self.query_one("#tool-log", ToolLog)
        if not snapshot_id:
            tl.write("[yellow]No snapshot found for this prompt; keeping current files[/yellow]")
            return
        ok = self.session_manager.restore_snapshot_files(snapshot_id)
        if ok:
            tl.write("[green]Reverted[/green] files to selected prompt snapshot")
        else:
            tl.write("[yellow]Could not revert files[/yellow]")

    # ── Model Selection ──

    def on_input_bar_model_switch_requested(self, event: InputBar.ModelSwitchRequested) -> None:
        """Handle click on the model button in the input bar."""
        self.show_model_selector()

    def show_model_selector(self) -> None:
        """Open the model picker modal."""
        if not self._live_mode:
            tl = self.query_one("#tool-log", ToolLog)
            tl.write("[yellow]Model switching is only available in live mode[/yellow]")
            return

        from prsm.tui.screens.model_selector import ModelSelectorScreen, ModelOption

        models_data = self.bridge.get_available_models()
        current = self.bridge.current_model

        options = [
            ModelOption(
                model_id=m["model_id"],
                provider=m["provider"],
                tier=m["tier"],
                available=m["available"],
                is_current=m["is_current"],
                display_name=m.get("display_name", m["model_id"]),
            )
            for m in models_data
        ]

        if not options:
            tl = self.query_one("#tool-log", ToolLog)
            tl.write("[yellow]No models available to select[/yellow]")
            return

        self.app.push_screen(
            ModelSelectorScreen(models=options, current_model=current),
            self._on_model_selected,
        )

    def _on_model_selected(self, model_id: str | None) -> None:
        """Callback from ModelSelectorScreen."""
        if model_id is None:
            return  # Cancelled

        current = self.bridge.current_model
        if model_id == current:
            tl = self.query_one("#tool-log", ToolLog)
            tl.write(f"[dim]Already using model {model_id}[/dim]")
            return

        self.switch_model(model_id)

    def switch_model(self, model_id: str) -> None:
        """Switch to a new model and show a visual indicator in the conversation."""
        tl = self.query_one("#tool-log", ToolLog)

        old_model = self.bridge.current_model
        try:
            resolved, provider = self.bridge.set_model(model_id)
        except Exception as exc:
            safe_err = str(exc).replace("[", "\\[")
            tl.write(f"[red]Failed to switch model:[/red] {safe_err}")
            return

        # Keep all model displays in sync after a model switch.
        self._refresh_model_display(show_demo_banner=not self._live_mode)

        # Add a visual separator in the conversation
        from prsm.tui.widgets.conversation import ModelSwitchIndicator

        conv = self.query_one("#conversation", ConversationView)
        indicator = ModelSwitchIndicator(old_model=old_model, new_model=resolved)
        conv.mount_before_queued(indicator)
        container = conv.query_one("#message-container")
        container.scroll_end(animate=False)

        # Update the last master agent's descriptor with the new model/provider
        # so that the next prompt can continue the conversation with the new model
        if self.bridge._engine and self.bridge._engine._last_master_id:
            master_id = self.bridge._engine._last_master_id
            manager = self.bridge._engine._manager

            # Check if the master is in completed agents (where restart_agent looks)
            if master_id in manager._completed_agents:
                descriptor = manager._completed_agents[master_id]
                descriptor.model = resolved
                descriptor.provider = provider
                tl.write(
                    f"[dim]Updated conversation context to use new model[/dim]"
                )

        tl.write(
            f"[green]Model switched:[/green] {old_model} → [bold cyan]{resolved}[/bold cyan]"
        )

    # ── Demo Mode (simulated streaming) ──

    @work(exclusive=True, name="stream-response")
    async def _stream_response(self, agent_id: str, prompt: str) -> None:
        """Background worker: simulate a streaming agent response."""
        conv = self.query_one("#conversation", ConversationView)
        sb = self.query_one("#status-bar", StatusBar)
        tl = self.query_one("#tool-log", ToolLog)
        tree = self.query_one("#agent-tree", AgentTree)

        # Update state to streaming
        agent = self.session.agents.get(agent_id)
        if agent:
            agent.state = AgentState.RUNNING
            tree.update_agent_state(agent_id, AgentState.RUNNING)
        sb.status = "streaming"
        tl.write(f"[yellow]Streaming[/yellow] response for [bold]{agent_id}[/bold]...")

        # Simulate a tool call mid-response
        await asyncio.sleep(0.3)
        tl.write(f"  [cyan]read_file[/cyan](src/auth/config.py)")
        conv.add_tool_call(
            agent_id, "read_file", "src/auth/config.py",
            "JWT_SECRET, TOKEN_EXPIRY=3600, REFRESH_EXPIRY=604800",
        )
        tl.write(f"  [green]\u2192 JWT_SECRET, TOKEN_EXPIRY=3600, REFRESH_EXPIRY=604800[/green]")
        await asyncio.sleep(0.2)

        # Stream the response
        async def _simulated_chunks():
            """Generate simulated markdown chunks with realistic timing."""
            response = _build_simulated_response(prompt)
            words = response.split(" ")
            for i, word in enumerate(words):
                sep = "" if i == 0 else " "
                yield sep + word
                delay = 0.02 + random.random() * 0.04
                if word.endswith((".", "\n", ":")):
                    delay += 0.05
                await asyncio.sleep(delay)

        msg = await conv.stream_assistant_message(agent_id, _simulated_chunks())

        # Update state back to connected
        if agent:
            agent.state = AgentState.COMPLETED
            tree.update_agent_state(agent_id, AgentState.COMPLETED)
        sb.status = "connected"
        sb.tokens_used = len(self.session.get_messages(agent_id)) * 150
        tl.write(f"[green]Done[/green] [dim](demo)[/dim] streaming for [bold]{agent_id}[/bold]")

    # ── Demo Data ──

    def _populate_demo_agents(self) -> None:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)

        agents = [
            AgentNode(
                id="root",
                name="Orchestrator",
                state=AgentState.RUNNING,
                role=AgentRole.MASTER,
                model="claude-opus-4-6",
                last_active=now,
            ),
            AgentNode(
                id="w1",
                name="Code Explorer",
                state=AgentState.COMPLETED,
                role=AgentRole.WORKER,
                model="claude-opus-4-6",
                parent_id="root",
                last_active=now - timedelta(minutes=12),
            ),
            AgentNode(
                id="w2",
                name="Test Runner",
                state=AgentState.RUNNING,
                role=AgentRole.WORKER,
                model="claude-opus-4-6",
                parent_id="root",
                last_active=now - timedelta(seconds=3),
            ),
            AgentNode(
                id="w3",
                name="Code Writer",
                state=AgentState.WAITING_FOR_CHILD,
                role=AgentRole.WORKER,
                model="claude-opus-4-6",
                parent_id="root",
                last_active=now - timedelta(seconds=15),
            ),
            AgentNode(
                id="w3-sub",
                name="Lint Check",
                state=AgentState.COMPLETED,
                role=AgentRole.WORKER,
                model="claude-opus-4-6",
                parent_id="w3",
                last_active=now - timedelta(minutes=2),
            ),
            AgentNode(
                id="e1",
                name="Rust Expert",
                state=AgentState.PENDING,
                role=AgentRole.EXPERT,
                model="claude-opus-4-6",
                parent_id="root",
                last_active=now - timedelta(minutes=5),
            ),
        ]

        tree = self.query_one("#agent-tree", AgentTree)

        for agent in agents:
            self.session.add_agent(agent)
            if agent.id == "root":
                tree.add_agent(None, agent)
                tree.root.expand()
            else:
                tree.add_agent(agent.parent_id, agent)

        tree.sort_by_activity()

    def _populate_demo_conversations(self) -> None:
        conv = self.query_one("#conversation", ConversationView)

        # ── Orchestrator (root) conversation ──
        conv.session.add_message(
            "root", MessageRole.SYSTEM,
            "Session started. Connected to orchestrator.",
        )
        conv.session.add_message(
            "root", MessageRole.USER,
            "Refactor the authentication module to use JWT tokens "
            "and add refresh token rotation.",
        )
        conv.session.add_message(
            "root", MessageRole.ASSISTANT,
            "I'll analyze the current auth implementation and plan the refactor. "
            "Let me spawn some workers to explore the codebase.\n\n"
            "Spawning Code Explorer to map the auth module...\n"
            "Spawning Test Runner to verify existing tests...",
        )
        conv.session.add_message(
            "root", MessageRole.TOOL, "",
            tool_calls=[ToolCall(
                id="tc-1",
                name="spawn_child",
                arguments='prompt="Explore src/auth/ and describe the implementation"',
                result="Spawned agent w1 (Code Explorer)",
            )],
        )
        conv.session.add_message(
            "root", MessageRole.ASSISTANT,
            "The Code Explorer found that auth currently uses session-based "
            "authentication with cookies. Here's my refactoring plan:\n\n"
            "1. Replace session store with JWT access + refresh tokens\n"
            "2. Add token rotation on refresh\n"
            "3. Update middleware to validate JWT signatures\n"
            "4. Migrate existing sessions to new token format",
        )

        # ── Code Explorer (w1) conversation ──
        conv.session.add_message(
            "w1", MessageRole.SYSTEM,
            "Agent spawned by Orchestrator.",
        )
        conv.session.add_message(
            "w1", MessageRole.ASSISTANT,
            "Exploring src/auth/ directory structure...",
        )
        conv.session.add_message(
            "w1", MessageRole.TOOL, "",
            tool_calls=[ToolCall(
                id="tc-2",
                name="read_file",
                arguments="src/auth/session.py",
                result="Found SessionStore class with Redis backend, cookie-based auth",
            )],
        )
        conv.session.add_message(
            "w1", MessageRole.TOOL, "",
            tool_calls=[ToolCall(
                id="tc-3",
                name="read_file",
                arguments="src/auth/middleware.py",
                result="AuthMiddleware checks session cookie, loads user from Redis",
            )],
        )
        conv.session.add_message(
            "w1", MessageRole.ASSISTANT,
            "Analysis complete. The auth module consists of:\n\n"
            "- session.py: SessionStore class using Redis for storage\n"
            "- middleware.py: Extracts session ID from cookies\n"
            "- routes.py: Login/logout endpoints\n"
            "- models.py: User and Session SQLAlchemy models\n\n"
            "No existing JWT support. The migration will need to add "
            "token generation, validation, and refresh rotation.",
        )

        # ── Test Runner (w2) conversation ──
        conv.session.add_message(
            "w2", MessageRole.SYSTEM,
            "Agent spawned by Orchestrator.",
        )
        conv.session.add_message(
            "w2", MessageRole.ASSISTANT,
            "Running existing auth test suite...",
        )
        conv.session.add_message(
            "w2", MessageRole.TOOL, "",
            tool_calls=[ToolCall(
                id="tc-4",
                name="run_tests",
                arguments="tests/auth/ -v",
                result="12 passed, 2 failed (test_session_expiry, test_concurrent_login)",
            )],
        )
        conv.session.add_message(
            "w2", MessageRole.ASSISTANT,
            "Test results: 12/14 passing.\n\n"
            "Two failures:\n"
            "- test_session_expiry: Race condition in Redis TTL check\n"
            "- test_concurrent_login: Session overwrite on simultaneous logins\n\n"
            "Both relate to the session-based approach and will be resolved "
            "by the JWT migration.",
        )

        # ── Code Writer (w3) — waiting, no messages yet ──
        conv.session.add_message(
            "w3", MessageRole.SYSTEM,
            "Agent spawned by Orchestrator. Waiting for exploration results...",
        )

        # ── Rust Expert (e1) — idle ──
        conv.session.add_message(
            "e1", MessageRole.SYSTEM,
            "Expert available. No queries received yet.",
        )


def _build_simulated_response(prompt: str) -> str:
    """Build a simulated markdown response based on the user prompt."""
    word_count = len(prompt.split())
    if word_count < 5:
        return (
            "I'll look into that for you.\n\n"
            "Based on my analysis, here's what I found:\n\n"
            "- The current implementation uses a straightforward approach\n"
            "- There are a few areas we could optimize\n"
            "- I'd recommend starting with the core module\n\n"
            "Want me to proceed with the changes?"
        )
    return (
        "I'll analyze the codebase and implement the requested changes.\n\n"
        "## Plan\n\n"
        "1. **Review** the existing implementation\n"
        "2. **Identify** the key integration points\n"
        "3. **Implement** the changes incrementally\n"
        "4. **Test** each modification\n\n"
        "## Findings\n\n"
        "After examining the relevant files, I can see that the current "
        "architecture follows a modular pattern with clear separation of "
        "concerns. The main entry point delegates to specialized handlers.\n\n"
        "```python\n"
        "def process_request(ctx):\n"
        "    handler = get_handler(ctx.route)\n"
        "    result = handler.execute(ctx.params)\n"
        "    return format_response(result)\n"
        "```\n\n"
        "I'll proceed with the implementation now."
    )
