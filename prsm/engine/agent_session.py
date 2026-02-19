"""Wraps a single Claude Agent SDK session with MCP tools.

Each agent is a separate query() call. The orchestration layer
injects a per-agent MCP server providing orchestration tools
(ask_parent, spawn_child, etc.). The SDK handles the agent loop;
our MCP tool handlers route messages through the MessageRouter.

TIMEOUT MODEL (two independent clocks):

1. Agent timeout (default 7200s):
   Cumulative wall-clock time the agent spends REASONING — i.e. the
   time between tool calls when Claude is thinking or generating text.
   Tool execution time is excluded via ToolTimeTracker. This prevents
   an agent from timing out just because its children are slow.

   Checked after every message yield:
       reasoning_time = elapsed - tracker.accumulated_tool_time
       if reasoning_time > agent_timeout: raise

2. Tool call timeout (default 7200s):
   Max wall-clock for any SINGLE tool call. Each MCP tool handler is
   wrapped with asyncio.wait_for(). If a single spawn_child or
   ask_parent blocks for more than this, it's cancelled and returns
   an error to the agent. The agent can then decide to retry or fail.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import shlex
import shutil
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .models import (
    AgentDescriptor,
    AgentResult,
    AgentState,
)
from .config import fire_event
from .lifecycle import validate_transition
from .errors import AgentTimeoutError
from prsm.shared.services.command_policy_store import CommandPolicyStore

if TYPE_CHECKING:
    from .agent_manager import AgentManager
    from .config import EventCallback, PermissionCallback, UserQuestionCallback
    from .message_router import MessageRouter
    from .expert_registry import ExpertRegistry
    from .providers.base import Provider

logger = logging.getLogger(__name__)


class BashRepeatGuard:
    """Detects and blocks consecutive identical bash commands.

    If an agent sends the exact same bash command more than `max_repeats`
    times in a row, the guard triggers — returning a denial message that
    tells the agent to try a different approach.
    """

    def __init__(self, max_repeats: int = 3) -> None:
        self._max_repeats = max_repeats
        self._last_command: str | None = None
        self._repeat_count: int = 0

    def check(self, command: str) -> tuple[bool, int]:
        """Check a command. Returns (allowed, repeat_count)."""
        command = command.strip()
        if command == self._last_command:
            self._repeat_count += 1
        else:
            self._last_command = command
            self._repeat_count = 1
        return self._repeat_count <= self._max_repeats, self._repeat_count


class AgentSession:
    """Manages the lifecycle of a single Claude agent session.

    Wraps the Claude Agent SDK query() call, provides the agent
    with MCP tools for inter-agent communication, and collects
    the final result.
    """

    def __init__(
        self,
        descriptor: AgentDescriptor,
        manager: AgentManager,
        router: MessageRouter,
        expert_registry: ExpertRegistry,
        agent_timeout_seconds: float = 7200.0,
        tool_call_timeout_seconds: float = 7200.0,
        user_question_timeout_seconds: float = 0.0,
        provider: Provider | None = None,
        event_callback: EventCallback | None = None,
        permission_callback: PermissionCallback | None = None,
        user_question_callback: UserQuestionCallback | None = None,
        plugin_mcp_servers: dict | None = None,
        always_allowed_tools: set[str] | None = None,
        conversation_store: object | None = None,
        child_default_model: str = "claude-opus-4-6",
        child_default_provider: str = "claude",
        model_registry: object | None = None,
        peer_models: dict | None = None,
        command_whitelist: list[str] | None = None,
        command_blacklist: list[str] | None = None,
        command_safety_model_enabled: bool = False,
        command_safety_model: str | None = None,
        orchestration_tools: object | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._manager = manager
        self._router = router
        self._expert_registry = expert_registry
        self._agent_timeout = agent_timeout_seconds
        self._tool_call_timeout = tool_call_timeout_seconds
        self._user_question_timeout = user_question_timeout_seconds
        self._provider = provider
        self._event_callback = event_callback
        self._permission_callback = permission_callback
        self._user_question_callback = user_question_callback
        self._plugin_mcp_servers = plugin_mcp_servers or {}
        # Shared mutable set — "Allow Always" propagates across all agents
        self._always_allowed_tools = always_allowed_tools if always_allowed_tools is not None else set()
        self._conversation_store = conversation_store
        # Default model/provider for child agents spawned by this agent
        self._child_default_model = child_default_model
        self._child_default_provider = child_default_provider
        # Model capability registry for intelligent model selection
        self._model_registry = model_registry
        # Peer models for child agent model restriction
        # Maps alias → (provider_instance, model_id)
        self._peer_models = peer_models
        self._command_safety_model_enabled = command_safety_model_enabled
        self._command_safety_model = command_safety_model
        self._start_time: float | None = None
        self._accumulated_text: list[str] = []
        self._queued_prompts: list[str] = []
        self._inject_after_tool: str | None = None
        self._bash_guard = BashRepeatGuard(max_repeats=3)
        # Track active subprocess for non-Claude providers so kill_agent
        # can terminate it promptly instead of waiting for natural exit.
        self._active_process: asyncio.subprocess.Process | None = None
        self._workspace_root = Path(self._descriptor.cwd or ".")
        self._policy_store = CommandPolicyStore(self._workspace_root)
        policy_rules = self._policy_store.load_compiled()
        self._whitelist_patterns = list(policy_rules.whitelist)
        self._blacklist_patterns = list(policy_rules.blacklist)
        self._whitelist_patterns.extend(
            self._compile_policy_patterns(command_whitelist or [])
        )
        self._blacklist_patterns.extend(
            self._compile_policy_patterns(command_blacklist or [])
        )
        self._orchestration_tools = orchestration_tools

    @property
    def agent_id(self) -> str:
        return self._descriptor.agent_id

    @property
    def state(self) -> AgentState:
        return self._descriptor.state

    def queue_prompt(self, prompt: str) -> None:
        """Queue a prompt to run after the current task completes."""
        self._queued_prompts.append(prompt)

    def inject_after_tool(self, prompt: str) -> None:
        """Set a prompt to inject after the current tool call finishes."""
        self._inject_after_tool = prompt

    def pop_queued_prompt(self) -> str | None:
        """Pop the next queued prompt, or None if empty."""
        if self._queued_prompts:
            return self._queued_prompts.pop(0)
        return None

    def _resolve_model_max_context(self) -> int | None:
        """Best-effort lookup of the model's context window size."""
        registry = getattr(self, "_model_registry", None)
        if registry is None:
            return None
        getter = getattr(registry, "get", None)
        if not callable(getter):
            return None
        try:
            capability = getter(self._descriptor.model)
        except Exception:
            return None
        max_context = getattr(capability, "max_context", None)
        if isinstance(max_context, int) and max_context > 0:
            return max_context
        return None

    async def _emit_context_window_usage(
        self,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> None:
        """Emit context window usage telemetry for UI display."""
        if total_tokens <= 0:
            return

        max_context_tokens = self._resolve_model_max_context()
        payload = {
            "event": "context_window_usage",
            "agent_id": self.agent_id,
            "model": self._descriptor.model,
            "input_tokens": max(0, int(input_tokens)),
            "cached_input_tokens": max(0, int(cached_input_tokens)),
            "output_tokens": max(0, int(output_tokens)),
            "total_tokens": max(0, int(total_tokens)),
        }
        if max_context_tokens:
            payload["max_context_tokens"] = max_context_tokens
            payload["percent_used"] = round(
                (payload["total_tokens"] / max_context_tokens) * 100.0,
                2,
            )

        await fire_event(self._event_callback, payload)

    @property
    def has_queued_prompts(self) -> bool:
        """Check if there are queued prompts waiting."""
        return bool(self._queued_prompts)

    def terminate_subprocess(self) -> bool:
        """Terminate the active provider subprocess, if any.

        Called by AgentManager.kill_agent() to ensure non-Claude provider
        subprocesses are stopped promptly when an agent is killed.
        """
        proc = self._active_process
        if proc is None:
            return False
        interrupted = False
        try:
            if proc.returncode is None:
                # Best-effort Ctrl+C semantics for provider-native tool calls.
                # Provider subprocesses are started in their own session, so we
                # can signal the whole process group.
                interrupted = False
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGINT)
                        interrupted = True
                except ProcessLookupError:
                    pass
                except Exception:
                    interrupted = False
                if not interrupted:
                    proc.terminate()
                    interrupted = True
                logger.info(
                    "Terminated provider subprocess for agent %s (pid=%s, interrupted=%s)",
                    self.agent_id[:8],
                    proc.pid,
                    interrupted,
                )
        except ProcessLookupError:
            return interrupted  # Already exited
        except Exception:
            # Last resort: try kill
            try:
                proc.kill()
                interrupted = True
            except Exception:
                pass
        return interrupted

    def kill_tool_call(self, tool_call_id: str) -> bool:
        """Kills a specific tool call by its ID, if supported.

        Currently only supports killing run_bash subprocesses.
        """
        if self._orchestration_tools:
            pid = None
            pid_getter = getattr(self._orchestration_tools, "get_bash_subprocess_pid", None)
            if callable(pid_getter):
                try:
                    pid = pid_getter(tool_call_id)
                except Exception:
                    pid = None
            logger.info(
                "Agent %s kill_tool_call request tool_call_id=%s pid=%s",
                self.agent_id[:8],
                tool_call_id[:12],
                pid if pid is not None else "unknown",
            )
            killed = bool(
                self._orchestration_tools.kill_bash_subprocess(tool_call_id)
            )
            logger.info(
                "Agent %s kill_tool_call result tool_call_id=%s killed=%s",
                self.agent_id[:8],
                tool_call_id[:12],
                killed,
            )
            return killed
        else:
            logger.warning(
                "Agent %s: OrchestrationTools not available to kill tool_call_id %s",
                self.agent_id[:8],
                tool_call_id[:8],
            )
            return False

    def get_latest_active_bash_tool_call_id(self) -> str | None:
        """Return the latest active run_bash tool_call_id, if any."""
        if not self._orchestration_tools:
            return None
        getter = getattr(
            self._orchestration_tools,
            "get_latest_active_bash_tool_call_id",
            None,
        )
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def get_active_bash_process_snapshot(self) -> list[dict[str, Any]]:
        """Return active run_bash process metadata for diagnostics."""
        if not self._orchestration_tools:
            return []
        getter = getattr(
            self._orchestration_tools,
            "get_active_bash_process_snapshot",
            None,
        )
        if not callable(getter):
            return []
        try:
            snapshot = getter()
            return snapshot if isinstance(snapshot, list) else []
        except Exception:
            return []

    def interrupt_active_execution(self) -> bool:
        """Best-effort interrupt for active tool/process execution.

        This handles both:
        - run_bash subprocesses tracked by OrchestrationTools
        - provider-native command execution running inside the provider CLI
        """
        interrupted = False
        latest_bash_tool_id = self.get_latest_active_bash_tool_call_id()
        logger.info(
            "Agent %s interrupt_active_execution start latest_bash_tool_id=%s has_provider_proc=%s",
            self.agent_id[:8],
            (latest_bash_tool_id[:12] if latest_bash_tool_id else "none"),
            bool(self._active_process and self._active_process.returncode is None),
        )
        if latest_bash_tool_id:
            interrupted = bool(self.kill_tool_call(latest_bash_tool_id)) or interrupted
        provider_interrupted = self.terminate_subprocess()
        interrupted = provider_interrupted or interrupted
        logger.info(
            "Agent %s interrupt_active_execution done interrupted=%s (bash=%s provider=%s)",
            self.agent_id[:8],
            interrupted,
            bool(latest_bash_tool_id),
            provider_interrupted,
        )
        return interrupted

    _RETRIABLE_TRANSPORT_PATTERNS: tuple[str, ...] = (
        "processtransport is not ready for writing",
        "tool permission stream closed before response received",
        "tool permission request failed: error: stream closed",
        "control request timeout: initialize",
        "initialize",
        "timeout",
        "stream closed",
        "cliconnectionerror",
    )
    _transport_failure_count: int = 0
    _breaker_open_until: float = 0.0
    _breaker_lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def _maybe_wait_for_circuit_breaker(cls) -> None:
        """Pause briefly when repeated transport failures open the breaker."""
        async with cls._breaker_lock:
            remaining = cls._breaker_open_until - time.monotonic()
        if remaining > 0:
            logger.warning(
                "Claude transport circuit breaker open; waiting %.2fs before retrying",
                remaining,
            )
            await asyncio.sleep(remaining)

    @classmethod
    async def _record_transport_failure(cls) -> None:
        """Track transport failures and open cooldown breaker when needed."""
        threshold = max(
            1, int(os.getenv("PRSM_TRANSPORT_BREAKER_THRESHOLD", "3"))
        )
        cooldown = max(
            1.0,
            float(os.getenv("PRSM_TRANSPORT_BREAKER_COOLDOWN_SECONDS", "20.0")),
        )
        async with cls._breaker_lock:
            cls._transport_failure_count += 1
            if cls._transport_failure_count >= threshold:
                cls._breaker_open_until = max(
                    cls._breaker_open_until,
                    time.monotonic() + cooldown,
                )
                logger.warning(
                    "Claude transport breaker opened after %d consecutive failures; "
                    "cooldown %.1fs",
                    cls._transport_failure_count,
                    cooldown,
                )

    @classmethod
    async def _record_transport_success(cls) -> None:
        async with cls._breaker_lock:
            cls._transport_failure_count = 0
            cls._breaker_open_until = 0.0

    async def _transition(self, new_state: AgentState) -> None:
        """Transition to a new state with validation."""
        validate_transition(self._descriptor.state, new_state)
        old = self._descriptor.state
        self._descriptor.state = new_state
        logger.info(
            "Agent %s: %s -> %s",
            self.agent_id[:8],
            old.value,
            new_state.value,
        )
        await fire_event(self._event_callback, {
            "event": "agent_state_changed",
            "agent_id": self.agent_id,
            "old_state": old.value,
            "new_state": new_state.value,
        })

    async def _wait_for_children_then_complete(
        self, result_text: str,
    ) -> None:
        """Wait for all active children to finish, then transition to COMPLETED.

        Called when a non-Claude provider subprocess exits but its children
        are still running. Transitions to WAITING_FOR_CHILD while waiting,
        then to COMPLETED when all children finish.

        Polls the manager every 2 seconds (lightweight) rather than blocking
        on futures, since the children were spawned via MCP tools and may
        have already finished by the time we check.
        """
        try:
            await self._transition(AgentState.WAITING_FOR_CHILD)
        except ValueError:
            # State machine doesn't allow this transition (e.g., already
            # moved to a terminal state) — just complete directly.
            if self._descriptor.state == AgentState.RUNNING:
                await self._transition(AgentState.COMPLETED)
            return

        poll_interval = 2.0
        max_wait = self._agent_timeout  # Don't wait longer than agent timeout
        waited = 0.0

        try:
            while self._manager.has_active_children(self.agent_id):
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                if waited >= max_wait:
                    logger.warning(
                        "Agent %s: timed out waiting for children after %.0fs",
                        self.agent_id[:8],
                        waited,
                    )
                    break
        except asyncio.CancelledError:
            # Agent was killed/interrupted while waiting for children.
            # Let the cancellation propagate so the task terminates cleanly.
            logger.info(
                "Agent %s: cancelled while waiting for children",
                self.agent_id[:8],
            )
            raise

        # Transition WAITING_FOR_CHILD → RUNNING → COMPLETED
        # (lifecycle requires going through RUNNING first)
        if self._descriptor.state == AgentState.WAITING_FOR_CHILD:
            await self._transition(AgentState.RUNNING)
        if self._descriptor.state == AgentState.RUNNING:
            await self._transition(AgentState.COMPLETED)

    async def run(self) -> AgentResult:
        """Execute the agent session to completion."""
        self._start_time = time.monotonic()
        await self._transition(AgentState.STARTING)
        max_attempts = max(
            1,
            int(os.getenv("PRSM_TRANSPORT_RETRY_ATTEMPTS", "5")),
        )
        base_delay = max(
            0.1,
            float(os.getenv("PRSM_TRANSPORT_RETRY_BASE_DELAY_SECONDS", "0.75")),
        )
        max_delay = max(
            base_delay,
            float(os.getenv("PRSM_TRANSPORT_RETRY_MAX_DELAY_SECONDS", "5.0")),
        )
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                await self._maybe_wait_for_circuit_breaker()
                # If a non-Claude provider is set, delegate to it
                if self._provider and self._provider.name != "claude":
                    # Master agents with MCP-capable providers get
                    # the full orchestration bridge; workers use
                    # the simple single-shot path.
                    from .models import AgentRole
                    if (
                        self._descriptor.role == AgentRole.MASTER
                        and self._provider.supports_master
                    ):
                        return await self._run_with_provider_mcp()
                    return await self._run_with_provider()

                # Import SDK lazily to avoid import errors when SDK
                # is not installed (e.g., during unit testing)
                from claude_agent_sdk import query, ClaudeAgentOptions

                from .mcp_server.server import (
                    build_agent_mcp_config,
                    ORCHESTRATION_TOOL_NAMES,
                )

                # Build per-agent in-process MCP server with orchestration tools
                mcp_config, orch_tools = build_agent_mcp_config(
                    agent_id=self.agent_id,
                    manager=self._manager,
                    router=self._router,
                    expert_registry=self._expert_registry,
                    tool_call_timeout=self._tool_call_timeout,
                    default_model=self._child_default_model,
                    default_provider=self._child_default_provider,
                    user_question_callback=self._user_question_callback,
                    conversation_store=self._conversation_store,
                    model_registry=self._model_registry,
                    peer_models=self._peer_models,
                    permission_callback=self._permission_callback,
                    cwd=self._descriptor.cwd,
                    event_callback=self._event_callback,
                )
                self._orchestration_tools = orch_tools # Assign the orch_tools instance
                tracker = orch_tools.time_tracker

                # Merge external plugin MCP servers
                if self._plugin_mcp_servers:
                    mcp_config.update(self._plugin_mcp_servers)

                # Build the system prompt with orchestration instructions
                system_suffix = self._build_orchestration_instructions()

                # Determine tool allowlist (base tools + orchestration tools).
                # Claude SDK may reference MCP tools in namespaced form
                # (mcp__orchestrator__tool_name), so include both forms.
                base_tools = self._descriptor.tools or [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
                ]
                orch_namespaced = [
                    f"mcp__orchestrator__{name}"
                    for name in ORCHESTRATION_TOOL_NAMES
                ]
                control_tools = list(self._CONTROL_TOOLS)
                all_tools = (
                    base_tools
                    + ORCHESTRATION_TOOL_NAMES
                    + orch_namespaced
                    + control_tools
                )

                # Collect stderr from the CLI for error diagnostics
                self._stderr_lines: list[str] = []

                def _capture_stderr(line: str) -> None:
                    self._stderr_lines.append(line)
                    logger.debug("claude stderr: %s", line.rstrip())

                # Build PreToolUse hooks to intercept dangerous bash commands.
                # Hooks run BEFORE permission mode in the CLI's evaluation
                # flow, so they work regardless of the permission_mode setting.
                # This is the reliable way to block dangerous commands like rm.
                bash_hooks = self._build_bash_permission_hooks()

                options_kwargs = dict(
                    system_prompt=system_suffix,
                    allowed_tools=all_tools,
                    permission_mode=self._descriptor.permission_mode.value,
                    cwd=self._descriptor.cwd,
                    model=self._descriptor.model,
                    mcp_servers=mcp_config,
                    stderr=_capture_stderr,
                    debug_stderr=None,  # Disable default stderr passthrough
                    hooks=bash_hooks,
                )
                # Enable token-by-token streaming if SDK supports it
                options_kwargs["include_partial_messages"] = True
                # Use the system Claude CLI only if explicitly configured.
                # Default to the SDK-bundled binary, which avoids the
                # "nested Claude Code session" error when PRSM is run
                # inside Claude Code (the system CLI refuses to launch
                # if CLAUDECODE env var is set).
                cli_path = os.getenv("PRSM_CLAUDE_CLI_PATH", "").strip()
                if cli_path:
                    resolved_cli = shutil.which(cli_path)
                    if resolved_cli:
                        options_kwargs["cli_path"] = resolved_cli
                    else:
                        logger.warning(
                            "Configured Claude CLI not found: %s; falling back to SDK default",
                            cli_path,
                        )
                # Clear CLAUDECODE env var so the SDK-bundled or system CLI
                # doesn't reject us as a nested session.
                os.environ.pop("CLAUDECODE", None)
                # Always wire can_use_tool so the BashRepeatGuard can
                # intercept runaway command loops.  For bypass-mode agents
                # the callback auto-allows everything except repetitions.
                options_kwargs["can_use_tool"] = self._check_permission
                logger.info(
                    "Agent %s starting query model=%s mode=%s tools=%d can_use_tool=%s cwd=%s cli=%s",
                    self.agent_id[:8],
                    self._descriptor.model,
                    self._descriptor.permission_mode.value,
                    len(all_tools),
                    "yes" if "can_use_tool" in options_kwargs else "no",
                    self._descriptor.cwd,
                    options_kwargs.get("cli_path", "<sdk-bundled>"),
                )
                options = ClaudeAgentOptions(**options_kwargs)

                if self._descriptor.state != AgentState.RUNNING:
                    await self._transition(AgentState.RUNNING)

                # Always use AsyncIterable prompt instead of a string.
                # For string prompts the SDK calls end_input() (closes
                # stdin) immediately after writing — before MCP control
                # requests can be serviced.
                #
                # The generator returns immediately after yielding.
                # The SDK's stream_input() then waits for the first
                # "result" message (_first_result_event) before calling
                # end_input().  The default 60s timeout is too short
                # for agent sessions, so we increase it via env var.
                # When the result arrives, stdin is closed and the CLI
                # finishes cleanly.
                close_timeout_seconds = max(
                    (
                        timeout
                        for timeout in (
                            self._agent_timeout,
                            self._tool_call_timeout,
                        )
                        if timeout > 0
                    ),
                    default=7200.0,
                )
                os.environ["CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"] = str(
                    int(close_timeout_seconds * 2 * 1000)
                )

                async def _prompt_stream():
                    yield {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": self._descriptor.prompt,
                        },
                    }

                prompt_input: AsyncIterator = _prompt_stream()

                # Log the initial user prompt to conversation store
                self._log_conversation(
                    "user_message",
                    content=self._descriptor.prompt,
                )

                # Run the query and collect output
                result_text = ""
                async for message in query(
                    prompt=prompt_input,
                    options=options,
                ):
                    # Check agent timeout (reasoning time only)
                    elapsed = time.monotonic() - self._start_time
                    reasoning_time = (
                        elapsed - tracker.accumulated_tool_time
                    )
                    if self._agent_timeout > 0 and reasoning_time > self._agent_timeout:
                        raise AgentTimeoutError(
                            self.agent_id, self._agent_timeout
                        )

                    # Emit events for message content
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "thinking"):
                                await fire_event(self._event_callback, {
                                    "event": "thinking",
                                    "agent_id": self.agent_id,
                                    "text": str(block.thinking or "")[:500],
                                })
                                self._log_conversation(
                                    "thinking",
                                    content=str(block.thinking or ""),
                                )
                            elif hasattr(block, "text"):
                                self._accumulated_text.append(block.text)
                                await fire_event(self._event_callback, {
                                    "event": "stream_chunk",
                                    "agent_id": self.agent_id,
                                    "text": block.text,
                                })
                                self._log_conversation(
                                    "text",
                                    content=block.text,
                                )
                            elif hasattr(block, "name") and hasattr(block, "input"):
                                logger.info(
                                    "Agent %s tool_use id=%s name=%s",
                                    self.agent_id[:8],
                                    str(getattr(block, "id", ""))[:12],
                                    block.name,
                                )
                                try:
                                    _args_str = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                                except (TypeError, ValueError):
                                    _args_str = str(block.input)
                                # For file-modifying tools, preserve full args so server
                                # can extract file_path, old_string, new_string.
                                _max_args = self._tool_argument_limit(block.name)
                                await fire_event(self._event_callback, {
                                    "event": "tool_call_started",
                                    "agent_id": self.agent_id,
                                    "tool_id": getattr(block, "id", ""),
                                    "tool_name": block.name,
                                    "arguments": _args_str[:_max_args],
                                })
                                self._log_conversation(
                                    "tool_call",
                                    tool_name=block.name,
                                    tool_id=getattr(block, "id", ""),
                                    tool_args=str(block.input)[:2000],
                                )
                            elif hasattr(block, "tool_use_id"):
                                logger.info(
                                    "Agent %s tool_result id=%s is_error=%s",
                                    self.agent_id[:8],
                                    str(getattr(block, "tool_use_id", ""))[:12],
                                    getattr(block, "is_error", False),
                                )
                                result_text = self._extract_tool_result_text(
                                    getattr(block, "content", "")
                                )
                                await fire_event(self._event_callback, {
                                    "event": "tool_call_completed",
                                    "agent_id": self.agent_id,
                                    "tool_id": getattr(block, "tool_use_id", ""),
                                    "result": result_text,
                                    "is_error": getattr(block, "is_error", False),
                                })
                                self._log_conversation(
                                    "tool_result",
                                    content=result_text[:2000],
                                    tool_id=getattr(block, "tool_use_id", ""),
                                    is_error=getattr(block, "is_error", False),
                                )

                    # Stop immediately if task_complete was called
                    if self._descriptor.result_summary is not None:
                        logger.info(
                            "Agent %s called task_complete — stopping session",
                            self.agent_id[:8],
                        )
                        break

                    # Collect result
                    if hasattr(message, "result"):
                        result_text = message.result
                        await fire_event(self._event_callback, {
                            "event": "agent_result",
                            "agent_id": self.agent_id,
                            "result": result_text[:500] if result_text else "",
                            "is_error": getattr(message, "is_error", False),
                        })

                if self._descriptor.state == AgentState.RUNNING:
                    await self._transition(AgentState.COMPLETED)
                await self._record_transport_success()

                duration = time.monotonic() - self._start_time
                return AgentResult(
                    agent_id=self.agent_id,
                    success=True,
                    summary=(
                        self._descriptor.result_summary
                        or result_text
                        or "\n".join(self._accumulated_text)
                    ),
                    artifacts=self._result_artifacts(),
                    duration_seconds=duration,
                )

            except AgentTimeoutError:
                await self._transition(AgentState.FAILED)
                self._descriptor.error = "timeout"
                duration = time.monotonic() - (
                    self._start_time or time.monotonic()
                )
                return AgentResult(
                    agent_id=self.agent_id,
                    success=False,
                    summary="",
                    error=(
                        f"Agent reasoning time exceeded "
                        f"{self._agent_timeout}s (excludes tool calls)"
                    ),
                    duration_seconds=duration,
                )

            except Exception as exc:
                retriable = self._is_retriable_transport_exception(exc)
                if retriable and attempt < max_attempts:
                    await self._record_transport_failure()
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warning(
                        "Agent %s transient transport failure on attempt %d/%d; retrying in %.2fs: %s",
                        self.agent_id[:8],
                        attempt,
                        max_attempts,
                        delay,
                        self._compact_exception(exc),
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.exception(
                    "Agent %s session error (attempt %d/%d)",
                    self.agent_id[:8],
                    attempt,
                    max_attempts,
                )
                try:
                    await self._transition(AgentState.FAILED)
                except ValueError:
                    pass  # Already in terminal state

                if retriable:
                    await self._record_transport_failure()
                    error_msg = (
                        "Temporary Claude transport instability persisted after "
                        f"{attempt} attempt(s). Please retry the request."
                    )
                else:
                    # Include captured stderr and accumulated text for diagnostics
                    error_msg = str(exc)
                    stderr_output = getattr(self, "_stderr_lines", [])
                    if stderr_output:
                        stderr_tail = "\n".join(stderr_output[-10:])
                        error_msg = f"{error_msg}\nstderr: {stderr_tail}"
                    # The SDK may emit error details as text blocks before
                    # raising the exception (e.g., "There's an issue with
                    # the selected model"). Include accumulated text.
                    if self._accumulated_text:
                        text_tail = "\n".join(self._accumulated_text[-5:])
                        error_msg = f"{error_msg}\noutput: {text_tail}"

                    # Detect model-not-accessible errors and mark the model
                    # unavailable in the registry so future spawns skip it.
                    combined_err = error_msg.lower()
                    if (
                        "issue with the selected model" in combined_err
                        or "not have access" in combined_err
                        or "does not exist" in combined_err
                    ):
                        self._mark_model_unavailable(self._descriptor.model)

                self._descriptor.error = error_msg
                duration = time.monotonic() - (
                    self._start_time or time.monotonic()
                )
                return AgentResult(
                    agent_id=self.agent_id,
                    success=False,
                    summary="",
                    error=error_msg,
                    duration_seconds=duration,
                )

        # Defensive fallback; loop always returns above.
        duration = time.monotonic() - (
            self._start_time or time.monotonic()
        )
        return AgentResult(
            agent_id=self.agent_id,
            success=False,
            summary="",
            error="Agent failed without a terminal result.",
            duration_seconds=duration,
        )

    def _result_artifacts(self) -> dict[str, Any]:
        artifacts = getattr(self._descriptor, "result_artifacts", None)
        if isinstance(artifacts, dict):
            return dict(artifacts)
        return {}

    def _mark_model_unavailable(self, model_id: str) -> None:
        """Mark a model as unavailable in the registry after a runtime failure.

        Called when the SDK reports that the user doesn't have access
        to a model. This prevents future spawns from selecting it.
        """
        if not self._model_registry:
            return
        cap = self._model_registry.get(model_id)
        if cap and cap.available:
            cap.available = False
            logger.warning(
                "Model '%s' marked unavailable at runtime "
                "(user account does not have access)",
                model_id,
            )

    @classmethod
    def _compact_exception(cls, exc: BaseException) -> str:
        text = cls._exception_text(exc)
        return " | ".join([p.strip() for p in text.splitlines() if p.strip()][:3])

    @classmethod
    def _exception_text(cls, exc: BaseException) -> str:
        parts: list[str] = []

        def visit(err: BaseException | None) -> None:
            if err is None:
                return
            parts.append(f"{type(err).__name__}: {err}")
            nested = getattr(err, "exceptions", None)
            if isinstance(nested, tuple):
                for child in nested:
                    if isinstance(child, BaseException):
                        visit(child)
            visit(getattr(err, "__cause__", None))
            visit(getattr(err, "__context__", None))

        visit(exc)
        return "\n".join(parts)

    @classmethod
    def _is_retriable_transport_exception(cls, exc: BaseException) -> bool:
        text = cls._exception_text(exc).lower()
        return any(pat in text for pat in cls._RETRIABLE_TRANSPORT_PATTERNS)

    @staticmethod
    def _sanitize_provider_stderr(
        provider_name: str,
        stderr_text: str,
    ) -> str:
        """Remove known noisy stderr lines from provider CLIs."""
        text = str(stderr_text or "")
        lines = text.splitlines()
        if provider_name in {"codex", "minimax", "alibaba"}:
            filtered = [
                line for line in lines
                if "state db missing rollout path for thread"
                not in line.lower()
            ]
            if filtered:
                return "\n".join(filtered).strip()
            return ""
        return text.strip()

    async def _run_with_provider(self) -> AgentResult:
        """Execute the agent session via an external provider."""
        await self._transition(AgentState.RUNNING)

        system_suffix = self._build_orchestration_instructions()
        result_text = ""
        streamed_text: list[str] = []
        had_error = False
        provider_name = str(getattr(self._provider, "name", "")).lower()
        codex_active_tools: dict[str, str] = {}
        fallback_active_tool_id: list[str | None] = [None]

        try:
            async for message in self._provider.run_agent(
                prompt=self._descriptor.prompt,
                system_prompt=system_suffix,
                model_id=self._descriptor.model,
                tools=self._descriptor.tools or [],
                permission_mode=self._descriptor.permission_mode.value,
                cwd=self._descriptor.cwd,
            ):
                chunk = str(getattr(message, "text", "") or "")
                if not message.is_result:
                    display_text: str | None = None
                    if provider_name in {"codex", "minimax"}:
                        display_text = await self._process_codex_jsonl(
                            chunk,
                            codex_active_tools,
                        )
                    elif provider_name == "gemini":
                        display_text = await self._process_gemini_stream_json(chunk)
                    else:
                        display_text = chunk
                        await self._detect_tool_from_text(
                            chunk,
                            fallback_active_tool_id,
                        )

                    if display_text:
                        display_text = self._sanitize_agent_text(display_text)
                    if display_text:
                        streamed_text.append(display_text)
                        await fire_event(self._event_callback, {
                            "event": "stream_chunk",
                            "agent_id": self.agent_id,
                            "text": display_text,
                        })
                        self._accumulated_text.append(display_text)
                        self._log_conversation("text", content=display_text)
                    continue

                result_text = chunk
                if getattr(message, "is_error", False):
                    had_error = True

                if result_text:
                    # Avoid duplicating output in the transcript when the
                    # provider already streamed its assistant/tool text.
                    if not streamed_text:
                        await fire_event(self._event_callback, {
                            "event": "stream_chunk",
                            "agent_id": self.agent_id,
                            "text": result_text,
                        })
                        self._accumulated_text.append(result_text)
                        self._log_conversation("text", content=result_text)
        except Exception as exc:
            logger.exception(
                "Provider %s run_agent failed for agent %s",
                self._provider.name,
                self.agent_id[:8],
            )
            had_error = True
            result_text = f"Provider error: {exc}"
            await fire_event(self._event_callback, {
                "event": "stream_chunk",
                "agent_id": self.agent_id,
                "text": result_text,
            })

        if not result_text and streamed_text:
            result_text = "".join(streamed_text).strip()

        if had_error:
            if self._descriptor.state == AgentState.RUNNING:
                await self._transition(AgentState.FAILED)
            self._descriptor.error = result_text[:500]
        elif self._descriptor.state == AgentState.RUNNING:
            # Check if this agent has active children before marking COMPLETED.
            # Non-Claude providers run as subprocesses that exit when done, but
            # their children (spawned via MCP tools) may still be running.
            if self._manager.has_active_children(self.agent_id):
                logger.info(
                    "Agent %s: provider subprocess exited but children "
                    "still active — staying in RUNNING state",
                    self.agent_id[:8],
                )
                await self._wait_for_children_then_complete(result_text)
            else:
                await self._transition(AgentState.COMPLETED)

        # Emit agent_result so TUI/server knows this agent finished
        await fire_event(self._event_callback, {
            "event": "agent_result",
            "agent_id": self.agent_id,
            "result": result_text[:500] if result_text else "",
            "is_error": had_error,
        })

        duration = time.monotonic() - self._start_time
        return AgentResult(
            agent_id=self.agent_id,
            success=not had_error,
            summary=(
                self._descriptor.result_summary
                or result_text
            ),
            artifacts=self._result_artifacts(),
            error=result_text[:500] if had_error else None,
            duration_seconds=duration,
        )

    @staticmethod
    async def _read_line_unbounded(stream: asyncio.StreamReader) -> bytes:
        """Read a full line from *stream* with no size limit.

        Unlike ``StreamReader.readline()``, this will never raise
        ``LimitOverrunError``.  When the internal buffer fills before a
        newline is found, we drain the buffered bytes and keep
        accumulating until the separator appears or EOF is reached.

        This is critical for Codex ``--json`` output where a single
        JSONL event (e.g. a tool result containing a large ``find``
        listing) can easily exceed the default 64 KiB StreamReader
        limit.
        """
        chunks: list[bytes] = []
        while True:
            try:
                chunk = await stream.readuntil(b"\n")
                # Got a complete line (including the trailing newline)
                chunks.append(chunk)
                return b"".join(chunks)
            except asyncio.LimitOverrunError as exc:
                # Buffer is full but no newline yet.
                # exc.consumed tells us how many bytes are available.
                chunk = await stream.read(exc.consumed)
                chunks.append(chunk)
                # Loop back to keep reading until we find the newline.
            except asyncio.IncompleteReadError as exc:
                # EOF before newline — return whatever is left.
                chunks.append(exc.partial)
                return b"".join(chunks)

    async def _run_with_provider_mcp(self) -> AgentResult:
        """Execute a non-Claude provider as master with MCP orchestration.

        Starts a TCP bridge, configures the provider's CLI with an
        MCP proxy pointing back to the bridge, then runs the CLI.

        Tool calls from the CLI flow through:
            CLI -> MCP -> orch_proxy.py -> TCP -> OrchBridge -> OrchestrationTools
        """
        await self._transition(AgentState.RUNNING)

        from .mcp_server.server import build_agent_mcp_config
        from .mcp_server.orch_bridge import OrchBridge

        # Create OrchestrationTools (same as Claude path)
        _, orch_tools = build_agent_mcp_config(
            agent_id=self.agent_id,
            manager=self._manager,
            router=self._router,
            expert_registry=self._expert_registry,
            tool_call_timeout=self._tool_call_timeout,
            default_model=self._child_default_model,
            default_provider=self._child_default_provider,
            user_question_callback=self._user_question_callback,
            conversation_store=self._conversation_store,
            model_registry=self._model_registry,
            peer_models=self._peer_models,
            permission_callback=self._permission_callback,
            cwd=self._descriptor.cwd,
            event_callback=self._event_callback,
        )
        # Ensure Stop/kill paths can inspect and terminate active run_bash
        # processes for provider-MCP sessions (Codex/Gemini/etc.).
        self._orchestration_tools = orch_tools

        # Start TCP bridge
        bridge = OrchBridge(
            orch_tools=orch_tools,
            manager=self._manager,
            expert_registry=self._expert_registry,
            event_callback=self._event_callback,
            agent_id=self.agent_id,
        )
        port = await bridge.start()

        system_prompt = self._build_orchestration_instructions()
        result_text = ""
        had_error = False

        try:
            # Build provider command with MCP config
            cmd, env, stdin_payload = self._provider.build_master_cmd(
                prompt=self._descriptor.prompt,
                bridge_port=port,
                system_prompt=system_prompt,
                model_id=self._descriptor.model,
                cwd=self._descriptor.cwd,
                plugin_mcp_servers=self._plugin_mcp_servers,
            )

            logger.info(
                "Agent %s starting provider-MCP master: %s cmd=%s",
                self.agent_id[:8],
                self._provider.name,
                cmd[0],
            )

            # Safe array-based subprocess — no shell, no injection risk
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_payload is not None
                    else None
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._descriptor.cwd,
                start_new_session=True,
            )
            # Track subprocess so kill_agent can terminate it
            self._active_process = proc
            if stdin_payload is not None and proc.stdin is not None:
                proc.stdin.write(stdin_payload.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()

            # Stream stdout to TUI while CLI runs, parsing tool calls.
            #
            # Codex CLI with --json emits structured JSONL events that
            # we parse for tool_call_started/completed events.
            # Other providers emit plain text — we use regex fallback.
            # Determine output parsing mode based on provider
            # - Codex/MiniMax: use --json JSONL events
            # - Gemini: use --output-format stream-json events
            # - Other: regex-based text fallback
            is_codex = self._provider.name in ("codex", "minimax", "alibaba")
            is_gemini = self._provider.name == "gemini"
            result_parts: list[str] = []
            # active_tools: item_id → tool_name (for JSONL mode)
            active_tools: dict[str, str] = {}
            # active_tool_id: mutable ref for regex fallback mode
            active_tool_id: list[str | None] = [None]

            # Create a task for the bridge task_completed event so we can
            # race it against readline.  When the master calls task_complete,
            # the bridge sets the event and we can terminate the CLI subprocess
            # without waiting for it to finish its output.
            task_done_event = asyncio.ensure_future(bridge.task_completed.wait())

            while True:
                read_task = asyncio.ensure_future(
                    self._read_line_unbounded(proc.stdout)
                )
                done, _ = await asyncio.wait(
                    {read_task, task_done_event},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if task_done_event in done:
                    # Master called task_complete — terminate the CLI
                    read_task.cancel()
                    logger.info(
                        "Agent %s bridge signaled task_completed — "
                        "terminating CLI subprocess",
                        self.agent_id[:8],
                    )
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                    break

                line = read_task.result()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")

                if is_codex:
                    # Structured JSONL parsing for Codex --json output
                    display = await self._process_codex_jsonl(
                        text, active_tools,
                    )
                    if display:
                        # Sanitize before display — non-JSON passthrough
                        # or error text may contain echoed tool blocks.
                        display = self._sanitize_agent_text(display)
                    if display:
                        result_parts.append(display)
                        await fire_event(self._event_callback, {
                            "event": "stream_chunk",
                            "agent_id": self.agent_id,
                            "text": display,
                        })
                    else:
                        # Still collect raw line for final result assembly
                        result_parts.append(text)
                elif is_gemini:
                    # Structured stream-JSON parsing for Gemini CLI
                    display = await self._process_gemini_stream_json(
                        text,
                    )
                    if display:
                        display = self._sanitize_agent_text(display)
                    if display:
                        result_parts.append(display)
                        await fire_event(self._event_callback, {
                            "event": "stream_chunk",
                            "agent_id": self.agent_id,
                            "text": display,
                        })
                    else:
                        # Still collect raw line for final result assembly
                        result_parts.append(text)
                else:
                    # Plain text mode — regex-based tool detection
                    sanitized_text = self._sanitize_agent_text(text)
                    result_parts.append(text)  # Keep raw for result assembly
                    await self._detect_tool_from_text(
                        text, active_tool_id,
                    )
                    if sanitized_text:
                        await fire_event(self._event_callback, {
                            "event": "stream_chunk",
                            "agent_id": self.agent_id,
                            "text": sanitized_text,
                        })

            # Clean up the event waiter task
            if not task_done_event.done():
                task_done_event.cancel()

            await proc.wait()
            stdout_all = "".join(result_parts)

            # If we terminated the process because task_complete was called,
            # treat it as a success regardless of the return code.
            if bridge.task_completed.is_set() and bridge.task_result:
                result_text = bridge.task_result
            elif proc.returncode != 0:
                stderr_bytes = await proc.stderr.read()
                raw_error = stderr_bytes.decode("utf-8", errors="replace")
                error = self._sanitize_provider_stderr(
                    self._provider.name, raw_error,
                )
                if not error:
                    error = raw_error.strip() or "provider emitted no stderr"
                result_text = (
                    f"{self._provider.name} master failed "
                    f"(rc={proc.returncode}): {error}"
                )
                had_error = True
                await fire_event(self._event_callback, {
                    "event": "stream_chunk",
                    "agent_id": self.agent_id,
                    "text": result_text,
                })
            else:
                # Prefer task_complete summary if the master called it
                if bridge.task_result:
                    result_text = bridge.task_result
                else:
                    result_text = stdout_all.strip()

                # Detect when the model produced only hallucinated content
                # (all text was stripped by _sanitize_agent_text), leaving
                # no useful output and no tool calls executed via the bridge.
                if (
                    not result_text
                    and not bridge.task_result
                    and not had_error
                    and not bridge.task_completed.is_set()
                ):
                    logger.warning(
                        "Agent %s: %s produced no useful output "
                        "(possible hallucinated tool calls stripped)",
                        self.agent_id[:8],
                        self._provider.name,
                    )
                    result_text = (
                        f"The {self._provider.name} model produced no "
                        f"actionable output. Its response contained only "
                        f"hallucinated tool calls (XML-formatted text that "
                        f"was not executed). Please retry the task."
                    )
                    had_error = True
                    await fire_event(self._event_callback, {
                        "event": "stream_chunk",
                        "agent_id": self.agent_id,
                        "text": result_text,
                    })

        except FileNotFoundError:
            result_text = (
                f"ERROR: '{self._provider.name}' CLI not found. "
                f"Install it to use as master agent."
            )
            had_error = True
            await fire_event(self._event_callback, {
                "event": "stream_chunk",
                "agent_id": self.agent_id,
                "text": result_text,
            })
        except Exception as exc:
            logger.exception(
                "Provider-MCP master %s failed for agent %s",
                self._provider.name,
                self.agent_id[:8],
            )
            had_error = True
            result_text = f"Provider-MCP master error: {exc}"
            await fire_event(self._event_callback, {
                "event": "stream_chunk",
                "agent_id": self.agent_id,
                "text": result_text,
            })
        finally:
            self._active_process = None
            await bridge.stop()
            # Clean up any provider-specific state (e.g., Gemini .gemini/settings.json)
            if hasattr(self._provider, "cleanup_master_settings"):
                try:
                    self._provider.cleanup_master_settings()
                except Exception:
                    pass

        if had_error:
            if self._descriptor.state == AgentState.RUNNING:
                await self._transition(AgentState.FAILED)
            self._descriptor.error = result_text[:500]
        elif self._descriptor.state == AgentState.RUNNING:
            # Check if this agent has active children before marking COMPLETED.
            # Non-Claude providers run as subprocesses that exit when done, but
            # their children (spawned via MCP tools) may still be running.
            if self._manager.has_active_children(self.agent_id):
                logger.info(
                    "Agent %s: provider-MCP subprocess exited but children "
                    "still active — staying in RUNNING state",
                    self.agent_id[:8],
                )
                await self._wait_for_children_then_complete(result_text)
            else:
                await self._transition(AgentState.COMPLETED)

        # Log result to conversation store
        if result_text:
            self._accumulated_text.append(result_text)
            self._log_conversation("text", content=result_text)

        await fire_event(self._event_callback, {
            "event": "agent_result",
            "agent_id": self.agent_id,
            "result": result_text[:500] if result_text else "",
            "is_error": had_error,
        })

        duration = time.monotonic() - self._start_time
        return AgentResult(
            agent_id=self.agent_id,
            success=not had_error,
            summary=(
                self._descriptor.result_summary
                or result_text
            ),
            artifacts=self._result_artifacts(),
            error=result_text[:500] if had_error else None,
            duration_seconds=duration,
        )

    # ── Provider stdout tool-call detection ──────────────────────

    # Regex fallback for providers that don't support structured JSON
    # output (e.g. Gemini CLI, or Codex without --json).
    _TOOL_CALL_PATTERN: re.Pattern = re.compile(
        r'(?:^|\s)(Read|Write|Edit|Bash|Glob|Grep|WebFetch|WebSearch|NotebookEdit)\s*\(',
    )
    _TOOL_RESULT_PATTERN: re.Pattern = re.compile(
        r'^⎿\s*',
    )

    # Patterns for detecting hallucinated tool calls in agent text output.
    # Some models (especially MiniMax) sometimes output XML-style tool
    # invocations as plain text instead of using native function calling.
    _THINK_TAG_RE: re.Pattern = re.compile(
        r'<think>.*?</think>\s*', re.DOTALL,
    )
    _HALLUCINATED_TOOL_RE: re.Pattern = re.compile(
        r'(?:'
        r'<(?:antml:)?(?:invoke|function_calls?|tool_call|parameter)\b[^>]*>.*?'
        r'</(?:antml:)?(?:invoke|function_calls?|tool_call|parameter)>'
        r'|'
        r'\[Tool call:.*?\].*?<parameter\b.*?</parameter>'
        r'|'
        r'</invoke>\s*</(?:minimax|anthropic):tool_call>'
        r')',
        re.DOTALL,
    )
    # Matches bracket-style tool call blocks echoed from conversation history
    # context (produced by _build_restart_prompt or similar reconstruction).
    # Three patterns applied in sequence for robust stripping:
    # 1) Full block with all three markers
    _BRACKET_FULL_BLOCK_RE: re.Pattern = re.compile(
        r'\[Tool call:\s*[^\]]+\]'
        r'\s*\[Tool args\]'
        r'[\s\S]*?'
        r'\[Tool result:\s*[^\]]+\]'
        r'[ \t]*\n?',
    )
    # 2) Partial block: call + args without result
    _BRACKET_CALL_ARGS_RE: re.Pattern = re.compile(
        r'\[Tool call:\s*[^\]]+\]'
        r'\s*\[Tool args\]'
        r'[\s\S]*?'
        r'(?=\n\n|\[Tool call:|\Z)',
    )
    # 3) Standalone markers
    _BRACKET_MARKER_RE: re.Pattern = re.compile(
        r'\[Tool (?:call|args|result)(?::\s*[^\]]*)?\][ \t]*\n?'
    )

    @staticmethod
    def _bare_tool_name(tool_name: str) -> str:
        """Normalize MCP-prefixed tool names to a bare tool identifier."""
        if not tool_name:
            return ""
        if tool_name.startswith("mcp__") and tool_name.count("__") >= 2:
            return tool_name.split("__", 2)[2]
        return tool_name

    @classmethod
    def _tool_argument_limit(cls, tool_name: str) -> int:
        """Return a per-tool argument cap for tool_call_started events."""
        bare_name = cls._bare_tool_name(tool_name)
        if bare_name in ("Write", "Edit", "write_file", "edit_file"):
            return 50000
        if bare_name in ("Bash", "bash", "run_bash"):
            return 250000
        # Parallel spawn payloads can legitimately exceed 2k JSON args.
        if bare_name == "spawn_children_parallel":
            return 250000
        # Orchestration tools are structured JSON and should stay parseable.
        if bare_name in cls._ORCHESTRATION_TOOLS:
            return 50000
        return 2000

    @staticmethod
    def _sanitize_agent_text(text: str) -> str:
        """Strip hallucinated <think> tags and XML tool calls from agent text.

        Some models (MiniMax-M2.5 via Codex) emit <think>...</think> blocks
        and XML-formatted tool invocations as plain text instead of using the
        native function-calling mechanism. These are non-functional and should
        be stripped before displaying to the user.

        Returns the cleaned text, which may be empty if the entire response
        was hallucinated artifacts.
        """
        # Strip <think>...</think> blocks
        cleaned = AgentSession._THINK_TAG_RE.sub('', text)

        # Strip hallucinated XML tool calls
        cleaned = AgentSession._HALLUCINATED_TOOL_RE.sub('', cleaned)

        # Also strip standalone XML fragments that look like tool calls
        # e.g. </invoke>, </minimax:tool_call>, <parameter name="...">
        cleaned = re.sub(
            r'</?(?:antml:)?(?:invoke|function_calls?|tool_call|parameter)\b[^>]*>',
            '', cleaned,
        )
        cleaned = re.sub(
            r'</?(?:minimax|anthropic):tool_call\s*>',
            '', cleaned,
        )

        # Strip bracket-style tool call blocks echoed from history context
        # (produced by _build_restart_prompt or similar reconstruction).
        # Applied in three phases: full blocks first, then partial, then markers.
        cleaned = AgentSession._BRACKET_FULL_BLOCK_RE.sub('', cleaned)
        cleaned = AgentSession._BRACKET_CALL_ARGS_RE.sub('', cleaned)
        cleaned = AgentSession._BRACKET_MARKER_RE.sub('', cleaned)

        return cleaned.strip()

    @staticmethod
    def _extract_tool_result_text(result: Any) -> str:
        """Extract best-effort plain text from structured tool result payloads."""
        if result is None:
            return ""

        def _from_payload(payload: Any) -> str:
            if isinstance(payload, str):
                return payload
            if isinstance(payload, dict):
                content = payload.get("content")
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                chunks.append(text)
                    if chunks:
                        return "\n".join(chunks)
                for key in ("text", "stdout", "output", "result", "content"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        return value
                    if isinstance(value, (dict, list)):
                        nested = _from_payload(value)
                        if nested:
                            return nested
            if isinstance(payload, list):
                chunks: list[str] = []
                for item in payload:
                    nested = _from_payload(item)
                    if nested:
                        chunks.append(nested)
                if chunks:
                    return "\n".join(chunks)
            return ""

        if isinstance(result, (dict, list)):
            extracted = _from_payload(result)
            return extracted if extracted else str(result)

        text = str(result).strip()
        if not text:
            return ""

        candidate = text
        for _ in range(2):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                break
            if isinstance(parsed, str):
                candidate = parsed.strip()
                continue
            extracted = _from_payload(parsed)
            return extracted if extracted else candidate

        for _ in range(2):
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                break
            if isinstance(parsed, str):
                candidate = parsed.strip()
                continue
            extracted = _from_payload(parsed)
            return extracted if extracted else candidate

        return text

    async def _process_codex_jsonl(
        self,
        line: str,
        active_tools: dict[str, str],
    ) -> str | None:
        """Parse a single JSONL line from Codex --json output.

        Emits tool_call_started / tool_call_completed events as
        appropriate and returns display text (or None to suppress).

        Codex --json event types:
          thread.started, turn.started, turn.completed, turn.failed,
          item.started, item.completed, error

        Item types of interest:
          command_execution — shell commands
          file_edit / file_write / file_read — file operations
          mcp_tool_call — MCP tool invocations (our orchestration tools)
          agent_message — text output
        """
        stripped = line.strip()
        if not stripped:
            return None

        try:
            event = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — pass through as plain text
            return line

        etype = event.get("type", "")
        item = event.get("item") or {}
        item_id = item.get("id", "")
        item_type = item.get("type", "")

        # ── item.started ──
        if etype == "item.started":
            tool_name = ""
            arguments = ""
            suppress_provider_tool_event = False

            if item_type == "command_execution":
                tool_name = "Bash"
                arguments = item.get("command", "")
            elif item_type in ("file_edit", "file_write", "file_read"):
                # Map to our tool names
                type_map = {
                    "file_edit": "Edit",
                    "file_write": "Write",
                    "file_read": "Read",
                }
                tool_name = type_map.get(item_type, item_type)
                arguments = item.get("file_path", "") or item.get("path", "")
            elif item_type == "mcp_tool_call":
                tool_name = item.get("tool_name", "") or item.get("name", "")
                normalized = self._bare_tool_name(tool_name)
                mcp_name_map = {
                    "read_file": "Read",
                    "file_read": "Read",
                    "write_file": "Write",
                    "file_write": "Write",
                    "edit_file": "Edit",
                    "file_edit": "Edit",
                }
                tool_name = mcp_name_map.get(normalized, tool_name)
                args = item.get("arguments", item.get("input", ""))
                if isinstance(args, dict):
                    arguments = json.dumps(args)
                else:
                    arguments = str(args) if args else ""
                # MCP proxy calls are already emitted by orch_bridge with the
                # canonical tool ID used for streaming deltas. Avoid duplicate
                # started/completed events from provider JSONL parsing.
                suppress_provider_tool_event = True
            elif item_type == "web_search":
                tool_name = "WebSearch"
                arguments = item.get("query", "")

            if tool_name and not suppress_provider_tool_event:
                tool_id = item_id or str(uuid.uuid4())
                active_tools[item_id] = tool_name
                _max_args = self._tool_argument_limit(tool_name)
                await fire_event(self._event_callback, {
                    "event": "tool_call_started",
                    "agent_id": self.agent_id,
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "arguments": str(arguments)[:_max_args],
                })
                self._log_conversation(
                    "tool_call",
                    tool_name=tool_name,
                    tool_id=tool_id,
                    tool_args=str(arguments)[:2000],
                )
                logger.info(
                    "Agent %s provider tool_call id=%s name=%s",
                    self.agent_id[:8],
                    tool_id[:12],
                    tool_name,
                )

            return None  # Suppress raw JSON from display

        # ── item.completed ──
        if etype == "item.completed":
            if item_type == "agent_message":
                # This is actual text output — display it.
                # Sanitize hallucinated <think> tags and XML tool calls
                # that some models (MiniMax) emit as plain text.
                raw_text = item.get("text", "")
                if raw_text:
                    text = self._sanitize_agent_text(raw_text)
                    if text != raw_text:
                        logger.warning(
                            "Agent %s: stripped hallucinated content "
                            "from agent_message (%d→%d chars)",
                            self.agent_id[:8],
                            len(raw_text),
                            len(text),
                        )
                    if text:
                        return text + "\n"
                return None

            # Tool completion
            tool_id = item_id or ""
            tool_name = active_tools.pop(item_id, "")
            result_text = ""
            is_error = item.get("status") == "failed"
            suppress_provider_tool_event = False

            if item_type == "command_execution":
                result_text = item.get("output", "")
            elif item_type in ("file_edit", "file_write", "file_read"):
                result_text = item.get("result", "done")
            elif item_type == "mcp_tool_call":
                raw_result = item.get("result", "")
                bare_tool_name = self._bare_tool_name(tool_name).lower()
                if bare_tool_name in {"bash", "run_bash", "run_shell_command"}:
                    result_text = self._extract_tool_result_text(raw_result)
                else:
                    result_text = str(raw_result)
                # Completion for MCP proxy calls is emitted by orch_bridge.
                suppress_provider_tool_event = True
            elif item_type == "web_search":
                result_text = str(item.get("results", ""))

            if tool_id and not suppress_provider_tool_event:
                await fire_event(self._event_callback, {
                    "event": "tool_call_completed",
                    "agent_id": self.agent_id,
                    "tool_id": tool_id,
                    "result": result_text,
                    "is_error": is_error,
                })
                self._log_conversation(
                    "tool_result",
                    content=result_text,
                    tool_id=tool_id,
                    is_error=is_error,
                )

            return None  # Suppress raw JSON from display

        # ── turn.completed ──
        if etype == "turn.completed":
            usage = event.get("usage", {})
            if usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                logger.info(
                    "Agent %s turn completed: input=%d cached=%d output=%d",
                    self.agent_id[:8],
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                )
                await self._emit_context_window_usage(
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + cached_input_tokens + output_tokens,
                )
            return None

        # ── error ──
        if etype == "error":
            error_msg = event.get("message", str(event))
            return f"[error] {error_msg}\n"

        # Other events — suppress
        return None

    async def _process_gemini_stream_json(
        self,
        line: str,
    ) -> str | None:
        """Parse a single stream-JSON line from Gemini --output-format stream-json.

        Emits tool_call_started / tool_call_completed events and returns
        display text (or None to suppress).

        Gemini stream-json event types:
          init — session metadata
          message — text content (role=user or assistant)
          tool_use — tool call started
          tool_result — tool call completed
          result — final stats
        """
        stripped = line.strip()
        if not stripped:
            return None

        try:
            event = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Not JSON (e.g. "Loaded cached credentials.") — suppress
            return None

        etype = event.get("type", "")

        # ── message (assistant text) ──
        if etype == "message" and event.get("role") == "assistant":
            text = event.get("content", "")
            if text:
                return text
            return None

        # ── tool_use (tool call started) ──
        if etype == "tool_use":
            tool_name_raw = event.get("tool_name", "")
            tool_id = event.get("tool_id", str(uuid.uuid4()))
            params = event.get("parameters", {})

            # Map Gemini tool names to our standard names
            gemini_tool_map = {
                "run_shell_command": "Bash",
                "read_file": "Read",
                "write_file": "Write",
                "edit_file": "Edit",
                "list_directory": "Glob",
                "search_files": "Grep",
                "web_search": "WebSearch",
            }
            tool_name = gemini_tool_map.get(tool_name_raw, tool_name_raw)

            if isinstance(params, dict):
                args_str = json.dumps(params)
            else:
                args_str = str(params) if params else ""

            _max_args = self._tool_argument_limit(tool_name)
            await fire_event(self._event_callback, {
                "event": "tool_call_started",
                "agent_id": self.agent_id,
                "tool_id": tool_id,
                "tool_name": tool_name,
                "arguments": args_str[:_max_args],
            })
            self._log_conversation(
                "tool_call",
                tool_name=tool_name,
                tool_id=tool_id,
                tool_args=args_str[:2000],
            )
            logger.info(
                "Agent %s gemini tool_call id=%s name=%s",
                self.agent_id[:8],
                str(tool_id)[:12],
                tool_name,
            )
            return None  # Suppress from display

        # ── tool_result (tool call completed) ──
        if etype == "tool_result":
            tool_id = event.get("tool_id", "")
            status = event.get("status", "")
            is_error = status != "success"
            output = event.get("output", "")

            if tool_id:
                await fire_event(self._event_callback, {
                    "event": "tool_call_completed",
                    "agent_id": self.agent_id,
                    "tool_id": tool_id,
                    "result": str(output),
                    "is_error": is_error,
                })
                self._log_conversation(
                    "tool_result",
                    content=str(output)[:2000],
                    tool_id=tool_id,
                    is_error=is_error,
                )
            return None  # Suppress from display

        # ── result (final stats) ──
        if etype == "result":
            stats = event.get("stats", {})
            if stats:
                input_tokens = int(stats.get("input_tokens", 0) or 0)
                output_tokens = int(stats.get("output_tokens", 0) or 0)
                total_tokens = int(
                    stats.get("total_tokens", input_tokens + output_tokens) or 0
                )
                logger.info(
                    "Agent %s gemini completed: tokens=%d input=%d output=%d tool_calls=%d duration=%dms",
                    self.agent_id[:8],
                    total_tokens,
                    input_tokens,
                    output_tokens,
                    stats.get("tool_calls", 0),
                    stats.get("duration_ms", 0),
                )
                await self._emit_context_window_usage(
                    input_tokens=input_tokens,
                    cached_input_tokens=0,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            return None

        # ── init, user messages, other — suppress ──
        return None

    async def _detect_tool_from_text(
        self,
        text: str,
        active_tool_id: list[str | None],
    ) -> None:
        """Regex-based tool call detection for non-JSON provider output.

        Used as fallback for providers that don't support structured
        JSONL/stream-JSON output.
        """
        stripped = text.strip()

        # Check for tool call pattern (e.g. "Read(", "Bash(", etc.)
        match = self._TOOL_CALL_PATTERN.search(stripped)
        if match:
            tool_name = match.group(1)
            tool_id = str(uuid.uuid4())
            active_tool_id[0] = tool_id
            await fire_event(self._event_callback, {
                "event": "tool_call_started",
                "agent_id": self.agent_id,
                "tool_id": tool_id,
                "tool_name": tool_name,
                "arguments": stripped[:2000],
            })
            self._log_conversation(
                "tool_call",
                tool_name=tool_name,
                tool_id=tool_id,
                tool_args=stripped[:2000],
            )
            return

        # Check for tool result pattern (⎿ prefix in Codex text output)
        if self._TOOL_RESULT_PATTERN.match(stripped) and active_tool_id[0]:
            await fire_event(self._event_callback, {
                "event": "tool_call_completed",
                "agent_id": self.agent_id,
                "tool_id": active_tool_id[0],
                "result": stripped,
                "is_error": False,
            })
            self._log_conversation(
                "tool_result",
                content=stripped[:2000],
                tool_id=active_tool_id[0],
                is_error=False,
            )
            active_tool_id[0] = None

    # Orchestration MCP tools — always allowed, never prompt the user
    _ORCHESTRATION_TOOLS: frozenset[str] = frozenset({
        "ask_parent",
        "ask_user",
        "spawn_child",
        "spawn_children_parallel",
        "restart_child",
        "consult_expert",
        "report_progress",
        "task_complete",
        "wait_for_message",
        "respond_to_child",
        "consult_peer",
        "get_child_history",
        "check_child_status",
        "send_child_prompt",
        "get_children_status",
        "recommend_model",
        "run_bash",
    })
    # Claude/Codex control tools that may appear in plan workflows.
    # These should never depend on interactive permission streams in
    # headless orchestrated sessions.
    # Control tools that should be auto-allowed.  Note: AskUserQuestion
    # and request_user_input are NOT here — they're intercepted earlier
    # by _handle_user_question_tool and routed through PRSM's UI.
    _CONTROL_TOOLS: frozenset[str] = frozenset({
        "ExitPlanMode",
        "update_plan",
        "functions.ExitPlanMode",
        "functions.update_plan",
        "mcp__functions__ExitPlanMode",
        "mcp__functions__update_plan",
    })
    _TERMINAL_TOOLS: frozenset[str] = frozenset({
        "bash", "shell", "terminal",
    })
    # SDK built-in tools for asking the user.  These require an
    # interactive terminal that doesn't exist in headless mode.
    # We intercept them in can_use_tool and route through PRSM's UI.
    _USER_QUESTION_TOOLS: frozenset[str] = frozenset({
        "askuserquestion",
        "request_user_input",
    })

    @classmethod
    def _is_terminal_tool(cls, tool_name: str) -> bool:
        name = tool_name.lower()
        return (
            name in cls._TERMINAL_TOOLS
            or name.endswith(".bash")
            or name.endswith("__bash")
            or name.endswith("_bash")
        )

    @staticmethod
    def _is_dangerous_terminal_command(tool_input: dict) -> bool:
        """Heuristic risk detector for shell commands.

        Prompts only for potentially destructive/escalating operations.
        """
        if not isinstance(tool_input, dict):
            return False
        command = (
            tool_input.get("command")
            or tool_input.get("cmd")
            or tool_input.get("script")
            or ""
        )
        if not isinstance(command, str):
            return False
        c = command.strip().lower()
        if not c:
            return False

        danger_patterns = [
            r"\bsudo\b",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bchgrp\b",
            r"\bdd\s+if=",
            r"\bdd\s+of=",
            r"\bmkfs(\.| )",
            r"\bfdisk\b",
            r"\bparted\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
            r"\bhalt\b",
            r"\bkillall\b",
            r"\bkill\s+-9\b",
            r"\bgit\s+reset\s+--hard\b",
            r"\bgit\s+clean\s+-[^\n]*f[^\n]*d[^\n]*x\b",
            r"\bdocker\s+volume\s+(rm|prune)\b",
            r"\bdocker\s+system\s+prune\b",
            r"(curl|wget)[^\n|]*\|\s*(sh|bash)\b",
            r"/dev/sd[a-z]",
            r":\(\)\s*\{",
        ]
        return any(re.search(p, c) for p in danger_patterns)

    @staticmethod
    def _is_read_only_sql(statement: str) -> bool:
        cleaned = re.sub(r"/\*.*?\*/", " ", statement, flags=re.DOTALL).strip()
        while cleaned.startswith(";"):
            cleaned = cleaned[1:].lstrip()
        lower = cleaned.lower()
        if not lower:
            return True
        if re.search(
            r"\b(insert|update|delete|drop|alter|truncate|create|replace|merge|upsert|grant|revoke|vacuum|attach|detach)\b",
            lower,
        ):
            return False
        return lower.startswith(
            ("select", "show", "describe", "desc", "explain", "pragma", "with")
        )

    @classmethod
    def _is_dangerous_nonterminal_tool(cls, tool_name: str, tool_input: dict) -> bool:
        if not isinstance(tool_input, dict):
            return False

        name = tool_name.lower()
        delete_tokens = ("delete", "remove", "unlink", "rmdir", "destroy", "drop")
        file_tokens = ("file", "path", "dir", "directory", "folder", "fs", "filesystem")

        # File deletion-like operations should require confirmation.
        if any(token in name for token in delete_tokens) and (
            any(token in name for token in file_tokens)
            or any(
                any(token in key.lower() for token in file_tokens)
                for key in tool_input.keys()
                if isinstance(key, str)
            )
        ):
            return True

        is_db_tool = (
            bool(re.search(r"(?:^|[_\-.])db(?:$|[_\-.])", name))
            or any(
                token in name
                for token in (
                    "database",
                    "sql",
                    "postgres",
                    "mysql",
                    "sqlite",
                    "mongo",
                    "redis",
                    "supabase",
                    "prisma",
                )
            )
        )
        if not is_db_tool:
            return False

        # Explicit read-only hints should skip prompts.
        readonly = tool_input.get("readonly")
        read_only = tool_input.get("read_only")
        if readonly is True or read_only is True:
            return False

        # SQL payloads must be read-only.
        for key in ("query", "sql", "statement", "command"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return not cls._is_read_only_sql(value)

        # Non-SQL DB APIs: method/operation names indicate mutation.
        mutating_ops = (
            "insert",
            "update",
            "delete",
            "remove",
            "upsert",
            "replace",
            "create",
            "drop",
            "alter",
            "truncate",
            "write",
        )
        for key in ("operation", "action", "method", "type"):
            value = tool_input.get(key)
            if isinstance(value, str) and any(tok in value.lower() for tok in mutating_ops):
                return True
        return False

    async def _request_permission_for_tool(
        self,
        tool_name: str,
        bare_name: str,
        tool_input: dict,
        *,
        deny_project_message: str,
    ):
        from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

        if not self._permission_callback:
            return PermissionResultAllow()

        try:
            args_str = str(tool_input)[:500]
            result = await self._permission_callback(
                self.agent_id, tool_name, args_str,
            )
            logger.info(
                "Permission callback result agent=%s tool=%s result=%s",
                self.agent_id[:8],
                tool_name,
                result,
            )
            if result == "allow_always":
                self._always_allowed_tools.add(tool_name)
                self._always_allowed_tools.add(bare_name)
                return PermissionResultAllow()
            if result == "deny_project":
                return PermissionResultDeny(message=deny_project_message)
            if result == "allow":
                return PermissionResultAllow()
            return PermissionResultDeny(message="User denied tool call")
        except Exception:
            logger.warning(
                "Permission callback error for %s, allowing",
                tool_name,
            )
            return PermissionResultAllow()  # Fail open

    @staticmethod
    def _compile_policy_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            cleaned = str(pattern or "").strip()
            if not cleaned:
                continue
            try:
                compiled.append(re.compile(cleaned, re.IGNORECASE | re.MULTILINE))
            except re.error:
                logger.warning("Invalid command policy regex ignored: %s", cleaned)
        return compiled

    @staticmethod
    def _extract_terminal_command(tool_input: dict) -> str:
        if not isinstance(tool_input, dict):
            return ""
        command = (
            tool_input.get("command")
            or tool_input.get("cmd")
            or tool_input.get("script")
            or ""
        )
        if not isinstance(command, str):
            return ""
        return command.strip()

    def _extract_script_path(self, command: str) -> Path | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        candidate: str | None = None
        launcher = Path(tokens[0]).name.lower()
        if launcher in {"bash", "sh", "zsh"} and len(tokens) > 1:
            if not tokens[1].startswith("-"):
                candidate = tokens[1]
        elif launcher in {"python", "python3"} and len(tokens) > 1:
            if tokens[1].endswith((".py", ".sh", ".bash", ".zsh")):
                candidate = tokens[1]
        elif tokens[0].endswith((".sh", ".bash", ".zsh", ".py")):
            candidate = tokens[0]

        if not candidate:
            return None
        script_path = Path(candidate)
        if not script_path.is_absolute():
            script_path = self._workspace_root / script_path
        return script_path

    def _load_script_content(self, command: str) -> str:
        script_path = self._extract_script_path(command)
        if not script_path or not script_path.exists() or not script_path.is_file():
            return ""
        try:
            # Keep lightweight: scan up to 200KB.
            return script_path.read_text(encoding="utf-8", errors="ignore")[:200_000]
        except OSError:
            return ""

    def _matches_any(self, patterns: list[re.Pattern[str]], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _evaluate_terminal_command_policy(self, tool_input: dict) -> tuple[str, str]:
        command = self._extract_terminal_command(tool_input)
        if not command:
            return "allow", ""
        command_lower = command.lower()
        script_content = self._load_script_content(command)
        scan_text = command_lower
        if script_content:
            scan_text += f"\n{script_content.lower()}"

        # Explicit deny list always requires explicit user confirmation.
        if self._matches_any(self._blacklist_patterns, scan_text):
            return "prompt", command

        # Explicit allow list bypasses prompts for non-mandatory commands.
        if self._matches_any(self._whitelist_patterns, scan_text):
            return "allow", command

        if self._is_dangerous_terminal_command({"command": scan_text}):
            return "prompt", command

        # Optional model hook; currently disabled by default.
        if self._command_safety_model_enabled and self._command_safety_model:
            logger.debug(
                "Command safety model hook enabled model=%s (heuristic fallback used)",
                self._command_safety_model,
            )
        return "allow", command

    def _build_bash_permission_hooks(self) -> dict:
        """Build PreToolUse hooks that intercept dangerous bash commands.

        Hooks are evaluated by the Claude CLI BEFORE permission mode, so they
        work even in bypassPermissions mode.  This is the reliable mechanism
        for blocking dangerous commands like ``rm``, ``sudo``, etc.

        The hook checks the command against the blacklist/whitelist from
        CommandPolicyStore.  Dangerous commands trigger PRSM's permission
        popup via the permission_callback.  Safe commands are auto-allowed.
        """
        from claude_agent_sdk import HookMatcher

        agent_session = self  # capture for closure

        async def _bash_permission_hook(input_data, tool_use_id, context):
            """PreToolUse hook for Bash commands."""
            command = ""
            if isinstance(input_data, dict):
                tool_input = input_data.get("tool_input", {})
                if isinstance(tool_input, dict):
                    command = (
                        tool_input.get("command")
                        or tool_input.get("cmd")
                        or tool_input.get("script")
                        or ""
                    )
            else:
                tool_input = getattr(input_data, "tool_input", {})
                if isinstance(tool_input, dict):
                    command = (
                        tool_input.get("command")
                        or tool_input.get("cmd")
                        or tool_input.get("script")
                        or ""
                    )

            if not command:
                return {}  # Allow — no command to check

            logger.info(
                "HOOK_BASH_CHECK agent=%s cmd=%s",
                agent_session.agent_id[:8],
                command.strip()[:120],
            )

            # Check blacklist/whitelist
            policy = agent_session._policy_store
            is_blacklisted = any(
                p.search(command) for p in policy._blacklist_patterns
            )
            is_whitelisted = any(
                p.search(command) for p in policy._whitelist_patterns
            )

            if is_whitelisted:
                logger.info("HOOK_BASH_ALLOW (whitelisted) cmd=%s", command.strip()[:80])
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "Whitelisted command",
                    }
                }

            if is_blacklisted:
                logger.info("HOOK_BASH_BLOCKED (blacklisted) cmd=%s", command.strip()[:80])

                # Route through PRSM's permission UI
                if agent_session._permission_callback:
                    try:
                        result = await agent_session._permission_callback(
                            agent_session.agent_id,
                            "Bash",
                            {"command": command},
                        )
                        if result and result.get("allowed"):
                            logger.info("HOOK_BASH_USER_APPROVED cmd=%s", command.strip()[:80])
                            return {
                                "hookSpecificOutput": {
                                    "hookEventName": "PreToolUse",
                                    "permissionDecision": "allow",
                                    "permissionDecisionReason": "User approved",
                                }
                            }
                    except Exception as exc:
                        logger.warning("Permission callback error: %s", exc)

                # Denied — block the command
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Blocked: '{command.strip()[:80]}' matches a "
                            f"blacklisted pattern. Use the permission popup "
                            f"to approve if needed."
                        ),
                    }
                }

            # Heuristic danger check for commands not in blacklist/whitelist
            danger = agent_session._is_dangerous_terminal_command({"command": command})
            if danger:
                logger.info("HOOK_BASH_DANGER (heuristic) cmd=%s", command.strip()[:80])

                if agent_session._permission_callback:
                    try:
                        result = await agent_session._permission_callback(
                            agent_session.agent_id,
                            "Bash",
                            {"command": command},
                        )
                        if result and result.get("allowed"):
                            logger.info("HOOK_BASH_USER_APPROVED cmd=%s", command.strip()[:80])
                            return {
                                "hookSpecificOutput": {
                                    "hookEventName": "PreToolUse",
                                    "permissionDecision": "allow",
                                    "permissionDecisionReason": "User approved",
                                }
                            }
                    except Exception as exc:
                        logger.warning("Permission callback error: %s", exc)

                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Blocked: '{command.strip()[:80]}' looks dangerous. "
                            f"Use the permission popup to approve if needed."
                        ),
                    }
                }

            # Safe command — auto-allow
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "Safe command",
                }
            }

        return {
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[_bash_permission_hook]),
            ],
        }

    async def _check_permission(
        self, tool_name: str, tool_input: dict, context: object = None,
    ):
        """Check with the TUI whether a tool call is allowed.

        Conforms to claude_agent_sdk can_use_tool callback signature:
        (tool_name, tool_input, context) -> PermissionResultAllow | PermissionResultDeny

        Always checks the BashRepeatGuard first (applies to ALL agents
        including bypass mode).  Then orchestration/control tools are
        auto-allowed.  "Allow Always" decisions are shared across all
        agents via the mutable set passed from AgentManager.
        """
        from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

        # Strip MCP server prefix (e.g. "mcp__orchestrator__task_complete" → "task_complete")
        bare_name = tool_name
        if tool_name.startswith("mcp__") and tool_name.count("__") >= 2:
            bare_name = tool_name.split("__", 2)[2]
        logger.info(
            "PERM_CHECK agent=%s tool=%s bare=%s mode=%s",
            self.agent_id[:8],
            tool_name,
            bare_name,
            self._descriptor.permission_mode.value,
        )

        # ── Bash repeat guard (applies to ALL agents, even bypass) ──
        if self._is_terminal_tool(bare_name) and isinstance(tool_input, dict):
            command = (
                tool_input.get("command")
                or tool_input.get("cmd")
                or tool_input.get("script")
                or ""
            )
            if isinstance(command, str) and command.strip():
                allowed, count = self._bash_guard.check(command)
                if not allowed:
                    logger.warning(
                        "BashRepeatGuard DENIED agent=%s repeat=%d cmd=%s",
                        self.agent_id[:8],
                        count,
                        command.strip()[:80],
                    )
                    return PermissionResultDeny(
                        message=(
                            f"BLOCKED: You have run the exact same bash command "
                            f"{count} times in a row and it keeps failing. "
                            f"Stop repeating it. Try a DIFFERENT approach — "
                            f"change the command, fix the underlying issue, "
                            f"or report the problem via task_complete."
                        ),
                    )

        # ── AskUserQuestion interception (ALL agents, even bypass) ──
        # The SDK's built-in AskUserQuestion needs a terminal which
        # doesn't exist in headless/orchestrated mode.  Intercept the
        # call, route through PRSM's UI, and return the user's answer
        # via the deny message so the agent receives the response.
        if bare_name.lower() in self._USER_QUESTION_TOOLS:
            return await self._handle_user_question_tool(
                tool_name, tool_input,
            )

        # Dangerous non-terminal operations still require explicit consent,
        # even for bypass agents (for example destructive file/db actions).
        if (
            not self._is_terminal_tool(bare_name)
            and self._is_dangerous_nonterminal_tool(bare_name, tool_input)
        ):
            logger.info(
                "Permission dangerous non-terminal eval agent=%s tool=%s",
                self.agent_id[:8],
                tool_name,
            )
            return await self._request_permission_for_tool(
                tool_name,
                bare_name,
                tool_input,
                deny_project_message="User permanently denied dangerous operation",
            )

        # ── Bypass mode: auto-allow everything else ──
        is_bypass = (
            self._descriptor.permission_mode.value == "bypassPermissions"
        )
        if is_bypass:
            return PermissionResultAllow()

        # ── Normal permission logic (non-bypass agents) ──

        # Orchestration tools never need user permission
        if bare_name in self._ORCHESTRATION_TOOLS:
            logger.debug("Permission auto-allow orchestration tool=%s", tool_name)
            return PermissionResultAllow()
        if tool_name.startswith("mcp__orchestrator__"):
            logger.debug("Permission auto-allow orchestrator namespace tool=%s", tool_name)
            return PermissionResultAllow()
        normalized = tool_name.lower()
        bare_normalized = bare_name.lower()
        if (
            tool_name in self._CONTROL_TOOLS
            or bare_name in self._CONTROL_TOOLS
            or "exitplanmode" in normalized
            or "exitplanmode" in bare_normalized
            or normalized.endswith(".update_plan")
        ):
            logger.debug("Permission auto-allow control tool=%s", tool_name)
            return PermissionResultAllow()

        if tool_name in self._always_allowed_tools or bare_name in self._always_allowed_tools:
            logger.debug("Permission allow_always hit tool=%s", tool_name)
            return PermissionResultAllow()

        # Only prompt for potentially dangerous terminal commands.
        # All other tools are auto-allowed.
        if not self._is_terminal_tool(bare_name):
            logger.debug("Permission auto-allow non-terminal tool=%s", tool_name)
            return PermissionResultAllow()
        decision, cmd = self._evaluate_terminal_command_policy(tool_input)
        logger.info(
            "Permission terminal eval agent=%s tool=%s decision=%s cmd=%.80s",
            self.agent_id[:8], tool_name, decision, cmd,
        )
        if decision != "prompt":
            logger.debug("Permission auto-allow safe terminal tool=%s", tool_name)
            return PermissionResultAllow()

        return await self._request_permission_for_tool(
            tool_name,
            bare_name,
            tool_input,
            deny_project_message="User permanently denied command",
        )

    async def _handle_user_question_tool(
        self,
        tool_name: str,
        tool_input: dict,
    ):
        """Intercept SDK's AskUserQuestion / request_user_input and
        route through PRSM's user-question callback.

        The SDK's built-in question tools need an interactive terminal
        that doesn't exist in headless orchestrated mode.  We extract
        the question, show it in the TUI / VSCode UI, wait for the
        user's answer, and return it via PermissionResultDeny.message
        so the agent receives the response.

        Returns PermissionResultDeny with the user's answer embedded.
        """
        from claude_agent_sdk.types import PermissionResultDeny

        question, options = self._extract_user_question(tool_input)

        if not question:
            logger.warning(
                "AskUserQuestion intercepted but no question found "
                "agent=%s input=%s",
                self.agent_id[:8],
                str(tool_input)[:200],
            )
            return PermissionResultDeny(
                message=(
                    "Could not extract question from tool input. "
                    "Use the ask_user() orchestration tool instead."
                ),
            )

        if not self._user_question_callback:
            logger.warning(
                "AskUserQuestion intercepted but no callback agent=%s",
                self.agent_id[:8],
            )
            return PermissionResultDeny(
                message=(
                    "User question handler not configured. "
                    "Use the ask_user() orchestration tool instead."
                ),
            )

        logger.info(
            "AskUserQuestion intercepted agent=%s question=%s options=%d",
            self.agent_id[:8],
            question[:80],
            len(options),
        )

        try:
            timeouts = [
                timeout
                for timeout in (
                    self._tool_call_timeout,
                    self._user_question_timeout,
                )
                if timeout > 0
            ]
            timeout = min(timeouts) if timeouts else None
            if timeout is None:
                answer = await self._user_question_callback(
                    self.agent_id, question, options,
                )
            else:
                answer = await asyncio.wait_for(
                    self._user_question_callback(
                        self.agent_id, question, options,
                    ),
                    timeout=timeout,
                )
            logger.info(
                "AskUserQuestion answered agent=%s answer=%s",
                self.agent_id[:8],
                str(answer)[:80],
            )
            return PermissionResultDeny(
                message=f"User responded: {answer}",
            )
        except asyncio.TimeoutError:
            return PermissionResultDeny(
                message="User did not respond within the timeout.",
            )
        except Exception as exc:
            logger.exception(
                "AskUserQuestion callback failed agent=%s",
                self.agent_id[:8],
            )
            return PermissionResultDeny(
                message=f"Failed to ask user: {exc}",
            )

    @staticmethod
    def _extract_user_question(
        tool_input: dict,
    ) -> tuple[str, list[dict]]:
        """Extract question text and options from AskUserQuestion input.

        Handles both the SDK's format::

            {questions: [{question, header, options: [{label, description}]}]}

        and simpler formats::

            {question: "...", options: [...]}
        """
        if not isinstance(tool_input, dict):
            return "", []

        # SDK format: {questions: [{question, options}]}
        questions = tool_input.get("questions")
        if isinstance(questions, list) and questions:
            first = questions[0] if isinstance(questions[0], dict) else {}
            question_text = str(first.get("question", "")).strip()
            options = first.get("options", [])
            if isinstance(options, list):
                return question_text, options
            return question_text, []

        # Simple format: {question: "...", options: [...]}
        question_text = str(tool_input.get("question", "")).strip()
        options = tool_input.get("options", [])
        if not isinstance(options, list):
            options = []
        return question_text, options

    def _log_conversation(
        self,
        entry_type: str,
        content: str = "",
        tool_name: str | None = None,
        tool_id: str | None = None,
        tool_args: str | None = None,
        is_error: bool = False,
    ) -> None:
        """Append an entry to the conversation store if available."""
        if self._conversation_store is None:
            return
        from .conversation_store import ConversationEntry, EntryType
        type_map = {
            "text": EntryType.TEXT,
            "thinking": EntryType.THINKING,
            "tool_call": EntryType.TOOL_CALL,
            "tool_result": EntryType.TOOL_RESULT,
            "user_message": EntryType.USER_MESSAGE,
        }
        et = type_map.get(entry_type)
        if et is None:
            return
        self._conversation_store.append(
            self.agent_id,
            ConversationEntry(
                entry_type=et,
                content=content,
                tool_name=tool_name,
                tool_id=tool_id,
                tool_args=tool_args,
                is_error=is_error,
            ),
        )

    def _build_orchestration_instructions(self) -> str:
        """Build system prompt instructions for orchestration tools.

        Master agents get full instructions including spawn/restart tools.
        Worker/expert agents get a focused set: ask_parent, ask_user,
        consult_expert, report_progress, task_complete.
        """
        from .models import AgentRole

        available_experts = self._expert_registry.list_ids()
        expert_list = (
            ", ".join(available_experts) if available_experts
            else "none registered"
        )

        is_master = self._descriptor.role == AgentRole.MASTER

        # Build available plugins section
        plugin_list = "none"
        if self._plugin_mcp_servers:
            plugin_list = ", ".join(self._plugin_mcp_servers.keys())

        # Common tools available to all agents
        parts = [
            "\n\n## Orchestration Tools Available\n\n"
            "You have access to inter-agent communication tools:\n\n"
            "- **ask_parent(question)**: Ask your parent agent a "
            "question and wait for an answer. Use when you need "
            "clarification, are unsure about requirements, or face "
            "an ambiguous decision. ALWAYS prefer asking over guessing.\n"
            "- **ask_user(question, options)**: Ask the USER directly "
            "with clickable options. Use when you need the user to "
            "make a decision or choose between approaches. Options "
            "are rendered as clickable buttons in the UI. Each option "
            "should have a 'label' and 'description'.\n"
            "- **wait_for_message(timeout_seconds)**: Wait for the next "
            "incoming routed message. For masters, this receives child "
            "questions/results. For workers, this can receive new parent "
            "instructions (for example from send_child_prompt). Default "
            "timeout is 0 (disabled — blocks until a message arrives).\n"
        ]

        # Master-only spawn/restart/interactive tools
        if is_master:
            parts.append(
                "- **spawn_child(prompt, wait, tools, model, cwd, "
                "mcp_servers, exclude_plugins, complexity)**: "
                "Spawn a child agent for any focused task — writing code, "
                "exploring a codebase, analyzing an implementation, "
                "researching a question, or reviewing changes. "
                "Use the **complexity** parameter for automatic model "
                "selection: 'trivial' (cheapest model), 'simple' (fast), "
                "'medium' (default), 'complex' (strong), 'frontier' (best). "
                "This allows you to use cheaper/faster models for simple "
                "tasks like file search, renaming, or exploration, and "
                "reserve expensive frontier models for complex reasoning. "
                "Use mcp_servers={name: config} to give the child specific "
                "MCP plugins (e.g., filesystem, database tools). "
                "Use exclude_plugins=[names] to remove global plugins. "
                "By default, plugins are auto-matched based on the "
                "child's prompt and role. "
                "Child launches are always non-blocking; if wait=true is "
                "passed it is ignored. Use wait_for_message + "
                "respond_to_child.\n"
                "- **spawn_children_parallel(children)**: Spawn multiple "
                "children simultaneously. Each child is a dict with: "
                "prompt (required), tools, model, cwd, mcp_servers, "
                "exclude_plugins, complexity.\n"
                "- **restart_child(child_agent_id, prompt, wait)**: Restart "
                "a completed or failed child agent with a new task. Reuses "
                "the child's identity, model, tools, and working directory. "
                "Use this instead of spawn_child when you want to send "
                "follow-up work to the same agent (e.g. continue a previous "
                "task, fix something it did, or assign a related task that "
                "benefits from the same configuration).\n"
                "- **recommend_model(task_description, complexity)**: Get a "
                "model recommendation from the capability registry. Returns "
                "the best model for the given task type and complexity.\n"
            )

        parts.append(
            "- **consult_expert(expert_id, question)**: Consult a "
            "specialist. Available experts: " + expert_list + "\n"
            "- **report_progress(status, percent_complete)**: Send "
            "non-blocking progress update to parent.\n"
            "- **task_complete(summary, artifacts)**: Signal "
            "completion. ALWAYS call this when your task is done.\n"
        )

        if is_master:
            parts.append(
                "- **respond_to_child(child_agent_id, correlation_id, "
                "response)**: Answer a child's ask_parent question.\n"
                "- **get_child_history(child_agent_id, detail_level)**: "
                "Review a child's conversation history. detail_level "
                "can be 'full' (everything including thinking and tool "
                "args) or 'summary' (text + tool names only).\n"
                "- **check_child_status(child_agent_id)**: Check a "
                "child's current state, error, timestamps, and "
                "children count.\n"
                "- **send_child_prompt(child_agent_id, prompt)**: "
                "Send a prompt/instruction to a child agent.\n"
            )

        parts.append(
            "\n### CRITICAL — Ask, Don't Guess\n"
            "Make as few assumptions or interpretations as possible.\n"
            "If anything is unclear, ambiguous, or unspecified, ask a "
            "clarifying question immediately before proceeding.\n"
            "If your task prompt is ambiguous or you encounter a "
            "decision that could go multiple ways, ALWAYS use "
            "ask_parent() to get clarification. Never assume what "
            "the user or your parent wants. Specific situations "
            "where you MUST ask:\n"
            "- The task could be interpreted in multiple valid ways\n"
            "- You need to choose between different implementation "
            "approaches\n"
            "- You are unsure about naming, placement, or behavior\n"
            "- Requirements conflict with existing code patterns\n"
            "- Something seems wrong but you are not sure\n\n"
            "The user is the decision-maker. Your job is to execute "
            "their decisions accurately, not to make decisions for "
            "them. When in doubt, ask.\n"
        )

        if not is_master:
            parts.append(
                "\n### Worker Agent Role\n"
                "You are a WORKER agent. Your job is to complete your "
                "assigned task directly using your available tools "
                "(Read, Write, Edit, Bash, Glob, Grep, etc.). "
                "Do NOT attempt to spawn child agents or delegate — "
                "only the orchestrator can do that. If your task is "
                "too large or needs to be broken into sub-tasks, use "
                "task_complete to report back to your parent with a "
                "recommendation for how to split the work.\n"
                "If your parent may send follow-up instructions while "
                "you work, call wait_for_message() to check for updates "
                "(or pass timeout_seconds=0 to wait indefinitely) and incorporate any "
                "USER_PROMPT messages.\n"
            )

        if is_master:
            parts.append(
                "\n### Available Plugins\n"
                f"Loaded plugins: {plugin_list}\n"
                "Plugins are auto-matched to child agents based on their "
                "prompt and role. Workers get all plugins. Experts get "
                "plugins whose tags match keywords in their prompt. "
                "You can override with explicit mcp_servers or "
                "exclude_plugins in spawn_child.\n\n"
                "### What Counts as a Task\n"
                "Tasks are NOT limited to writing code. Valuable tasks "
                "include:\n"
                "- Deeply exploring and describing an existing "
                "implementation so your parent doesn't have to\n"
                "- Researching how a subsystem works and summarizing it\n"
                "- Analyzing code for patterns, bugs, or architectural "
                "issues\n"
                "- Reviewing changes for correctness and style\n"
                "- Gathering information from multiple files into a "
                "coherent summary\n\n"
                "### Smart Delegation with Model Selection\n"
                "You have access to a model capability registry that "
                "allows child agents to run on different models than you. "
                "This makes delegation much more cost-effective:\n\n"
                "- **trivial tasks** (file search, listing, renaming): "
                "Use complexity='trivial' — runs on the cheapest model\n"
                "- **simple tasks** (exploration, grep, reading files): "
                "Use complexity='simple' — runs on a fast model\n"
                "- **medium tasks** (coding, editing, documentation): "
                "Use complexity='medium' (default)\n"
                "- **complex tasks** (architecture, debugging, analysis): "
                "Use complexity='complex' — runs on a strong model\n"
                "- **frontier tasks** (critical decisions, complex reasoning): "
                "Use complexity='frontier' — runs on the best available\n\n"
                "Because simpler tasks use cheaper/faster models, you "
                "should delegate more freely than you would otherwise. "
                "Spawning a child with complexity='simple' is very cheap "
                "and fast. Use delegation whenever it improves:\n"
                "- Parallelism (multiple independent tasks at once)\n"
                "- Context isolation (keeping your context clean)\n"
                "- Specialist knowledge (expert profiles)\n"
                "- Cost efficiency (simple model for simple tasks)\n\n"
                "### Interactive Child Pattern (avoids deadlocks)\n"
                "When spawning children that may ask questions:\n"
                "1. spawn_child(prompt, wait=false) → get child_id\n"
                "2. wait_for_message() → receive questions/results\n"
                "3. respond_to_child(id, corr_id, answer) → answer\n"
                "4. Repeat until TASK_RESULT received\n\n"
                "IMPORTANT: Child launches are always non-blocking "
                "(wait=true is ignored).\n\n"
                "### Do NOT Abandon Progressing Children\n"
                "If you check on children (via get_children_status() or "
                "check_child_status()) and see they are still running, "
                "waiting, or starting — they are making progress. Do NOT "
                "give up on them, do NOT start doing their work yourself, "
                "and do NOT kill them unless the user explicitly asks. "
                "Always return to wait_for_message() to collect their "
                "eventual results. Children may take several minutes for "
                "non-trivial tasks.\n"
            )

        parts.append(
            "\n### Shell Command Execution\n"
            "- **run_bash(command, timeout, cwd)**: Execute a bash command "
            "with permission checking. Dangerous commands (rm, sudo, etc.) "
            "require explicit user approval before execution. You MUST use "
            "this tool for ALL shell/bash commands instead of any native "
            "Bash tool. This ensures the user can review and approve "
            "potentially dangerous operations.\n\n"
            "### Shell Command Notes\n"
            "Do NOT prefix commands with `builtin`. The `builtin` "
            "keyword only works for actual shell built-in commands "
            "(cd, echo, type, etc.) — it will FAIL for external "
            "programs like npm, npx, python, pip, node, git, etc. "
            "Run external commands directly without any prefix.\n"
            "If `cd` fails due to shell aliases, use the full path "
            "to the target directory instead (e.g., "
            "npm run build --prefix /path/to/project).\n"
            "Do NOT redirect shell output into workspace scratch files "
            "(for example `.tmp*`, `.*.txt`, `.file_index.txt`, "
            "`.grep_agent_manager.txt`). Prefer streamed stdout/stderr.\n"
            "If a temporary output file is absolutely necessary, write it "
            "under `$PRSM_SCRATCH_DIR` (tmp-only), never in the workspace.\n"
            "If a command fails, do NOT retry the exact same command. "
            "Analyze the error and try a different approach.\n"
        )

        parts.append(
            "\n### Asking the User\n"
            "You can ask the user questions using either:\n"
            "- **AskUserQuestion** (the standard tool) — works normally, "
            "PRSM routes it through its UI automatically.\n"
            "- **ask_user()** (orchestration tool) — supports structured "
            "clickable options.\n"
            "Both work in orchestrated mode. Use whichever is natural.\n"
        )

        parts.append(
            "\n### Tool Availability Notes\n"
            "Do NOT call host-runtime control tools such as "
            "ExitPlanMode or EnterPlanMode unless "
            "they are explicitly listed in your available tools for "
            "this run. If a mode transition is needed, continue the "
            "task and report completion via task_complete instead.\n"
        )

        parts.append(
            "\n### Documentation Workflow\n"
            "Before exploring source files, read relevant architecture docs in "
            "@docs/ first (especially @docs/architecture.md, @docs/engine.md, "
            "@docs/data-flow.md, @docs/adapters.md, @docs/shared.md, "
            "@docs/configuration.md, @docs/tui.md, @docs/vscode.md). "
            "Use docs context to guide targeted exploration and avoid broad scans.\n"
            "When launching child agents, explicitly instruct them to review "
            "relevant @docs/ architecture files before deep code exploration.\n"
        )

        return "".join(parts)
