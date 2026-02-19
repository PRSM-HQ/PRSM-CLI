"""Agent manager: creates, tracks, and kills agent sessions.

Central registry for all active agents. Enforces depth limits,
concurrency limits, and provides cascading failure propagation.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import (
    AgentDescriptor,
    AgentResult,
    AgentRole,
    AgentState,
    MessageType,
    PermissionMode,
    RoutedMessage,
    SpawnRequest,
)
from .agent_session import AgentSession
from .config import fire_event
from .mcp_server.tools import strip_injected_prompt_prefix
from .errors import (
    AgentSpawnError,
    MaxDepthExceededError,
    MessageRoutingError,
    ProviderNotAvailableError,
)

if TYPE_CHECKING:
    from .message_router import MessageRouter
    from .expert_registry import ExpertRegistry
    from .conversation_store import ConversationStore
    from .config import EngineConfig, EventCallback, PermissionCallback, UserQuestionCallback
    from .providers.registry import ProviderRegistry
    from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


def _fallback_agent_name(role: AgentRole, prompt: str) -> str:
    """Deterministic child-agent name when model naming is unavailable."""
    clean_prompt = strip_injected_prompt_prefix(prompt or "")
    if role == AgentRole.MASTER:
        return "Orchestrator"
    if role == AgentRole.EXPERT:
        name = clean_prompt[:40].strip() if clean_prompt else "Expert"
        if clean_prompt and len(clean_prompt) > 40:
            name += "..."
        return name
    if role == AgentRole.REVIEWER:
        name = clean_prompt[:50].strip() if clean_prompt else "Reviewer"
        if clean_prompt and len(clean_prompt) > 50:
            name += "..."
        return name

    # Worker
    name = clean_prompt[:50].strip()
    if len(clean_prompt) > 50:
        name += "..."
    return name or "Worker"


class AgentManager:
    """Creates, tracks, and manages agent session lifecycles.

    Enforces:
    - Maximum nesting depth (default 5)
    - Maximum concurrent agents (default 5)
    - Clean teardown when parent completes/fails
    """

    def __init__(
        self,
        router: MessageRouter,
        expert_registry: ExpertRegistry,
        config: EngineConfig,
        provider_registry: ProviderRegistry | None = None,
        event_callback: EventCallback | None = None,
        permission_callback: PermissionCallback | None = None,
        user_question_callback: UserQuestionCallback | None = None,
        plugin_mcp_servers: dict | None = None,
        plugin_manager: object | None = None,
        initial_allowed_tools: set[str] | None = None,
        conversation_store: ConversationStore | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        self._router = router
        self._expert_registry = expert_registry
        self._config = config
        self._provider_registry = provider_registry
        self._event_callback = event_callback
        self._permission_callback = permission_callback
        self._user_question_callback = user_question_callback
        self._plugin_mcp_servers = plugin_mcp_servers or {}
        self._conversation_store = conversation_store
        self._plugin_manager = plugin_manager
        self._resource_manager = resource_manager
        self._agents: dict[str, AgentDescriptor] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._result_futures: dict[str, asyncio.Future[AgentResult]] = {}
        # Completed/failed agents kept for restart. Maps agent_id → descriptor.
        self._completed_agents: dict[str, AgentDescriptor] = {}
        # Shared across all agents — "Allow Always" propagates globally
        # Seeded with persisted permissions from PermissionStore
        self._always_allowed_tools: set[str] = set(initial_allowed_tools or ())
        # Guard flag: prevents new agent spawns during shutdown/interrupt
        self._shutting_down: bool = False

    async def _derive_agent_name(self, role: AgentRole, prompt: str) -> str:
        """Generate a concise display name for non-master agents."""
        fallback = _fallback_agent_name(role, prompt)
        if role == AgentRole.MASTER:
            return fallback
        task_prompt = strip_injected_prompt_prefix(prompt or "").strip()
        if not task_prompt:
            return fallback
        try:
            from prsm.shared.services.session_naming import generate_session_name

            generated = await generate_session_name(task_prompt, model_id="gpt-5-3-spark")
            if generated and re.search(r"[A-Za-z]", generated):
                return generated.strip()
        except Exception:
            logger.debug("Agent naming failed; using fallback", exc_info=True)
        return fallback

    def get_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        """Get an agent's descriptor by ID."""
        return self._agents.get(agent_id)

    async def transition_agent_state(
        self, agent_id: str, new_state: AgentState,
    ) -> bool:
        """Transition an agent's state and emit the event.

        Used by OrchestrationTools to set WAITING_FOR_* states when
        agents block on parent/child/expert operations. Returns True
        if the transition succeeded, False if the agent was not found
        or the transition was invalid.
        """
        from .lifecycle import validate_transition

        descriptor = self._agents.get(agent_id)
        if descriptor is None:
            return False

        try:
            validate_transition(descriptor.state, new_state)
        except ValueError:
            logger.debug(
                "Skipping state transition for agent %s: %s -> %s (invalid)",
                agent_id[:8], descriptor.state.value, new_state.value,
            )
            return False

        old_state = descriptor.state
        descriptor.state = new_state
        logger.info(
            "Agent %s: %s -> %s (via manager)",
            agent_id[:8], old_state.value, new_state.value,
        )
        await fire_event(self._event_callback, {
            "event": "agent_state_changed",
            "agent_id": agent_id,
            "old_state": old_state.value,
            "new_state": new_state.value,
        })
        return True

    @property
    def active_count(self) -> int:
        """Number of agents not in a terminal state."""
        terminal = {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.KILLED,
        }
        return sum(
            1 for d in self._agents.values()
            if d.state not in terminal
        )

    async def spawn_agent(
        self,
        request: SpawnRequest,
    ) -> AgentDescriptor:
        """Spawn a new agent session.

        Creates the descriptor, registers with the router,
        starts the session as a background asyncio.Task.

        Raises:
            MaxDepthExceededError: If nesting limit exceeded.
            AgentSpawnError: If concurrency limit reached or engine is shutting down.
        """
        # Reject spawns during shutdown/interrupt to prevent race conditions
        if self._shutting_down:
            raise AgentSpawnError(
                "new",
                "Engine is shutting down — cannot spawn new agents",
            )

        parent = (
            self._agents.get(request.parent_id)
            if request.parent_id
            else None
        )
        depth = (parent.depth + 1) if parent else 0

        if depth > self._config.max_agent_depth:
            raise MaxDepthExceededError(
                "new", depth, self._config.max_agent_depth
            )

        if self.active_count >= self._config.max_concurrent_agents:
            raise AgentSpawnError(
                "new",
                f"Concurrency limit: "
                f"{self.active_count}/{self._config.max_concurrent_agents}",
            )

        project_id = self._config.project_id or "default"
        if self._resource_manager is not None:
            if request.parent_id:
                allowed, reason = self._resource_manager.check_circuit_breaker(
                    request.parent_id
                )
                if not allowed:
                    raise AgentSpawnError(
                        "new",
                        f"Circuit breaker open for parent {request.parent_id[:8]}: {reason}",
                    )
            allowed, reason = self._resource_manager.check_budget(
                project_id,
                "spawn_agent",
            )
            if not allowed:
                raise AgentSpawnError("new", f"Resource budget exceeded: {reason}")

        # Validate that the requested provider is available
        if request.provider and self._provider_registry:
            provider_obj = self._provider_registry.get(request.provider)
            if provider_obj is None:
                available = self._provider_registry.list_names()
                raise ProviderNotAvailableError(
                    request.provider,
                    available,
                )
            if not provider_obj.is_available():
                available = self._provider_registry.list_available()
                raise ProviderNotAvailableError(
                    request.provider,
                    available,
                )

        descriptor = AgentDescriptor(
            parent_id=request.parent_id,
            role=request.role,
            prompt=request.prompt,
            tools=request.tools,
            model=request.model,
            permission_mode=request.permission_mode,
            cwd=request.cwd or self._config.default_cwd,
            mcp_servers=request.mcp_servers,
            exclude_plugins=request.exclude_plugins,
            depth=depth,
            max_depth=self._config.max_agent_depth,
            provider=request.provider,
        )

        if parent:
            parent.children.append(descriptor.agent_id)

        self._agents[descriptor.agent_id] = descriptor
        self._router.register_agent(descriptor)
        if self._resource_manager is not None:
            self._resource_manager.record_usage(project_id, agent_spawn=True)

        # Create result future
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AgentResult] = loop.create_future()
        self._result_futures[descriptor.agent_id] = future

        # Resolve provider from registry if available
        provider = None
        if self._provider_registry and request.provider:
            provider = self._provider_registry.get(request.provider)

        # Resolve effective plugin set for this agent
        effective_plugins = self._resolve_agent_plugins(request)

        # Start session as background task
        session = AgentSession(
            descriptor=descriptor,
            manager=self,
            router=self._router,
            expert_registry=self._expert_registry,
            agent_timeout_seconds=self._config.agent_timeout_seconds,
            tool_call_timeout_seconds=(
                self._config.tool_call_timeout_seconds
            ),
            user_question_timeout_seconds=(
                self._config.user_question_timeout_seconds
            ),
            provider=provider,
            event_callback=self._event_callback,
            permission_callback=self._permission_callback,
            user_question_callback=self._user_question_callback,
            plugin_mcp_servers=effective_plugins,
            always_allowed_tools=self._always_allowed_tools,
            conversation_store=self._conversation_store,
            child_default_model=self._config.default_model,
            child_default_provider=self._config.default_provider,
            model_registry=self._config.model_registry,
            peer_models=self._config.peer_models,
            command_whitelist=self._config.command_whitelist,
            command_blacklist=self._config.command_blacklist,
            command_safety_model_enabled=self._config.command_safety_model_enabled,
            command_safety_model=self._config.command_safety_model,
        )
        self._sessions[descriptor.agent_id] = session

        task = asyncio.create_task(
            self._run_session(descriptor.agent_id, session, future),
            name=f"agent-{descriptor.agent_id[:8]}",
        )
        self._tasks[descriptor.agent_id] = task

        logger.info(
            "Agent spawned: %s (parent=%s, role=%s, depth=%d)",
            descriptor.agent_id[:8],
            (request.parent_id or "none")[:8],
            descriptor.role.value,
            depth,
        )

        await fire_event(self._event_callback, {
            "event": "agent_spawned",
            "agent_id": descriptor.agent_id,
            "parent_id": descriptor.parent_id,
            "role": descriptor.role.value,
            "model": descriptor.model,
            "depth": depth,
            "prompt": strip_injected_prompt_prefix(descriptor.prompt)[:200],
            "name": await self._derive_agent_name(descriptor.role, descriptor.prompt),
        })

        return descriptor

    def _resolve_agent_plugins(self, request: SpawnRequest) -> dict:
        """Resolve effective MCP server plugins for an agent.

        Resolution order:
        1. If request.mcp_servers is explicitly set, merge on top of global.
        2. If plugin_manager available, auto-match by prompt/role.
        3. Otherwise, use all global plugins.
        4. Apply exclude_plugins removals last.
        """
        if request.mcp_servers is not None:
            # Explicit per-agent: start from global, merge explicit on top
            effective = dict(self._plugin_mcp_servers)
            effective.update(request.mcp_servers)
        elif self._plugin_manager:
            # Auto-match: filter plugins by prompt/role relevance
            role_str = (
                request.role.value
                if hasattr(request.role, "value")
                else str(request.role)
            )
            effective = self._plugin_manager.get_plugins_for_agent(
                prompt=request.prompt, role=role_str,
            )
        else:
            # Fallback: all global plugins
            effective = dict(self._plugin_mcp_servers)

        # Apply exclusions
        if request.exclude_plugins:
            for name in request.exclude_plugins:
                effective.pop(name, None)

        return effective

    async def _run_session(
        self,
        agent_id: str,
        session: AgentSession,
        future: asyncio.Future[AgentResult],
    ) -> None:
        """Run an agent session and resolve its result future.

        After the session completes, if the agent has a parent and
        didn't explicitly call task_complete (which sends a TASK_RESULT
        message via the router), we send one automatically. This
        ensures that parents using wait_for_message always get notified
        when a child finishes, regardless of whether the child called
        task_complete.
        """
        result: AgentResult | None = None
        try:
            result = await session.run()
            if not future.done():
                future.set_result(result)
        except asyncio.CancelledError:
            # Task was cancelled (kill_agent or interrupt).
            # Resolve the future so parents don't hang.
            result = AgentResult(
                agent_id=agent_id,
                success=False,
                summary="",
                error="Agent task was cancelled",
            )
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            result = AgentResult(
                agent_id=agent_id,
                success=False,
                summary="",
                error=str(exc),
            )
            if not future.done():
                future.set_result(result)
        finally:
            if self._resource_manager is not None:
                if result and result.success:
                    self._resource_manager.record_success(agent_id)
                else:
                    self._resource_manager.record_failure(agent_id)
                self._resource_manager.record_agent_completed(
                    self._config.project_id or "default"
                )
            # Send a TASK_RESULT to the parent if the child didn't
            # already do so via task_complete.  The descriptor's
            # result_summary is set by task_complete; if it's empty
            # the child completed without calling it.
            descriptor = self._agents.get(agent_id)
            if descriptor and descriptor.parent_id and not descriptor.result_summary:
                try:
                    summary = (
                        result.summary if result and result.summary
                        else result.error if result and result.error
                        else "Child agent completed without explicit task_complete."
                    )
                    msg = RoutedMessage(
                        message_type=MessageType.TASK_RESULT,
                        source_agent_id=agent_id,
                        target_agent_id=descriptor.parent_id,
                        payload={
                            "summary": summary,
                            "artifacts": {},
                        },
                    )
                    await self._router.send(msg)
                    logger.info(
                        "Auto-sent TASK_RESULT for agent %s to parent %s",
                        agent_id[:8],
                        descriptor.parent_id[:8],
                    )
                except MessageRoutingError:
                    logger.debug(
                        "Could not auto-send TASK_RESULT for agent %s "
                        "(parent %s may be unregistered)",
                        agent_id[:8],
                        (descriptor.parent_id or "none")[:8],
                    )
                except Exception:
                    logger.debug(
                        "Auto-send TASK_RESULT failed for agent %s",
                        agent_id[:8],
                        exc_info=True,
                    )

            # Only cleanup if we're still the active task for this agent.
            # If the agent was killed and restarted, a new task owns the
            # agent_id — cleaning up here would destroy the restarted agent.
            active_task = self._tasks.get(agent_id)
            if active_task is None or active_task is asyncio.current_task():
                self._cleanup_agent(agent_id)

    def _cleanup_agent(self, agent_id: str) -> None:
        """Clean up after an agent completes.

        Preserves the descriptor in _completed_agents so parents can
        restart it later. The agent is removed from active tracking
        (sessions, tasks, futures, router) but remains discoverable.
        """
        descriptor = self._agents.get(agent_id)
        if descriptor:
            descriptor.completed_at = datetime.now(timezone.utc)
            # Keep descriptor in completed pool for potential restart
            self._completed_agents[agent_id] = descriptor
        self._sessions.pop(agent_id, None)
        self._tasks.pop(agent_id, None)
        self._result_futures.pop(agent_id, None)
        self._router.unregister_agent(agent_id)
        self._agents.pop(agent_id, None)
        logger.info("Agent cleaned up: %s", agent_id[:8])

    def get_children(self, parent_id: str) -> list:
        """Return all agents whose parent_id matches."""
        return [
            d for d in self._agents.values()
            if d.parent_id == parent_id
        ]

    def has_active_children(self, parent_id: str) -> bool:
        """Check if an agent has any children still in a non-terminal state."""
        terminal = {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.KILLED,
        }
        return any(
            d.state not in terminal
            for d in self._agents.values()
            if d.parent_id == parent_id
        )

    async def wait_for_result(self, agent_id: str) -> AgentResult:
        """Block until an agent completes and return its result.

        Uses the configured agent timeout to prevent hanging indefinitely.
        On timeout, logs a warning and returns a failed AgentResult.
        """
        future = self._result_futures.get(agent_id)
        if future is None:
            raise AgentSpawnError(agent_id, "no result future found")
        timeout = self._config.agent_timeout_seconds
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "wait_for_result timed out for agent %s after %.0fs",
                agent_id[:8], timeout,
            )
            return AgentResult(
                agent_id=agent_id,
                success=False,
                summary="",
                error=(
                    f"Timed out waiting for agent {agent_id[:8]} to complete "
                    f"after {timeout:.0f}s"
                ),
            )

    async def wait_for_multiple(
        self,
        agent_ids: list[str],
    ) -> list[AgentResult]:
        """Wait for multiple agents to complete in parallel.

        Uses the configured agent timeout to prevent hanging indefinitely.
        On timeout, individual agents that did not finish are returned as
        failed AgentResults.
        """
        futures = []
        for aid in agent_ids:
            f = self._result_futures.get(aid)
            if f is None:
                raise AgentSpawnError(aid, "no result future found")
            futures.append(f)
        timeout = self._config.agent_timeout_seconds
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "wait_for_multiple timed out after %.0fs for agents: %s",
                timeout,
                ", ".join(aid[:8] for aid in agent_ids),
            )
            # Build results: use done futures where available, timeout for rest
            results_list: list[AgentResult] = []
            for i, f in enumerate(futures):
                if f.done() and not f.cancelled():
                    try:
                        results_list.append(f.result())
                    except Exception as exc:
                        results_list.append(AgentResult(
                            agent_id=agent_ids[i],
                            success=False,
                            summary="",
                            error=str(exc),
                        ))
                else:
                    results_list.append(AgentResult(
                        agent_id=agent_ids[i],
                        success=False,
                        summary="",
                        error=(
                            f"Timed out waiting for agent {agent_ids[i][:8]} "
                            f"to complete after {timeout:.0f}s"
                        ),
                    ))
            return results_list
        # Convert exceptions to failed AgentResults
        final: list[AgentResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final.append(AgentResult(
                    agent_id=agent_ids[i],
                    success=False,
                    summary="",
                    error=str(r),
                ))
            else:
                final.append(r)
        return final

    async def force_fail_agent(
        self,
        agent_id: str,
        reason: str,
    ) -> None:
        """Force an agent into FAILED state and cancel its task."""
        descriptor = self._agents.get(agent_id)
        if descriptor is None:
            return

        logger.warning(
            "Force-failing agent %s: %s", agent_id[:8], reason
        )

        old_state = descriptor.state
        descriptor.state = AgentState.FAILED
        descriptor.error = reason

        # Fire state-change event so TUI/server update
        await fire_event(self._event_callback, {
            "event": "agent_state_changed",
            "agent_id": agent_id,
            "old_state": old_state.value,
            "new_state": AgentState.FAILED.value,
        })

        # Terminate any active provider subprocess
        session = self._sessions.get(agent_id)
        if session is not None:
            session.terminate_subprocess()

        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()

        future = self._result_futures.get(agent_id)
        if future and not future.done():
            future.set_result(AgentResult(
                agent_id=agent_id,
                success=False,
                summary="",
                error=reason,
            ))

        # Recursively kill all children
        for child_id in list(descriptor.children):
            await self.force_fail_agent(
                child_id, f"Parent {agent_id[:8]} failed"
            )

    async def kill_agent(
        self, agent_id: str, *, keep_restartable: bool = False,
    ) -> None:
        """Force-kill an agent and all its children."""
        descriptor = self._agents.get(agent_id)
        if descriptor is None:
            # Also remove from completed pool if present
            self._completed_agents.pop(agent_id, None)
            return

        descriptor.state = AgentState.KILLED
        if not keep_restartable:
            await fire_event(self._event_callback, {
                "event": "agent_killed",
                "agent_id": agent_id,
            })

        # Terminate any active provider subprocess before cancelling the
        # asyncio task.  This ensures non-Claude CLIs (Codex, Gemini,
        # MiniMax) are stopped promptly rather than lingering.
        session = self._sessions.get(agent_id)
        if session is not None:
            session.terminate_subprocess()

        # Resolve the result future with partial output before cancelling
        future = self._result_futures.get(agent_id)
        if future and not future.done():
            partial = "\n".join(getattr(session, '_accumulated_text', []) if session else [])
            future.set_result(AgentResult(
                agent_id=agent_id,
                success=False,
                summary=partial,
                error=(
                    f"Agent was killed by user. Partial output ({len(partial)} chars):\n"
                    f"{partial[:2000]}"
                    if partial else
                    "Agent was killed by user before producing output."
                ),
            ))

        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()

        for child_id in list(descriptor.children):
            await self.kill_agent(child_id)

        self._cleanup_agent(agent_id)
        if not keep_restartable:
            # Killed agents should not be restartable unless explicitly requested
            self._completed_agents.pop(agent_id, None)

    def get_session(self, agent_id: str) -> AgentSession | None:
        """Get a running agent's session by ID."""
        return self._sessions.get(agent_id)

    def get_completed_descriptor(self, agent_id: str) -> AgentDescriptor | None:
        """Get a completed/failed agent's descriptor by ID."""
        return self._completed_agents.get(agent_id)

    async def restart_agent(
        self,
        agent_id: str,
        new_prompt: str,
    ) -> AgentDescriptor:
        """Restart a completed or failed agent with a new prompt.

        Reuses the agent's identity (ID, parent, role, model, tools, cwd)
        but resets its state and runs a new session. The parent's children
        list is re-linked automatically.

        Raises:
            AgentSpawnError: If the agent is not found or not in a
                restartable state, concurrency limit is reached,
                or engine is shutting down.
        """
        # Reject restarts during shutdown/interrupt
        if self._shutting_down:
            raise AgentSpawnError(
                agent_id,
                "Engine is shutting down — cannot restart agents",
            )

        descriptor = self._completed_agents.pop(agent_id, None)
        if descriptor is None:
            raise AgentSpawnError(
                agent_id, "Agent not found in completed pool"
            )
        if descriptor.state not in (AgentState.COMPLETED, AgentState.FAILED, AgentState.KILLED):
            raise AgentSpawnError(
                agent_id,
                f"Agent is in state {descriptor.state.value}, "
                f"not restartable",
            )

        if self.active_count >= self._config.max_concurrent_agents:
            # Put it back so it can be retried later
            self._completed_agents[agent_id] = descriptor
            raise AgentSpawnError(
                agent_id,
                f"Concurrency limit: "
                f"{self.active_count}/{self._config.max_concurrent_agents}",
            )

        # Auto-detect provider based on model name if model_registry is available.
        # This ensures the correct provider is used even if the descriptor's
        # provider field is outdated (e.g., from a restored session).
        effective_provider = descriptor.provider
        effective_model = descriptor.model
        if self._config and self._config.model_registry and descriptor.model:
            cap = self._config.model_registry.get(descriptor.model)
            if cap:
                if not cap.available:
                    # Try to fall back to a similar available model
                    available_models = self._config.model_registry.list_available()
                    if not available_models:
                        # Put it back so it can be retried later
                        self._completed_agents[agent_id] = descriptor
                        raise ProviderNotAvailableError(
                            f"Model '{descriptor.model}' provider '{cap.provider}'",
                            [],
                        )

                    # Prefer a model from the same tier, or just pick the first available
                    fallback = None
                    for m in available_models:
                        if m.tier == cap.tier:
                            fallback = m
                            break
                    if not fallback:
                        fallback = available_models[0]

                    logger.warning(
                        "Agent %s model '%s' is not available (provider '%s' not installed). "
                        "Falling back to '%s' (%s provider).",
                        agent_id[:8],
                        descriptor.model,
                        cap.provider,
                        fallback.model_id,
                        fallback.provider,
                    )
                    effective_model = fallback.model_id
                    effective_provider = fallback.provider
                else:
                    effective_provider = cap.provider
                    logger.info(
                        "Auto-detected provider for agent %s model %s: %s",
                        agent_id[:8],
                        descriptor.model,
                        effective_provider,
                    )
                # Update descriptor with the correct provider and model
                descriptor.provider = effective_provider
                descriptor.model = effective_model

        # Validate that the provider is still available
        if effective_provider and self._provider_registry:
            provider_obj = self._provider_registry.get(effective_provider)
            if provider_obj is None or not provider_obj.is_available():
                # Put it back so it can be retried later
                self._completed_agents[agent_id] = descriptor
                available = (
                    self._provider_registry.list_available()
                    if self._provider_registry else []
                )
                raise ProviderNotAvailableError(
                    effective_provider,
                    available,
                )

        # Reset descriptor for new run
        descriptor.prompt = new_prompt
        descriptor.state = AgentState.PENDING
        descriptor.error = None
        descriptor.result_summary = None
        descriptor.result_artifacts = {}
        descriptor.completed_at = None
        descriptor.created_at = datetime.now(timezone.utc)

        # The PreToolUse hook in AgentSession handles bash permission
        # checking regardless of permission_mode.  Heal old descriptors
        # to BYPASS for maximum speed.
        if descriptor.role == AgentRole.MASTER:
            descriptor.permission_mode = PermissionMode.BYPASS

        # Re-link to parent's children list
        if descriptor.parent_id:
            parent = (
                self._agents.get(descriptor.parent_id)
                or self._completed_agents.get(descriptor.parent_id)
            )
            if parent and agent_id not in parent.children:
                parent.children.append(agent_id)

        self._agents[agent_id] = descriptor
        self._router.register_agent(descriptor)

        # Create new result future
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AgentResult] = loop.create_future()
        self._result_futures[agent_id] = future

        # Resolve provider
        provider = None
        if self._provider_registry and descriptor.provider:
            provider = self._provider_registry.get(descriptor.provider)

        # Resolve plugins using original settings
        request = SpawnRequest(
            parent_id=descriptor.parent_id,
            prompt=new_prompt,
            role=descriptor.role,
            tools=descriptor.tools,
            model=descriptor.model,
            permission_mode=descriptor.permission_mode,
            cwd=descriptor.cwd,
            mcp_servers=descriptor.mcp_servers,
            exclude_plugins=descriptor.exclude_plugins,
            provider=descriptor.provider,
        )
        effective_plugins = self._resolve_agent_plugins(request)

        # Start new session
        session = AgentSession(
            descriptor=descriptor,
            manager=self,
            router=self._router,
            expert_registry=self._expert_registry,
            agent_timeout_seconds=self._config.agent_timeout_seconds,
            tool_call_timeout_seconds=(
                self._config.tool_call_timeout_seconds
            ),
            user_question_timeout_seconds=(
                self._config.user_question_timeout_seconds
            ),
            provider=provider,
            event_callback=self._event_callback,
            permission_callback=self._permission_callback,
            user_question_callback=self._user_question_callback,
            plugin_mcp_servers=effective_plugins,
            always_allowed_tools=self._always_allowed_tools,
            conversation_store=self._conversation_store,
            child_default_model=self._config.default_model,
            child_default_provider=self._config.default_provider,
            model_registry=self._config.model_registry,
            peer_models=self._config.peer_models,
        )
        self._sessions[agent_id] = session

        task = asyncio.create_task(
            self._run_session(agent_id, session, future),
            name=f"agent-{agent_id[:8]}-restart",
        )
        self._tasks[agent_id] = task

        logger.info(
            "Agent restarted: %s (parent=%s, role=%s)",
            agent_id[:8],
            (descriptor.parent_id or "none")[:8],
            descriptor.role.value,
        )

        await fire_event(self._event_callback, {
            "event": "agent_restarted",
            "agent_id": agent_id,
            "parent_id": descriptor.parent_id,
            "role": descriptor.role.value,
            "model": descriptor.model,
            "prompt": strip_injected_prompt_prefix(new_prompt)[:200],
            "name": await self._derive_agent_name(descriptor.role, new_prompt),
        })

        return descriptor

    def get_all_descriptors(self) -> list[AgentDescriptor]:
        """Return all agent descriptors (active + completed, for monitoring)."""
        all_descs = dict(self._completed_agents)
        all_descs.update(self._agents)  # Active agents override
        return list(all_descs.values())
