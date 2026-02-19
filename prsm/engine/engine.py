"""Top-level orchestration engine.

Wires together AgentManager, MessageRouter, ExpertRegistry, and
the deadlock detector. Single entry point for running orchestrated
tasks.

Usage:
    from claude_orchestrator import OrchestrationEngine, ExpertProfile

    engine = OrchestrationEngine()
    engine.register_expert(ExpertProfile(...))
    result = await engine.run("Build feature X with tests")
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .models import (
    AgentRole,
    ExpertProfile,
    PermissionMode,
    SpawnRequest,
)
from .agent_manager import AgentManager
from .message_router import MessageRouter
from .expert_registry import ExpertRegistry
from .conversation_store import ConversationStore
from .deadlock import run_deadlock_detector
from .context import DEFAULT_MASTER_PROMPT_TEMPLATE
from .config import EngineConfig, EventCallback, PermissionCallback, fire_event
from .resource_manager import ResourceBudget, ResourceManager
from .shadow_triage import ShadowTriageModel
from .telemetry import TelemetryCollector

logger = logging.getLogger(__name__)


class OrchestrationEngine:
    """Main orchestration engine.

    Usage:
        engine = OrchestrationEngine()
        engine.register_expert(ExpertProfile(...))
        result = await engine.run("Build feature X with tests")
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        master_prompt_template: str | None = None,
        master_prompt_vars: dict[str, str] | None = None,
        provider_registry: object | None = None,
        plugin_mcp_servers: dict | None = None,
        plugin_manager: object | None = None,
        initial_allowed_tools: set[str] | None = None,
    ) -> None:
        self._config = config or EngineConfig.from_env()
        self._event_callback = self._build_scoped_event_callback(
            self._config.event_callback,
            self._config.project_id,
        )
        self._permission_callback = self._config.permission_callback
        self._user_question_callback = self._config.user_question_callback
        self._master_prompt_template = (
            master_prompt_template or DEFAULT_MASTER_PROMPT_TEMPLATE
        )
        self._master_prompt_vars = master_prompt_vars or {}
        self._provider_registry = provider_registry
        self._resource_manager = ResourceManager()
        self._telemetry_collector: TelemetryCollector | None = None
        if self._config.telemetry_db_path:
            self._telemetry_collector = TelemetryCollector(
                Path(self._config.telemetry_db_path)
            )
        self._shadow_triage_model: ShadowTriageModel | None = None
        if self._config.triage_model_shadow_enabled:
            self._shadow_triage_model = ShadowTriageModel(
                model=self._config.triage_shadow_model
            )
        for project_id, budget_cfg in (self._config.resource_budgets or {}).items():
            self._resource_manager.configure_budget(
                ResourceBudget(
                    project_id=project_id,
                    max_total_tokens=int(budget_cfg.get("max_total_tokens", 0)),
                    max_concurrent_agents=int(
                        budget_cfg.get("max_concurrent_agents", 10)
                    ),
                    max_agent_spawns_per_hour=int(
                        budget_cfg.get("max_agent_spawns_per_hour", 50)
                    ),
                    max_tool_calls_per_hour=int(
                        budget_cfg.get("max_tool_calls_per_hour", 500)
                    ),
                )
            )
        self._router = MessageRouter(
            queue_maxsize=self._config.message_queue_size,
            event_callback=self._event_callback,
            telemetry_collector=self._telemetry_collector,
            shadow_triage_model=self._shadow_triage_model,
        )
        self._expert_registry = ExpertRegistry()
        self._conversation_store = ConversationStore()
        self._manager = AgentManager(
            router=self._router,
            expert_registry=self._expert_registry,
            config=self._config,
            provider_registry=provider_registry,
            event_callback=self._event_callback,
            permission_callback=self._permission_callback,
            user_question_callback=self._user_question_callback,
            plugin_mcp_servers=plugin_mcp_servers,
            plugin_manager=plugin_manager,
            initial_allowed_tools=initial_allowed_tools,
            conversation_store=self._conversation_store,
            resource_manager=self._resource_manager,
        )
        self._deadlock_task: asyncio.Task | None = None
        self._last_master_id: str | None = None
        # Set by shutdown() to prevent run() from emitting a duplicate
        # engine_finished event when its wait_for_result() resolves
        # after the master was killed during shutdown.
        self._shutdown_finished_fired = False
        self._shutdown_lock = asyncio.Lock()

    @staticmethod
    def _build_scoped_event_callback(
        callback: EventCallback | None,
        project_id: str | None,
    ) -> EventCallback | None:
        """Inject project_id into emitted events when configured."""
        if callback is None:
            return None
        if not project_id:
            return callback

        async def _wrapped(event: dict[str, object]) -> None:
            scoped = dict(event)
            scoped.setdefault("project_id", project_id)
            await callback(scoped)

        return _wrapped

    def register_expert(self, profile: ExpertProfile) -> None:
        """Register a domain expert profile."""
        self._expert_registry.register(profile)

    def unregister_expert(self, expert_id: str) -> None:
        """Remove a domain expert profile."""
        self._expert_registry.unregister(expert_id)

    def set_master_prompt(
        self,
        template: str,
        vars: dict[str, str] | None = None,
    ) -> None:
        """Replace the master prompt template and variables.

        The template can use {task_definition} and {expert_list}
        (auto-populated) plus any custom variables.
        """
        self._master_prompt_template = template
        if vars is not None:
            self._master_prompt_vars = vars

    @property
    def expert_registry(self) -> ExpertRegistry:
        """Access the expert registry directly."""
        return self._expert_registry

    @property
    def agent_manager(self) -> AgentManager:
        """Access the agent manager directly."""
        return self._manager

    @property
    def conversation_store(self) -> ConversationStore:
        """Access the conversation store directly."""
        return self._conversation_store

    @property
    def last_master_id(self) -> str | None:
        """ID of the most recently completed master agent, if any."""
        return self._last_master_id

    async def run(
        self,
        task_definition: str,
        master_model: str | None = None,
        master_tools: list[str] | None = None,
    ) -> str:
        """Run a task through the full orchestration pipeline.

        Args:
            task_definition: High-level description of what to do.
            master_model: Override model for the master agent.
            master_tools: Override tools for the master agent.

        Returns:
            The master agent's final synthesis summary.
        """
        logger.info("Engine starting: %s", task_definition[:200])
        self._shutdown_finished_fired = False
        await fire_event(self._event_callback, {
            "event": "engine_started",
            "task_definition": task_definition[:500],
        })

        # Start deadlock detector
        self._deadlock_task = asyncio.create_task(
            run_deadlock_detector(
                router=self._router,
                manager=self._manager,
                check_interval=(
                    self._config.deadlock_check_interval_seconds
                ),
                max_wait_seconds=(
                    self._config.deadlock_max_wait_seconds
                ),
            ),
            name="deadlock-detector",
        )

        try:
            # Build master prompt
            expert_list = ", ".join(
                self._expert_registry.list_ids()
            ) or "none registered"

            template_vars = {
                "task_definition": task_definition,
                "expert_list": expert_list,
                **self._master_prompt_vars,
            }

            master_prompt = self._master_prompt_template.format(
                **template_vars
            )

            # Spawn master agent (no parent).
            # Claude uses in-process MCP; other providers use the
            # TCP bridge + MCP proxy (see orch_bridge.py).

            # Determine effective model and provider.
            # If master_model is provided, auto-detect its provider from
            # the model registry. Otherwise use config defaults.
            effective_model = master_model or self._config.master_model
            effective_provider = self._config.master_provider

            # Auto-detect provider based on model name if model_registry is available
            if self._config.model_registry:
                cap = self._config.model_registry.get(effective_model)
                if cap:
                    if not cap.available:
                        # Try to fall back to a similar available model
                        available_models = self._config.model_registry.list_available()
                        if not available_models:
                            raise ValueError(
                                f"Model '{effective_model}' is not available "
                                f"(provider '{cap.provider}' not installed) and no other models are available. "
                                f"Please install at least one provider CLI (claude, codex, or gemini)."
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
                            "Model '%s' is not available (provider '%s' not installed). "
                            "Falling back to '%s' (%s provider).",
                            effective_model,
                            cap.provider,
                            fallback.model_id,
                            fallback.provider,
                        )
                        effective_model = fallback.model_id
                        effective_provider = fallback.provider
                    else:
                        effective_provider = cap.provider
                        logger.info(
                            "Auto-detected provider for master model %s: %s",
                            effective_model,
                            effective_provider,
                        )

            request = SpawnRequest(
                parent_id=None,
                prompt=master_prompt,
                role=AgentRole.MASTER,
                model=effective_model,
                provider=effective_provider,
                # Master gets no file-editing tools by default
                tools=master_tools or [
                    "Read", "Glob", "Grep",
                ],
                # Use BYPASS mode for maximum speed — the PreToolUse hook
                # in AgentSession._build_bash_permission_hooks() handles
                # blocking dangerous bash commands.  Hooks run BEFORE
                # permission mode in the CLI evaluation flow, so they work
                # even with bypassPermissions.
                permission_mode=PermissionMode.BYPASS,
                cwd=self._config.default_cwd,
            )

            master_desc = await self._manager.spawn_agent(request)
            self._last_master_id = master_desc.agent_id
            result = await self._manager.wait_for_result(
                master_desc.agent_id,
            )

            logger.info(
                "Engine completed (success=%s, duration=%.1fs)",
                result.success,
                result.duration_seconds,
            )

            if not result.success:
                logger.error("Engine failed: %s", result.error)

            # Skip if shutdown() already fired engine_finished (prevents
            # duplicate events when a stopped run's wait_for_result
            # resolves after the shutdown event was already emitted).
            if not self._shutdown_finished_fired:
                await fire_event(self._event_callback, {
                    "event": "engine_finished",
                    "success": result.success,
                    "summary": result.summary or "",
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                })

            return result.summary

        finally:
            if self._deadlock_task:
                self._deadlock_task.cancel()
                try:
                    await self._deadlock_task
                except asyncio.CancelledError:
                    pass

    async def run_continuation(
        self,
        task_definition: str,
        master_agent_id: str,
    ) -> str:
        """Continue a task on an existing (completed) master agent.

        Instead of spawning a brand-new master, this restarts the given
        agent with a new prompt.  The agent keeps its identity (UUID,
        tree position, model, tools) so the TUI conversation view is
        *not* cleared — it just continues where it left off.

        Args:
            task_definition: The follow-up prompt from the user.
            master_agent_id: ID of the completed master to restart.

        Returns:
            The master agent's final synthesis summary.
        """
        logger.info("Engine continuing on %s: %s", master_agent_id[:8], task_definition[:200])
        self._shutdown_finished_fired = False
        await fire_event(self._event_callback, {
            "event": "engine_started",
            "task_definition": task_definition[:500],
        })

        # Start deadlock detector
        self._deadlock_task = asyncio.create_task(
            run_deadlock_detector(
                router=self._router,
                manager=self._manager,
                check_interval=(
                    self._config.deadlock_check_interval_seconds
                ),
                max_wait_seconds=(
                    self._config.deadlock_max_wait_seconds
                ),
            ),
            name="deadlock-detector",
        )

        try:
            # Build the continuation prompt — just the user's follow-up,
            # wrapped so the agent understands it's a continuation.
            continuation_prompt = (
                "The user has sent a follow-up message. Continue the conversation.\n\n"
                f"User message:\n{task_definition}"
            )

            master_desc = await self._manager.restart_agent(
                master_agent_id, continuation_prompt,
            )
            self._last_master_id = master_desc.agent_id
            result = await self._manager.wait_for_result(
                master_desc.agent_id,
            )

            logger.info(
                "Engine continuation completed (success=%s, duration=%.1fs)",
                result.success,
                result.duration_seconds,
            )

            if not result.success:
                logger.error("Engine continuation failed: %s", result.error)

            if not self._shutdown_finished_fired:
                await fire_event(self._event_callback, {
                    "event": "engine_finished",
                    "success": result.success,
                    "summary": result.summary or "",
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                })

            return result.summary

        finally:
            if self._deadlock_task:
                self._deadlock_task.cancel()
                try:
                    await self._deadlock_task
                except asyncio.CancelledError:
                    pass

    async def shutdown(self) -> None:
        """Gracefully shut down all agents."""
        if self._shutdown_lock.locked():
            logger.info("Shutdown already in progress, skipping concurrent call")
            return
        async with self._shutdown_lock:
            # Set shutdown guard FIRST to prevent new spawns during teardown
            self._manager._shutting_down = True
            try:
                for desc in self._manager.get_all_descriptors():
                    await self._manager.kill_agent(desc.agent_id)
                # Mark that shutdown has fired engine_finished so that
                # a concurrent run()/run_continuation() doesn't emit
                # a duplicate when its wait_for_result() resolves.
                self._shutdown_finished_fired = True
                # Fire engine_finished so the server layer saves the session
                await fire_event(self._event_callback, {
                    "event": "engine_finished",
                    "success": False,
                    "summary": "",
                    "error": "Shutdown requested",
                    "duration_seconds": 0.0,
                })
                logger.info("Engine shutdown complete")
            finally:
                self._manager._shutting_down = False

    async def interrupt(self) -> str | None:
        """Interrupt the current orchestration, keeping the master restartable.

        Kills all child agents and stops the master agent, but preserves
        the master in the completed pool so it can be restarted via
        ``run_continuation``. Returns the master agent ID if one was
        found, or None.

        Unlike ``shutdown``, this does NOT fire ``engine_finished`` —
        the caller is expected to immediately start a continuation.
        """
        # Prevent the still-running run()/run_continuation() from emitting
        # a stale engine_finished after the master's result future resolves.
        self._shutdown_finished_fired = True
        # Set shutdown guard to prevent new spawns during teardown
        self._manager._shutting_down = True
        try:
            master_id = self._last_master_id
            all_descs = self._manager.get_all_descriptors()

            # Kill children first (any agent that has a parent)
            for desc in all_descs:
                if desc.parent_id is not None:
                    await self._manager.kill_agent(desc.agent_id)

            # Kill any non-master orphan agents (agents without a parent
            # that are NOT the master — shouldn't normally exist but
            # guards against edge cases)
            for desc in self._manager.get_all_descriptors():
                if desc.agent_id != master_id and desc.parent_id is None:
                    await self._manager.kill_agent(desc.agent_id)

            # Kill the master with keep_restartable=True so it stays in
            # the completed pool and can be restarted
            if master_id:
                desc = self._manager._agents.get(master_id)
                if desc:
                    await self._manager.kill_agent(
                        master_id, keep_restartable=True,
                    )

            # Cancel the deadlock detector if running
            if self._deadlock_task:
                self._deadlock_task.cancel()
                try:
                    await self._deadlock_task
                except asyncio.CancelledError:
                    pass
                self._deadlock_task = None

            logger.info("Engine interrupt complete (master=%s)", master_id)
            return master_id
        finally:
            self._manager._shutting_down = False
