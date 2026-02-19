"""Bridge between the orchestrator engine and the PRSM TUI.

Creates an EngineConfig with event_callback wired to an EventBus,
manages the engine lifecycle, and maps orchestrator agent descriptors
to TUI AgentNode models.
"""
from __future__ import annotations

import asyncio
import ast
import copy
import json
import logging
import os
import shutil
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.adapters.event_bus import EventBus
from prsm.adapters.events import PermissionRequest, UserQuestionRequest
from prsm.adapters.permission_store import PermissionStore
from prsm.shared.services.command_policy_store import CommandPolicyStore
from prsm.shared.services.project import ProjectManager

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)
DEFAULT_USER_QUESTION_TIMEOUT_SECONDS = 0.0
SUPEREGO_MULTI_PROJECT = os.getenv("PRSM_MULTI_PROJECT", "0") == "1"
_CLAUDE_FAMILY_TOKENS = {"opus", "sonnet", "haiku"}
_CLAUDE_ALIAS_RE = re.compile(
    r"^(?:claude-)?(opus|sonnet|haiku)(?:-(.+))?$",
    re.IGNORECASE,
)
_SPARK_ALIAS_RE = re.compile(r"^(?:claude-)?(spark)(?:-(.+))?$", re.IGNORECASE)


def _strip_claude_date_suffix(version: str) -> str:
    """Strip trailing runtime date suffix from a version token (YYYYMMDD)."""
    parts = version.split("-")
    if len(parts) > 1 and re.fullmatch(r"\d{6,8}", parts[-1] or ""):
        parts.pop()
    return "-".join(parts)


def _claude_family_version_parts(model_id: str) -> tuple[str, tuple] | None:
    """Extract family + comparable version tuple from Claude model IDs."""
    parts = model_id.replace(".", "-").replace("_", "-").lower().split("-")
    if len(parts) < 3 or parts[0] != "claude":
        return None
    for idx in range(1, len(parts) - 1):
        family = parts[idx]
        if family not in _CLAUDE_FAMILY_TOKENS:
            continue
        prefix = "-".join(parts[1:idx])
        version = _strip_claude_date_suffix("-".join(parts[idx + 1 :]).strip("-"))
        if not version:
            continue
        if prefix:
            version = f"{prefix}-{version}"
        chunks: list[tuple[int, int | str]] = []
        for bit in re.findall(r"[0-9]+|[a-z]+", version):
            if bit.isdigit():
                chunks.append((0, int(bit)))
            else:
                chunks.append((1, bit))
        return family, tuple(chunks)
    return None


def _family_version_from_model_id(family: str, model_id: str) -> str | None:
    normalized = (model_id or "").split("::reasoning_effort=", 1)[0].strip().lower()
    normalized = normalized.replace("_", "-")
    if family == "spark":
        match = re.match(
            r"^gpt-([0-9]+(?:[.-][0-9]+)*)-spark(?:-|$)",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).replace("-", ".")

    match = re.match(
        rf"^claude-{re.escape(family)}-(.+)$",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    version = match.group(1).strip("-")
    if not version:
        return None
    return _strip_claude_date_suffix(version).replace("-", ".")


def _humanize_model_alias(alias: str, model_id: str | None = None) -> str:
    """Convert model aliases to human-friendly display form."""
    if not alias:
        return alias

    match = _CLAUDE_ALIAS_RE.match(alias)
    if match:
        family = match.group(1).lower()
        suffix = match.group(2)
        family_label = family[0].upper() + family[1:]
        if suffix:
            return f"{family_label} {suffix.replace('-', '.')}"
        version = _family_version_from_model_id(family, model_id or alias)
        if version:
            return f"{family_label} {version}"
        return family_label

    match = _SPARK_ALIAS_RE.match(alias)
    if match:
        suffix = match.group(2)
        family_label = "Spark"
        if suffix:
            return f"{family_label} {suffix.replace('-', '.')}"
        version = _family_version_from_model_id("spark", model_id or alias)
        if version:
            return f"{family_label} {version}"
        return family_label

    return alias


def _alias_sort_key(alias: str) -> tuple[int, int, str]:
    """Prefer versioned family aliases over non-versioned versions."""
    family_match = _CLAUDE_ALIAS_RE.match(alias)
    if family_match:
        has_version = bool(family_match.group(2))
        return (0 if has_version else 1, len(alias), alias)

    spark_match = _SPARK_ALIAS_RE.match(alias)
    if spark_match:
        has_version = bool(spark_match.group(2))
        return (0 if has_version else 1, len(alias), alias)

    return (2, len(alias), alias)


def _model_display_name(
    model_id: str,
    aliases_by_target: dict[str, list[str]],
) -> str:
    """Prefer explicit aliases, then fall back to readable runtime variants."""
    aliases = aliases_by_target.get(model_id, [])
    if aliases:
        aliases_sorted = sorted(aliases, key=_alias_sort_key)
        return _humanize_model_alias(aliases_sorted[0], model_id=model_id)
    return _fallback_model_display_name(model_id)


def _canonical_claude_family_model_ids(model_ids: set[str]) -> dict[str, str]:
    """Pick the latest model_id per Claude family (opus/sonnet/haiku)."""
    latest_by_family: dict[str, str] = {}
    latest_scores: dict[str, tuple] = {}
    for model_id in sorted(model_ids):
        family_info = _claude_family_version_parts(model_id)
        if not family_info:
            continue
        family, score = family_info
        if family not in latest_scores or score > latest_scores[family]:
            latest_scores[family] = score
            latest_by_family[family] = model_id
    return latest_by_family

# ── Model mappings: orchestrator string → engine enum ──

_ROLE_MAP: dict[str, AgentRole] = {
    "master": AgentRole.MASTER,
    "worker": AgentRole.WORKER,
    "expert": AgentRole.EXPERT,
    "reviewer": AgentRole.REVIEWER,
}

_STATE_MAP: dict[str, AgentState] = {
    "pending": AgentState.PENDING,
    "starting": AgentState.STARTING,
    "running": AgentState.RUNNING,
    "waiting_for_parent": AgentState.WAITING_FOR_PARENT,
    "waiting_for_child": AgentState.WAITING_FOR_CHILD,
    "waiting_for_expert": AgentState.WAITING_FOR_EXPERT,
    "completed": AgentState.COMPLETED,
    "failed": AgentState.FAILED,
    "killed": AgentState.KILLED,
}


def _strip_prompt_prefix(prompt: str) -> str:
    """Strip injected system instruction prefixes from agent prompts.

    The orchestration tools prepend docs-first and assumption-minimization
    instructions to child prompts, delimited by a ``TASK:\\n`` marker.
    This function returns only the user-facing task description so that
    display names and previews are meaningful.

    Also handles truncated prompts where the ``TASK:\\n`` marker was cut
    off by ``[:200]`` slicing — detects the ``[DOCS-FIRST-ARCHITECTURE]``
    sentinel prefix and returns empty string in that case.
    """
    if not prompt:
        return prompt
    # Full prompt with TASK marker
    marker = "TASK:\n"
    idx = prompt.find(marker)
    if idx != -1:
        return prompt[idx + len(marker):]
    # Truncated: the sentinel prefix is present but TASK marker was sliced off
    if prompt.startswith("[DOCS-FIRST-ARCHITECTURE]"):
        return ""
    return prompt


def _agent_name(role: str, prompt: str) -> str:
    """Derive a display name for an agent from its role and prompt."""
    prompt = _strip_prompt_prefix(prompt)
    if role == "master":
        return "Orchestrator"
    if role == "expert":
        return prompt[:40] if prompt else "Expert"
    if role == "reviewer":
        name = prompt[:50].strip() if prompt else "Reviewer"
        if prompt and len(prompt) > 50:
            name += "..."
        return name
    # Worker: use first 50 chars of prompt
    name = prompt[:50].strip()
    if len(prompt) > 50:
        name += "..."
    return name or "Worker"


def _fallback_model_display_name(model_id: str) -> str:
    """Format runtime-encoded model IDs into readable labels when possible."""
    marker = "::reasoning_effort="
    if marker not in model_id:
        return model_id

    base, effort = model_id.split(marker, 1)
    effort = effort.strip().lower()
    if base and effort in {"low", "medium", "high"}:
        return f"{base}-{effort}"
    return model_id


class OrchestratorBridge:
    """Connects the orchestrator engine to the TUI.

    Usage:
        bridge = OrchestratorBridge()
        bridge.configure(model="claude-opus-4-6", cwd=".")
        # Then in a @work worker:
        await bridge.run(prompt)
    """

    def __init__(self) -> None:
        self.event_bus = EventBus()
        self._engine = None
        self._configured = False
        self._running = False
        # Monotonically increasing counter; incremented each time run()
        # or run_continuation() starts.  Used by the finally block to
        # avoid clobbering ``_running`` when a stale run() returns after
        # a new run has already begun (stop-then-restart race).
        self._run_generation: int = 0
        self._preferred_model_alias: str | None = None
        # Maps orchestrator agent_id → TUI AgentNode
        self.agent_map: dict[str, AgentNode] = {}
        # Pending permission requests: request_id → Future[str]
        self._permission_futures: dict[str, asyncio.Future[str]] = {}
        # Pending user question requests: request_id → Future[str]
        self._question_futures: dict[str, asyncio.Future[str]] = {}
        # Track agent_id → set of request_ids for cleanup
        self._agent_permission_requests: dict[str, set[str]] = {}
        self._agent_question_requests: dict[str, set[str]] = {}
        # Permission persistence
        self._permission_store: PermissionStore | None = None
        self._command_policy_store: CommandPolicyStore | None = None
        # Persisted allowed tools loaded at configure time
        self._persisted_allowed_tools: set[str] = set()
        self._project_registry = None
        self._project_id: str | None = None
        self._user_question_timeout_seconds: float = 0.0
        # Track the YAML-configured default model so current_model can
        # return it even if the engine hasn't been built yet (e.g. Claude
        # CLI missing but a non-Claude default was configured).
        self._configured_default_model: str | None = None

    @property
    def available(self) -> bool:
        """Whether the orchestrator can run.

        Requires:
        1. At least one provider CLI on PATH (claude, codex, or gemini)
        2. The orchestrator engine importable (vendored in prsm.orchestrator)

        The system no longer hard-gates on the ``claude`` CLI alone —
        non-Claude providers (Codex, Gemini, etc.) are sufficient when
        Claude is not installed.
        """
        has_any_cli = (
            shutil.which("claude")
            or shutil.which("codex")
            or shutil.which("gemini")
        )
        if not has_any_cli:
            return False
        try:
            from prsm.engine import OrchestrationEngine  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def running(self) -> bool:
        """Whether an orchestration is currently in progress."""
        return self._running

    @property
    def last_master_id(self) -> str | None:
        """ID of the most recently completed master agent, if any."""
        if self._engine:
            return self._engine.last_master_id
        return None

    def configure(
        self,
        model: str = "claude-opus-4-6",
        cwd: str = ".",
        experts: list | None = None,
        plugin_mcp_servers: dict | None = None,
        plugin_manager: object | None = None,
        project_dir: "Path | None" = None,
        yaml_config: object | None = None,
        project_registry: object | None = None,
        project_id: str | None = None,
    ) -> bool:
        """Configure the orchestrator engine. Returns False if unavailable.

        If yaml_config (an OrchestrationConfig from yaml_config.py) is
        provided, engine settings, model, and experts are loaded from it.
        Individual parameters (model, cwd, experts) still override.
        """
        if not self.available:
            return False

        try:
            from prsm.engine import OrchestrationEngine
            from prsm.engine.config import EngineConfig
            from prsm.engine.providers.registry import build_provider_registry

            # Load persisted tool permissions
            self._permission_store = PermissionStore(project_dir)
            self._persisted_allowed_tools = self._permission_store.load()
            self._command_policy_store = CommandPolicyStore(Path(cwd))
            self._command_policy_store.ensure_files()
            if self._persisted_allowed_tools:
                logger.info(
                    "Loaded %d persisted tool permissions",
                    len(self._persisted_allowed_tools),
                )

            # Build EngineConfig — from YAML or defaults
            if yaml_config is not None:
                logger.info(
                    "OrchestratorBridge.configure: using YAML config "
                    "(providers=%d, models=%d, experts=%d)",
                    len(yaml_config.providers),
                    len(yaml_config.models),
                    len(yaml_config.experts),
                )
                # Copy the shared EngineConfig so per-session mutations
                # (cwd, callbacks) don't bleed across sessions.
                config = copy.copy(yaml_config.engine)
                config.event_callback = self.event_bus.make_callback()
                config.permission_callback = self._handle_permission
                config.user_question_callback = self._handle_user_question
                # Allow parameter overrides
                if model != "claude-opus-4-6":
                    logger.debug(
                        "OrchestratorBridge.configure: overriding model from config "
                        "(was %s, now %s)",
                        config.default_model, model
                    )
                    config.default_model = model
                if cwd != ".":
                    logger.debug(
                        "OrchestratorBridge.configure: overriding cwd from config "
                        "(was %s, now %s)",
                        config.default_cwd, cwd
                    )
                    config.default_cwd = cwd
                self._user_question_timeout_seconds = (
                    config.user_question_timeout_seconds
                )
                defaults = getattr(yaml_config, "defaults", None)
                if defaults is not None:
                    # Prefer the configured default alias for display labels.
                    self._preferred_model_alias = getattr(defaults, "model", None)
                # Track the YAML-configured master model for current_model
                # fallback when no engine is available.
                self._configured_default_model = config.master_model
            else:
                logger.info(
                    "OrchestratorBridge.configure: no YAML config provided, "
                    "using defaults (model=%s, cwd=%s)",
                    model, cwd
                )
                config = EngineConfig(
                    default_model=model,
                    default_cwd=cwd,
                    event_callback=self.event_bus.make_callback(),
                    permission_callback=self._handle_permission,
                    user_question_callback=self._handle_user_question,
                )
                self._user_question_timeout_seconds = (
                    config.user_question_timeout_seconds
                )
                self._preferred_model_alias = None

            if project_id:
                config.project_id = project_id
            if not config.project_id:
                config.project_id = ProjectManager.get_repo_identity(Path(cwd))
            self._project_id = config.project_id

            provider_registry = build_provider_registry(
                yaml_config.providers if yaml_config is not None else None
            )

            # Log provider availability at startup
            available = provider_registry.list_available()
            unavailable = provider_registry.list_unavailable()
            if available:
                logger.info(
                    "Providers ready: %s", ", ".join(available)
                )
            if unavailable:
                logger.warning(
                    "Providers NOT available (agents using these will fail): %s",
                    ", ".join(unavailable),
                )

            # Build model capability registry for intelligent model selection
            from prsm.engine.model_registry import (
                build_default_registry,
                load_model_registry_from_yaml,
            )
            from prsm.engine.model_intelligence import ModelIntelligence

            model_registry = build_default_registry()
            yaml_models = yaml_config.models if (
                yaml_config is not None and hasattr(yaml_config, "models")
            ) else None
            if yaml_config is not None and hasattr(yaml_config, "model_registry_raw"):
                model_registry = load_model_registry_from_yaml(
                    yaml_config.model_registry_raw or {}, model_registry,
                    model_aliases=yaml_models,
                )
            # Sync model availability with actual provider availability
            provider_report = provider_registry.get_availability_report()
            changed = model_registry.sync_availability(provider_report)
            if changed:
                unavail_models = [m for m, ok in changed.items() if not ok]
                if unavail_models:
                    logger.warning(
                        "Models marked unavailable (provider not installed): %s",
                        ", ".join(unavail_models),
                    )
            # Optional Claude model probe is deferred to run() (requires async)
            self._model_registry_needs_probe = os.getenv(
                "PRSM_PROBE_CLAUDE_MODELS_ON_STARTUP", "0",
            ).lower() in {"1", "true", "yes"}
            # Model discovery is deferred to run() (requires async).
            # Discovers models from CLI tools and updates models.yaml.
            self._model_discovery_needed = os.getenv(
                "PRSM_MODEL_DISCOVERY", "1",
            ).lower() in {"1", "true", "yes", "on"}
            self._update_clis_on_discovery = os.getenv(
                "PRSM_UPDATE_CLIS", "1",
            ).lower() in {"1", "true", "yes", "on"}
            # Load persistent model intelligence (learned rankings)
            model_intelligence = ModelIntelligence()
            model_intelligence.load()
            model_registry.set_intelligence(model_intelligence)
            config.model_registry = model_registry

            available_models = model_registry.list_available()
            logger.info(
                "Model registry loaded: %d models (%d available)",
                model_registry.count,
                len(available_models),
            )

            # Resolve peer_models from YAML config — restricts which
            # models can be used for child agents.
            if yaml_config is not None and hasattr(yaml_config, "defaults"):
                from prsm.engine.yaml_config import resolve_model_alias
                peer_models_dict: dict[str, tuple] = {}
                defaults = yaml_config.defaults
                if defaults.peer_models:
                    for peer_alias in defaults.peer_models:
                        p_name, p_model_id = resolve_model_alias(
                            peer_alias, yaml_config.models,
                        )
                        p_provider = provider_registry.get(p_name)
                        if p_provider:
                            peer_models_dict[peer_alias] = (
                                p_provider, p_model_id,
                            )
                            logger.info(
                                "Peer model registered: %s (%s, model=%s)",
                                peer_alias, p_name, p_model_id,
                            )
                if peer_models_dict:
                    config.peer_models = peer_models_dict
                    logger.info(
                        "Peer models configured: %d models — "
                        "child agents restricted to these models",
                        len(peer_models_dict),
                    )

            def _build_engine(local_config: EngineConfig):
                return OrchestrationEngine(
                    config=local_config,
                    plugin_mcp_servers=plugin_mcp_servers,
                    plugin_manager=plugin_manager,
                    provider_registry=provider_registry,
                    initial_allowed_tools=self._persisted_allowed_tools,
                )

            if SUPEREGO_MULTI_PROJECT:
                from prsm.engine.project_registry import ProjectRegistry

                registry_is_external = project_registry is not None
                self._project_registry = (
                    project_registry
                    if isinstance(project_registry, ProjectRegistry)
                    else ProjectRegistry()
                )
                memory_scope = str(ProjectManager.get_memory_path(
                    ProjectManager.get_project_dir(Path(cwd))
                ))
                self._project_registry.register_project(
                    config.project_id or "default",
                    config,
                    bridge=self,
                    memory_scope=memory_scope,
                )
                if registry_is_external:
                    # Keep phase-1 server behavior stable: one engine per session.
                    self._engine = _build_engine(config)
                else:
                    self._engine = self._project_registry.get_or_create_engine(
                        config.project_id or "default",
                        engine_factory=_build_engine,
                    )
            else:
                self._engine = _build_engine(config)

            # Register experts — from YAML config and/or explicit list
            if yaml_config is not None and hasattr(yaml_config, "experts"):
                logger.info(
                    "OrchestratorBridge.configure: registering %d experts from YAML config",
                    len(yaml_config.experts)
                )
                for expert in yaml_config.experts:
                    self._engine.register_expert(expert)
                    logger.debug(
                        "Registered expert: %s (%s)",
                        expert.expert_id, expert.name
                    )

            if experts:
                logger.info(
                    "OrchestratorBridge.configure: registering %d additional experts",
                    len(experts)
                )
                for expert in experts:
                    self._engine.register_expert(expert)
                    logger.debug(
                        "Registered additional expert: %s",
                        getattr(expert, "expert_id", "<unknown>")
                    )

            self._configured = True
            effective_model = getattr(config, "default_model", model)
            effective_provider = getattr(config, "default_provider", "claude")
            logger.info(
                "OrchestratorBridge configured (model=%s provider=%s)",
                effective_model,
                effective_provider,
            )
            return True

        except Exception:
            logger.exception("Failed to configure orchestrator")
            return False

    async def run(self, prompt: str) -> str:
        """Run a prompt through the orchestrator. Call from a @work worker."""
        if not self._configured or self._engine is None:
            raise RuntimeError("Orchestrator not configured")
        if self._running:
            raise RuntimeError("Orchestration already in progress")

        self._run_generation += 1
        my_generation = self._run_generation
        self._running = True
        try:
            # Probe Claude models on first run, but do it in background so
            # first response latency is not blocked by CLI probing.
            if getattr(self, "_model_registry_needs_probe", False):
                self._model_registry_needs_probe = False
                try:
                    mr = getattr(self._engine, "_config", None)
                    mr = getattr(mr, "model_registry", None) if mr else None
                    if mr and hasattr(mr, "probe_claude_models"):
                        asyncio.create_task(self._probe_claude_models_background(mr))
                except Exception as exc:
                    logger.debug("Claude model probe skipped: %s", exc)

            # Model discovery: query CLI tools for available models,
            # optionally update CLIs, and merge new models into
            # ~/.prsm/models.yaml.  Runs as a background task.
            if getattr(self, "_model_discovery_needed", False):
                self._model_discovery_needed = False
                asyncio.create_task(
                    self._discover_models_background(
                        update_clis=getattr(
                            self, "_update_clis_on_discovery", True
                        )
                    )
                )

            result = await self._engine.run(prompt)
            return result
        except Exception:
            logger.exception("Orchestrator run failed")
            raise
        finally:
            # Only clear _running if no newer run has started (prevents
            # a stale run() returning after stop+restart from clobbering
            # the new run's state).
            if self._run_generation == my_generation:
                self._running = False

    async def _probe_claude_models_background(self, model_registry: object) -> None:
        """Run Claude model probing asynchronously after orchestration starts."""
        try:
            probe_changed = await model_registry.probe_claude_models()
            if probe_changed:
                logger.warning(
                    "Claude models not accessible: %s",
                    ", ".join(probe_changed.keys()),
                )
        except Exception as exc:
            logger.debug("Claude model probe skipped: %s", exc)

    async def _discover_models_background(
        self, *, update_clis: bool = True
    ) -> None:
        """Discover available models from CLI tools in the background.

        Updates ``~/.prsm/models.yaml`` with newly found models.
        This runs as a fire-and-forget background task so the first
        orchestration run is not delayed.
        """
        try:
            from prsm.engine.model_discovery import discover_and_update_models

            result = await discover_and_update_models(
                update_clis=update_clis,
                force_overwrite=True,
            )
            if result.discovered_models:
                logger.info(
                    "Model discovery: %d models found%s%s",
                    len(result.discovered_models),
                    f", CLIs updated: {', '.join(result.updated_clis)}"
                    if result.updated_clis else "",
                    ", models.yaml updated" if result.models_yaml_updated else "",
                )
            if result.errors:
                for err in result.errors:
                    logger.warning("Model discovery error: %s", err)
        except Exception as exc:
            logger.debug("Model discovery skipped: %s", exc)

    async def run_continuation(self, prompt: str, master_agent_id: str) -> str:
        """Continue a conversation on an existing master agent.

        Restarts the completed master agent with the follow-up prompt
        instead of spawning a brand-new one. The TUI conversation view
        stays on the same agent, so the user sees a seamless continuation.
        """
        if not self._configured or self._engine is None:
            raise RuntimeError("Orchestrator not configured")
        if self._running:
            raise RuntimeError("Orchestration already in progress")

        self._run_generation += 1
        my_generation = self._run_generation
        self._running = True
        try:
            result = await self._engine.run_continuation(prompt, master_agent_id)
            return result
        except Exception:
            logger.exception("Orchestrator continuation failed")
            raise
        finally:
            if self._run_generation == my_generation:
                self._running = False

    async def shutdown(self) -> None:
        """Shut down the orchestrator engine."""
        if self._engine:
            await self._engine.shutdown()
        self._running = False
        # Cancel all pending futures to prevent leaks
        self._cancel_pending_futures()
        self.event_bus.close()

    async def interrupt(self) -> str | None:
        """Interrupt the current orchestration, keeping master restartable.

        Returns the master agent ID if one was found, or None.
        Unlike shutdown, does not fire engine_finished — the caller
        should immediately start a continuation.
        """
        master_id = None
        if self._engine:
            master_id = await self._engine.interrupt()
        self._running = False
        self._cancel_pending_futures()
        # Don't reset or close the event bus here — agent_killed events
        # for children may still be in the queue and should be consumed
        # by the old event consumer before the new run starts.
        return master_id

    def _cancel_pending_futures(self) -> None:
        """Cancel all pending permission and question futures."""
        for request_id, future in list(self._permission_futures.items()):
            if not future.done():
                future.cancel()
                logger.debug("Cancelled pending permission future: %s", request_id[:8])
        self._permission_futures.clear()

        for request_id, future in list(self._question_futures.items()):
            if not future.done():
                future.cancel()
                logger.debug("Cancelled pending question future: %s", request_id[:8])
        self._question_futures.clear()

    def cancel_agent_futures(self, agent_id: str) -> None:
        """Cancel all pending futures for a specific agent.

        Called when an agent is killed, fails, or times out to prevent
        memory leaks from orphaned futures.
        """
        cancelled_count = 0

        # Cancel permission futures for this agent
        request_ids = self._agent_permission_requests.pop(agent_id, set())
        for request_id in request_ids:
            future = self._permission_futures.pop(request_id, None)
            if future and not future.done():
                future.cancel()
                cancelled_count += 1
                logger.debug(
                    "Cancelled permission future for dead agent: %s request: %s",
                    agent_id[:8],
                    request_id[:8],
                )

        # Cancel question futures for this agent
        request_ids = self._agent_question_requests.pop(agent_id, set())
        for request_id in request_ids:
            future = self._question_futures.pop(request_id, None)
            if future and not future.done():
                future.cancel()
                cancelled_count += 1
                logger.debug(
                    "Cancelled question future for dead agent: %s request: %s",
                    agent_id[:8],
                    request_id[:8],
                )

        if cancelled_count > 0:
            logger.info(
                "Cleaned up %d pending futures for agent %s",
                cancelled_count,
                agent_id[:8],
            )

    # ── Permission handling ──

    async def _handle_permission(
        self, agent_id: str, tool_name: str, arguments: str,
    ) -> str:
        """Called by the engine when an agent wants to use a tool.

        Creates a Future, emits a PermissionRequest event, and awaits
        the Future which will be resolved by the TUI event consumer.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._permission_futures[request_id] = future
        # Track for cleanup
        if agent_id not in self._agent_permission_requests:
            self._agent_permission_requests[agent_id] = set()
        self._agent_permission_requests[agent_id].add(request_id)

        # Get agent name for display
        agent_name = "Unknown"
        node = self.agent_map.get(agent_id)
        if node:
            agent_name = node.name

        await self.event_bus.emit(PermissionRequest(
            agent_id=agent_id,
            request_id=request_id,
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
        ))
        logger.info(
            "Permission request queued agent=%s request_id=%s tool=%s",
            agent_id[:8],
            request_id[:8],
            tool_name,
        )

        # Block until TUI resolves the permission
        try:
            result = await asyncio.wait_for(future, timeout=300.0)
            logger.info(
                "Permission request resolved agent=%s request_id=%s result=%s",
                agent_id[:8],
                request_id[:8],
                result,
            )
            # Persist "always" decisions
            if result == "allow_project" and self._permission_store:
                if self._is_terminal_tool(tool_name):
                    self._persist_command_pattern(arguments, allow=True)
                    return "allow"
                self._permission_store.add_project(tool_name)
                self._persisted_allowed_tools.add(tool_name)
                return "allow_always"  # Engine treats as always-allow
            if result == "allow_global" and self._permission_store:
                self._permission_store.add_global(tool_name)
                self._persisted_allowed_tools.add(tool_name)
                return "allow_always"  # Engine treats as always-allow
            if result == "deny_project":
                if self._is_terminal_tool(tool_name):
                    self._persist_command_pattern(arguments, allow=False)
                return "deny"
            return result
        except asyncio.TimeoutError:
            logger.warning("Permission request timed out, denying")
            return "deny"
        finally:
            self._permission_futures.pop(request_id, None)
            # Clean up tracking
            if agent_id in self._agent_permission_requests:
                self._agent_permission_requests[agent_id].discard(request_id)

    def resolve_permission(self, request_id: str, result: str) -> None:
        """Resolve a pending permission request from the TUI."""
        future = self._permission_futures.get(request_id)
        if future and not future.done():
            future.set_result(result)
            logger.info(
                "Permission future set request_id=%s result=%s",
                request_id[:8],
                result,
            )
        else:
            logger.warning(
                "Permission resolve ignored request_id=%s (missing or already done)",
                request_id[:8],
            )

    @staticmethod
    def _is_terminal_tool(tool_name: str) -> bool:
        normalized = tool_name.lower()
        bare = normalized.split("__")[-1]
        return (
            bare in {"bash", "shell", "terminal"}
            or bare.endswith(".bash")
            or bare.endswith("_bash")
        )

    def _persist_command_pattern(self, arguments: str, *, allow: bool) -> None:
        if not self._command_policy_store:
            return
        command = self._extract_command(arguments)
        if not command:
            return
        pattern = CommandPolicyStore.build_command_pattern(command, allow=allow)
        if not pattern:
            return
        if allow:
            self._command_policy_store.add_whitelist_pattern(pattern)
        else:
            self._command_policy_store.add_blacklist_pattern(pattern)

    @staticmethod
    def _extract_command(arguments: str) -> str:
        parsed: object | None = None
        try:
            parsed = ast.literal_eval(arguments)
        except Exception:
            try:
                parsed = json.loads(arguments)
            except Exception:
                return ""
        if isinstance(parsed, dict):
            value = parsed.get("command") or parsed.get("cmd") or parsed.get("script") or ""
            if isinstance(value, str):
                return value
        return ""

    # ── User question handling ──

    async def _handle_user_question(
        self, agent_id: str, question: str, options: list,
    ) -> str:
        """Called by the engine when an agent asks the user a question.

        Creates a Future, emits a UserQuestionRequest event, and awaits
        the Future which will be resolved by the TUI.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._question_futures[request_id] = future
        # Track for cleanup
        if agent_id not in self._agent_question_requests:
            self._agent_question_requests[agent_id] = set()
        self._agent_question_requests[agent_id].add(request_id)

        agent_name = "Unknown"
        node = self.agent_map.get(agent_id)
        if node:
            agent_name = node.name

        await self.event_bus.emit(UserQuestionRequest(
            agent_id=agent_id,
            request_id=request_id,
            agent_name=agent_name,
            question=question,
            options=options or [],
        ))

        timeout_seconds = self._user_question_timeout_seconds
        if self._engine is not None:
            config_timeout = getattr(
                self._engine._config,
                "user_question_timeout_seconds",
                None,
            )
            if config_timeout is not None:
                try:
                    parsed = float(config_timeout)
                    timeout_seconds = parsed
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid user question timeout '%s'; using default %.1fs",
                        config_timeout,
                        DEFAULT_USER_QUESTION_TIMEOUT_SECONDS,
                    )

        try:
            if timeout_seconds <= 0:
                result = await future
            else:
                result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            logger.warning("User question timed out")
            return "No response (timed out)"
        finally:
            self._question_futures.pop(request_id, None)
            # Clean up tracking
            if agent_id in self._agent_question_requests:
                self._agent_question_requests[agent_id].discard(request_id)

    def resolve_user_question(self, request_id: str, answer: str) -> None:
        """Resolve a pending user question from the TUI."""
        future = self._question_futures.get(request_id)
        if future and not future.done():
            future.set_result(answer)

    # ── Model mapping ──

    def map_agent(
        self,
        agent_id: str,
        parent_id: str | None,
        role: str,
        model: str,
        prompt: str,
        name: str | None = None,
    ) -> AgentNode:
        """Map an orchestrator agent descriptor to a TUI AgentNode."""
        tui_role = _ROLE_MAP.get(role, AgentRole.WORKER)
        resolved_name = name.strip() if isinstance(name, str) else ""
        if not resolved_name:
            resolved_name = _agent_name(role, prompt)

        node = AgentNode(
            id=agent_id,
            name=resolved_name,
            state=AgentState.PENDING,
            role=tui_role,
            model=self.get_model_display_name(model),
            parent_id=parent_id,
        )
        self.agent_map[agent_id] = node
        return node

    def map_state(self, state_str: str) -> AgentState:
        """Map an orchestrator state string to an AgentState."""
        return _STATE_MAP.get(state_str, AgentState.PENDING)

    # ── Model selection ──

    @property
    def current_model(self) -> str:
        """The model currently configured for master agent runs."""
        if self._engine:
            return self._engine._config.master_model
        # Fall back to the configured default from YAML, not a hardcoded
        # Claude model. This matters when Claude CLI is not installed but
        # a non-Claude model was set as default in .prism/prsm.yaml.
        if self._configured_default_model:
            return self._configured_default_model
        return "claude-opus-4-6"

    def get_model_display_name(self, model_id: str) -> str:
        """Resolve a model ID to a concise display name (alias when available)."""
        if not model_id:
            return _fallback_model_display_name(model_id)
        if not self._engine:
            return _fallback_model_display_name(model_id)
        mr = getattr(self._engine._config, "model_registry", None)
        if not mr:
            return _fallback_model_display_name(model_id)
        try:
            strip_runtime = getattr(mr, "_strip_runtime_model_options", None)
            normalized = (
                strip_runtime(model_id)
                if callable(strip_runtime)
                else str(model_id)
            )
            preferred = self._preferred_model_alias
            if preferred:
                resolved_preferred = mr.resolve_alias(preferred)
                if (
                    resolved_preferred == model_id
                    or (
                        callable(strip_runtime)
                        and strip_runtime(resolved_preferred) == normalized
                    )
                ):
                    return _humanize_model_alias(preferred, model_id=model_id)

            aliases = mr.list_aliases()
            matches = [
                alias for alias, target in aliases.items()
                if target == model_id
                or (
                    callable(strip_runtime)
                    and strip_runtime(target) == normalized
                )
            ]
            if matches:
                chosen = sorted(matches, key=_alias_sort_key)[0]
                return _humanize_model_alias(chosen, model_id=model_id)
        except Exception:
            logger.exception("Failed to resolve model display name for %s", model_id)
        return _fallback_model_display_name(model_id)

    def model_display_name(self, model_id: str) -> str:
        """Backward-compatible model display helper."""
        return self.get_model_display_name(model_id)

    @property
    def current_model_display_name(self) -> str:
        """Backward-compatible UI label accessor."""
        return self.get_model_display_name(self.current_model)

    @property
    def current_model_display(self) -> str:
        """User-facing model name, preferring configured aliases when possible."""
        return self.get_model_display_name(self.current_model)

    def get_available_models(self) -> list[dict]:
        """Return a list of available models for the UI model selector.

        Each entry:
        {model_id, provider, tier, available, display_name, is_current}
        + is_legacy when model is an older/oppossed Claude family version.
        """
        if not self._engine:
            return []
        mr = getattr(self._engine._config, "model_registry", None)
        if not mr:
            return []
        try:
            all_models = mr._models.values()
            current = self._engine._config.master_model
            aliases_by_target: dict[str, list[str]] = {}
            list_aliases = getattr(mr, "list_aliases", None)
            if callable(list_aliases):
                for alias, target in list_aliases().items():
                    target_id = str(target)
                    aliases_by_target.setdefault(target_id, []).append(str(alias))
            result: list[dict] = []
            model_ids: set[str] = set()
            for cap in all_models:
                model_id = str(cap.model_id)
                model_ids.add(model_id)
                result.append({
                    "model_id": model_id,
                    "provider": cap.provider,
                    "tier": cap.tier.value if hasattr(cap.tier, "value") else str(cap.tier),
                    "available": cap.available,
                    "is_current": model_id == current,
                    "display_name": _model_display_name(model_id, aliases_by_target),
                    "is_legacy": False,
                })

            canonical_by_family = _canonical_claude_family_model_ids(model_ids)
            for model in result:
                family_info = _claude_family_version_parts(model["model_id"])
                if family_info:
                    family = family_info[0]
                    if model["model_id"] != canonical_by_family.get(family):
                        model["is_legacy"] = True

            # Sort: available first, then by tier (frontier > strong > fast > economy)
            tier_order = {"frontier": 0, "strong": 1, "fast": 2, "economy": 3}
            result.sort(key=lambda m: (
                0 if m["available"] else 1,
                tier_order.get(m["tier"], 9),
                m["provider"],
                m["model_id"],
            ))
            return result
        except Exception:
            logger.exception("Failed to get available models")
            return []

    def set_model(self, model_id: str) -> tuple[str, str]:
        """Change the master model for the next orchestration run.

        Resolves aliases (e.g. "opus" → "claude-opus-4-6") and updates
        the engine config. Also detects the provider from the model
        registry.

        Returns a tuple of (resolved_model_id, provider).
        """
        if not self._engine:
            raise RuntimeError("Orchestrator not configured")

        mr = getattr(self._engine._config, "model_registry", None)
        resolved = model_id
        provider = "claude"

        if mr:
            # Resolve alias
            resolved = mr.resolve_alias(model_id)
            # Detect provider
            cap = mr.get(resolved)
            if cap:
                provider = cap.provider

        self._engine._config.master_model = resolved
        self._engine._config.master_provider = provider
        logger.info(
            "Model switched: %s (provider=%s)", resolved, provider,
        )
        return resolved, provider
