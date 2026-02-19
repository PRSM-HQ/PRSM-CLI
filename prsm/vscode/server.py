"""HTTP + SSE server for the PRSM orchestrator.

Exposes a multi-session REST API with Server-Sent Events for real-time
agent tree updates. Each session manages its own OrchestratorBridge and
can run orchestrations concurrently.

Usage:
    prsm --server [--port PORT]
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import MessageRole, ToolCall
from prsm.shared.models.session import (
    FORKED_PREFIX,
    Session,
    WorktreeMetadata,
    format_forked_name,
    is_default_session_name,
)
from prsm.adapters.events import (
    OrchestratorEvent,
    AgentSpawned,
    AgentRestarted,
    AgentStateChanged,
    AgentKilled,
    StreamChunk,
    ToolCallStarted,
    ToolCallCompleted,
    ToolCallDelta,
    AgentResult,
    PermissionRequest,
    UserQuestionRequest,
    EngineStarted,
    EngineFinished,
    Thinking,
    UserPrompt,
    event_to_dict,
)
from prsm.adapters.orchestrator import OrchestratorBridge, _strip_prompt_prefix
from prsm.adapters.file_tracker import FileChangeRecord, FileChangeTracker, normalize_tool_name
from prsm.shared.commands import COMMAND_HELP, parse_command
from prsm.shared.services.persistence import SessionPersistence
from prsm.shared.services.command_policy_store import CommandPolicyStore
from prsm.shared.services.project_memory import ProjectMemory
from prsm.shared.services.preferences import UserPreferences
from prsm.shared.services.project import ProjectManager

logger = logging.getLogger(__name__)


# ── Session state ──


@dataclass
class SessionState:
    """State for a single orchestration session."""

    session_id: str
    name: str
    project_id: str
    bridge: OrchestratorBridge
    session: Session
    summary: str | None = None
    forked_from: str | None = None
    file_tracker: FileChangeTracker = field(default_factory=FileChangeTracker)
    _event_task: asyncio.Task | None = field(default=None, repr=False)
    _pending_user_prompt: str | None = None
    _pending_user_prompt_snapshot_id: str | None = None
    _stop_after_tool: bool = False
    # Per-agent inject: agent_id → (display_prompt, resolved_prompt)
    _inject_after_tool_agents: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Per-agent queue: agent_id → list of (display_prompt, resolved_prompt)
    _agent_prompt_queue: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # Agents restarted directly via _handle_agent_message (not full orchestration).
    _directly_restarted_agents: set[str] = field(default_factory=set)
    # Most recent snapshot ID for snapshot lineage (parent linkage).
    _last_snapshot_id: str | None = None
    # Last time this session handled API or event activity.
    _last_touched_at: float = field(default_factory=time.time)
    # Current plan file path emitted to clients for this session.
    _plan_file_path: str | None = None
    _workspace_root: str | None = None
    _worktree_path: str | None = None
    _file_sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _pre_tool_workspace_paths: dict[str, set[str]] = field(default_factory=dict)
    # Monotonically increasing generation counter for orchestration runs.
    # Incremented at the start of each _run_orchestration call so that
    # stale engine_finished events from a previous (stopped) run can be
    # detected and discarded.
    _run_generation: int = 0
    # Phase 8 governance observability caches.
    _governance_events: list[dict[str, Any]] = field(default_factory=list)
    _memory_entries: list[dict[str, Any]] = field(default_factory=list)


# ── Server ──


class PrsmServer:
    """Multi-session HTTP + SSE server wrapping OrchestratorBridge.

    Thin adapter: all orchestration state lives in OrchestratorBridge and
    Session. This class only handles HTTP routing, SSE fan-out, and event
    dispatch.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        cwd: str | None = None,
        model: str = "claude-opus-4-6",
        config_path: str | None = None,
        claude_preflight_enabled: bool = False,
        claude_preflight_ok: bool | None = None,
        claude_preflight_detail: str | None = None,
        claude_preflight_checked_at: float | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._cwd = cwd or str(Path.cwd())
        self._model = model
        self._config_path = config_path
        self._claude_preflight_enabled = claude_preflight_enabled
        self._claude_preflight_ok = claude_preflight_ok
        self._claude_preflight_detail = claude_preflight_detail
        self._claude_preflight_checked_at = claude_preflight_checked_at
        self._sessions: dict[str, SessionState] = {}
        # Cold session cache loaded from disk (no bridge/event task).
        self._cold_sessions: dict[str, Session] = {}
        self._cold_last_touched_at: dict[str, float] = {}
        # Metadata index for lazily loaded sessions (session_id -> saved metadata).
        self._session_index: dict[str, dict[str, Any]] = {}
        self._sse_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._started_at = time.time()
        self._app = web.Application(middlewares=[self._request_logging_middleware])
        self._persistence = SessionPersistence(cwd=Path(self._cwd))
        self._autosave_task: asyncio.Task | None = None
        self._file_index: Any = None
        self._plan_migration_ran = False
        self._plan_index_counter: int = 0  # Next plan index (sequential, 1-based)
        inactivity_minutes_raw = os.getenv(
            "PRSM_SESSION_INACTIVITY_MINUTES",
            "15",
        )
        try:
            inactivity_minutes = float(inactivity_minutes_raw)
        except ValueError:
            inactivity_minutes = 15.0
        # <= 0 disables inactivity unloading.
        self._session_inactivity_seconds = max(0.0, inactivity_minutes * 60.0)
        self._setup_routes()
        logger.info(
            "PrsmServer init host=%s port=%s cwd=%s model=%s config=%s pid=%s",
            self._host, self._port, self._cwd, self._model,
            self._config_path or "<none>", os.getpid(),
        )
        logger.info(
            "Session inactivity timeout: %.1f minutes",
            self._session_inactivity_seconds / 60.0
            if self._session_inactivity_seconds > 0
            else 0.0,
        )

        self._yaml_config = None
        self._project_registry = None
        self._project_id = ProjectManager.get_repo_identity(Path(self._cwd))
        self._default_project_id = self._project_id
        self._multi_project_enabled = os.getenv("PRSM_MULTI_PROJECT", "0") == "1"
        self._known_projects: dict[str, dict[str, Any]] = {
            self._project_id: {"project_id": self._project_id, "label": self._project_id}
        }
        self._session_projects: dict[str, str] = {}
        self._provider_registry = None
        self._model_registry = None
        self._config_path = config_path
        resolved_config_path = self._resolve_initial_config_path(config_path)
        if resolved_config_path:
            config_path = str(resolved_config_path)
            self._config_path = config_path
            logger.info(
                "PrsmServer: attempting to load YAML config from %s",
                config_path
            )
            try:
                from prsm.engine.yaml_config import load_yaml_config
                self._yaml_config = load_yaml_config(config_path)
                logger.info(
                    "PrsmServer: successfully loaded YAML config from %s "
                    "(providers=%d, models=%d, experts=%d)",
                    config_path,
                    len(self._yaml_config.providers),
                    len(self._yaml_config.models),
                    len(self._yaml_config.experts),
                )
            except FileNotFoundError:
                logger.error(
                    "PrsmServer: config file not found at %s",
                    config_path
                )
            except Exception as exc:
                logger.exception(
                    "PrsmServer: failed to load YAML config from %s: %s",
                    config_path, exc
                )
        else:
            logger.info(
                "PrsmServer: no resolvable config path found, using defaults"
            )

        if (
            self._yaml_config is not None
            and getattr(self._yaml_config, "projects", None)
        ):
            for proj_id, proj_cfg in self._yaml_config.projects.items():
                if not proj_id:
                    continue
                cfg_dict = proj_cfg if isinstance(proj_cfg, dict) else {}
                self._known_projects[str(proj_id)] = {
                    "project_id": str(proj_id),
                    "label": str(cfg_dict.get("name") or proj_id),
                    "cwd": str(cfg_dict.get("cwd") or self._cwd),
                }

        if self._multi_project_enabled:
            try:
                from prsm.engine.project_registry import ProjectRegistry
                from prsm.engine.config import EngineConfig

                self._project_registry = ProjectRegistry()
                for proj_id, proj_meta in self._known_projects.items():
                    runtime_config = (
                        deepcopy(self._yaml_config.engine)
                        if self._yaml_config is not None
                        else EngineConfig(
                            default_model=self._model,
                            default_cwd=str(proj_meta.get("cwd") or self._cwd),
                        )
                    )
                    runtime_config.project_id = proj_id
                    runtime_config.default_cwd = str(
                        proj_meta.get("cwd") or runtime_config.default_cwd or self._cwd
                    )
                    self._project_registry.register_project(
                        project_id=proj_id,
                        config=runtime_config,
                        memory_scope=str(
                            ProjectManager.get_memory_path(
                                ProjectManager.get_project_dir(
                                    Path(str(proj_meta.get("cwd") or self._cwd))
                                )
                            )
                        ),
                    )
            except Exception:
                logger.warning(
                    "Failed to initialize project registry; continuing in single-project mode",
                    exc_info=True,
                )

        self._build_registries()

    def _resolve_initial_config_path(self, explicit_config_path: str | None) -> Path | None:
        """Resolve an initial config path for server bootstrap.

        Fallback order:
        1. Explicit path (if provided and exists).
        2. Workspace ``.prism/prsm.yaml``.
        3. Workspace ``prsm.yaml``.
        4. Global ``~/.prsm/models.yaml``.
        """
        if explicit_config_path:
            candidate = Path(explicit_config_path).expanduser()
            if candidate.exists():
                return candidate
            logger.warning(
                "PrsmServer: explicit config path %s does not exist; checking fallbacks",
                candidate,
            )

        candidates = [
            Path(self._cwd) / ".prism" / "prsm.yaml",
            Path(self._cwd) / "prsm.yaml",
            Path.home() / ".prsm" / "models.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    # ── Bridge configuration (single source of truth) ──

    def _configure_bridge(
        self,
        bridge: OrchestratorBridge,
        *,
        project_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        """Configure a bridge with current server settings."""
        resolved_project_id = project_id or self._default_project_id
        workspace_dir = self._workspace_dir_for_project(resolved_project_id)
        resolved_cwd = cwd or str(workspace_dir)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "cwd": resolved_cwd,
            "project_dir": workspace_dir,
            "project_registry": self._project_registry,
            "project_id": resolved_project_id,
        }
        if self._yaml_config:
            kwargs["yaml_config"] = self._yaml_config
        bridge.configure(**kwargs)

    def _build_registries(self) -> None:
        """Build ProviderRegistry and ModelRegistry from config.

        Uses the same logic as OrchestratorBridge.configure() so the
        settings panel shows the exact same provider/model data that
        the engine uses at runtime.
        """
        try:
            from prsm.engine.providers.registry import build_provider_registry
            from prsm.engine.model_registry import (
                build_default_registry,
                load_model_registry_from_yaml,
            )

            provider_configs = (
                self._yaml_config.providers if self._yaml_config else None
            )
            self._provider_registry = build_provider_registry(provider_configs)

            self._model_registry = build_default_registry()
            if (
                self._yaml_config
                and getattr(self._yaml_config, "model_registry_raw", None)
            ):
                yaml_models = getattr(self._yaml_config, "models", None)
                self._model_registry = load_model_registry_from_yaml(
                    self._yaml_config.model_registry_raw,
                    self._model_registry,
                    model_aliases=yaml_models,
                )
            # Sync model availability with provider availability
            if self._provider_registry:
                avail = self._provider_registry.get_availability_report()
                self._model_registry.sync_availability(avail)
            logger.info(
                "Built registries: %d providers, %d models",
                self._provider_registry.count if self._provider_registry else 0,
                len(self._model_registry.list_models()) if self._model_registry else 0,
            )
        except Exception:
            logger.exception("Failed to build provider/model registries")

    @staticmethod
    def _normalize_model_alias_id(value: str) -> str:
        raw = (value or "").split("::reasoning_effort=", 1)[0].strip().lower()
        if not raw:
            return ""
        normalized = raw.replace("_", "-").replace(".", "-")
        tokens = [token for token in normalized.split("-") if token]
        if len(tokens) > 1 and re.fullmatch(r"\d{6,8}", tokens[-1] or ""):
            tokens.pop()
        return "-".join(tokens)

    @staticmethod
    def _has_alpha_numeric_digit(value: str) -> bool:
        for char in value:
            code = ord(char)
            if 48 <= code <= 57:
                return True
        return False

    @staticmethod
    def _is_technical_alias(alias: str, model_id: str | None) -> bool:
        normalized = (alias or "").strip().lower()
        if not normalized:
            return False

        if "::reasoning_effort=" in normalized:
            return True

        normalized_technical = normalized.replace("_", "-").replace(".", "-")
        if not re.fullmatch(r"[a-z0-9-]+", normalized_technical):
            return False

        has_upper = any("A" <= ch <= "Z" for ch in alias)
        has_space = any(ord(ch) <= 32 for ch in alias)
        if has_upper or has_space:
            return False

        normalized_model_id = PrsmServer._normalize_model_alias_id(model_id or alias)
        if not normalized_model_id:
            return False

        technical_prefixes = {
            "opus",
            "sonnet",
            "haiku",
            "claude",
            "gpt",
            "gemini",
            "minimax",
            "spark",
            "codex",
        }

        if normalized_technical in technical_prefixes:
            return True
        if normalized == normalized_model_id:
            return True
        if normalized.startswith(f"{normalized_model_id}-") and re.search(
            r"-(?:low|medium|high)$",
            normalized,
        ):
            return True
        if not PrsmServer._has_alpha_numeric_digit(normalized):
            return any(
                normalized_technical.startswith(f"{prefix}-")
                for prefix in technical_prefixes
            )
        return any(
            normalized_technical.startswith(f"{prefix}-")
            for prefix in technical_prefixes
        )

    @staticmethod
    def _alias_display_score(alias: str) -> int:
        score = 0
        if " " in alias:
            score += 10
        if any("A" <= ch <= "Z" for ch in alias):
            score += 5
        if "-" not in alias:
            score += 3
        if "(" in alias:
            score += 2
        if "::" not in alias:
            score += 1
        return score

    def _build_runtime_alias_map(
        self,
        model_aliases: dict[str, dict[str, str]],
        detected_models: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        if not model_aliases:
            return {}

        detected_model_ids: set[str] = set()
        for model_id in detected_models:
            normalized_model_id = self._normalize_model_alias_id(model_id)
            if normalized_model_id:
                detected_model_ids.add(normalized_model_id)

        selected: dict[str, tuple[str, dict[str, str], int]] = {}
        for alias, cfg in model_aliases.items():
            if not isinstance(alias, str) or not alias.strip():
                continue
            if not isinstance(cfg, dict):
                cfg = {"provider": "unknown", "model_id": str(alias)}

            provider = cfg.get("provider", "unknown")
            model_id = str(cfg.get("model_id", alias))
            if not model_id:
                continue

            normalized_model_id = self._normalize_model_alias_id(model_id)
            if not normalized_model_id or normalized_model_id not in detected_model_ids:
                continue
            if self._is_technical_alias(alias, model_id):
                continue

            score = self._alias_display_score(alias)
            existing = selected.get(normalized_model_id)
            if existing is None or score > existing[2]:
                selected[normalized_model_id] = (
                    alias,
                    {
                        "provider": str(provider),
                        "model_id": model_id,
                        **(
                            {"reasoning_effort": cfg["reasoning_effort"]}
                            if isinstance(cfg.get("reasoning_effort"), str)
                            and cfg["reasoning_effort"]
                            else {}
                        ),
                    },
                    score,
                )

        return {
            alias: info
            for alias, info, _ in selected.values()
        }

    def _get_runtime_info(self) -> dict[str, Any]:
        """Return detected providers and models from the real registries.

        This is the SAME data the engine uses at runtime — not a
        separate shutil.which() reimplementation.
        """
        detected_providers: dict[str, dict[str, Any]] = {}
        detected_models: dict[str, dict[str, Any]] = {}

        if self._provider_registry:
            for name in self._provider_registry.list_names():
                provider = self._provider_registry.get(name)
                if provider is None:
                    continue
                detected_providers[name] = {
                    "type": provider.name,
                    "available": provider.is_available(),
                    "supports_master": provider.supports_master,
                }

        if self._model_registry:
            for cap in self._model_registry.list_models():
                detected_models[cap.model_id] = {
                    "provider": cap.provider,
                    "model_id": cap.model_id,
                    "tier": cap.tier.value if hasattr(cap.tier, "value") else str(cap.tier),
                    "available": cap.available,
                    "cost_factor": cap.cost_factor,
                    "speed_factor": cap.speed_factor,
                    "affinities": cap.affinities,
                }

        # Also build model aliases from YAML config
        model_aliases: dict[str, dict[str, str]] = {}
        if self._yaml_config and hasattr(self._yaml_config, "models"):
            configured_models = getattr(self._yaml_config, "models")
            if isinstance(configured_models, dict):
                for alias, mcfg in configured_models.items():
                    if not alias or not isinstance(alias, str):
                        continue
                    try:
                        provider = mcfg.provider
                        model_id = mcfg.model_id
                        reasoning_effort = getattr(mcfg, "reasoning_effort", None)
                    except Exception:
                        if not isinstance(mcfg, dict):
                            continue
                        provider = mcfg.get("provider")
                        model_id = mcfg.get("model_id")
                        reasoning_effort = mcfg.get("reasoning_effort")
                    if not model_id:
                        continue
                    model_aliases[alias] = {
                        "provider": str(provider) if provider else "unknown",
                        "model_id": str(model_id),
                    }
                    if reasoning_effort:
                        model_aliases[alias]["reasoning_effort"] = str(reasoning_effort)

        # Fallback: derive aliases from discovered registry entries so the
        # settings UI stays populated even when YAML models are temporarily
        # unavailable or discovery just ran and config has not been reloaded yet.
        if self._model_registry is not None:
            registry_aliases = self._model_registry.list_aliases()
            if isinstance(registry_aliases, dict):
                for alias, model_id in registry_aliases.items():
                    if not isinstance(alias, str) or not isinstance(model_id, str):
                        continue
                    if alias in model_aliases:
                        continue
                    cap = self._model_registry.get(model_id)
                    model_aliases[alias] = {
                        "provider": cap.provider if cap is not None else "unknown",
                        "model_id": model_id,
                    }

            for cap in self._model_registry.list_models():
                if cap.model_id not in model_aliases:
                    model_aliases[cap.model_id] = {
                        "provider": cap.provider,
                        "model_id": cap.model_id,
                    }

            # Last resort: if registry is present but still no aliases for
            # any reason (e.g., alias normalization produced collisions),
            # expose all detected runtime models directly.
            if not model_aliases:
                for model_id, model_info in detected_models.items():
                    provider = model_info.get("provider", "unknown")
                    model_aliases[model_id] = {
                        "provider": str(provider),
                        "model_id": model_id,
                    }

        model_aliases = self._build_runtime_alias_map(model_aliases, detected_models)

        return {
            "providers": detected_providers,
            "models": detected_models,
            "model_aliases": model_aliases,
        }

    def _get_snapshot_service(self, state: SessionState | None = None):
        """Create a SnapshotService for the current project.

        If *state* is provided and has a worktree, the service captures diffs
        against the worktree (isolating per-session changes).
        """
        from prsm.shared.services.snapshot import SnapshotService
        cwd = Path(self._cwd)
        if state and state._worktree_path:
            cwd = Path(state._worktree_path)
        return SnapshotService(
            self._persistence.project_dir or Path.home() / ".prsm",
            cwd,
        )

    def _latest_snapshot_for_session(self, session_id: str) -> str | None:
        """Return the most recent snapshot ID for a session, if any."""
        try:
            snapshots = self._get_snapshot_service().list_snapshots_by_session(session_id)
            if not snapshots:
                return None
            snapshots.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
            return snapshots[0].get("snapshot_id")
        except Exception:
            logger.debug(
                "Failed to resolve latest snapshot for session %s",
                session_id,
                exc_info=True,
            )
            return None

    def _create_snapshot(
        self,
        state: SessionState,
        description: str,
        *,
        agent_id: str | None = None,
    ) -> str:
        """Create snapshot and broadcast metadata."""
        snapshot_agent_id = agent_id
        if not snapshot_agent_id:
            snapshot_agent_id = (
                state.session.active_agent_id
                or self._get_master_agent_id(state)
            )
        snapshot_agent = (
            state.session.agents.get(snapshot_agent_id)
            if snapshot_agent_id
            else None
        )
        snapshot_id = self._get_snapshot_service(state).create(
            state.session,
            state.name,
            description,
            file_tracker=state.file_tracker,
            session_id=state.session_id,
            parent_snapshot_id=state._last_snapshot_id,
            agent_id=snapshot_agent_id,
            agent_name=snapshot_agent.name if snapshot_agent else None,
            parent_agent_id=snapshot_agent.parent_id if snapshot_agent else None,
        )
        state._last_snapshot_id = snapshot_id
        meta = self._get_snapshot_service().get_meta(snapshot_id)
        self._broadcast_sse("snapshot_created", {
            "session_id": state.session_id,
            "snapshot_id": snapshot_id,
            "description": meta.get("description", description),
            "timestamp": meta.get("timestamp"),
            "git_branch": meta.get("git_branch"),
            "parent_snapshot_id": meta.get("parent_snapshot_id"),
            "agent_id": meta.get("agent_id"),
            "agent_name": meta.get("agent_name"),
            "parent_agent_id": meta.get("parent_agent_id"),
        })
        # Eagerly persist session index/state alongside snapshot creation so
        # restart/reload recovery does not rely on graceful shutdown/autosave.
        self._save_session(state)
        return snapshot_id

    # ── Middleware ──

    @web.middleware
    async def _request_logging_middleware(self, request: web.Request, handler) -> web.StreamResponse:
        req_id = request.headers.get("x-prsm-request-id", str(uuid.uuid4())[:8])
        request["req_id"] = req_id
        start = time.monotonic()
        logger.info("HTTP %s %s req=%s from=%s", request.method, request.path_qs, req_id, request.remote)
        try:
            response = await handler(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "HTTP %s %s req=%s status=%s duration_ms=%.1f",
                request.method, request.path_qs, req_id,
                getattr(response, "status", "?"), elapsed_ms,
            )
            return response
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("HTTP %s %s req=%s failed duration_ms=%.1f", request.method, request.path_qs, req_id, elapsed_ms)
            raise

    # ── Route setup ──

    def _setup_routes(self) -> None:
        r = self._app.router
        r.add_get("/health", self._handle_health)
        r.add_get("/events", self._handle_sse)
        r.add_get("/projects", self._handle_list_projects)
        r.add_get("/projects/events", self._handle_list_project_events)
        r.add_post("/projects/{project_id}/subscriptions", self._handle_subscribe_project)
        r.add_delete("/projects/{project_id}/subscriptions", self._handle_unsubscribe_project)
        r.add_post("/projects/{project_id}/events", self._handle_publish_project_event)
        # Session CRUD
        r.add_get("/sessions", self._handle_list_sessions)
        r.add_post("/sessions", self._handle_create_session)
        r.add_post("/sessions/{id}/fork", self._handle_fork_session)
        r.add_delete("/sessions/{id}", self._handle_remove_session)
        r.add_patch("/sessions/{id}", self._handle_rename_session)
        # Session data
        r.add_get("/sessions/{id}/agents", self._handle_get_agents)
        r.add_get("/sessions/{id}/agents/{agent_id}/messages", self._handle_get_messages)
        # Transcript import + slash command execution
        r.add_get("/import/sessions", self._handle_import_list)
        r.add_get("/import/preview", self._handle_import_preview)
        r.add_post("/sessions/{id}/import", self._handle_import_run)
        r.add_post("/sessions/{id}/import-all", self._handle_import_all)
        r.add_post("/sessions/{id}/command", self._handle_command)
        # Orchestration
        r.add_post("/sessions/{id}/run", self._handle_run)
        r.add_post("/sessions/{id}/resolve-permission", self._handle_resolve_permission)
        r.add_post("/sessions/{id}/resolve-question", self._handle_resolve_question)
        r.add_post("/sessions/{id}/agents/{agent_id}/message", self._handle_agent_message)
        r.add_post("/sessions/{id}/kill-agent", self._handle_kill_agent)
        r.add_post("/sessions/{id}/shutdown", self._handle_shutdown_session)
        r.add_post("/sessions/{id}/cancel-latest-tool-call", self._handle_cancel_latest_tool_call)
        r.add_post("/sessions/{id}/stop-after-tool", self._handle_stop_after_tool)
        r.add_post("/sessions/{id}/agents/{agent_id}/inject-prompt", self._handle_inject_prompt)
        # Persistence + snapshots
        r.add_post("/sessions/{id}/save", self._handle_save_session)
        r.add_get("/sessions/restore", self._handle_list_restorable)
        r.add_post("/sessions/restore/{name}", self._handle_restore_session)
        r.add_get("/sessions/{id}/snapshots", self._handle_list_snapshots)
        r.add_post("/sessions/{id}/snapshots", self._handle_create_snapshot)
        r.add_post("/sessions/{id}/snapshots/{snap_id}/restore", self._handle_restore_snapshot)
        r.add_post("/sessions/{id}/snapshots/{snap_id}/fork", self._handle_fork_snapshot)
        r.add_delete("/sessions/{id}/snapshots/{snap_id}", self._handle_delete_snapshot)
        # File tracking + completion
        r.add_get("/sessions/{id}/file-changes", self._handle_get_file_changes)
        r.add_post("/sessions/{id}/file-changes/{tool_call_id}/accept", self._handle_accept_change)
        r.add_post("/sessions/{id}/file-changes/{tool_call_id}/reject", self._handle_reject_change)
        r.add_post("/sessions/{id}/file-changes/accept-all", self._handle_accept_all_changes)
        r.add_post("/sessions/{id}/file-changes/reject-all", self._handle_reject_all_changes)
        r.add_get("/files/complete", self._handle_file_complete)
        # Agent history
        r.add_get("/api/sessions/{id}/agents/{agent_id}/history", self._handle_get_agent_history)
        r.add_get("/api/sessions/{id}/agents/{agent_id}/tool-rationale/{tool_call_id}", self._handle_get_tool_rationale)
        # Model selection
        r.add_post("/sessions/{id}/model", self._handle_set_model)
        r.add_get("/sessions/{id}/models", self._handle_get_available_models)
        # Configuration
        r.add_get("/config", self._handle_get_config)
        r.add_put("/config", self._handle_update_config)
        r.add_get("/config/preferences", self._handle_get_preferences)
        r.add_put("/config/preferences", self._handle_update_preferences)
        r.add_get("/config/thinking-verbs", self._handle_get_thinking_verbs)
        r.add_get("/config/detect-providers", self._handle_detect_providers)
        # Phase 8 parity + governance endpoints
        r.add_get("/sessions/{id}/command-policy", self._handle_get_command_policy)
        r.add_put("/sessions/{id}/command-policy", self._handle_update_command_policy)
        r.add_get("/sessions/{id}/project-memory", self._handle_get_project_memory)
        r.add_put("/sessions/{id}/project-memory", self._handle_update_project_memory)
        r.add_get("/sessions/{id}/policy", self._handle_get_policy)
        r.add_get("/sessions/{id}/leases", self._handle_get_leases)
        r.add_get("/sessions/{id}/audit", self._handle_get_audit)
        r.add_get("/sessions/{id}/memory", self._handle_get_memory)
        r.add_post("/sessions/{id}/memory", self._handle_add_memory)
        r.add_get("/sessions/{id}/experts/stats", self._handle_get_expert_stats)
        r.add_get("/sessions/{id}/budget", self._handle_get_budget)
        r.add_get("/sessions/{id}/decisions", self._handle_get_decisions)
        r.add_post("/sessions/{id}/telemetry/export", self._handle_export_telemetry)
        # Archive import / export
        r.add_post("/archive/import", self._handle_archive_import)
        r.add_post("/archive/export", self._handle_archive_export)
        r.add_post("/archive/preview", self._handle_archive_preview)

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the server and print the port to stdout."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()

        actual_port = self._resolve_port(site, runner)
        if actual_port is None:
            raise RuntimeError("PRSM server started but no listening socket was reported.")
        self._port = actual_port

        sys.stdout.write(json.dumps({"port": actual_port}) + "\n")
        sys.stdout.flush()
        logger.info("PRSM server listening on %s:%d", self._host, actual_port)

        self._restore_saved_sessions()
        self._prune_orphaned_worktrees()
        self._autosave_task = asyncio.create_task(self._autosave_loop())

        # Model discovery: query installed CLIs for their supported models,
        # optionally update the CLIs first, and merge new models into
        # ~/.prsm/models.yaml.  Runs as a background task so server
        # startup is not blocked.  Controlled by PRSM_MODEL_DISCOVERY env
        # var (defaults to "1" / enabled).
        discovery_enabled = os.getenv(
            "PRSM_MODEL_DISCOVERY", "1"
        ).lower() in {"1", "true", "yes", "on"}
        update_clis_enabled = os.getenv(
            "PRSM_UPDATE_CLIS", "1"
        ).lower() in {"1", "true", "yes", "on"}
        if discovery_enabled:
            asyncio.create_task(
                self._discover_models_background(
                    update_clis=update_clis_enabled,
                    models_yaml_path=Path(self._resolve_models_yaml_path()),
                )
            )

        # Optional startup probe (disabled by default to avoid spawning Claude
        # subprocesses before the user prompts an agent).
        probe_on_startup = os.getenv("PRSM_PROBE_CLAUDE_MODELS_ON_STARTUP", "0").lower() in {
            "1", "true", "yes", "on",
        }
        if self._model_registry and probe_on_startup:
            asyncio.create_task(self._probe_claude_models_background())

        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Server shutting down")
        finally:
            if self._autosave_task:
                self._autosave_task.cancel()
            for state in self._sessions.values():
                self._save_session(state)
                await state.bridge.shutdown()
            await runner.cleanup()

    async def _discover_models_background(
        self,
        *,
        update_clis: bool = True,
        models_yaml_path: Path | None = None,
    ) -> None:
        """Discover available models from CLI tools and update models.yaml.

        Runs as a background task so server startup is not delayed.
        After discovery, rebuilds the model registry so that newly
        discovered models are available immediately.
        """
        try:
            from prsm.engine.model_discovery import discover_and_update_models

            if models_yaml_path is None:
                models_yaml_path = Path(self._resolve_models_yaml_path())

            result = await discover_and_update_models(
                update_clis=update_clis,
                models_yaml_path=models_yaml_path,
                force_overwrite=True,
                sync_global_models_yaml=True,
            )
            if result.discovered_models:
                logger.info(
                    "Model discovery found %d models (%d new in YAML)%s",
                    len(result.discovered_models),
                    sum(1 for _ in result.discovered_models),
                    f", updated CLIs: {', '.join(result.updated_clis)}"
                    if result.updated_clis else "",
                )
            if result.errors:
                for err in result.errors:
                    logger.warning("Model discovery error: %s", err)

            # If models.yaml was updated, rebuild registries so the
            # engine uses the new models immediately.
            if result.models_yaml_updated:
                logger.info(
                    "models.yaml was updated — rebuilding registries"
                )
                if self._yaml_config:
                    try:
                        from prsm.engine.yaml_config import load_yaml_config
                        # Reload the YAML config to pick up new models
                        config_path = self._resolve_config_path()
                        if config_path:
                            self._yaml_config = load_yaml_config(config_path)
                    except Exception as exc:
                        logger.debug(
                            "Could not reload YAML config: %s", exc
                        )
                self._build_registries()
        except Exception as exc:
            logger.warning(
                "Model discovery failed (non-fatal): %s", exc,
                exc_info=True,
            )

    async def _probe_claude_models_background(self) -> None:
        """Probe Claude models to check account-level access.

        Runs as a background task so server startup isn't delayed.
        Marks models as unavailable if the user's plan doesn't include them.
        """
        try:
            changed = await self._model_registry.probe_claude_models()
            if changed:
                unavail = [m for m, ok in changed.items() if not ok]
                if unavail:
                    logger.warning(
                        "Claude models not accessible on this account: %s",
                        ", ".join(unavail),
                    )
        except Exception as exc:
            logger.warning(
                "Claude model probe failed (non-fatal): %s", exc,
            )

    @staticmethod
    def _resolve_port(site, runner) -> int | None:
        sockets = getattr(getattr(site, "_server", None), "sockets", None) or ()
        if sockets:
            return sockets[0].getsockname()[1]
        addresses = getattr(runner, "addresses", None) or ()
        if addresses:
            first = addresses[0]
            if isinstance(first, tuple) and len(first) >= 2:
                return int(first[1])
        return None

    # ── @ reference resolution ──

    def _resolve_at_references(self, prompt: str) -> str:
        """Resolve @file and @directory references in a prompt."""
        try:
            from prsm.shared.file_utils import resolve_references
            resolved_text, attachments = resolve_references(prompt, Path(self._cwd))
            if attachments:
                parts = []
                for att in attachments:
                    tag = "directory" if att.is_directory else "file"
                    warning = " (truncated)" if att.truncated else ""
                    parts.append(f'<{tag} path="{att.path}"{warning}>\n{att.content}\n</{tag}>')
                return resolved_text + "\n\n" + "\n\n".join(parts)
            return prompt
        except Exception:
            logger.warning("Failed to resolve @references, using original prompt", exc_info=True)
            return prompt

    # ── SSE fan-out ──

    def _broadcast_sse(self, event_type: str, data: dict[str, Any]) -> None:
        payload = dict(data)
        if "project_id" not in payload:
            session_id = payload.get("session_id")
            if isinstance(session_id, str):
                project_id = self._session_projects.get(session_id)
                if project_id:
                    payload["project_id"] = project_id
        msg = {"event": event_type, "data": payload}
        for queue in self._sse_queues:
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                logger.warning("SSE queue full, dropping event")

    # ── Session management ──

    def _touch_session(self, state: SessionState) -> None:
        state._last_touched_at = time.time()

    def _resolve_project_id(self, requested: str | None) -> str:
        if requested and requested in self._known_projects:
            return requested
        return self._default_project_id

    def _workspace_dir_for_project(self, project_id: str) -> Path:
        meta = self._known_projects.get(project_id, {})
        cwd = str(meta.get("cwd") or self._cwd)
        return Path(cwd)

    def _policy_store_for_state(self, state: SessionState) -> CommandPolicyStore:
        return CommandPolicyStore(self._workspace_dir_for_project(state.project_id))

    def _project_memory_for_state(self, state: SessionState) -> ProjectMemory:
        project_dir = ProjectManager.get_project_dir(
            self._workspace_dir_for_project(state.project_id)
        )
        memory_path = ProjectManager.get_memory_path(project_dir)
        return ProjectMemory(memory_path)

    def _last_activity_for_session(self, session: Session) -> datetime | None:
        last_activity = session.created_at
        if last_activity and last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)
        for agent in session.agents.values():
            agent_ts = agent.last_active or agent.completed_at or agent.created_at
            if agent_ts and agent_ts.tzinfo is None:
                agent_ts = agent_ts.replace(tzinfo=timezone.utc)
            if agent_ts and (not last_activity or agent_ts > last_activity):
                last_activity = agent_ts
        return last_activity

    def _state_to_session_summary(self, state: SessionState) -> dict[str, Any]:
        last_activity = self._last_activity_for_session(state.session)
        return {
            "sessionId": state.session_id,
            "name": state.name or state.session_id,
            "summary": state.summary,
            "projectId": state.project_id,
            "forkedFrom": state.forked_from,
            "agentCount": len(state.session.agents),
            "messageCount": state.session.message_count,
            "running": state.bridge.running,
            "createdAt": state.session.created_at.isoformat() if state.session.created_at else None,
            "lastActivity": last_activity.isoformat() if last_activity else None,
            "currentModel": state.bridge.current_model,
            "currentModelDisplay": state.bridge.current_model_display,
        }

    def _touch_cold_session(self, session_id: str) -> None:
        self._cold_last_touched_at[session_id] = time.time()

    def _get_cold_session(self, session_id: str) -> Session | None:
        """Return a disk-backed session without creating runtime bridge state."""
        session = self._cold_sessions.get(session_id)
        if session is not None:
            self._touch_cold_session(session_id)
            return session

        meta = self._session_index.get(session_id)
        file_stem = meta.get("file_stem") if meta else session_id
        try:
            loaded, _ = self._persistence.load_with_meta(str(file_stem))
        except Exception:
            return None
        self._cold_sessions[session_id] = loaded
        self._touch_cold_session(session_id)
        return loaded

    def _build_state(
        self,
        *,
        session_id: str,
        name: str,
        session: Session,
        project_id: str,
        summary: str | None = None,
        forked_from: str | None = None,
    ) -> SessionState:
        bridge = OrchestratorBridge()
        session.name = name
        session.session_id = session_id
        state = SessionState(
            session_id=session_id,
            name=name,
            project_id=project_id,
            bridge=bridge,
            session=session,
            summary=summary,
            forked_from=forked_from,
        )
        workspace_root = self._workspace_dir_for_project(project_id).resolve()
        state._workspace_root = str(workspace_root)
        # Worktree is created lazily on first orchestration run via _ensure_session_worktree.
        self._configure_bridge(bridge, project_id=project_id, cwd=str(workspace_root))
        state._last_snapshot_id = self._latest_snapshot_for_session(session_id)
        state._last_touched_at = time.time()
        self._sessions[session_id] = state
        self._session_projects[session_id] = project_id
        # Promote from cold cache when this session becomes active.
        self._cold_sessions.pop(session_id, None)
        self._cold_last_touched_at.pop(session_id, None)
        state._event_task = asyncio.create_task(self._consume_session_events(state))
        return state

    def _index_saved_sessions(self) -> None:
        """Build a metadata index of persisted sessions without loading bridges."""
        self._session_index.clear()
        try:
            session_dir = getattr(self._persistence, "_dir", None)
            if not isinstance(session_dir, Path):
                return
            for stem in self._persistence.list_sessions_by_mtime():
                path = session_dir / f"{stem}.json"
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                session_id = str(data.get("session_id") or stem)
                # Prefer the newest file for duplicate session IDs.
                if session_id in self._session_index:
                    continue
                messages = data.get("messages", {}) or {}
                self._session_index[session_id] = {
                    "file_stem": stem,
                    "session_id": session_id,
                    "name": data.get("name") or session_id,
                    "summary": data.get("summary"),
                    "project_id": data.get("project_id") or self._default_project_id,
                    "forked_from": data.get("forked_from"),
                    "agent_count": len(data.get("agents", {}) or {}),
                    "message_count": sum(
                        len(msgs) for msgs in messages.values()
                        if isinstance(msgs, list)
                    ),
                    "created_at": data.get("created_at"),
                    "last_activity": data.get("saved_at") or data.get("created_at"),
                }
            logger.info("Indexed %d persisted sessions for lazy load", len(self._session_index))
        except Exception:
            logger.warning("Failed to index saved sessions", exc_info=True)

    def _try_load_indexed_session(self, session_id: str) -> SessionState | None:
        """Load a session from disk on first access."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        meta = self._session_index.get(session_id)
        session = self._get_cold_session(session_id)
        if session is None:
            return None
        display_name = str(
            (meta.get("name") if meta else None)
            or session.name
            or session_id
        )
        forked_from = session.forked_from or (meta.get("forked_from") if meta else None)
        project_id = str(meta.get("project_id") or self._default_project_id) if meta else self._default_project_id
        state = self._build_state(
            session_id=session_id,
            name=display_name,
            session=session,
            project_id=project_id,
            summary=str(meta.get("summary")) if meta and meta.get("summary") else None,
            forked_from=forked_from,
        )
        self._load_file_changes(state)
        self._touch_session(state)
        logger.info(
            "Lazy-loaded session '%s' (%s) with %d agents, %d messages",
            display_name, session_id, len(session.agents), session.message_count,
        )
        return state

    def _create_session_state(
        self,
        name: str,
        session: Session | None = None,
        summary: str | None = None,
        forked_from: str | None = None,
        project_id: str | None = None,
    ) -> SessionState:
        session_id = str(uuid.uuid4())
        resolved_project_id = self._resolve_project_id(project_id)
        state = self._build_state(
            session_id=session_id,
            name=name,
            session=session or Session(),
            project_id=resolved_project_id,
            summary=summary,
            forked_from=forked_from,
        )
        self._session_index.pop(session_id, None)
        return state

    def _require_session(self, request: web.Request) -> tuple[SessionState | None, web.Response | None]:
        """Return (state, None) or (None, 404 response)."""
        session_id = request.match_info["id"]
        state = self._sessions.get(session_id)
        if state is None:
            state = self._try_load_indexed_session(session_id)
        if state is None:
            return None, web.json_response(
                {"error": f"Session {session_id} not found"},
                status=404,
            )
        self._touch_session(state)
        return state, None

    # ── Event processing ──

    async def _consume_session_events(self, state: SessionState) -> None:
        try:
            async for event in state.bridge.event_bus.consume():
                try:
                    await self._process_event(state, event)
                except Exception:
                    logger.exception(
                        "Error processing event %s for session %s (consumer continues)",
                        getattr(event, 'event_type', '?'), state.session_id,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Event consumer fatal error for session %s", state.session_id)

    async def _process_event(self, state: SessionState, event: OrchestratorEvent) -> None:
        """Process an event: update session state and broadcast via SSE."""
        self._touch_session(state)
        event_dict = event_to_dict(event)
        event_dict["session_id"] = state.session_id
        event_dict.setdefault("project_id", state.project_id)
        if event.event_type in {
            "lease_status",
            "audit_entry",
            "budget_status",
            "decision_report",
            "expert_status",
            "topic_event",
            "digest_event",
        }:
            state._governance_events.append(dict(event_dict))
            if len(state._governance_events) > 1000:
                state._governance_events = state._governance_events[-1000:]

        if isinstance(event, AgentSpawned):
            self._on_agent_spawned(state, event, event_dict)
        elif isinstance(event, AgentRestarted):
            self._on_agent_restarted(state, event, event_dict)
        elif isinstance(event, AgentStateChanged):
            self._on_agent_state_changed(state, event)
        elif isinstance(event, AgentKilled):
            state.bridge.cancel_agent_futures(event.agent_id)
            killed = state.session.agents.get(event.agent_id)
            if killed:
                killed.state = AgentState.KILLED
                killed.completed_at = datetime.now(timezone.utc)
        elif isinstance(event, AgentResult):
            await self._on_agent_result(state, event)
        elif isinstance(event, StreamChunk):
            self._on_stream_chunk(state, event)
        elif isinstance(event, ToolCallStarted):
            self._on_tool_call_started(state, event)
        elif isinstance(event, ToolCallDelta):
            self._on_tool_call_delta(state, event)
        elif isinstance(event, ToolCallCompleted):
            await self._on_tool_call_completed(state, event)
        elif isinstance(event, EngineFinished):
            await self._on_engine_finished(state)
        elif isinstance(event, UserPrompt):
            agent_id = event.agent_id or self._get_master_agent_id(state)
            if agent_id:
                state.session.add_message(agent_id, MessageRole.USER, event.text)
        elif isinstance(event, (PermissionRequest, UserQuestionRequest)):
            logger.info(
                "Event %s session=%s agent=%s request_id=%s",
                event.event_type, state.session_id,
                event.agent_id[:8], event.request_id[:8],
            )
        elif isinstance(event, EngineStarted):
            logger.info("Event engine_started session=%s sse_clients=%d", state.session_id, len(self._sse_queues))

        self._broadcast_sse(event.event_type, event_dict)

    # ── Event handlers (extracted from _process_event) ──

    def _on_agent_spawned(self, state: SessionState, event: AgentSpawned, event_dict: dict) -> None:
        logger.info(
            "Event agent_spawned session=%s agent=%s parent=%s role=%s",
            state.session_id, event.agent_id[:8], (event.parent_id or "")[:8], event.role,
        )
        node = state.bridge.map_agent(
            agent_id=event.agent_id, parent_id=event.parent_id,
            role=event.role, model=event.model, prompt=event.prompt, name=event.name,
        )
        state.session.add_agent(node)
        event_dict["name"] = node.name
        # Include the initial state so the extension can show the correct
        # icon immediately, rather than defaulting to "idle" and waiting
        # for the first agent_state_changed event.
        event_dict.setdefault("state", "pending")

        if event.parent_id:
            parent = state.session.agents.get(event.parent_id)
            if parent and event.agent_id not in parent.children_ids:
                parent.children_ids.append(event.agent_id)
        elif state._pending_user_prompt:
            state.session.add_message(
                event.agent_id,
                MessageRole.USER,
                state._pending_user_prompt,
                snapshot_id=state._pending_user_prompt_snapshot_id,
            )
            state._pending_user_prompt = None
            state._pending_user_prompt_snapshot_id = None

    def _on_agent_restarted(self, state: SessionState, event: AgentRestarted, event_dict: dict) -> None:
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.state = "idle"
            clean_prompt = _strip_prompt_prefix(event.prompt) if event.prompt else ""
            agent.prompt_preview = clean_prompt[:200]
            agent.last_active = datetime.now(timezone.utc)
            event_dict["name"] = agent.name
            if event.parent_id:
                parent = state.session.agents.get(event.parent_id)
                if parent and event.agent_id not in parent.children_ids:
                    parent.children_ids.append(event.agent_id)
        else:
            node = state.bridge.map_agent(
                agent_id=event.agent_id, parent_id=event.parent_id,
                role=event.role, model=event.model, prompt=event.prompt, name=event.name,
            )
            state.session.add_agent(node)
            event_dict["name"] = node.name
            if event.parent_id:
                parent = state.session.agents.get(event.parent_id)
                if parent and event.agent_id not in parent.children_ids:
                    parent.children_ids.append(event.agent_id)

    def _on_agent_state_changed(self, state: SessionState, event: AgentStateChanged) -> None:
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.state = state.bridge.map_state(event.new_state)
            agent.last_active = datetime.now(timezone.utc)
            if event.new_state in ("completed", "failed", "killed"):
                agent.completed_at = datetime.now(timezone.utc)
        if event.new_state in ("failed", "killed"):
            state.bridge.cancel_agent_futures(event.agent_id)

    async def _on_agent_result(self, state: SessionState, event: AgentResult) -> None:
        # Per-agent queue: restart agent with queued prompt after completion
        queued = state._agent_prompt_queue.get(event.agent_id)
        if queued:
            display_prompt, resolved = queued.pop(0)
            if not queued:
                del state._agent_prompt_queue[event.agent_id]
            asyncio.create_task(self._deferred_restart_agent(state, event.agent_id, display_prompt, resolved))

        # If directly restarted, fire engine_finished so extension clears busy state
        if event.agent_id in state._directly_restarted_agents:
            state._directly_restarted_agents.discard(event.agent_id)
            self._broadcast_sse("engine_finished", {
                "session_id": state.session_id,
                "success": not event.is_error,
                "summary": event.result[:200] if event.result else "",
                "error": None,
            })
            self._save_session(state)

    def _on_stream_chunk(self, state: SessionState, event: StreamChunk) -> None:
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)
        msgs = state.session.get_messages(event.agent_id)
        if msgs and msgs[-1].role == MessageRole.ASSISTANT:
            msgs[-1].content += event.text
        else:
            state.session.add_message(event.agent_id, MessageRole.ASSISTANT, event.text)
        self._write_plan_chunk(state, event.agent_id, event.text)

    def _on_tool_call_started(self, state: SessionState, event: ToolCallStarted) -> None:
        canonical_tool_name = normalize_tool_name(event.tool_name)
        logger.info(
            "Event tool_call_started session=%s agent=%s tool_id=%s tool=%s",
            state.session_id, event.agent_id[:8], event.tool_id[:12], canonical_tool_name,
        )
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)
        state.session.add_message(
            event.agent_id, MessageRole.TOOL, "",
            tool_calls=[ToolCall(id=event.tool_id, name=canonical_tool_name, arguments=event.arguments)],
        )
        agent_cwd = self._agent_cwd_for_state(state)
        state.file_tracker.capture_pre_tool(
            event.tool_id,
            canonical_tool_name,
            event.arguments,
            cwd=agent_cwd,
        )
        state._pre_tool_workspace_paths[event.tool_id] = self._status_path_set(agent_cwd)

    def _on_tool_call_delta(self, state: SessionState, event: ToolCallDelta) -> None:
        """Accumulate streaming tool output in session history.

        This preserves partial tool output if a run is stopped before a final
        tool_call_completed event arrives.
        """
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)
        agent_messages = state.session.get_messages(event.agent_id)
        for idx in range(len(agent_messages) - 1, -1, -1):
            msg = agent_messages[idx]
            for tc in msg.tool_calls:
                if tc.id != event.tool_id:
                    continue
                if tc.result is None:
                    tc.result = ""
                if event.stream == "stderr" and "STDERR:\n" not in tc.result:
                    tc.result += "\n\nSTDERR:\n"
                tc.result += event.delta
                return

    async def _on_tool_call_completed(self, state: SessionState, event: ToolCallCompleted) -> None:
        logger.info(
            "Event tool_call_completed session=%s agent=%s tool_id=%s is_error=%s",
            state.session_id, event.agent_id[:8], event.tool_id[:12], event.is_error,
        )
        agent = state.session.agents.get(event.agent_id)
        if agent:
            agent.last_active = datetime.now(timezone.utc)
        # Update the tool call result in existing messages
        tool_message_index = 0
        task_complete_summary: str | None = None
        agent_messages = state.session.get_messages(event.agent_id)
        for idx in range(len(agent_messages) - 1, -1, -1):
            msg = agent_messages[idx]
            found = False
            for tc in msg.tool_calls:
                if tc.id == event.tool_id:
                    tc.result = event.result
                    tc.success = not event.is_error
                    tool_message_index = idx
                    found = True
                    # Detect task_complete and extract summary for injection
                    if not event.is_error:
                        task_complete_summary = self._extract_task_complete_summary(tc)
                    break
            if found:
                break

        # Inject task_complete summary as a visible assistant message
        if task_complete_summary:
            state.session.add_message(event.agent_id, MessageRole.ASSISTANT, task_complete_summary)
            self._broadcast_sse("stream_chunk", {
                "agent_id": event.agent_id,
                "text": task_complete_summary,
            })

        # Track file changes — detect what the tool modified and broadcast.
        persisted_changes = False
        if not event.is_error:
            async with state._file_sync_lock:
                records = state.file_tracker.track_changes(
                    event.agent_id,
                    event.tool_id,
                    message_index=tool_message_index,
                )
                # Worktree fallback: snapshot-based tracking may miss changes
                # when the ToolCallStarted event is processed after the tool
                # already executed (event timing race).
                if not records and state._worktree_path:
                    records = self._worktree_file_change_fallback(
                        state, event.tool_id, event.agent_id, tool_message_index,
                    )
                for record in records:
                    self._normalize_file_change_record_path(state, record)
                    self._sync_file_to_workspace(state, record)
                    display_path = self._display_file_path_for_state(state, record.file_path)
                    self._broadcast_sse("file_changed", {
                        "session_id": state.session_id,
                        "agent_id": event.agent_id,
                        "file_path": display_path,
                        "change_type": record.change_type,
                        "tool_call_id": record.tool_call_id,
                        "tool_name": record.tool_name,
                        "message_index": record.message_index,
                        "old_content": record.old_content,
                        "new_content": record.new_content,
                        "pre_tool_content": record.pre_tool_content,
                        "added_ranges": record.added_ranges,
                        "removed_ranges": record.removed_ranges,
                        "timestamp": record.timestamp,
                    })
                    persisted_changes = True
                if persisted_changes:
                    self._persist_file_changes(state)
        else:
            state._pre_tool_workspace_paths.pop(event.tool_id, None)

        # Per-agent inject
        inject_info = state._inject_after_tool_agents.pop(event.agent_id, None)
        if inject_info:
            await self._execute_per_agent_inject(state, event.agent_id, inject_info)

        # Stop-after-tool
        if state._stop_after_tool:
            state._stop_after_tool = False
            await self._execute_stop_after_tool(state)

    async def _on_engine_finished(self, state: SessionState) -> None:
        logger.info("Event engine_finished session=%s sse_clients=%d", state.session_id, len(self._sse_queues))
        # NOTE: Session save is deferred to _run_orchestration which drains
        # the event bus first to ensure all agents and messages are captured.
        # Saving here would race with the drain and produce incomplete state.

        # Process queued prompts
        queued = getattr(state, '_queued_prompts', [])
        if queued:
            prompt, resolved_prompt = queued.pop(0)
            logger.info("Processing queued prompt session=%s prompt_len=%d", state.session_id, len(prompt))
            self._ensure_session_worktree(state)
            self._configure_bridge(
                state.bridge,
                project_id=state.project_id,
                cwd=str(self._agent_cwd_for_state(state)),
            )
            asyncio.create_task(self._run_orchestration(state, resolved_prompt, original_prompt=prompt))
        else:
            # No more work queued — prune worktree if all changes are resolved.
            self._maybe_cleanup_empty_worktree(state)

    # ── Task-complete summary injection ──

    @staticmethod
    def _extract_task_complete_summary(tc) -> str | None:
        """Extract the summary from a task_complete ToolCall if present.

        Returns the non-empty summary string, or None if this is not a
        task_complete call or has no summary.
        """
        from prsm.shared.formatters.tool_call import parse_args

        bare_name = str(tc.name).strip()
        # Handle both prefixed and already-normalized names
        if bare_name.startswith("mcp__") and "__" in bare_name:
            bare_name = bare_name.split("__", 2)[-1]
        if bare_name != "task_complete":
            return None
        args = parse_args(tc.arguments)
        summary = str(args.get("summary", "")).strip()
        return summary if summary else None

    # ── Inject / stop-after-tool helpers ──

    async def _execute_per_agent_inject(
        self, state: SessionState, agent_id: str, inject_info: tuple[str, str],
    ) -> None:
        display_prompt, resolved = inject_info
        manager = state.bridge._engine._manager if state.bridge._engine else None
        if not manager:
            return
        try:
            await manager.kill_agent(agent_id, keep_restartable=True)
        except Exception:
            logger.warning("Per-agent inject kill failed for %s", agent_id[:8], exc_info=True)
        state.session.add_message(agent_id, MessageRole.USER, display_prompt)
        self._broadcast_sse("agent_message", {
            "session_id": state.session_id, "agent_id": agent_id,
            "content": display_prompt, "role": "user",
        })
        restart_prompt = self._build_restart_prompt(
            manager, agent_id, resolved, state=state,
        )
        try:
            await manager.restart_agent(agent_id, restart_prompt)
            state._directly_restarted_agents.add(agent_id)
        except Exception:
            logger.warning("Per-agent inject restart failed for %s", agent_id[:8], exc_info=True)

    async def _execute_stop_after_tool(self, state: SessionState) -> None:
        await state.bridge.shutdown()
        inject_prompt = getattr(state, '_pending_inject_prompt', None)
        inject_resolved = getattr(state, '_pending_inject_resolved', None)
        if inject_prompt and inject_resolved:
            state._pending_inject_prompt = None
            state._pending_inject_resolved = None
            self._save_session(state)
            self._ensure_session_worktree(state)
            self._configure_bridge(
                state.bridge,
                project_id=state.project_id,
                cwd=str(self._agent_cwd_for_state(state)),
            )
            asyncio.create_task(self._run_orchestration(state, inject_resolved, original_prompt=inject_prompt))

    # ── Serialization ──

    def _serialize_agent(self, agent: AgentNode) -> dict[str, Any]:
        return {
            "id": agent.id,
            "name": agent.name,
            "state": agent.state.value if agent.state else "idle",
            "role": agent.role.value if agent.role else "worker",
            "model": agent.model,
            "parent_id": agent.parent_id,
            "children_ids": agent.children_ids,
            "prompt_preview": agent.prompt_preview,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "completed_at": agent.completed_at.isoformat() if agent.completed_at else None,
            "last_active": agent.last_active.isoformat() if agent.last_active else None,
        }

    def _serialize_message(self, msg: Any) -> dict[str, Any]:
        return {
            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
            "content": msg.content,
            "agent_id": msg.agent_id,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "snapshot_id": getattr(msg, "snapshot_id", None),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "result": tc.result, "success": tc.success}
                for tc in (msg.tool_calls or [])
            ],
            "streaming": getattr(msg, "streaming", False),
        }

    def _serialize_session_snapshot(self, state: SessionState) -> dict[str, Any]:
        """Serialize full session state for extension-side replacement."""
        return {
            "session_id": state.session_id,
            "name": state.name,
            "summary": state.summary,
            "forked_from": state.forked_from,
            "worktree": {
                "branch": state.session.worktree.branch if state.session.worktree else None,
                "worktreePath": state.session.worktree.root if state.session.worktree else None,
                "isWorktree": bool(state.session.worktree and state.session.worktree.root),
            } if state.session.worktree else None,
            "created_at": (
                state.session.created_at.isoformat()
                if state.session.created_at
                else None
            ),
            "running": state.bridge.running,
            "current_model": state.bridge.current_model,
            "current_model_display": state.bridge.current_model_display,
            "agents": [
                self._serialize_agent(agent)
                for agent in state.session.agents.values()
            ],
            "messages": {
                aid: [self._serialize_message(m) for m in msgs]
                for aid, msgs in state.session.messages.items()
            },
            "active_agent_id": state.session.active_agent_id,
            "imported_from": state.session.imported_from,
        }

    # ── HTTP handlers ──

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "pid": os.getpid(),
            "uptime_seconds": round(max(0.0, time.time() - self._started_at), 3),
            "cwd": self._cwd,
            "claude_preflight": {
                "enabled": self._claude_preflight_enabled,
                "ok": self._claude_preflight_ok,
                "detail": self._claude_preflight_detail,
                "checked_at_epoch_seconds": self._claude_preflight_checked_at,
            },
            "session_inactivity_minutes": (
                self._session_inactivity_seconds / 60.0
                if self._session_inactivity_seconds > 0
                else 0.0
            ),
        })

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5000)
        self._sse_queues.append(queue)
        logger.info("SSE client connected req=%s active_clients=%d", request.get("req_id", "unknown"), len(self._sse_queues))

        try:
            await response.write(
                f"event: connected\ndata: {json.dumps({'sessions': list(self._sessions.keys())})}\n\n".encode()
            )
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = json.dumps(msg["data"])
                    await response.write(f"event: {msg['event']}\ndata: {data}\n\n".encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                except ConnectionResetError:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._sse_queues.remove(queue)
            logger.info("SSE client disconnected req=%s active_clients=%d", request.get("req_id", "unknown"), len(self._sse_queues))
        return response

    async def _handle_list_projects(self, request: web.Request) -> web.Response:
        session_counts: dict[str, int] = {}
        for project_id in self._session_projects.values():
            session_counts[project_id] = session_counts.get(project_id, 0) + 1
        projects: list[dict[str, Any]] = []
        for project_id, meta in self._known_projects.items():
            projects.append(
                {
                    "project_id": project_id,
                    "name": meta.get("label") or project_id,
                    "is_default": project_id == self._default_project_id,
                    "session_count": session_counts.get(project_id, 0),
                }
            )
        projects.sort(key=lambda p: (not p["is_default"], p["name"]))
        return web.json_response({"projects": projects})

    def _project_broker_or_error(
        self, project_id: str,
    ) -> tuple[Any | None, web.Response | None]:
        if not self._multi_project_enabled or self._project_registry is None:
            return None, web.json_response(
                {"error": "Multi-project broker is disabled"},
                status=409,
            )
        if project_id not in self._known_projects:
            return None, web.json_response(
                {"error": f"Unknown project_id: {project_id}"},
                status=400,
            )
        return self._project_registry.event_broker, None

    async def _handle_subscribe_project(self, request: web.Request) -> web.Response:
        project_id = request.match_info["project_id"]
        broker, err = self._project_broker_or_error(project_id)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        topic_filter = str(body.get("topic_filter", "")).strip()
        if not topic_filter:
            return web.json_response({"error": "topic_filter is required"}, status=400)
        broker.subscribe(project_id, topic_filter)
        return web.json_response(
            {
                "status": "subscribed",
                "project_id": project_id,
                "topic_filter": topic_filter,
                "subscriptions": broker.get_subscriptions(project_id),
            },
        )

    async def _handle_unsubscribe_project(self, request: web.Request) -> web.Response:
        project_id = request.match_info["project_id"]
        broker, err = self._project_broker_or_error(project_id)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        topic_filter = str(body.get("topic_filter", "")).strip()
        if not topic_filter:
            return web.json_response({"error": "topic_filter is required"}, status=400)
        broker.unsubscribe(project_id, topic_filter)
        return web.json_response(
            {
                "status": "unsubscribed",
                "project_id": project_id,
                "topic_filter": topic_filter,
                "subscriptions": broker.get_subscriptions(project_id),
            },
        )

    async def _handle_publish_project_event(self, request: web.Request) -> web.Response:
        project_id = request.match_info["project_id"]
        broker, err = self._project_broker_or_error(project_id)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        topic = str(body.get("topic", "")).strip()
        if not topic:
            return web.json_response({"error": "topic is required"}, status=400)

        from prsm.engine.project_registry import CrossProjectEvent

        event = CrossProjectEvent(
            topic=topic,
            source_project_id=project_id,
            scope=str(body.get("scope") or "global"),
            urgency=str(body.get("urgency") or "normal"),
            ttl=body.get("ttl"),
            payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
        )
        await broker.publish(event)
        published = broker.get_event_log(limit=1)[0]
        self._broadcast_sse("cross_project_event", published)
        return web.json_response({"status": "accepted", "event": published}, status=202)

    async def _handle_list_project_events(self, request: web.Request) -> web.Response:
        if not self._multi_project_enabled or self._project_registry is None:
            return web.json_response(
                {"error": "Multi-project broker is disabled"},
                status=409,
            )
        broker = self._project_registry.event_broker
        limit_raw = request.query.get("limit")
        limit: int | None = None
        if limit_raw:
            try:
                limit = int(limit_raw)
            except ValueError:
                return web.json_response({"error": "limit must be an integer"}, status=400)
        events = broker.get_event_log(limit=limit)
        topic = request.query.get("topic")
        source_project_id = request.query.get("source_project_id")
        if topic:
            events = [evt for evt in events if evt.get("topic") == topic]
        if source_project_id:
            events = [
                evt
                for evt in events
                if evt.get("source_project_id") == source_project_id
            ]
        return web.json_response({"events": events})

    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        self._refresh_session_metadata_from_disk()
        sessions: list[dict[str, Any]] = []
        loaded_ids = set(self._sessions.keys())
        for state in self._sessions.values():
            sessions.append(self._state_to_session_summary(state))

        for session_id, meta in self._session_index.items():
            if session_id in loaded_ids:
                continue
            sessions.append({
                "sessionId": session_id,
                "name": meta.get("name") or session_id,
                "summary": meta.get("summary"),
                "projectId": meta.get("project_id") or self._default_project_id,
                "forkedFrom": meta.get("forked_from"),
                "agentCount": int(meta.get("agent_count") or 0),
                "messageCount": int(meta.get("message_count") or 0),
                "running": False,
                "createdAt": meta.get("created_at"),
                "lastActivity": meta.get("last_activity"),
            })

        sessions.sort(
            key=lambda s: (
                s.get("lastActivity") or s.get("createdAt") or "",
                s.get("sessionId") or "",
            ),
            reverse=True,
        )
        return web.json_response({"sessions": sessions})

    def _refresh_session_metadata_from_disk(self) -> None:
        """Refresh indexed metadata and reconcile idle loaded sessions."""
        self._index_saved_sessions()
        for session_id, state in self._sessions.items():
            if state.bridge.running:
                continue
            meta = self._session_index.get(session_id) or {}
            persisted_name = str(meta.get("name") or "").strip()
            if persisted_name and persisted_name != (state.name or "").strip():
                state.name = persisted_name
                state.session.name = persisted_name

    async def _handle_create_session(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        name = body.get("name", "Untitled Session")
        requested_project = body.get("project_id")
        if requested_project and requested_project not in self._known_projects:
            return web.json_response(
                {"error": f"Unknown project_id: {requested_project}"},
                status=400,
            )
        state = self._create_session_state(name=name, project_id=requested_project)
        self._broadcast_sse(
            "session_created",
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "forked_from": None,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
        )
        return web.json_response(
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
            status=201,
        )

    async def _handle_fork_session(self, request: web.Request) -> web.Response:
        source_id = request.match_info["id"]
        source = self._sessions.get(source_id)
        if not source:
            return web.json_response({"error": f"Session {source_id} not found"}, status=404)

        body = await request.json() if request.can_read_body else {}
        base_name = (body.get("name") or "").strip() or (source.name or source_id)
        name = format_forked_name(base_name)
        forked_session = source.session.fork(new_name=name)
        state = self._create_session_state(
            name=name,
            session=forked_session,
            forked_from=source_id,
            project_id=source.project_id,
        )

        self._broadcast_sse(
            "session_created",
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "forked_from": source_id,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
        )
        self._broadcast_agent_tree(state, forked_session)
        return web.json_response(
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "forked_from": source_id,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
            status=201,
        )

    async def _handle_remove_session(self, request: web.Request) -> web.Response:
        session_id = request.match_info["id"]
        state = self._sessions.get(session_id)
        project_id = self._session_projects.get(session_id, self._default_project_id)
        if state:
            await state.bridge.shutdown()
            if state._event_task:
                state._event_task.cancel()
            self._save_session(state)
            self._cleanup_session_worktree(state)

            self._sessions.pop(session_id, None)
            self._session_projects.pop(session_id, None)

        # Delete persisted file, resolving legacy name-based stems if needed.
        deleted = self._persistence.delete(session_id)
        meta = self._session_index.pop(session_id, None)
        if not deleted and meta and meta.get("file_stem") and meta.get("file_stem") != session_id:
            deleted = self._persistence.delete(str(meta.get("file_stem")))
        if not deleted and not state:
            return web.json_response(
                {"error": f"Session {session_id} not found"},
                status=404,
            )
        self._broadcast_sse(
            "session_removed",
            {
                "session_id": session_id,
                "project_id": project_id,
            },
        )
        return web.json_response({"status": "removed"})

    async def _handle_rename_session(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        new_name = body.get("name", "").strip()
        if not new_name:
            return web.json_response({"error": "Name is required"}, status=400)
        state.name = new_name
        state.session.name = new_name
        self._broadcast_sse(
            "session_renamed",
            {
                "session_id": state.session_id,
                "name": new_name,
                "summary": state.summary,
                "project_id": state.project_id,
            },
        )
        self._save_session(state)
        return web.json_response({"session_id": state.session_id, "name": new_name, "summary": state.summary})

    async def _handle_get_agents(self, request: web.Request) -> web.Response:
        session_id = request.match_info["id"]
        state = self._sessions.get(session_id)
        if state:
            self._touch_session(state)
            agents = [self._serialize_agent(a) for a in state.session.agents.values()]
            return web.json_response({"agents": agents})

        session = self._get_cold_session(session_id)
        if not session:
            return web.json_response({"error": f"Session {session_id} not found"}, status=404)
        agents = [self._serialize_agent(a) for a in session.agents.values()]
        return web.json_response({"agents": agents})

    async def _handle_get_messages(self, request: web.Request) -> web.Response:
        session_id = request.match_info["id"]
        state = self._sessions.get(session_id)
        agent_id = request.match_info["agent_id"]
        if state:
            self._touch_session(state)
            msgs = state.session.get_messages(agent_id)
            return web.json_response({"messages": [self._serialize_message(m) for m in msgs]})

        session = self._get_cold_session(session_id)
        if not session:
            return web.json_response({"error": f"Session {session_id} not found"}, status=404)
        msgs = session.get_messages(agent_id)
        return web.json_response({"messages": [self._serialize_message(m) for m in msgs]})

    async def _handle_import_list(self, request: web.Request) -> web.Response:
        provider = str(request.query.get("provider", "all")).strip().lower() or "all"
        if provider not in {"all", "codex", "claude", "prsm"}:
            return web.json_response({"error": f"Invalid provider: {provider}"}, status=400)
        try:
            limit = int(str(request.query.get("limit", "50")))
        except Exception:
            return web.json_response({"error": "limit must be an integer"}, status=400)
        if limit <= 0:
            return web.json_response({"error": "limit must be > 0"}, status=400)

        from prsm.shared.services.transcript_import.service import TranscriptImportService

        service = TranscriptImportService()
        sessions = service.list_sessions(provider=provider, limit=limit)
        return web.json_response(
            {
                "sessions": [
                    {
                        "provider": s.provider,
                        "source_id": s.source_session_id,
                        "title": s.title,
                        "turn_count": s.turn_count,
                        "source_path": str(s.source_path),
                        "started_at": s.started_at.isoformat() if s.started_at else None,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    }
                    for s in sessions
                ]
            }
        )

    async def _handle_import_preview(self, request: web.Request) -> web.Response:
        provider = str(request.query.get("provider", "")).strip().lower()
        source_id = str(request.query.get("source_id", "")).strip()
        if provider not in {"codex", "claude", "prsm"}:
            return web.json_response({"error": "provider must be codex or claude or prsm"}, status=400)
        if not source_id:
            return web.json_response({"error": "source_id is required"}, status=400)

        from prsm.shared.services.transcript_import.service import TranscriptImportService

        service = TranscriptImportService()
        try:
            transcript = service.load_transcript(provider, source_id)
        except FileNotFoundError:
            return web.json_response({"error": f"Import source not found: {provider}:{source_id}"}, status=404)
        preview_turns = transcript.turns[:12]
        return web.json_response(
            {
                "summary": {
                    "provider": transcript.summary.provider,
                    "source_id": transcript.summary.source_session_id,
                    "title": transcript.summary.title,
                    "turn_count": len(transcript.turns),
                    "started_at": (
                        transcript.summary.started_at.isoformat()
                        if transcript.summary.started_at
                        else None
                    ),
                    "updated_at": (
                        transcript.summary.updated_at.isoformat()
                        if transcript.summary.updated_at
                        else None
                    ),
                    "source_path": str(transcript.summary.source_path),
                },
                "preview_turns": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        "tool_call_count": len(t.tool_calls),
                    }
                    for t in preview_turns
                ],
                "warnings": transcript.warnings,
            }
        )

    async def _handle_import_run(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        if state.bridge.running:
            return web.json_response({"error": "Orchestration already in progress"}, status=409)

        body = await request.json()
        provider = str(body.get("provider", "")).strip().lower()
        source_id = str(body.get("source_id", "")).strip()
        session_name = body.get("session_name")
        if isinstance(session_name, str):
            session_name = session_name.strip() or None
        else:
            session_name = None

        max_turns_raw = body.get("max_turns")
        max_turns: int | None = None
        if max_turns_raw is not None:
            try:
                max_turns = int(max_turns_raw)
            except Exception:
                return web.json_response({"error": "max_turns must be an integer"}, status=400)
            if max_turns <= 0:
                return web.json_response({"error": "max_turns must be > 0"}, status=400)

        if provider not in {"codex", "claude", "prsm"}:
            return web.json_response({"error": "provider must be codex or claude or prsm"}, status=400)
        if not source_id:
            return web.json_response({"error": "source_id is required"}, status=400)

        from prsm.shared.services.transcript_import.service import TranscriptImportService

        service = TranscriptImportService()
        try:
            result = service.import_to_session(
                provider,
                source_id,
                session_name=session_name,
                max_turns=max_turns,
            )
        except FileNotFoundError:
            return web.json_response({"error": f"Import source not found: {provider}:{source_id}"}, status=404)

        imported_session = result.session
        imported_session.imported_from = TranscriptImportService.session_import_metadata(result)
        imported_session.session_id = state.session_id
        for node in imported_session.agents.values():
            normalized_model, normalized_provider = (
                self._normalize_agent_model_for_restart(
                    state=state,
                    model=node.model,
                    provider=node.provider,
                )
            )
            node.model = normalized_model
            node.provider = normalized_provider

        state.session = imported_session
        state.name = imported_session.name or state.name
        state.summary = (
            f"Imported {provider} transcript ({result.imported_turns} turns"
            + (f", dropped {result.dropped_turns}" if result.dropped_turns else "")
            + ")"
        )
        state.forked_from = imported_session.forked_from
        state.file_tracker = FileChangeTracker()
        state._pending_user_prompt = None
        state._pending_user_prompt_snapshot_id = None
        state._inject_after_tool_agents.clear()
        state._agent_prompt_queue.clear()
        state._directly_restarted_agents.clear()

        self._save_session(state)
        return web.json_response(
            {
                "status": "ok",
                "message": state.summary,
                "warnings": result.metadata.get("warnings", []),
                "session": self._serialize_session_snapshot(state),
            }
        )

    async def _handle_import_all(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err

        body = await request.json()
        provider = str(body.get("provider", "")).strip().lower()
        if provider not in {"codex", "claude", "prsm"}:
            return web.json_response({"error": "provider must be codex or claude or prsm"}, status=400)

        max_turns_raw = body.get("max_turns")
        max_turns: int | None = None
        if max_turns_raw is not None:
            try:
                max_turns = int(max_turns_raw)
            except Exception:
                return web.json_response({"error": "max_turns must be an integer"}, status=400)
            if max_turns <= 0:
                return web.json_response({"error": "max_turns must be > 0"}, status=400)

        from prsm.shared.services.transcript_import.service import TranscriptImportService

        service = TranscriptImportService()
        results = service.import_all_sessions(provider, max_turns=max_turns)

        summaries = []
        for result in results:
            imported_session = result.session
            imported_session.imported_from = TranscriptImportService.session_import_metadata(result)
            session_name = imported_session.name or result.source.source_session_id
            session_id = imported_session.session_id if hasattr(imported_session, "session_id") and imported_session.session_id else None
            self._persistence.save(
                imported_session,
                session_name,
                session_id=session_id,
            )
            summaries.append(
                {
                    "provider": result.source.provider,
                    "source_id": result.source.source_session_id,
                    "title": result.source.title,
                    "imported_turns": result.imported_turns,
                    "dropped_turns": result.dropped_turns,
                    "session_name": session_name,
                }
            )

        return web.json_response(
            {
                "status": "ok",
                "count": len(summaries),
                "sessions": summaries,
            }
        )

    async def _handle_import_all(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err

        body = await request.json()
        provider = str(body.get("provider", "")).strip().lower()
        if provider not in {"codex", "claude", "prsm"}:
            return web.json_response({"error": "provider must be codex or claude or prsm"}, status=400)

        max_turns_raw = body.get("max_turns")
        max_turns: int | None = None
        if max_turns_raw is not None:
            try:
                max_turns = int(max_turns_raw)
            except Exception:
                return web.json_response({"error": "max_turns must be an integer"}, status=400)
            if max_turns <= 0:
                return web.json_response({"error": "max_turns must be > 0"}, status=400)

        from prsm.shared.services.transcript_import.service import TranscriptImportService

        service = TranscriptImportService()
        results = service.import_all_sessions(provider, max_turns=max_turns)

        summaries = []
        for result in results:
            imported_session = result.session
            imported_session.imported_from = TranscriptImportService.session_import_metadata(result)
            session_name = imported_session.name or result.source.source_session_id
            session_id = imported_session.session_id if hasattr(imported_session, "session_id") and imported_session.session_id else None
            self._persistence.save(
                imported_session,
                session_name,
                session_id=session_id,
            )
            summaries.append(
                {
                    "provider": result.source.provider,
                    "source_id": result.source.source_session_id,
                    "title": result.source.title,
                    "imported_turns": result.imported_turns,
                    "dropped_turns": result.dropped_turns,
                    "session_name": session_name,
                }
            )

        return web.json_response(
            {
                "status": "ok",
                "count": len(summaries),
                "sessions": summaries,
            }
        )

    async def _handle_command(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        raw = str(body.get("command", "")).strip()
        if not raw:
            return web.json_response({"error": "command is required"}, status=400)

        parsed = parse_command(raw)
        if parsed is None:
            return web.json_response({"error": "Not a slash command"}, status=400)
        name = parsed.name.lower()
        args = parsed.args

        if name == "help":
            lines = [f"/{cmd} — {desc}" for cmd, desc in COMMAND_HELP.items()]
            return web.json_response({"status": "ok", "kind": "help", "lines": lines})

        if name == "import":
            action = args[0].lower() if args else "list"
            if action == "list":
                provider = args[1].lower() if len(args) > 1 else "all"
                if provider not in {"all", "codex", "claude", "prsm"}:
                    return web.json_response({"error": f"Invalid provider: {provider}"}, status=400)
                from prsm.shared.services.transcript_import.service import TranscriptImportService

                service = TranscriptImportService()
                sessions = service.list_sessions(provider=provider, limit=25)
                return web.json_response(
                    {
                        "status": "ok",
                        "kind": "import_list",
                        "sessions": [
                            {
                                "provider": s.provider,
                                "source_id": s.source_session_id,
                                "title": s.title,
                                "turn_count": s.turn_count,
                                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                            }
                            for s in sessions
                        ],
                    }
                )

            return web.json_response(
                {
                    "status": "error",
                    "kind": "unsupported",
                    "message": "Use /import list|preview|run in this client path.",
                },
                status=400,
            )

        return web.json_response(
            {"status": "error", "kind": "unsupported", "message": f"Unknown command: /{name}"},
            status=400,
        )

    async def _handle_run(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)
        if state.bridge.running:
            return web.json_response({"error": "Orchestration already in progress"}, status=409)

        resolved_prompt = self._resolve_at_references(prompt)
        logger.info(
            "Run requested session=%s prompt_len=%d resolved_len=%d",
            state.session_id, len(prompt), len(resolved_prompt),
        )
        asyncio.create_task(self._run_orchestration(state, resolved_prompt, original_prompt=prompt))
        return web.json_response({"status": "started", "session_id": state.session_id})

    async def _handle_resolve_permission(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        request_id = body.get("request_id", "")
        result = body.get("result", "deny")
        logger.info("Resolve permission session=%s request_id=%s result=%s", state.session_id, str(request_id)[:8], result)
        state.bridge.resolve_permission(request_id, result)
        return web.json_response({"status": "resolved"})

    async def _handle_resolve_question(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        request_id = body.get("request_id", "")
        answer = body.get("answer", "")
        logger.info("Resolve question session=%s request_id=%s answer_len=%d", state.session_id, str(request_id)[:8], len(str(answer)))
        resolved_answer = self._resolve_at_references(answer)
        state.bridge.resolve_user_question(request_id, resolved_answer)
        return web.json_response({"status": "resolved"})

    async def _handle_agent_message(self, request: web.Request) -> web.Response:
        """Send a user message to a specific agent."""
        state, err = self._require_session(request)
        if err:
            return err
        agent_id = request.match_info["agent_id"]
        agent = state.session.agents.get(agent_id)
        if not agent:
            return web.json_response({"error": f"Agent {agent_id} not found"}, status=404)

        body = await request.json()
        prompt = body.get("prompt", "").strip()
        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)

        resolved_prompt = self._resolve_at_references(prompt)
        snapshot_id = None
        try:
            snapshot_id = self._create_snapshot(
                state,
                prompt,
                agent_id=agent_id,
            )
        except Exception:
            logger.debug(
                "Auto-snapshot failed before agent message session=%s agent=%s",
                state.session_id,
                agent_id[:8],
                exc_info=True,
            )
        state.session.add_message(
            agent_id,
            MessageRole.USER,
            resolved_prompt,
            snapshot_id=snapshot_id,
        )

        # Try to restart completed/failed agent
        restarted = await self._try_restart_agent(state, agent_id, resolved_prompt)

        # Broadcast original prompt so clients see what the user typed
        self._broadcast_sse("agent_message", {
            "session_id": state.session_id,
            "agent_id": agent_id,
            "content": prompt,
            "role": "user",
            "snapshot_id": snapshot_id,
        })

        # Deliver to running agent via message router if not restarted
        if not restarted:
            await self._deliver_to_running_agent(state, agent_id, resolved_prompt)

        return web.json_response({"status": "sent", "session_id": state.session_id, "agent_id": agent_id})

    async def _handle_kill_agent(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json()
        agent_id = body.get("agent_id", "")
        if state.bridge._engine:
            try:
                await state.bridge._engine._manager.kill_agent(agent_id)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"status": "killed"})

    async def _handle_shutdown_session(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        await state.bridge.shutdown()
        self._save_session(state)
        return web.json_response({"status": "shutdown"})

    def _find_latest_inflight_tool_agent_id(self, state: SessionState) -> str | None:
        latest_ts: datetime | None = None
        latest_agent_id: str | None = None

        for agent_id, messages in state.session.messages.items():
            for msg in reversed(messages):
                if not msg.tool_calls:
                    continue
                has_inflight = any(tc.result is None for tc in msg.tool_calls)
                if not has_inflight:
                    continue
                ts = msg.timestamp
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest_agent_id = agent_id
                break

        if latest_agent_id:
            return latest_agent_id

        def _state_value(agent: AgentNode) -> str:
            state_val = agent.state
            return state_val.value if hasattr(state_val, "value") else str(state_val)

        candidate_states = {
            "pending",
            "starting",
            "running",
            "waiting_for_parent",
            "waiting_for_child",
            "waiting_for_expert",
        }
        best_agent: AgentNode | None = None
        best_activity: datetime | None = None
        for agent in state.session.agents.values():
            if _state_value(agent) not in candidate_states:
                continue
            activity = agent.last_active or agent.created_at
            if activity is None:
                continue
            if best_activity is None or activity > best_activity:
                best_activity = activity
                best_agent = agent

        return best_agent.id if best_agent else None

    async def _handle_cancel_latest_tool_call(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err

        if not state.bridge._engine:
            return web.json_response({"error": "Engine is not running"}, status=409)

        try:
            # Stop the active orchestration run completely. This handles both:
            # - in-flight tool calls
            # - pure thinking/generation periods with no active tool call
            await state.bridge.shutdown()
        except Exception:
            logger.warning(
                "Stop run failed for session=%s",
                state.session_id,
                exc_info=True,
            )
            return web.json_response(
                {"error": "Failed to stop run"},
                status=500,
            )

        self._save_session(state)
        return web.json_response(
            {"status": "stopped"},
        )

    async def _handle_stop_after_tool(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        state._stop_after_tool = True
        return web.json_response({"status": "ok"})

    async def _handle_inject_prompt(self, request: web.Request) -> web.Response:
        """Inject a user prompt with a delivery mode (interrupt/inject/queue)."""
        state, err = self._require_session(request)
        if err:
            return err
        agent_id = request.match_info["agent_id"]
        if agent_id not in state.session.agents:
            return web.json_response(
                {"error": f"Agent {agent_id} not found"},
                status=404,
            )
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        mode = body.get("mode", "queue")

        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)
        if mode not in ("interrupt", "inject", "queue"):
            return web.json_response({"error": f"Invalid mode: {mode}"}, status=400)

        resolved_prompt = self._resolve_at_references(prompt)
        try:
            self._create_snapshot(
                state,
                prompt,
                agent_id=agent_id,
            )
        except Exception:
            logger.debug(
                "Auto-snapshot failed before inject prompt session=%s agent=%s mode=%s",
                state.session_id,
                agent_id[:8],
                mode,
                exc_info=True,
            )
        logger.info(
            "Inject prompt session=%s agent=%s mode=%s prompt_len=%d",
            state.session_id, agent_id[:12], mode, len(prompt),
        )

        if mode == "interrupt":
            return await self._inject_interrupt(state, agent_id, prompt, resolved_prompt)
        elif mode == "inject":
            return self._inject_inject(state, agent_id, prompt, resolved_prompt)
        else:
            return self._inject_queue(state, agent_id, prompt, resolved_prompt)

    # ── Inject mode helpers ──

    async def _inject_interrupt(
        self, state: SessionState, agent_id: str,
        prompt: str, resolved: str,
    ) -> web.Response:
        if not state.bridge._engine:
            return web.json_response(
                {"error": "Engine is not running"},
                status=409,
            )
        manager = state.bridge._engine._manager
        try:
            await manager.kill_agent(agent_id, keep_restartable=True)
        except Exception:
            logger.debug("Kill during interrupt failed for %s", agent_id)
        self._broadcast_sse("agent_message", {
            "session_id": state.session_id, "agent_id": agent_id,
            "content": prompt, "role": "user",
        })
        state.session.add_message(agent_id, MessageRole.USER, prompt)
        restart_prompt = self._build_restart_prompt(
            manager, agent_id, resolved, state=state,
        )
        try:
            await manager.restart_agent(agent_id, restart_prompt)
            state._directly_restarted_agents.add(agent_id)
        except Exception:
            logger.warning(
                "Interrupt restart failed for %s",
                agent_id[:8],
                exc_info=True,
            )
            return web.json_response(
                {"error": "Failed to restart interrupted agent"},
                status=500,
            )
        return web.json_response({"status": "interrupted", "mode": "interrupt"})

    def _inject_inject(
        self, state: SessionState, agent_id: str,
        prompt: str, resolved: str,
    ) -> web.Response:
        state._inject_after_tool_agents[agent_id] = (prompt, resolved)
        return web.json_response({"status": "injecting", "mode": "inject"})

    def _inject_queue(
        self, state: SessionState, agent_id: str,
        prompt: str, resolved: str,
    ) -> web.Response:
        if agent_id not in state._agent_prompt_queue:
            state._agent_prompt_queue[agent_id] = []
        state._agent_prompt_queue[agent_id].append((prompt, resolved))
        return web.json_response({"status": "queued", "mode": "queue"})

    # ── Persistence + snapshot handlers ──

    async def _handle_save_session(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        self._save_session(state)
        return web.json_response({"status": "saved"})

    async def _handle_list_restorable(self, request: web.Request) -> web.Response:
        saved = self._persistence.list_sessions_detailed()
        loaded_names = {s.name for s in self._sessions.values()}
        restorable = [s for s in saved if s["name"] not in loaded_names]
        return web.json_response({"sessions": restorable})

    async def _handle_restore_session(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        try:
            session = self._persistence.load(name)
        except FileNotFoundError:
            return web.json_response({"error": f"Session '{name}' not found on disk"}, status=404)
        except Exception as e:
            return web.json_response({"error": f"Failed to load session: {e}"}, status=500)

        state = self._create_session_state(name=name, session=session)
        self._broadcast_sse(
            "session_created",
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "forked_from": None,
                "restored": True,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
        )
        self._broadcast_agent_tree(state, session)
        return web.json_response(
            {
                "session_id": state.session_id,
                "name": state.name,
                "summary": state.summary,
                "project_id": state.project_id,
                "current_model": state.bridge.current_model,
                "current_model_display": state.bridge.current_model_display,
            },
            status=201,
        )

    async def _handle_list_snapshots(self, request: web.Request) -> web.Response:
        session_id = request.match_info["id"]
        if session_id not in self._sessions and session_id not in self._session_index:
            return web.json_response({"error": f"Session {session_id} not found"}, status=404)
        try:
            return web.json_response({
                "snapshots": self._get_snapshot_service().list_snapshots_by_session(session_id),
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_snapshot(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        description = body.get("description", "")
        try:
            snapshot_id = self._create_snapshot(state, description)
            return web.json_response({"snapshot_id": snapshot_id}, status=201)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_restore_snapshot(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        snap_id = request.match_info["snap_id"]
        try:
            async with state._file_sync_lock:
                state.session, state.file_tracker = self._get_snapshot_service().restore(snap_id)
                self._normalize_loaded_file_change_paths(state)
                state._last_snapshot_id = snap_id
                # Persist restored file changes to the session's on-disk directory
                self._persist_file_changes(state)
                self._broadcast_sse("snapshot_restored", {"session_id": state.session_id, "snapshot_id": snap_id})
                return web.json_response({"status": "restored"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_fork_snapshot(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        snap_id = request.match_info["snap_id"]
        body = await request.json() if request.can_read_body else {}
        requested_name = (body.get("name") or "").strip()

        try:
            meta = self._get_snapshot_service().get_meta(snap_id)
            base_session, file_tracker = self._get_snapshot_service().load_session(snap_id)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        base_name = (
            requested_name
            or meta.get("session_name")
            or base_session.name
            or f"Snapshot {snap_id}"
        )
        name = format_forked_name(base_name)
        session_ref = meta.get("session_name") or meta.get("session_id") or "unknown"
        forked_from = f"snapshot:{snap_id} ({session_ref})"

        forked_session = base_session.fork(new_name=name)
        forked_session.forked_from = forked_from

        new_state = self._create_session_state(
            name=name,
            session=forked_session,
            forked_from=forked_from,
            project_id=state.project_id,
        )
        new_state.file_tracker = file_tracker
        self._normalize_loaded_file_change_paths(new_state)
        if new_state.file_tracker.file_changes:
            self._persist_file_changes(new_state)

        self._broadcast_sse(
            "session_created",
            {
                "session_id": new_state.session_id,
                "name": new_state.name,
                "summary": new_state.summary,
                "forked_from": forked_from,
                "project_id": new_state.project_id,
                "current_model": new_state.bridge.current_model,
                "current_model_display": new_state.bridge.current_model_display,
            },
        )
        self._broadcast_agent_tree(new_state, forked_session)
        return web.json_response(
            {
                "session_id": new_state.session_id,
                "name": new_state.name,
                "summary": new_state.summary,
                "forked_from": forked_from,
                "project_id": new_state.project_id,
                "current_model": new_state.bridge.current_model,
                "current_model_display": new_state.bridge.current_model_display,
            },
            status=201,
        )

    async def _handle_delete_snapshot(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        snap_id = request.match_info["snap_id"]
        try:
            self._get_snapshot_service().delete(snap_id)
            return web.json_response({"status": "deleted"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    def _workspace_root_for_state(self, state: SessionState) -> Path:
        root = getattr(state, "_workspace_root", None)
        if root:
            return Path(str(root)).resolve()
        return Path(self._cwd).resolve()

    def _agent_cwd_for_state(self, state: SessionState) -> Path:
        """Return the directory where agents run — worktree if available, workspace otherwise."""
        if state._worktree_path:
            return Path(state._worktree_path)
        return self._workspace_root_for_state(state)

    def _worktree_to_workspace_path(self, state: SessionState, file_path: str) -> str:
        """Map a worktree-absolute path to the equivalent workspace path."""
        if not state._worktree_path:
            return file_path
        wt_root = Path(state._worktree_path).resolve()
        ws_root = self._workspace_root_for_state(state)
        p = Path(file_path).resolve()
        try:
            rel = p.relative_to(wt_root)
            return str((ws_root / rel).resolve())
        except ValueError:
            return file_path

    def _workspace_to_worktree_path(self, state: SessionState, file_path: str) -> str:
        """Map a workspace-absolute path to the equivalent worktree path."""
        if not state._worktree_path:
            return file_path
        ws_root = self._workspace_root_for_state(state)
        wt_root = Path(state._worktree_path).resolve()
        p = Path(file_path).resolve()
        try:
            rel = p.relative_to(ws_root)
            return str((wt_root / rel).resolve())
        except ValueError:
            return file_path

    def _ensure_git_repo(self, workspace_root: Path) -> bool:
        """Ensure the workspace is a git repo, running git init if needed."""
        if ProjectManager.is_git_repo(workspace_root):
            return True
        try:
            result = subprocess.run(
                ["git", "init"],
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("Initialized git repo at %s for session worktree support", workspace_root)
                return True
            logger.warning("git init failed at %s: %s", workspace_root, result.stderr.strip())
        except Exception as e:
            logger.warning("git init failed at %s: %s", workspace_root, e)
        return False

    def _setup_session_worktree(self, state: SessionState) -> None:
        """Create a detached-HEAD git worktree for session isolation."""
        workspace_root = self._workspace_root_for_state(state)
        if not self._ensure_git_repo(workspace_root):
            return
        wt_path = f"/tmp/prsm-wt-{state.session_id}"
        # Clean up stale worktree from previous server run (e.g. crash recovery)
        if Path(wt_path).exists():
            ProjectManager.remove_worktree(wt_path, force=True, cwd=workspace_root)
        ok, msg = ProjectManager.create_worktree(wt_path, cwd=workspace_root)
        if ok:
            state._worktree_path = wt_path
            state.session.worktree = WorktreeMetadata(root=wt_path)
            logger.info("Created session worktree session=%s path=%s", state.session_id[:8], wt_path)
        else:
            logger.warning("Failed to create worktree session=%s: %s", state.session_id[:8], msg)

    def _cleanup_session_worktree(self, state: SessionState) -> None:
        """Remove a session's worktree."""
        if not state._worktree_path:
            return
        wt_path = state._worktree_path
        workspace_root = self._workspace_root_for_state(state)
        ok, msg = ProjectManager.remove_worktree(wt_path, force=True, cwd=workspace_root)
        if ok:
            logger.info("Removed session worktree session=%s path=%s", state.session_id[:8], wt_path)
        else:
            logger.warning("Failed to remove worktree session=%s: %s", state.session_id[:8], msg)
        state._worktree_path = None

    def _has_pending_file_changes(self, state: SessionState) -> bool:
        """Return True if any file change records are still pending."""
        for records in state.file_tracker.file_changes.values():
            for r in records:
                if r.status == "pending":
                    return True
        return False

    def _maybe_cleanup_empty_worktree(self, state: SessionState) -> None:
        """Remove the session worktree if no pending file changes remain."""
        if not state._worktree_path:
            return
        if self._has_pending_file_changes(state):
            return
        self._cleanup_session_worktree(state)
        logger.info("Cleaned up empty worktree session=%s (no pending changes)", state.session_id[:8])

    def _ensure_session_worktree(self, state: SessionState) -> None:
        """Create a worktree for the session if one doesn't exist, then reconfigure the bridge."""
        if state._worktree_path:
            return  # Already has a worktree
        self._setup_session_worktree(state)
        if state._worktree_path:
            # Reconfigure bridge to use the new worktree cwd
            self._configure_bridge(
                state.bridge,
                project_id=state.project_id,
                cwd=str(self._agent_cwd_for_state(state)),
            )

    def _prune_orphaned_worktrees(self) -> None:
        """Remove /tmp/prsm-wt-* dirs that don't belong to any known session.

        Called once at server startup to clean up stale worktrees left by
        a previous server process that crashed or exited uncleanly.
        """
        import glob as glob_mod
        known_ids: set[str] = set()
        # Collect IDs from loaded sessions and the on-disk index.
        for sid in self._sessions:
            known_ids.add(sid)
        for sid in self._session_index:
            known_ids.add(sid)

        pruned = 0
        for wt_dir in glob_mod.glob("/tmp/prsm-wt-*"):
            # Extract session ID from the directory name.
            basename = Path(wt_dir).name  # "prsm-wt-<session_id>"
            sid = basename.removeprefix("prsm-wt-")
            if sid in known_ids:
                continue  # Belongs to a known session — leave it.
            try:
                shutil.rmtree(wt_dir, ignore_errors=True)
                pruned += 1
            except Exception:
                logger.debug("Failed to prune orphaned worktree %s", wt_dir)
        # Also ask git to prune its worktree bookkeeping.
        if pruned > 0:
            for project_id in self._known_projects:
                try:
                    cwd = str(self._workspace_dir_for_project(project_id))
                    subprocess.run(
                        ["git", "worktree", "prune"],
                        cwd=cwd,
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass
            logger.info("Pruned %d orphaned worktrees at startup", pruned)

    def _sync_file_to_workspace(self, state: SessionState, record: FileChangeRecord) -> None:
        """Apply a file change as a patch to the workspace.

        Rather than copying the whole worktree file (which would overwrite
        workspace edits from other accepted sessions), we compute the diff
        between pre_tool_content and new_content and apply only that delta.
        This preserves concurrent workspace changes when multiple sessions
        edit the same file.
        """
        logger.info(
            "sync_file_to_workspace: session=%s worktree_path=%s record.file_path=%s change_type=%s",
            state.session_id[:8], state._worktree_path, record.file_path, record.change_type,
        )

        # Resolve the workspace path for this record.
        if state._worktree_path:
            ws_path = Path(self._worktree_to_workspace_path(state, record.file_path))
        else:
            # Check if this was a worktree session whose worktree is now gone.
            wt_meta = state.session.worktree
            was_worktree = (
                wt_meta
                and wt_meta.root
                and wt_meta.root.startswith("/tmp/prsm-wt-")
            )
            if not was_worktree:
                return  # Genuine direct-workspace mode — file already there
            ws_path = self._resolve_workspace_path_from_record(state, record)

        if record.change_type == "delete":
            ws_path.unlink(missing_ok=True)
            return

        new_content = record.new_content
        if new_content is None:
            logger.warning(
                "Cannot sync change — no new_content in record session=%s tool_call=%s",
                state.session_id[:8], record.tool_call_id[:12],
            )
            return

        ws_path.parent.mkdir(parents=True, exist_ok=True)

        if record.change_type == "create" or not ws_path.exists():
            ws_path.write_text(new_content, encoding="utf-8")
            logger.info("sync_file_to_workspace: created %s", ws_path.name)
            return

        # For modify: apply as a diff patch to preserve concurrent workspace edits.
        pre_content = record.pre_tool_content if record.pre_tool_content is not None else record.old_content
        self._apply_diff_to_workspace_file(
            ws_path, pre_content, new_content, state.session_id, record.tool_call_id
        )

    def _apply_diff_to_workspace_file(
        self,
        ws_path: Path,
        pre_content: str | None,
        new_content: str,
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Apply the delta (pre_content → new_content) to the workspace file as a patch.

        If the workspace file exactly matches pre_content (no divergence), the
        new content is written directly.  Otherwise a unified diff is generated
        and applied with `patch -u` so that concurrent edits from other sessions
        are preserved.  Falls back to a direct write if patching fails.
        """
        try:
            current = ws_path.read_text(encoding="utf-8")
        except Exception:
            ws_path.write_text(new_content, encoding="utf-8")
            logger.info("sync_file_to_workspace: wrote (unreadable) %s", ws_path.name)
            return

        if pre_content is None or current == pre_content:
            # No divergence — write directly.
            ws_path.write_text(new_content, encoding="utf-8")
            logger.info("sync_file_to_workspace: wrote (no divergence) %s", ws_path.name)
            return

        # Workspace has diverged from the session base — apply as a patch.
        pre_lines = pre_content.splitlines()
        new_lines = new_content.splitlines()
        diff_lines = list(difflib.unified_diff(
            pre_lines, new_lines,
            fromfile="original", tofile="modified",
            lineterm="",
        ))
        if not diff_lines:
            return  # No actual change

        patch_text = "\n".join(diff_lines) + "\n"
        patch_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        )
        try:
            patch_file.write(patch_text)
            patch_file.close()
            result = subprocess.run(
                ["patch", "-u", str(ws_path), patch_file.name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(
                    "sync_file_to_workspace: patched %s session=%s",
                    ws_path.name, session_id[:8],
                )
            else:
                logger.warning(
                    "patch failed for %s (session=%s tool=%s): %s — writing directly",
                    ws_path.name, session_id[:8], tool_call_id[:12], result.stderr.strip(),
                )
                ws_path.write_text(new_content, encoding="utf-8")
        except FileNotFoundError:
            # `patch` not available — fall back to direct write.
            logger.warning(
                "`patch` command not found — writing %s directly", ws_path.name
            )
            ws_path.write_text(new_content, encoding="utf-8")
        finally:
            os.unlink(patch_file.name)

    def _resolve_workspace_path_from_record(
        self, state: SessionState, record: FileChangeRecord
    ) -> Path:
        """Resolve the workspace path for a record when worktree is gone.

        The record's file_path may be a workspace-absolute path or a
        (now-dead) worktree path.  Strip the worktree prefix if present
        and resolve relative to the workspace root.
        """
        fp = record.file_path
        wt_meta = state.session.worktree
        if wt_meta and wt_meta.root:
            wt_root = wt_meta.root.rstrip("/")
            # Also handle /private/tmp/... (macOS)
            for prefix in [wt_root, f"/private{wt_root}"]:
                if fp.startswith(prefix + "/"):
                    rel = fp[len(prefix) + 1:]
                    return self._workspace_root_for_state(state) / rel
        # Already a workspace path or can't determine — use as-is
        return Path(fp)

    def _git_head_content(self, cwd: Path, rel_path: str) -> str | None:
        """Return file content from HEAD in the given git working directory."""
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{rel_path}"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None

    def _worktree_file_change_fallback(
        self,
        state: SessionState,
        tool_id: str,
        agent_id: str,
        message_index: int,
    ) -> list[FileChangeRecord]:
        """Fallback file change detection for worktree sessions.

        When the snapshot-based tracker produces no records (due to event
        timing — the ToolCallStarted event is processed after the tool
        already ran), compare worktree files against git HEAD to detect
        changes. Uses tool call arguments from the session to attribute
        changes to specific tool calls.
        """
        if not state._worktree_path:
            return []

        wt_root = Path(state._worktree_path)

        # Find the tool call arguments in session messages.
        tool_name = ""
        file_path = ""
        args: dict = {}
        agent_messages = state.session.get_messages(agent_id)
        for msg in reversed(agent_messages):
            for tc in msg.tool_calls:
                if tc.id == tool_id:
                    tool_name = normalize_tool_name(tc.name)
                    raw_args = tc.arguments
                    if isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args)
                            if isinstance(parsed, dict):
                                args = parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    file_path = args.get("file_path", "")
                    break
            if tool_name:
                break

        # Only handle file-writing tools.
        if tool_name not in ("Edit", "Write"):
            # For Bash and other tools, fall through to git-diff-based detection below.
            return self._worktree_diff_fallback(state, tool_id, agent_id, message_index, tool_name)

        if not file_path:
            return []

        abs_path = Path(file_path)
        if not abs_path.is_absolute():
            abs_path = (wt_root / file_path).resolve()

        # Read current file content (post-edit).
        current_content = None
        if abs_path.exists() and abs_path.is_file():
            try:
                current_content = abs_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # Get relative path for git show.
        try:
            rel_path = str(abs_path.resolve().relative_to(wt_root.resolve()))
        except ValueError:
            return []

        # Check if this file is already tracked with a record for this tool_call_id.
        for records in state.file_tracker.file_changes.values():
            for r in records:
                if r.tool_call_id == tool_id:
                    return []  # Already tracked

        # Get HEAD content as the base.
        head_content = self._git_head_content(wt_root, rel_path)

        # Use last known record's new_content as pre-content if available.
        # Records are keyed by workspace path, but abs_path here is a worktree
        # path, so canonicalize before lookup.
        canonical_fp = self._canonicalize_file_path_for_state(state, str(abs_path))
        existing_records = state.file_tracker.file_changes.get(canonical_fp, [])
        if existing_records:
            pre_content = existing_records[-1].new_content
        else:
            pre_content = head_content

        # Determine change type.
        if pre_content is None and current_content is not None:
            change_type = "create"
        elif pre_content is not None and current_content is None:
            change_type = "delete"
        elif pre_content == current_content:
            return []  # No actual change
        else:
            change_type = "modify"

        record = FileChangeRecord(
            file_path=str(abs_path),
            agent_id=agent_id,
            change_type=change_type,
            tool_call_id=tool_id,
            tool_name=tool_name,
            message_index=message_index,
            old_content=pre_content,
            new_content=current_content,
            pre_tool_content=pre_content,
            added_ranges=[],
            removed_ranges=[],
            timestamp=datetime.now().isoformat(),
        )
        state.file_tracker.file_changes.setdefault(str(abs_path), []).append(record)
        logger.info(
            "Worktree fallback: tracked %s change file=%s tool=%s session=%s",
            change_type, rel_path, tool_id[:12], state.session_id[:8],
        )
        return [record]

    def _worktree_diff_fallback(
        self,
        state: SessionState,
        tool_id: str,
        agent_id: str,
        message_index: int,
        tool_name: str,
    ) -> list[FileChangeRecord]:
        """Detect file changes via git diff HEAD for Bash and other multi-file tools."""
        if not state._worktree_path:
            return []
        wt_root = Path(state._worktree_path)
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--name-only"],
                cwd=str(wt_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []

        # Also include untracked files.
        try:
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(wt_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            untracked_paths = set(untracked.stdout.strip().splitlines()) if untracked.returncode == 0 else set()
        except Exception:
            untracked_paths = set()

        changed_rel_paths = set(result.stdout.strip().splitlines()) | untracked_paths
        if not changed_rel_paths:
            return []

        # Filter to files not already tracked.
        # Build both workspace-root and worktree-root resolved paths so we
        # match regardless of which form the existing record uses.
        tracked_abs_paths: set[str] = set()
        ws_root = self._workspace_root_for_state(state)
        for fp in state.file_tracker.file_changes:
            resolved = Path(fp).resolve()
            tracked_abs_paths.add(str(resolved))
            # Also add the equivalent worktree path so a workspace-keyed record
            # blocks duplicate worktree-path records (and vice-versa).
            try:
                rel = resolved.relative_to(ws_root)
                tracked_abs_paths.add(str((wt_root / rel).resolve()))
            except ValueError:
                pass
            try:
                rel = resolved.relative_to(wt_root)
                tracked_abs_paths.add(str((ws_root / rel).resolve()))
            except ValueError:
                pass

        records: list[FileChangeRecord] = []
        for rel_path in sorted(changed_rel_paths):
            if not rel_path or rel_path.startswith(".git/"):
                continue
            abs_path = (wt_root / rel_path).resolve()
            if str(abs_path) in tracked_abs_paths:
                continue
            if self._is_filtered_workspace_artifact_path(rel_path):
                continue

            current_content = None
            if abs_path.exists() and abs_path.is_file():
                try:
                    current_content = abs_path.read_text(encoding="utf-8")
                except Exception:
                    continue

            head_content = self._git_head_content(wt_root, rel_path)

            if head_content is None and current_content is not None:
                change_type = "create"
            elif head_content is not None and current_content is None:
                change_type = "delete"
            elif head_content == current_content:
                continue
            else:
                change_type = "modify"

            record_tool_id = tool_id if len(changed_rel_paths) == 1 else f"{tool_id}:{len(records)}"
            record = FileChangeRecord(
                file_path=str(abs_path),
                agent_id=agent_id,
                change_type=change_type,
                tool_call_id=record_tool_id,
                tool_name=tool_name or "Bash",
                message_index=message_index,
                old_content=head_content,
                new_content=current_content,
                pre_tool_content=head_content,
                added_ranges=[],
                removed_ranges=[],
                timestamp=datetime.now().isoformat(),
            )
            state.file_tracker.file_changes.setdefault(str(abs_path), []).append(record)
            records.append(record)

        if records:
            logger.info(
                "Worktree diff fallback: tracked %d changes tool=%s session=%s",
                len(records), tool_id[:12], state.session_id[:8],
            )
        return records

    def _canonicalize_file_path_for_state(self, state: SessionState, file_path: str) -> str:
        """Normalize file paths to absolute workspace-rooted form."""
        if not file_path:
            return file_path
        workspace_root = self._workspace_root_for_state(state)
        p = Path(file_path)
        abs_path = p.resolve() if p.is_absolute() else (workspace_root / p).resolve()

        try:
            abs_path.relative_to(workspace_root)
            return str(abs_path)
        except ValueError:
            pass

        # Map worktree paths to workspace paths.
        if state._worktree_path:
            try:
                wt_root = Path(state._worktree_path).resolve()
                rel = abs_path.relative_to(wt_root)
                return str((workspace_root / rel).resolve())
            except ValueError:
                pass

        # Legacy migration: map /tmp/{session_id}/... paths from old worktree records.
        if state.session.worktree and state.session.worktree.root:
            try:
                rel = abs_path.relative_to(Path(state.session.worktree.root).resolve())
                return str((workspace_root / rel).resolve())
            except ValueError:
                pass

        return str(abs_path)

    def _normalize_file_change_record_path(self, state: SessionState, record: FileChangeRecord) -> None:
        """Keep file-change tracker keys and record payload paths canonical to workspace root."""
        old_path = record.file_path
        canonical = self._canonicalize_file_path_for_state(state, record.file_path)
        old_key = record.file_path
        record.file_path = canonical

        if canonical != old_key:
            old_records = state.file_tracker.file_changes.get(old_key, [])
            state.file_tracker.file_changes[old_key] = [
                r for r in old_records if r.tool_call_id != record.tool_call_id
            ]
            if not state.file_tracker.file_changes[old_key]:
                state.file_tracker.file_changes.pop(old_key, None)

            new_records = state.file_tracker.file_changes.setdefault(canonical, [])
            if all(r.tool_call_id != record.tool_call_id for r in new_records):
                new_records.append(record)

            logger.debug(
                "File change path canonicalized session=%s tool=%s old=%s new=%s",
                state.session_id[:8],
                record.tool_call_id[:12],
                old_key,
                canonical,
            )

        canonical_path = Path(record.file_path).resolve()
        workspace_root = self._workspace_root_for_state(state)
        try:
            canonical_path.relative_to(workspace_root)
        except ValueError:
            logger.warning(
                "Normalized file change path outside workspace session=%s tool=%s path=%s pre=%s",
                state.session_id[:8],
                record.tool_call_id[:12],
                canonical_path,
                old_path,
            )

    def _workspace_path_for_state(self, state: SessionState, file_path: str) -> Path:
        workspace_root = self._workspace_root_for_state(state)
        p = Path(file_path)
        return p.resolve() if p.is_absolute() else (workspace_root / p).resolve()

    def _display_file_path_for_state(self, state: SessionState, file_path: str) -> str:
        mapped = self._worktree_to_workspace_path(state, file_path)
        return str(self._workspace_path_for_state(state, mapped))

    def _display_file_path_for_session_id(self, session_id: str, file_path: str) -> str:
        if not file_path:
            return file_path
        state = self._sessions.get(session_id)
        if state:
            return self._display_file_path_for_state(state, file_path)
        project_id = self._session_projects.get(session_id)
        if not project_id:
            meta = self._session_index.get(session_id)
            project_id = str(meta.get("project_id") or self._default_project_id) if meta else self._default_project_id
        workspace_root = self._workspace_dir_for_project(project_id).resolve()
        p = Path(file_path)
        abs_path = p.resolve() if p.is_absolute() else (workspace_root / p).resolve()
        tmp_root = (Path("/tmp") / session_id).resolve()
        try:
            rel = abs_path.relative_to(tmp_root)
            return str((workspace_root / rel).resolve())
        except ValueError:
            pass
        try:
            rel = abs_path.relative_to(workspace_root)
            return str((workspace_root / rel).resolve())
        except ValueError:
            return str(abs_path)

    @staticmethod
    def _safe_read_text(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    @staticmethod
    def _paths_equal(path_a: Path, path_b: Path) -> bool:
        if not path_a.exists() and not path_b.exists():
            return True
        if path_a.exists() != path_b.exists():
            return False
        if path_a.is_dir() or path_b.is_dir():
            return False
        try:
            if path_a.stat().st_size != path_b.stat().st_size:
                return False
            return path_a.read_bytes() == path_b.read_bytes()
        except Exception:
            return False

    @staticmethod
    def _is_filtered_workspace_artifact_path(path_str: str) -> bool:
        if not path_str:
            return False
        try:
            p = Path(path_str)
            parts = p.parts
        except Exception:
            return False
        if any(part.startswith(".tmp") for part in parts if part and part != "."):
            return True
        name = p.name
        if name == ".file_index.txt":
            return True
        if re.fullmatch(r"\.grep_.*\.txt", name):
            return True
        if re.fullmatch(r"\.pytest_.*\.txt", name):
            return True
        if re.fullmatch(r"\.diff_.*\.txt", name):
            return True
        if re.fullmatch(r"\.compile_.*\.txt", name):
            return True
        return False

    def _status_path_set(self, root: Path) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return set()
        if proc.returncode != 0:
            return set()

        paths: set[str] = set()
        for raw_line in proc.stdout.splitlines():
            line = raw_line.rstrip()
            if len(line) < 4:
                continue
            path_part = line[3:]
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1]
            rel_path = path_part.strip()
            if not rel_path or rel_path.startswith(".git/"):
                continue
            if self._is_filtered_workspace_artifact_path(rel_path):
                continue
            paths.add(rel_path)
        return paths

    def _revert_workspace_file(self, record: FileChangeRecord, state: SessionState | None = None) -> None:
        """Restore a file to its pre-tool state.

        When the session has a worktree, the revert targets the worktree copy
        (agents write there, not the workspace).  ``record.file_path`` may be
        a worktree path *or* a workspace path depending on how the record was
        created, so we always resolve to the worktree path when available.
        """
        if state and state._worktree_path:
            path = Path(self._workspace_to_worktree_path(state, record.file_path))
        else:
            path = Path(record.file_path)
        if record.change_type == "create":
            path.unlink(missing_ok=True)
            return
        restore = record.pre_tool_content if record.pre_tool_content is not None else record.old_content
        if restore is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(restore, encoding="utf-8")

    def _latest_patch_file_for_session(self, session_id: str, state: SessionState | None = None) -> Path | None:
        snapshot_root = self._persistence.project_dir / "snapshots"
        if not snapshot_root.exists():
            return None
        latest: Path | None = None
        latest_mtime = -1.0
        for meta_path in snapshot_root.glob("*/meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(data.get("session_id") or "") != session_id:
                continue
            patch_path = meta_path.parent / "working_tree.patch"
            if not patch_path.exists():
                continue
            mtime = patch_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = patch_path
        return latest

    def _parse_working_tree_patch(self, patch_text: str) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                if current is not None:
                    changes.append(current)
                current = {
                    "file_path": "",
                    "change_type": "modify",
                    "added_ranges": [],
                    "removed_ranges": [],
                }
                continue
            if current is None:
                continue
            if line.startswith("--- "):
                if line.strip() == "--- /dev/null":
                    current["change_type"] = "create"
                else:
                    lhs = line[4:].strip()
                    if lhs.startswith("a/"):
                        lhs = lhs[2:]
                    if lhs != "/dev/null":
                        current["file_path"] = lhs
                continue
            if line.startswith("+++ "):
                rhs = line[4:].strip()
                if rhs == "/dev/null":
                    current["change_type"] = "delete"
                    continue
                if rhs.startswith("b/"):
                    rhs = rhs[2:]
                current["file_path"] = rhs
                continue
            match = hunk_re.match(line)
            if not match:
                continue
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            if old_count > 0:
                current["removed_ranges"].append(
                    {"startLine": old_start - 1, "endLine": old_start + old_count - 2}
                )
            if new_count > 0:
                current["added_ranges"].append(
                    {"startLine": new_start - 1, "endLine": new_start + new_count - 2}
                )
        if current is not None:
            changes.append(current)
        return [c for c in changes if c.get("file_path")]

    def _file_changes_from_patch_fallback(
        self,
        session_id: str,
        state: SessionState,
        include_content: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        patch_path = self._latest_patch_file_for_session(session_id, state)
        if patch_path is None or not patch_path.exists():
            return {}
        try:
            parsed = self._parse_working_tree_patch(patch_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        workspace_root = self._workspace_root_for_state(state)
        result: dict[str, list[dict[str, Any]]] = {}
        for idx, change in enumerate(parsed):
            abs_path = str((workspace_root / str(change["file_path"])).resolve())
            payload = {
                "file_path": abs_path,
                "agent_id": "git-patch-fallback",
                "change_type": change.get("change_type", "modify"),
                "tool_call_id": f"git-patch-{session_id}-{idx}",
                "tool_name": "Write",
                "message_index": 0,
                "old_content": None,
                "new_content": None,
                "pre_tool_content": None,
                "added_ranges": change.get("added_ranges", []),
                "removed_ranges": change.get("removed_ranges", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "has_content": False,
            }
            if include_content:
                payload["old_content"] = None
                payload["new_content"] = None
                payload["pre_tool_content"] = None
            result.setdefault(abs_path, []).append(payload)
        return result

    def _find_file_change_record(
        self,
        state: SessionState,
        tool_call_id: str,
    ) -> tuple[str, int, FileChangeRecord] | None:
        for file_path, records in state.file_tracker.file_changes.items():
            for idx, record in enumerate(records):
                if record.tool_call_id == tool_call_id:
                    return file_path, idx, record
        return None

    def _git_show_head_content(self, file_path: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "show", f"HEAD:{file_path}"],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout

    async def _handle_get_file_changes(self, request: web.Request) -> web.Response:
        session_id = request.match_info["id"]
        state = self._sessions.get(session_id)
        if state:
            self._touch_session(state)
            tracker = state.file_tracker
        else:
            if session_id not in self._session_index:
                return web.json_response({"error": f"Session {session_id} not found"}, status=404)
            tracker = FileChangeTracker()
            changes_dir = self._persistence.project_dir / "sessions" / session_id / "file-changes"
            tracker.load(changes_dir)
            if self._normalize_cold_file_change_paths(session_id, tracker):
                tracker.persist(changes_dir)

        changes = {}
        for file_path, records in tracker.file_changes.items():
            pending = [r for r in records if r.status == "pending"]
            if not pending:
                continue
            display_path = (
                self._display_file_path_for_state(state, file_path)
                if state
                else self._display_file_path_for_session_id(session_id, file_path)
            )
            payloads = [
                {
                    "file_path": (
                        self._display_file_path_for_state(state, r.file_path)
                        if state
                        else self._display_file_path_for_session_id(session_id, r.file_path)
                    ),
                    "agent_id": r.agent_id,
                    "change_type": r.change_type, "tool_call_id": r.tool_call_id,
                    "tool_name": r.tool_name, "message_index": r.message_index,
                    "old_content": r.old_content, "new_content": r.new_content,
                    "pre_tool_content": r.pre_tool_content,
                    "added_ranges": r.added_ranges, "removed_ranges": r.removed_ranges,
                    "timestamp": r.timestamp, "status": r.status,
                }
                for r in pending
            ]
            changes.setdefault(display_path, []).extend(payloads)
        return web.json_response({"file_changes": changes})

    async def _handle_accept_change(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        tool_call_id = request.match_info["tool_call_id"]
        logger.info(
            "accept_change: session=%s tool_call_id=%s worktree=%s",
            state.session_id[:8], tool_call_id[:16], state._worktree_path,
        )
        async with state._file_sync_lock:
            found = self._find_file_change_record(state, tool_call_id)
            if found:
                file_path, idx, record = found
                logger.info(
                    "accept_change: found record file=%s change_type=%s",
                    record.file_path, record.change_type,
                )
                # Mark the record as accepted and sync from worktree to workspace.
                record.status = "accepted"
                self._sync_file_to_workspace(state, record)
                self._broadcast_sse("file_change_status", {
                    "session_id": state.session_id,
                    "tool_call_id": tool_call_id,
                    "status": "accepted",
                    "file_path": self._display_file_path_for_state(state, record.file_path),
                })
                self._persist_file_changes(state)
                self._maybe_cleanup_empty_worktree(state)
                return web.json_response({"status": "accepted"})
            return web.json_response({"error": "Change not found"}, status=404)

    async def _handle_reject_change(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        tool_call_id = request.match_info["tool_call_id"]
        async with state._file_sync_lock:
            found = self._find_file_change_record(state, tool_call_id)
            if found:
                file_path, idx, record = found
                # Revert the workspace file to pre-tool state
                self._revert_workspace_file(record, state)

                # Mark target as rejected
                record.status = "rejected"

                # Cascade-reject: all later pending records for this file are
                # invalidated because we restored pre_tool_content, undoing
                # subsequent changes too.
                records = state.file_tracker.file_changes.get(file_path, [])
                cascade_rejected: list[str] = []
                for r in records:
                    if r.tool_call_id != tool_call_id and r.message_index > record.message_index and r.status == "pending":
                        r.status = "rejected"
                        cascade_rejected.append(r.tool_call_id)

                # Broadcast status for target + cascaded records
                display = self._display_file_path_for_state(state, record.file_path)
                for tcid in [tool_call_id, *cascade_rejected]:
                    self._broadcast_sse("file_change_status", {
                        "session_id": state.session_id,
                        "tool_call_id": tcid,
                        "status": "rejected",
                        "file_path": display,
                    })
                self._persist_file_changes(state)
                self._maybe_cleanup_empty_worktree(state)

                return web.json_response({
                    "status": "rejected",
                    "old_content": record.old_content,
                    "new_content": record.new_content,
                    "pre_tool_content": record.pre_tool_content,
                    "cascade_rejected": cascade_rejected,
                })
            return web.json_response({"error": "Change not found"}, status=404)

    async def _handle_accept_all_changes(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        logger.info(
            "accept_all_changes: session=%s worktree=%s",
            state.session_id[:8], state._worktree_path,
        )
        async with state._file_sync_lock:
            count = 0
            for file_path in list(state.file_tracker.file_changes.keys()):
                records = state.file_tracker.file_changes.get(file_path, [])
                for r in records:
                    if r.status == "pending":
                        r.status = "accepted"
                        self._sync_file_to_workspace(state, r)
                        count += 1
            if count > 0:
                self._broadcast_sse("file_changes_bulk_status", {
                    "session_id": state.session_id,
                    "status": "accepted",
                    "count": count,
                })
                self._persist_file_changes(state)
            self._maybe_cleanup_empty_worktree(state)
            return web.json_response({"status": "accepted", "count": count})

    async def _handle_reject_all_changes(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        async with state._file_sync_lock:
            count = 0
            rejected_changes: list[dict[str, Any]] = []
            for file_path in list(state.file_tracker.file_changes.keys()):
                records = state.file_tracker.file_changes.get(file_path, [])
                pending = [r for r in records if r.status == "pending"]
                if not pending:
                    continue
                # Revert to the earliest pending record's pre-tool state
                earliest = min(pending, key=lambda r: r.message_index)
                self._revert_workspace_file(earliest, state)
                count += len(pending)
                for r in pending:
                    r.status = "rejected"
                    rejected_changes.append({
                        "tool_call_id": r.tool_call_id,
                        "file_path": self._display_file_path_for_state(state, r.file_path),
                        "old_content": r.old_content,
                        "new_content": r.new_content,
                        "pre_tool_content": r.pre_tool_content,
                    })
            if count > 0:
                self._broadcast_sse("file_changes_bulk_status", {
                    "session_id": state.session_id,
                    "status": "rejected",
                    "count": count,
                })
                self._persist_file_changes(state)
            self._maybe_cleanup_empty_worktree(state)
            return web.json_response({"status": "rejected", "count": count, "changes": rejected_changes})

    def _file_changes_dir(self, state: SessionState) -> Path:
        """Return the on-disk directory for a session's file-change records."""
        return self._persistence.project_dir / "sessions" / state.session_id / "file-changes"

    def _persist_file_changes(self, state: SessionState) -> None:
        """Persist file change records to disk for recovery across restarts."""
        state.file_tracker.persist(self._file_changes_dir(state))
        self._save_session(state)

    def _load_file_changes(self, state: SessionState) -> None:
        """Load persisted file change records from disk."""
        state.file_tracker.load(self._file_changes_dir(state))
        changed = self._normalize_loaded_file_change_paths(state)
        if changed:
            # Rewrite legacy/tmp-rooted records in canonical form.
            state.file_tracker.persist(self._file_changes_dir(state))
        # Reconnect to an existing worktree and reconcile untracked changes.
        self._reconnect_session_worktree(state)

    def _reconnect_session_worktree(self, state: SessionState) -> None:
        """Reconnect a loaded session to its existing worktree and reconcile changes.

        When a server restarts, sessions loaded from disk may have a worktree
        on disk but no _worktree_path set.  This method restores the link and
        uses git diff to create FileChangeRecords for any changes in the
        worktree that were not previously tracked.
        """
        if state._worktree_path:
            return  # Already connected

        # Check if session metadata references a worktree that still exists.
        wt_root: str | None = None
        if state.session.worktree and state.session.worktree.root:
            candidate = state.session.worktree.root
            if Path(candidate).is_dir():
                wt_root = candidate

        # Also check the conventional path.
        if not wt_root:
            candidate = f"/tmp/prsm-wt-{state.session_id}"
            if Path(candidate).is_dir():
                wt_root = candidate

        if not wt_root:
            return

        # Verify it's still a valid git worktree.
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=wt_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return
        except Exception:
            return

        state._worktree_path = wt_root
        logger.info(
            "Reconnected session worktree session=%s path=%s",
            state.session_id[:8], wt_root,
        )

        # Reconcile: create records for worktree changes not already tracked.
        self._reconcile_worktree_changes(state)

        # If the worktree is clean (no pending changes after reconciliation),
        # prune it immediately so stale empty worktrees don't accumulate.
        self._maybe_cleanup_empty_worktree(state)

    def _reconcile_worktree_changes(self, state: SessionState) -> None:
        """Create FileChangeRecords for worktree changes not already tracked."""
        if not state._worktree_path:
            return
        wt_root = Path(state._worktree_path)

        # Get all files changed from HEAD in the worktree.
        try:
            diff_result = subprocess.run(
                ["git", "diff", "HEAD", "--name-only"],
                cwd=str(wt_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return
        if diff_result.returncode != 0:
            return

        # Also untracked files.
        try:
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(wt_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            untracked = set(untracked_result.stdout.strip().splitlines()) if untracked_result.returncode == 0 else set()
        except Exception:
            untracked = set()

        changed_rel_paths = set(diff_result.stdout.strip().splitlines()) | untracked
        if not changed_rel_paths:
            return

        # Collect already-tracked absolute paths — both workspace and worktree
        # forms so a workspace-rooted record blocks duplicate worktree entries.
        ws_root = self._workspace_root_for_state(state)
        tracked_abs: set[str] = set()
        for fp in state.file_tracker.file_changes:
            resolved = Path(fp).resolve()
            tracked_abs.add(str(resolved))
            try:
                rel = resolved.relative_to(ws_root)
                tracked_abs.add(str((wt_root / rel).resolve()))
            except ValueError:
                pass
            try:
                rel = resolved.relative_to(wt_root)
                tracked_abs.add(str((ws_root / rel).resolve()))
            except ValueError:
                pass

        count = 0
        for rel_path in sorted(changed_rel_paths):
            if not rel_path or rel_path.startswith(".git/"):
                continue
            abs_path = (wt_root / rel_path).resolve()
            if str(abs_path) in tracked_abs:
                continue
            if self._is_filtered_workspace_artifact_path(rel_path):
                continue

            current_content = None
            if abs_path.exists() and abs_path.is_file():
                try:
                    current_content = abs_path.read_text(encoding="utf-8")
                except Exception:
                    continue

            head_content = self._git_head_content(wt_root, rel_path)

            if head_content is None and current_content is not None:
                change_type = "create"
            elif head_content is not None and current_content is None:
                change_type = "delete"
            elif head_content == current_content:
                continue
            else:
                change_type = "modify"

            record = FileChangeRecord(
                file_path=str(abs_path),
                agent_id="unknown",
                change_type=change_type,
                tool_call_id=f"reconcile-{state.session_id[:8]}-{count}",
                tool_name="Edit",
                message_index=0,
                old_content=head_content,
                new_content=current_content,
                pre_tool_content=head_content,
                added_ranges=[],
                removed_ranges=[],
                timestamp=datetime.now().isoformat(),
            )
            state.file_tracker.file_changes.setdefault(str(abs_path), []).append(record)
            # Normalize path to workspace-rooted immediately so the record is
            # consistent with all other tracked changes (worktree is connected here).
            self._normalize_file_change_record_path(state, record)
            self._sync_file_to_workspace(state, record)
            count += 1

        if count > 0:
            self._persist_file_changes(state)
            logger.info(
                "Reconciled %d untracked worktree changes session=%s",
                count, state.session_id[:8],
            )

    def _normalize_loaded_file_change_paths(self, state: SessionState) -> bool:
        changed = False
        for records in list(state.file_tracker.file_changes.values()):
            for record in list(records):
                original = record.file_path
                self._normalize_file_change_record_path(state, record)
                if record.file_path != original:
                    changed = True

        seen_tool_ids: set[str] = set()
        for file_path in list(state.file_tracker.file_changes.keys()):
            records = state.file_tracker.file_changes.get(file_path, [])
            deduped: list[FileChangeRecord] = []
            for record in records:
                if self._is_filtered_workspace_artifact_path(record.file_path):
                    changed = True
                    continue
                if record.tool_call_id in seen_tool_ids:
                    changed = True
                    continue
                seen_tool_ids.add(record.tool_call_id)
                deduped.append(record)
            if len(deduped) != len(records):
                state.file_tracker.file_changes[file_path] = deduped
            if not deduped:
                state.file_tracker.file_changes.pop(file_path, None)
                changed = True
        return changed

    def _normalize_cold_file_change_paths(
        self,
        session_id: str,
        tracker: FileChangeTracker,
    ) -> bool:
        changed = False
        remapped: dict[str, list[FileChangeRecord]] = {}
        seen_tool_ids: set[str] = set()

        for file_path, records in tracker.file_changes.items():
            mapped_key = self._display_file_path_for_session_id(session_id, file_path)
            if mapped_key != file_path:
                changed = True
            if self._is_filtered_workspace_artifact_path(mapped_key):
                changed = True
                continue
            for record in records:
                mapped_path = self._display_file_path_for_session_id(session_id, record.file_path)
                if mapped_path != record.file_path:
                    changed = True
                if self._is_filtered_workspace_artifact_path(mapped_path):
                    changed = True
                    continue
                if record.tool_call_id in seen_tool_ids:
                    changed = True
                    continue
                seen_tool_ids.add(record.tool_call_id)
                if mapped_path != record.file_path:
                    record.file_path = mapped_path
                remapped.setdefault(mapped_key, []).append(record)

        if changed:
            tracker.file_changes = remapped
        return changed

    async def _handle_file_complete(self, request: web.Request) -> web.Response:
        prefix = request.query.get("prefix", "")
        limit = int(request.query.get("limit", "10"))
        try:
            index = self._get_file_index()
            matches = index.search(prefix)[:limit]
            results = [{"path": e.path, "is_directory": e.is_dir, "size": e.size} for e in matches]
            return web.json_response({"completions": results})
        except Exception as e:
            logger.exception("File completion failed: %s", e)
            return web.json_response({"completions": [], "error": str(e)})

    async def _handle_get_agent_history(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        agent_id = request.match_info["agent_id"]
        detail_level = request.query.get("detail_level", "full")
        if detail_level not in ("full", "summary"):
            return web.json_response({"error": "detail_level must be 'full' or 'summary'"}, status=400)
        if not state.bridge._engine:
            return web.json_response({"error": "Engine not available"}, status=503)

        history = state.bridge._engine.conversation_store.get_history(agent_id, detail_level=detail_level)
        return web.json_response({"agent_id": agent_id, "session_id": state.session_id, "detail_level": detail_level, "history": history})

    async def _handle_get_tool_rationale(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        agent_id = request.match_info["agent_id"]
        tool_call_id = request.match_info["tool_call_id"]

        if not state.bridge._engine:
            return web.json_response({"error": "Engine not available"}, status=503)
        try:
            from prsm.engine.rationale_extractor import extract_change_rationale
        except ImportError:
            return web.json_response({"error": "Rationale extractor module not available"}, status=501)

        conversation_store = state.bridge._engine.conversation_store
        agent_node = state.bridge._engine.get_agent(agent_id)
        agent_name = agent_node.name if agent_node else agent_id

        # Find tool name
        history = conversation_store.get_history(agent_id, detail_level="full")
        tool_name = "Unknown"
        for entry in history:
            if entry.get("type") == "tool_call" and entry.get("tool_id") == tool_call_id:
                tool_name = entry.get("tool_name", "Unknown")
                break

        try:
            rationale = extract_change_rationale(
                agent_id=agent_id, tool_call_id=tool_call_id,
                conversation_store=conversation_store, max_sentences=3,
            )
            return web.json_response({
                "agent_id": agent_id, "agent_name": agent_name,
                "session_id": state.session_id, "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "rationale": rationale or "No rationale found for this change.",
            })
        except Exception as e:
            logger.exception("Failed to extract tool rationale")
            return web.json_response({"error": f"Failed to extract rationale: {str(e)}"}, status=500)

    async def _handle_set_model(self, request: web.Request) -> web.Response:
        """Set the default model for the session.

        Also updates the completed master agent's descriptor so that the
        next prompt continues the conversation with the new model (context
        transfer), mirroring the TUI's switch_model() behaviour.
        """
        state, err = self._require_session(request)
        if err:
            return err

        try:
            body = await request.json()
            model_id = body.get("model_id", "").strip()
            if not model_id:
                return web.json_response({"error": "model_id is required"}, status=400)

            old_model = state.bridge.current_model

            # set_model returns (resolved_id, provider)
            resolved_id, provider = state.bridge.set_model(model_id)

            # Context transfer: update master descriptors in both active and
            # completed pools so continuation/restart always picks up the new
            # model/provider with the existing conversation history.
            engine = getattr(state.bridge, "_engine", None)
            if engine:
                master_id = getattr(engine, "_last_master_id", None)
                if not master_id:
                    master_id = self._get_master_agent_id(state)
                manager = getattr(engine, "_manager", None)
                if master_id and manager:
                    updated_pool_names: list[str] = []
                    for pool_name in ("_completed_agents", "_agents"):
                        pool = getattr(manager, pool_name, {})
                        if isinstance(pool, dict) and master_id in pool:
                            descriptor = pool[master_id]
                            descriptor.model = resolved_id
                            descriptor.provider = provider
                            updated_pool_names.append(pool_name)
                    if updated_pool_names:
                        logger.info(
                            "Updated master agent %s descriptor pools=%s model=%s provider=%s",
                            master_id[:8], ",".join(updated_pool_names), resolved_id, provider,
                        )
                    agent_node = state.session.agents.get(master_id)
                    if agent_node:
                        agent_node.model = resolved_id

            # Broadcast a model_switched SSE event so the extension can
            # show a visual indicator in the conversation.
            if old_model != resolved_id:
                self._broadcast_sse("model_switched", {
                    "session_id": state.session_id,
                    "old_model": old_model,
                    "new_model": resolved_id,
                    "provider": provider,
                })

            return web.json_response({
                "status": "ok",
                "model_id": resolved_id,
                "provider": provider,
                "old_model": old_model,
            })
        except Exception as e:
            logger.exception("Failed to set model")
            return web.json_response({"error": f"Failed to set model: {str(e)}"}, status=500)

    async def _handle_get_available_models(self, request: web.Request) -> web.Response:
        """Get list of available models for the session."""
        state, err = self._require_session(request)
        if err:
            return err

        try:
            models = state.bridge.get_available_models()
            runtime_info = self._get_runtime_info()
            runtime_aliases = runtime_info.get("model_aliases", {})
            if runtime_aliases and models:
                allowed_model_ids: set[str] = set()
                for alias_cfg in runtime_aliases.values():
                    if not isinstance(alias_cfg, dict):
                        continue
                    cfg_model = alias_cfg.get("model_id")
                    normalized_model_id = self._normalize_model_alias_id(str(cfg_model or ""))
                    if normalized_model_id:
                        allowed_model_ids.add(normalized_model_id)
                filtered_models = []
                for model in models:
                    model_id = self._normalize_model_alias_id(
                        str(model.get("model_id", ""))
                    )
                    if model_id in allowed_model_ids:
                        filtered_models.append(model)
                    elif not allowed_model_ids:
                        filtered_models.append(model)
                models = filtered_models
            return web.json_response({"models": models})
        except Exception as e:
            logger.exception("Failed to get available models")
            return web.json_response({"error": f"Failed to get available models: {str(e)}"}, status=500)

    # ── Configuration ──

    def _resolve_config_path(self) -> str:
        """Resolve config path for settings read/write.

        Preferred location is ``<cwd>/.prism/prsm.yaml``.  Falls back
        to ``<cwd>/prsm.yaml`` for legacy setups.  If launched with a
        generated temp config (e.g. /tmp/prsm-config.yaml), redirect to
        the project ``.prism/prsm.yaml`` so settings live alongside the
        sibling ``models.yaml``.
        """
        project_prism_prsm = os.path.join(self._cwd, ".prism", "prsm.yaml")
        legacy_project_prsm = os.path.join(self._cwd, "prsm.yaml")
        if os.path.exists(project_prism_prsm):
            return project_prism_prsm
        if self._config_path:
            cfg_path = Path(self._config_path)
            if (
                cfg_path.name == "prsm-config.yaml"
                and cfg_path.parent == Path(tempfile.gettempdir())
                and os.path.exists(project_prism_prsm)
            ):
                return project_prism_prsm
            return self._config_path
        return legacy_project_prsm

    def _resolve_models_yaml_path(self) -> str:
        """Resolve the models.yaml path used by model discovery.

        Discovery writes to the global ``~/.prsm/models.yaml`` so all
        projects share the same discovered model list.  This keeps runtime
        behavior and settings dropdowns consistent across workspaces.
        """
        return str(Path.home() / ".prsm" / "models.yaml")

    @staticmethod
    def _load_yaml_file(path: str) -> dict[str, Any]:
        """Load YAML file as dict; returns empty dict if missing/empty."""
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _models_path_candidates(config_path: str, cwd: str | None = None) -> list[Path]:
        """Return candidate ``models.yaml`` paths for compatibility.

        Runtime should use only the global file so all workspaces share the same
        model registry and project-local overrides are intentionally ignored.
        """
        return [Path.home() / ".prsm" / "models.yaml"]

    def _load_merged_config(self, config_path: str) -> dict[str, Any]:
        """Load prsm.yaml merged with ``models.yaml`` sections.

        ``models.yaml`` contributes ``models`` and ``model_registry``; ``prsm.yaml``
        wins on key conflicts so local overrides still work.

        Discovery uses ``~/.prsm/models.yaml`` only so all projects share the same
        model set.
        """
        prsm_raw = self._load_yaml_file(config_path)
        candidates = self._models_path_candidates(config_path, getattr(self, "_cwd", None))
        models_path_obj: Path = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                models_path_obj = candidate
                break
        models_path = str(models_path_obj)
        models_raw = self._load_yaml_file(models_path)

        merged: dict[str, Any] = dict(prsm_raw)

        merged_models = {
            **(models_raw.get("models", {}) or {}),
            **(prsm_raw.get("models", {}) or {}),
        }
        if merged_models:
            merged["models"] = merged_models

        merged_registry = {
            **(models_raw.get("model_registry", {}) or {}),
            **(prsm_raw.get("model_registry", {}) or {}),
        }
        if not merged_registry and merged_models:
            # Backward-compat: some configs place capability fields under
            # models entries instead of top-level model_registry.
            merged_registry = self._derive_model_registry_from_models(merged_models)
        if merged_registry:
            merged["model_registry"] = merged_registry

        return merged

    @staticmethod
    def _derive_model_registry_from_models(
        models: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Derive model_registry-like entries from capability-style models.

        Supports configs where models entries include fields like tier/cost/affinities
        rather than providing a separate top-level model_registry section.
        """
        derived: dict[str, dict[str, Any]] = {}
        for alias, cfg in (models or {}).items():
            if not isinstance(cfg, dict):
                continue
            if not any(
                key in cfg
                for key in ("tier", "cost_factor", "speed_factor", "affinities")
            ):
                continue
            model_id = str(cfg.get("model_id") or alias)
            effort = cfg.get("reasoning_effort")
            registry_id = (
                f"{model_id}::reasoning_effort={effort}"
                if effort
                else model_id
            )
            entry: dict[str, Any] = {}
            for key in (
                "tier",
                "provider",
                "cost_factor",
                "speed_factor",
                "available",
                "affinities",
            ):
                if key in cfg:
                    entry[key] = cfg[key]
            entry["model_id"] = model_id
            if effort:
                entry["reasoning_effort"] = effort
            if entry:
                derived[registry_id] = entry
        return derived

    def _write_split_config(
        self, config_path: str, new_config: dict[str, Any]
    ) -> None:
        """Persist config with model data in ``~/.prsm/models.yaml``.

        - ``prsm.yaml``: everything except ``models`` and ``model_registry``.
        - ``models.yaml``: ``models`` and ``model_registry`` only
        """
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        models_path = Path.home() / ".prsm" / "models.yaml"
        models_path.parent.mkdir(parents=True, exist_ok=True)

        prsm_out = dict(new_config)
        models_out: dict[str, Any] = {}

        if "models" in prsm_out:
            models_out["models"] = prsm_out.pop("models")
        if "model_registry" in prsm_out:
            models_out["model_registry"] = prsm_out.pop("model_registry")

        with open(config_path, "w") as f:
            yaml.dump(
                prsm_out,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        if models_out:
            with open(models_path, "w") as f:
                yaml.dump(
                    models_out,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

    async def _handle_get_config(self, request: web.Request) -> web.Response:
        """Return current configuration + detected runtime info as JSON.

        The response includes:
        - config: raw YAML contents
        - config_path: path to prsm.yaml
        - exists: whether the file exists
        - runtime: detected providers, models, and aliases from the
          real ProviderRegistry and ModelRegistry (same data the engine
          uses at runtime)
        """
        config_path = self._resolve_config_path()

        result: dict[str, Any] = {"config_path": config_path}

        if config_path and os.path.exists(config_path):
            result["config"] = self._load_merged_config(config_path)
            result["exists"] = True
        else:
            result["config"] = {}
            result["exists"] = False

        # Include runtime provider/model detection
        result["runtime"] = self._get_runtime_info()

        return web.json_response(result)

    async def _handle_update_config(self, request: web.Request) -> web.Response:
        """Update configuration: write to YAML and reload."""
        body = await request.json()
        new_config = body.get("config", {})

        config_path = self._resolve_config_path()
        self._config_path = config_path

        # Persist split config files.
        self._write_split_config(config_path, new_config)

        # Reload config
        try:
            from prsm.engine.yaml_config import load_yaml_config
            self._yaml_config = load_yaml_config(config_path)
            logger.info("Reloaded YAML config from %s", config_path)
        except Exception as exc:
            logger.exception("Failed to reload YAML config")
            return web.json_response(
                {"error": f"Config saved but reload failed: {exc}"},
                status=500,
            )

        # Rebuild registries with new config
        self._build_registries()

        # Reconfigure all active bridges
        for ss in self._sessions.values():
            self._configure_bridge(
                ss.bridge,
                project_id=ss.project_id,
                cwd=str(self._agent_cwd_for_state(ss)),
            )

        return web.json_response({"ok": True, "config_path": config_path})

    async def _handle_detect_providers(self, request: web.Request) -> web.Response:
        """Return runtime provider/model info from the real registries.

        Kept for backward compat — the /config endpoint now includes
        this same data in its 'runtime' field.
        """
        return web.json_response(self._get_runtime_info())

    async def _handle_get_thinking_verbs(self, request: web.Request) -> web.Response:
        """Return thinking verb lists loaded from shared_ui .txt files."""
        shared_ui_dir = Path(__file__).parent.parent / "shared_ui"

        def _load(name: str) -> list[str]:
            p = shared_ui_dir / name
            if not p.exists():
                return []
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]

        return web.json_response({
            "safe": _load("thinking_verbs.txt"),
            "nsfw": _load("nsfw_thinking_verbs.txt"),
        })

    async def _handle_get_preferences(self, request: web.Request) -> web.Response:
        prefs = UserPreferences.load()
        return web.json_response({"preferences": asdict(prefs)})

    async def _handle_update_preferences(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        raw = body.get("preferences", body)
        if not isinstance(raw, dict):
            return web.json_response({"error": "preferences must be an object"}, status=400)
        prefs = UserPreferences.load()
        for key in UserPreferences.__dataclass_fields__.keys():
            if key in raw:
                setattr(prefs, key, raw[key])
        prefs.validate()
        prefs.save()
        return web.json_response({"ok": True, "preferences": asdict(prefs)})

    @staticmethod
    def _normalize_pattern_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    async def _handle_get_command_policy(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        store = self._policy_store_for_state(state)
        store.ensure_files()
        return web.json_response(
            {
                "whitelist": store.read_whitelist(),
                "blacklist": store.read_blacklist(),
                "workspace": str(self._workspace_dir_for_project(state.project_id)),
            }
        )

    async def _handle_update_command_policy(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        whitelist = self._normalize_pattern_list(body.get("whitelist", []))
        blacklist = self._normalize_pattern_list(body.get("blacklist", []))
        store = self._policy_store_for_state(state)
        store.ensure_files()
        store.whitelist_path.write_text(
            ("\n".join(whitelist) + "\n") if whitelist else "",
            encoding="utf-8",
        )
        store.blacklist_path.write_text(
            ("\n".join(blacklist) + "\n") if blacklist else "",
            encoding="utf-8",
        )
        return web.json_response(
            {
                "ok": True,
                "whitelist": whitelist,
                "blacklist": blacklist,
            }
        )

    async def _handle_get_project_memory(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        memory = self._project_memory_for_state(state)
        content = memory.load()
        return web.json_response(
            {
                "content": content,
                "exists": memory.exists(),
                "path": str(memory.path),
            }
        )

    async def _handle_update_project_memory(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        content = body.get("content")
        if not isinstance(content, str):
            return web.json_response({"error": "content must be a string"}, status=400)
        memory = self._project_memory_for_state(state)
        memory.save(content)
        return web.json_response({"ok": True, "path": str(memory.path), "size": len(content)})

    async def _handle_get_policy(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        engine = state.bridge._engine
        triage_rules = []
        command_whitelist = []
        command_blacklist = []
        if engine is not None:
            triage_rules = list(getattr(engine._router, "_triage_rules", []))
            command_whitelist = list(getattr(engine._config, "command_whitelist", []))
            command_blacklist = list(getattr(engine._config, "command_blacklist", []))
        return web.json_response(
            {
                "project_id": state.project_id,
                "triage_rules": triage_rules,
                "command_whitelist": command_whitelist,
                "command_blacklist": command_blacklist,
            }
        )

    async def _handle_get_leases(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        entries = [
            evt for evt in state._governance_events if evt.get("event") == "lease_status"
        ]
        return web.json_response({"leases": entries})

    async def _handle_get_audit(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, int(limit_raw))
        except ValueError:
            return web.json_response({"error": "limit must be an integer"}, status=400)
        entries = [
            evt for evt in state._governance_events if evt.get("event") == "audit_entry"
        ][-limit:]
        return web.json_response({"entries": entries, "count": len(entries)})

    async def _handle_get_memory(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        return web.json_response({"entries": list(state._memory_entries)})

    async def _handle_add_memory(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        body = await request.json() if request.can_read_body else {}
        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            return web.json_response({"error": "content is required"}, status=400)
        entry = {
            "id": str(uuid.uuid4())[:12],
            "content": content.strip(),
            "scope": str(body.get("scope") or "project"),
            "category": str(body.get("category") or "general"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        state._memory_entries.append(entry)
        if len(state._memory_entries) > 1000:
            state._memory_entries = state._memory_entries[-1000:]
        return web.json_response({"entry": entry}, status=201)

    async def _handle_get_expert_stats(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        engine = state.bridge._engine
        if engine is None:
            return web.json_response({"experts": []})
        registry = engine.expert_registry
        profiles = registry.list_profiles()
        rankings = dict(registry.get_utility_rankings())
        experts = []
        for profile in profiles:
            experts.append(
                {
                    "expert_id": profile.expert_id,
                    "name": profile.name,
                    "lifecycle_state": profile.lifecycle_state,
                    "utility_score": rankings.get(profile.expert_id, profile.utility_score),
                    "consultation_count": profile.consultation_count,
                    "success_count": profile.success_count,
                    "failure_count": profile.failure_count,
                }
            )
        experts.sort(key=lambda e: float(e["utility_score"]), reverse=True)
        return web.json_response({"experts": experts})

    async def _handle_get_budget(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        engine = state.bridge._engine
        if engine is None:
            return web.json_response({"budget": None, "project_id": state.project_id})
        budget = engine._resource_manager.get_budget(state.project_id)
        return web.json_response(
            {
                "project_id": state.project_id,
                "budget": {
                    "max_total_tokens": budget.max_total_tokens,
                    "max_concurrent_agents": budget.max_concurrent_agents,
                    "max_agent_spawns_per_hour": budget.max_agent_spawns_per_hour,
                    "max_tool_calls_per_hour": budget.max_tool_calls_per_hour,
                    "current_token_usage": budget.current_token_usage,
                    "current_agent_count": budget.current_agent_count,
                    "spawns_this_hour": budget.spawns_this_hour,
                    "tool_calls_this_hour": budget.tool_calls_this_hour,
                },
            }
        )

    async def _handle_get_decisions(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        engine = state.bridge._engine
        if engine is None:
            return web.json_response({"reports": []})
        reports: list[dict[str, Any]] = []
        for agent_id in state.session.agents.keys():
            reports.extend(engine.conversation_store.get_decision_reports(agent_id))
        reports.sort(key=lambda r: float(r.get("timestamp", 0.0)), reverse=True)
        return web.json_response({"reports": reports})

    async def _handle_export_telemetry(self, request: web.Request) -> web.Response:
        state, err = self._require_session(request)
        if err:
            return err
        engine = state.bridge._engine
        if engine is None or engine._telemetry_collector is None:
            return web.json_response({"error": "Telemetry is not enabled"}, status=409)
        body = await request.json() if request.can_read_body else {}
        metric_type = str(body.get("metric_type") or "triage_decision").strip()
        output_path = Path(tempfile.gettempdir()) / (
            f"prsm-telemetry-{state.session_id}-{metric_type}.csv"
        )
        engine._telemetry_collector.export_csv(metric_type, output_path)
        csv_text = output_path.read_text(encoding="utf-8")
        return web.Response(
            status=200,
            text=csv_text,
            headers={"Content-Type": "text/csv; charset=utf-8"},
        )

    # ── Archive import / export ──

    async def _handle_archive_import(self, request: web.Request) -> web.Response:
        """Import a .prsm session archive (zip/tar.gz) into ~/.prsm/."""
        try:
            body = await request.json()
            archive_path = Path(body.get("archive_path", ""))
            conflict_mode = str(body.get("conflict_mode", "skip"))
            if not archive_path.exists():
                return web.json_response(
                    {"success": False, "error": f"File not found: {archive_path}"},
                    status=400,
                )
            from prsm.shared.services.session_archive_import import (
                SessionArchiveImportService,
            )

            svc = SessionArchiveImportService()
            result = svc.import_archive(archive_path, conflict_mode=conflict_mode)
            return web.json_response({
                "success": result.success,
                "sessions_imported": result.sessions_imported,
                "sessions_skipped": result.sessions_skipped,
                "files_imported": result.files_imported,
                "files_skipped": result.files_skipped,
                "warnings": result.warnings,
                "error": result.error,
                "manifest": result.manifest,
            })
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    async def _handle_archive_export(self, request: web.Request) -> web.Response:
        """Export all sessions for a repo identity to an archive file."""
        try:
            body = await request.json()
            output_path = Path(body.get("output_path", ""))
            archive_format = str(body.get("format", "tar.gz"))
            repo_identity = str(body.get("repo_identity", ""))

            from prsm.shared.services.session_export import SessionExportService

            svc = SessionExportService()

            # If no repo_identity provided, export all repo identities
            if not repo_identity:
                sessions_dir = Path.home() / ".prsm" / "sessions"
                if not sessions_dir.exists():
                    return web.json_response(
                        {"success": False, "error": "No sessions directory found"},
                        status=400,
                    )
                # Find the first repo identity that has sessions
                repo_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
                if not repo_dirs:
                    return web.json_response(
                        {"success": False, "error": "No session repos found"},
                        status=400,
                    )
                # Export from the first repo identity found
                repo_identity = repo_dirs[0].name

            result = svc.export_all_sessions(
                repo_identity=repo_identity,
                output_path=output_path,
                archive_format=archive_format,
            )
            return web.json_response({
                "success": result.success,
                "archive_path": str(result.archive_path),
                "manifest": result.manifest.to_dict() if result.manifest else None,
                "error": result.error,
            })
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    async def _handle_archive_preview(self, request: web.Request) -> web.Response:
        """Preview what a session archive contains without importing."""
        try:
            body = await request.json()
            archive_path = Path(body.get("archive_path", ""))
            if not archive_path.exists():
                return web.json_response(
                    {"success": False, "error": f"File not found: {archive_path}"},
                    status=400,
                )
            from prsm.shared.services.session_archive_import import (
                SessionArchiveImportService,
            )

            svc = SessionArchiveImportService()
            result = svc.preview_archive(archive_path)
            return web.json_response({
                "success": result.success,
                "sessions_imported": result.sessions_imported,
                "files_imported": result.files_imported,
                "warnings": result.warnings,
                "error": result.error,
                "manifest": result.manifest,
            })
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": str(exc)}, status=500
            )

    # ── Orchestration ──

    def _get_master_agent_id(self, state: SessionState) -> str | None:
        master_id = None
        for agent in state.session.agents.values():
            if agent.parent_id is None:
                master_id = agent.id
        return master_id

    def _sanitize_plan_name(self, name: str, fallback: str) -> str:
        safe_name = "".join(
            c if c.isalnum() or c in "-_ " else "_"
            for c in name
        ).strip()
        return safe_name or fallback

    def _plan_created_at(self, path: Path) -> float:
        """Return the file's creation (birth) time as a float timestamp.

        Tries in order:
        1. os.stat st_birthtime (macOS)
        2. Linux statx via ``stat -c %W`` (ext4/btrfs/xfs with kernel 4.11+)
        3. Fallback to st_mtime (safest – st_ctime is inode *change* time on
           Linux and gets updated by renames, so it must NOT be used here)
        """
        st = path.stat()
        # macOS exposes st_birthtime directly
        birth = getattr(st, "st_birthtime", 0.0)
        if birth and birth > 0:
            return float(birth)
        # Linux: query real birth time via stat(1) which uses statx(2)
        try:
            import subprocess
            result = subprocess.run(
                ["stat", "-c", "%W", str(path)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val and val != "0" and val != "-":
                    return float(val)
        except Exception:
            pass
        # Final fallback – use mtime (NOT ctime, which is inode change time)
        return float(st.st_mtime)

    def _next_plan_index(self) -> int:
        """Return the next sequential plan index (1-based)."""
        self._plan_index_counter += 1
        return self._plan_index_counter

    def _make_unique_indexed_plan_path(
        self,
        plans_dir: Path,
        index: int,
        base_name: str,
        exclude_path: Path | None = None,
    ) -> Path:
        candidate = plans_dir / f"{index} {base_name}.md"
        if exclude_path and candidate == exclude_path:
            return candidate
        if not candidate.exists():
            return candidate
        suffix = 2
        while True:
            candidate = plans_dir / f"{index} {base_name} ({suffix}).md"
            if exclude_path and candidate == exclude_path:
                return candidate
            if not candidate.exists():
                return candidate
            suffix += 1

    def _ensure_plan_name_prefixes(self, plans_dir: Path) -> None:
        """Migrate existing plan files from timestamp prefixes to sequential indexes.

        Renames files like '20260214004322 Name.md' → '1 Name.md',
        ordered by creation time so older plans get lower indexes.
        Also handles files with no numeric prefix (legacy).
        """
        if self._plan_migration_ran:
            return
        self._plan_migration_ran = True
        # Match files already using sequential index prefix (e.g. "1 Name.md", "42 Name.md")
        indexed_pattern = re.compile(r"^(\d+) .+\.md$")
        # Match old timestamp prefix (14 digits, e.g. "20260214004322 Name.md")
        timestamp_pattern = re.compile(r"^\d{14} (.+)\.md$")
        try:
            plans_dir.mkdir(exist_ok=True)
            all_files = list(plans_dir.glob("*.md"))
            if not all_files:
                return

            # Separate already-indexed files from those needing migration
            already_indexed: list[tuple[int, Path]] = []
            needs_migration: list[Path] = []

            for path in all_files:
                idx_match = indexed_pattern.match(path.name)
                if idx_match:
                    idx = int(idx_match.group(1))
                    # Only treat as indexed if it's NOT a 14-digit timestamp
                    if len(idx_match.group(1)) < 14:
                        already_indexed.append((idx, path))
                        continue
                needs_migration.append(path)

            # Start index counter after any existing indexed files
            if already_indexed:
                self._plan_index_counter = max(idx for idx, _ in already_indexed)

            if not needs_migration:
                return

            # Sort by creation time so older plans get lower indexes
            needs_migration.sort(key=lambda p: (self._plan_created_at(p), p.name.lower()))

            for plan_path in needs_migration:
                # Strip old timestamp prefix if present
                ts_match = timestamp_pattern.match(plan_path.name)
                if ts_match:
                    base_name = self._sanitize_plan_name(ts_match.group(1), plan_path.stem)
                else:
                    base_name = self._sanitize_plan_name(plan_path.stem, plan_path.stem)

                idx = self._next_plan_index()
                target_path = self._make_unique_indexed_plan_path(
                    plans_dir=plans_dir,
                    index=idx,
                    base_name=base_name,
                    exclude_path=plan_path,
                )
                if target_path != plan_path:
                    plan_path.rename(target_path)
                    logger.info("Renamed plan %s -> %s", plan_path.name, target_path.name)
        except Exception:
            logger.debug("Failed migrating plan filenames", exc_info=True)

    def _write_plan_chunk(self, state: SessionState, agent_id: str, text: str) -> None:
        master_id = self._get_master_agent_id(state)
        if not master_id or agent_id != master_id:
            return
        try:
            plans_dir = Path(self._cwd) / "plans"
            self._ensure_plan_name_prefixes(plans_dir)
            if state._plan_file_path:
                plan_path = Path(state._plan_file_path)
            else:
                safe_name = self._sanitize_plan_name(state.name, state.session_id)
                idx = self._next_plan_index()
                plan_path = self._make_unique_indexed_plan_path(
                    plans_dir=plans_dir,
                    index=idx,
                    base_name=safe_name,
                )
            with open(plan_path, "a", encoding="utf-8") as f:
                f.write(text)
            plan_path_str = str(plan_path)
            if state._plan_file_path != plan_path_str:
                state._plan_file_path = plan_path_str
                self._broadcast_sse("plan_file_updated", {
                    "session_id": state.session_id,
                    "file_path": plan_path_str,
                })
        except Exception:
            logger.debug("Failed to write plan chunk for session %s", state.session_id)

    async def _run_orchestration(
        self, state: SessionState, prompt: str, original_prompt: str | None = None,
    ) -> None:
        display_prompt = original_prompt or prompt
        # Bump the run generation so stale engine_finished events from a
        # previous (stopped) run are discarded by the event consumer.
        state._run_generation += 1
        run_gen = state._run_generation
        try:
            logger.info("Run orchestration start session=%s gen=%d prompt_len=%d", state.session_id, run_gen, len(prompt))
            # Lazily create worktree when session becomes active.
            self._ensure_session_worktree(state)
            state._plan_file_path = None

            # Auto-snapshot before run
            snapshot_id = None
            try:
                snapshot_id = self._create_snapshot(
                    state,
                    display_prompt,
                )
            except Exception:
                logger.debug("Auto-snapshot failed for session %s", state.session_id)

            # Auto-name session
            if is_default_session_name(state.name):
                asyncio.create_task(self._auto_name_session(state, display_prompt))
            elif not state.summary:
                state.summary = self._derive_session_summary(display_prompt)
                self._broadcast_sse(
                    "session_renamed",
                    {
                        "session_id": state.session_id,
                        "name": state.name,
                        "summary": state.summary,
                        "project_id": state.project_id,
                    },
                )

            state._pending_user_prompt = display_prompt
            state._pending_user_prompt_snapshot_id = snapshot_id

            self._broadcast_sse("user_prompt", {
                "session_id": state.session_id,
                "agent_id": "",
                "text": display_prompt,
                "snapshot_id": snapshot_id,
            })

            state.bridge.event_bus.reset()
            if state._event_task:
                state._event_task.cancel()
            state._event_task = asyncio.create_task(self._consume_session_events(state))

            result = await state.bridge.run(prompt)
            # If a newer run has started while we were awaiting (stop then
            # restart race), this task is stale — skip post-run actions.
            if state._run_generation != run_gen:
                logger.info(
                    "Session %s orchestration gen=%d superseded by gen=%d, skipping post-run",
                    state.session_id, run_gen, state._run_generation,
                )
                return
            # Wait for the event consumer to finish processing all queued
            # events (AgentSpawned, StreamChunk, etc.) before saving.
            # Without this, a session save can race ahead of event
            # processing and miss agents or messages that are still in
            # the EventBus queue.
            drained = await state.bridge.event_bus.drain(timeout=5.0)
            if not drained:
                logger.warning(
                    "Session %s: event bus did not drain within timeout, "
                    "saving with potentially incomplete state",
                    state.session_id,
                )
            logger.info("Session %s orchestration completed: %s", state.session_id, result[:100] if result else "empty")
            self._save_session(state)
        except Exception as e:
            # If a newer run has started, this failure is stale — don't
            # broadcast a spurious engine_finished or overwrite the session.
            if state._run_generation != run_gen:
                logger.info(
                    "Session %s orchestration gen=%d error superseded by gen=%d",
                    state.session_id, run_gen, state._run_generation,
                )
                return
            logger.exception("Session %s orchestration failed", state.session_id)
            error_msg = str(e) if str(e) else "Orchestration failed"
            # Drain remaining events before saving to capture any agents/messages
            await state.bridge.event_bus.drain(timeout=3.0)
            self._broadcast_sse("engine_finished", {
                "session_id": state.session_id, "success": False,
                "summary": "", "error": error_msg, "duration_seconds": 0.0,
            })
            self._save_session(state)

    # ── Agent restart helpers ──

    async def _try_restart_agent(
        self, state: SessionState, agent_id: str, resolved_prompt: str,
    ) -> bool:
        """Try to restart a completed/failed agent. Returns True if restarted."""
        if not state.bridge._engine:
            return False

        manager = state.bridge._engine._manager
        desc = manager.get_completed_descriptor(agent_id)

        # Fallback 1: check active pool for completed agents not yet cleaned up
        if not desc:
            active_desc = manager._agents.get(agent_id)
            if active_desc and active_desc.state in (AgentState.COMPLETED, AgentState.FAILED):
                manager._completed_agents[agent_id] = active_desc
                manager._agents.pop(agent_id, None)
                desc = active_desc
                logger.info("Agent %s moved from active to completed pool (state=%s)", agent_id[:8], active_desc.state.value)

        # Fallback 2: reconstitute from session persistence
        if not desc and agent_id in state.session.agents:
            desc = self._reconstitute_descriptor(state, agent_id)
            if desc:
                manager._completed_agents[agent_id] = desc

        if not desc:
            return False

        # Ensure worktree exists before restarting an agent.
        self._ensure_session_worktree(state)

        # Imported transcript placeholders (e.g. imported:codex) are not
        # runnable model IDs. Normalize before restart so retries do not loop.
        if not str(desc.model or "").strip() or str(desc.model).startswith("imported:"):
            model, provider = self._normalize_agent_model_for_restart(
                state=state,
                model=desc.model,
                provider=desc.provider,
            )
            desc.model = model
            desc.provider = provider

        restart_prompt = self._build_restart_prompt(manager, agent_id, resolved_prompt, state=state)
        try:
            await manager.restart_agent(agent_id, restart_prompt)
            state._directly_restarted_agents.add(agent_id)
            logger.info("Restarted completed agent %s with user prompt", agent_id[:8])
            return True
        except Exception:
            logger.warning("Failed to restart agent %s", agent_id[:8], exc_info=True)
            self._broadcast_sse("agent_state_changed", {
                "session_id": state.session_id, "agent_id": agent_id,
                "old_state": "running", "new_state": desc.state.value if desc else "completed",
            })
            return False

    async def _deliver_to_running_agent(
        self, state: SessionState, agent_id: str, resolved_prompt: str,
    ) -> None:
        """Deliver a user message to a currently running agent via message router."""
        try:
            if state.bridge._engine:
                from prsm.engine.models import RoutedMessage, MessageType
                user_msg = RoutedMessage(
                    message_type=MessageType.ANSWER,
                    source_agent_id="user",
                    target_agent_id=agent_id,
                    payload=resolved_prompt,
                )
                await state.bridge._engine._router.send(user_msg)
                logger.info("User message delivered to agent %s via router", agent_id[:8])
        except Exception as e:
            logger.warning("Failed to deliver user message to agent %s: %s", agent_id[:8], str(e))

    def _build_restart_prompt(
        self, manager: object, agent_id: str, new_prompt: str, state: SessionState | None = None,
    ) -> str:
        """Build a prompt for a restarted agent that includes conversation history.

        Tool calls are formatted using XML-style ``<system-reminder>`` tags
        rather than bracket-style ``[Tool call: ...]`` markers.  This prevents
        the model from echoing the history markers as plain text into the
        streamed response, which would cause the UI to render raw metadata
        instead of formatted tool-call widgets.
        """
        lines: list[str] = []

        # Try ConversationStore first (full history including tool outputs).
        store = getattr(manager, "_conversation_store", None)
        if store and store.has_history(agent_id):
            history = store.get_history(agent_id, detail_level="full")
            for entry in history:
                etype = entry.get("type", "")
                content = str(entry.get("content", "") or "")
                if etype == "user_message":
                    lines.append(f"User: {content}")
                elif etype == "text":
                    lines.append(f"Assistant: {content}")
                elif etype == "thinking":
                    pass  # Omit thinking from context — not actionable for continuation
                elif etype == "tool_call":
                    tool_name = entry.get("tool_name", "unknown")
                    tool_args = str(entry.get("tool_args", "") or "")
                    lines.append(f"<system-reminder>\nCalled the {tool_name} tool"
                                 + (f" with the following input: {tool_args}" if tool_args else "")
                                 + "\n</system-reminder>")
                elif etype == "tool_result":
                    status = "error" if entry.get("is_error", False) else "success"
                    tool_name = entry.get("tool_name", "unknown")
                    result_preview = content[:2000] if content else ""
                    lines.append(f"<system-reminder>\nResult of calling the {tool_name} tool: "
                                 + f"{result_preview}\n</system-reminder>")

        # Fallback: use session messages
        if not lines and state:
            messages = state.session.get_messages(agent_id)
            for msg in messages:
                if msg.role == MessageRole.USER:
                    lines.append(f"User: {msg.content}")
                elif msg.role == MessageRole.ASSISTANT:
                    lines.append(f"Assistant: {msg.content}")
                elif msg.role == MessageRole.TOOL and msg.tool_calls:
                    for tc in msg.tool_calls:
                        lines.append(
                            f"<system-reminder>\nCalled the {tc.name} tool"
                            + (f" with the following input: {tc.arguments}" if tc.arguments else "")
                            + "\n</system-reminder>"
                        )
                        if tc.result:
                            status = "error" if not tc.success else "success"
                            result_preview = tc.result[:2000]
                            lines.append(
                                f"<system-reminder>\nResult of calling the {tc.name} tool: "
                                + f"{result_preview}\n</system-reminder>"
                            )

        if not lines:
            return new_prompt

        conversation_context = "\n".join(lines)
        return (
            f"<previous-conversation>\n{conversation_context}\n</previous-conversation>\n\n"
            f"The user is continuing the conversation above. Respond to their new message:\n\n"
            f"{new_prompt}"
        )

    def _reconstitute_descriptor(self, state: SessionState, agent_id: str):
        """Reconstruct an AgentDescriptor from persisted session data."""
        from prsm.engine.models import AgentDescriptor, PermissionMode

        node = state.session.agents.get(agent_id)
        if not node:
            return None

        role = node.role or AgentRole.WORKER
        if role == AgentRole.MASTER:
            tools = ["Read", "Glob", "Grep"]
            perm_mode = PermissionMode.BYPASS
        else:
            tools = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
            perm_mode = PermissionMode.BYPASS

        model, provider = self._normalize_agent_model_for_restart(
            state=state,
            model=node.model,
            provider=node.provider,
        )

        descriptor = AgentDescriptor(
            agent_id=agent_id, parent_id=node.parent_id, role=role,
            state=AgentState.COMPLETED, prompt=node.prompt_preview or "",
            tools=tools, model=model, permission_mode=perm_mode,
            cwd=self._cwd, children=list(node.children_ids), provider=provider,
        )
        logger.info(
            "Reconstituted agent %s from session data (role=%s, model=%s, provider=%s)",
            agent_id[:8],
            role.value,
            model,
            provider,
        )
        return descriptor

    def _normalize_agent_model_for_restart(
        self,
        *,
        state: SessionState,
        model: str | None,
        provider: str | None,
    ) -> tuple[str, str]:
        """Return a runnable model/provider pair for restartable descriptors.

        Imported transcript sessions may persist placeholder model IDs like
        ``imported:codex`` that are not valid runtime model IDs. Normalize those
        to the session's current configured model before restart.
        """
        model_text = (model or "").strip()
        provider_text = (provider or "").strip()

        default_provider = "claude"
        if self._yaml_config and hasattr(self._yaml_config, "engine"):
            default_provider = getattr(
                self._yaml_config.engine, "default_provider", "claude"
            )
        if not provider_text:
            provider_text = default_provider

        # Imported transcript placeholder IDs are for provenance, not execution.
        if not model_text or model_text.startswith("imported:"):
            model_text = state.bridge.current_model or "claude-opus-4-6"
            engine_cfg = getattr(getattr(state.bridge, "_engine", None), "_config", None)
            if engine_cfg is not None:
                provider_text = (
                    getattr(engine_cfg, "master_provider", None)
                    or getattr(engine_cfg, "default_provider", None)
                    or provider_text
                )

        engine_cfg = getattr(getattr(state.bridge, "_engine", None), "_config", None)
        model_registry = getattr(engine_cfg, "model_registry", None)
        if model_registry is not None and model_text:
            try:
                resolve_with_provider = getattr(
                    model_registry, "resolve_alias_with_provider", None
                )
                if callable(resolve_with_provider):
                    resolved_model, resolved_provider = resolve_with_provider(model_text)
                else:
                    resolve_alias = getattr(model_registry, "resolve_alias", None)
                    resolved_model = (
                        resolve_alias(model_text) if callable(resolve_alias) else model_text
                    )
                    cap_getter = getattr(model_registry, "get", None)
                    cap = cap_getter(resolved_model) if callable(cap_getter) else None
                    resolved_provider = getattr(cap, "provider", None) if cap else None
                model_text = resolved_model or model_text
                if resolved_provider:
                    provider_text = resolved_provider
            except Exception:
                logger.debug(
                    "Model/provider normalization failed for model=%s provider=%s",
                    model_text,
                    provider_text,
                    exc_info=True,
                )

        return model_text, provider_text

    async def _deferred_restart_agent(
        self, state: SessionState, agent_id: str, display_prompt: str, resolved: str,
    ) -> None:
        """Wait for agent cleanup, then restart with queued prompt."""
        manager = state.bridge._engine._manager if state.bridge._engine else None
        if not manager:
            return
        for _ in range(20):
            await asyncio.sleep(0.1)
            if manager.get_completed_descriptor(agent_id):
                break
        else:
            reconstituted = self._reconstitute_descriptor(state, agent_id)
            if reconstituted:
                manager._completed_agents[agent_id] = reconstituted
            else:
                logger.warning("Agent %s not in completed pool, cannot restart", agent_id[:8])
                return

        state.session.add_message(agent_id, MessageRole.USER, display_prompt)
        self._broadcast_sse("agent_message", {
            "session_id": state.session_id, "agent_id": agent_id,
            "content": display_prompt, "role": "user",
        })
        restart_prompt = self._build_restart_prompt(manager, agent_id, resolved, state=state)
        try:
            await manager.restart_agent(agent_id, restart_prompt)
        except Exception:
            logger.warning("Deferred restart failed for %s", agent_id[:8], exc_info=True)

    # ── Persistence helpers ──

    def _restore_saved_sessions(self) -> None:
        """Eagerly load sessions into memory on startup."""
        self._index_saved_sessions()
        try:
            for session_id in list(self._session_index.keys()):
                try:
                    self._try_load_indexed_session(session_id)
                except Exception:
                    continue
        except Exception:
            logger.warning("Failed to restore saved sessions", exc_info=True)

    def _derive_session_summary(self, prompt: str) -> str:
        text = " ".join((prompt or "").strip().split())
        if not text:
            return ""
        words = text.split(" ")
        if len(words) <= 30:
            return text
        return " ".join(words[:30]).rstrip() + "..."

    async def _auto_name_session(self, state: SessionState, prompt: str) -> None:
        try:
            from prsm.shared.services.session_naming import generate_session_metadata

            name, summary = await generate_session_metadata(prompt)
            if name:
                needs_prefix = bool(state.forked_from) or (state.name or "").startswith(FORKED_PREFIX)
                new_name = format_forked_name(name) if needs_prefix else name
                state.name = new_name
                state.session.name = new_name
                state.summary = summary or self._derive_session_summary(prompt)
                self._broadcast_sse(
                    "session_renamed",
                    {
                        "session_id": state.session_id,
                        "name": new_name,
                        "summary": state.summary,
                        "project_id": state.project_id,
                    },
                )
        except Exception:
            logger.debug("Auto-naming failed for session %s", state.session_id, exc_info=True)

    def _save_session(self, state: SessionState) -> None:
        try:
            self._persistence.save(
                state.session,
                state.name or state.session_id,
                session_id=state.session_id,
                summary=state.summary,
                project_id=state.project_id,
            )
            last_activity = self._last_activity_for_session(state.session)
            self._session_index[state.session_id] = {
                "file_stem": state.session_id,
                "session_id": state.session_id,
                "name": state.name or state.session_id,
                "summary": state.summary,
                "project_id": state.project_id,
                "forked_from": state.forked_from,
                "agent_count": len(state.session.agents),
                "message_count": state.session.message_count,
                "created_at": state.session.created_at.isoformat() if state.session.created_at else None,
                "last_activity": last_activity.isoformat() if last_activity else None,
            }
            logger.debug("Auto-saved session %s", state.session_id)
        except Exception:
            logger.debug("Failed to save session %s", state.session_id)

    async def _autosave_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(30)
                for state in list(self._sessions.values()):
                    if state.bridge.running:
                        self._save_session(state)
                await self._unload_inactive_sessions()
        except asyncio.CancelledError:
            pass

    async def _unload_inactive_sessions(self) -> None:
        """Shutdown and unload idle sessions after configured inactivity."""
        if self._session_inactivity_seconds <= 0:
            return
        now = time.time()
        for session_id, state in list(self._sessions.items()):
            if state.bridge.running:
                continue
            idle_for = now - state._last_touched_at
            if idle_for < self._session_inactivity_seconds:
                continue
            try:
                self._save_session(state)
                # Clean up the worktree if no pending changes remain;
                # keeps /tmp free of stale empty worktrees.
                self._maybe_cleanup_empty_worktree(state)
                await state.bridge.shutdown()
                if state._event_task:
                    state._event_task.cancel()
                self._sessions.pop(session_id, None)
                self._session_projects.pop(session_id, None)
                logger.info(
                    "Unloaded inactive session %s after %.1f minutes",
                    session_id,
                    idle_for / 60.0,
                )
            except Exception:
                logger.warning(
                    "Failed to unload inactive session %s",
                    session_id,
                    exc_info=True,
                )

    def _broadcast_agent_tree(self, state: SessionState, session: Session) -> None:
        """Broadcast all agents in a session for clients to render."""
        for agent in session.agents.values():
            self._broadcast_sse("agent_spawned", {
                "session_id": state.session_id,
                "project_id": state.project_id,
                "agent_id": agent.id,
                "parent_id": agent.parent_id,
                "role": agent.role.value if agent.role else "worker",
                "model": agent.model,
                "depth": 0,
                "prompt": agent.prompt_preview,
                "name": agent.name,
                "state": agent.state.value if agent.state else "idle",
            })

    def _get_file_index(self):
        if self._file_index is None:
            from prsm.shared.file_utils import FileIndex
            self._file_index = FileIndex(Path(self._cwd))
        return self._file_index
