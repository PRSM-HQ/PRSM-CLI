"""Configuration loaded from environment variables.

All settings have sensible defaults. Override via ORCH_* env vars.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Optional async callback for real-time event observation.
# Signature: async def callback(event: dict[str, Any]) -> None
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Optional async callback for tool permission requests.
# Signature: async def callback(agent_id, tool_name, arguments) -> str
# Returns: "allow", "deny", "allow_always", "allow_project", or "allow_global"
PermissionCallback = Callable[[str, str, str], Awaitable[str]]

# Optional async callback for user-facing questions from agents.
# Signature: async def callback(agent_id, question, options) -> str
# options: list of {"label": str, "description": str}
# Returns: the user's selected option text or custom input
UserQuestionCallback = Callable[[str, str, list], Awaitable[str]]


async def fire_event(
    callback: EventCallback | None,
    event: dict[str, Any],
) -> None:
    """Fire an event callback if set, silently swallowing errors."""
    if callback is None:
        return
    try:
        await callback(event)
    except Exception:
        pass  # Never let callback errors break the engine


@dataclass
class EngineConfig:
    """Orchestration engine configuration."""

    # Agent defaults (for worker/child agents)
    default_model: str = "claude-opus-4-6"
    default_provider: str = "claude"
    default_cwd: str = "."

    # Master/orchestrator agent model.
    # Claude uses in-process MCP; other providers (Codex, Gemini,
    # MiniMax) use the TCP bridge + MCP proxy for orchestration.
    master_model: str = "claude-sonnet-4-5-20250929"
    master_provider: str = "claude"
    max_agent_depth: int = 5
    max_concurrent_agents: int = 10
    # Max cumulative time an agent spends reasoning (excludes tool calls).
    # Set to 0 (or a negative value) to disable timeout.
    agent_timeout_seconds: float = 7200.0
    # Max wall-clock time for any single tool call.
    # Set to 0 (or a negative value) to disable timeout.
    tool_call_timeout_seconds: float = 7200.0
    # Max wait for user responses to ask_user/request_user_input flows.
    # Set to 0 (or a negative value) to disable timeout.
    user_question_timeout_seconds: float = 0.0
    # Regex rules for command policy enforcement on bash-like tools.
    # These are merged with workspace .prism command lists.
    command_whitelist: list[str] = field(default_factory=list)
    command_blacklist: list[str] = field(default_factory=list)
    # Optional lightweight safety model hook for secondary screening.
    command_safety_model_enabled: bool = False
    command_safety_model: str | None = None

    # Deadlock detection
    deadlock_check_interval_seconds: float = 5.0
    deadlock_max_wait_seconds: float = 120.0

    # Message queue sizes
    message_queue_size: int = 1000

    # Logging
    log_level: str = "INFO"

    # Optional project identity used for event scoping and multi-project
    # control-plane coordination.
    project_id: str | None = None
    # Optional resource budget map keyed by project_id.
    resource_budgets: dict[str, dict[str, int]] = field(default_factory=dict)
    # Phase 7 shadow triage and telemetry settings.
    triage_model_shadow_enabled: bool = False
    triage_shadow_model: str = "claude-haiku"
    telemetry_db_path: str | None = None

    # Peer models for child agent model restriction.
    # When set, only models in this dict are allowed for child agents.
    # Maps alias → (provider_instance, model_id).
    # If empty/None, all models in the registry are allowed (backward compat).
    peer_models: dict | None = field(default=None, repr=False)

    # Model capability registry for intelligent model selection.
    # When set, child agents are automatically routed to cheaper/faster
    # models based on task complexity.  Built from build_default_registry()
    # and YAML overrides, then attached here so the engine propagates it.
    model_registry: object | None = field(default=None, repr=False)

    # Optional async callback for real-time event observation.
    # Receives dicts like {"event": "stream_chunk", "agent_id": "...", ...}
    event_callback: EventCallback | None = field(default=None, repr=False)

    # Optional async callback for tool permission requests.
    # Called before each tool execution. Returns "allow"/"deny"/"allow_always"/"allow_project"/"allow_global".
    permission_callback: PermissionCallback | None = field(
        default=None, repr=False,
    )

    # Optional async callback for user-facing questions from agents.
    # Called when an agent uses the ask_user tool.
    user_question_callback: UserQuestionCallback | None = field(
        default=None, repr=False,
    )

    @classmethod
    def from_env(cls) -> EngineConfig:
        """Load configuration from ORCH_* environment variables."""
        # Log which ORCH_* env vars are set for config debugging
        orch_vars = {
            k: v for k, v in os.environ.items() if k.startswith("ORCH_")
        }
        if orch_vars:
            logger.info(
                "EngineConfig.from_env: ORCH_* env overrides: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(orch_vars.items())),
            )
        else:
            logger.debug("EngineConfig.from_env: no ORCH_* env vars set, using defaults")

        config = cls(
            default_model=os.getenv(
                "ORCH_DEFAULT_MODEL", cls.default_model
            ),
            default_provider=os.getenv(
                "ORCH_DEFAULT_PROVIDER", cls.default_provider
            ),
            default_cwd=os.getenv("ORCH_DEFAULT_CWD", cls.default_cwd),
            max_agent_depth=int(os.getenv(
                "ORCH_MAX_DEPTH", str(cls.max_agent_depth)
            )),
            max_concurrent_agents=int(os.getenv(
                "ORCH_MAX_AGENTS", str(cls.max_concurrent_agents)
            )),
            agent_timeout_seconds=float(os.getenv(
                "ORCH_AGENT_TIMEOUT", str(cls.agent_timeout_seconds)
            )),
            tool_call_timeout_seconds=float(os.getenv(
                "ORCH_TOOL_TIMEOUT",
                str(cls.tool_call_timeout_seconds),
            )),
            user_question_timeout_seconds=float(os.getenv(
                "ORCH_USER_QUESTION_TIMEOUT",
                str(cls.user_question_timeout_seconds),
            )),
            command_safety_model_enabled=(
                os.getenv("ORCH_COMMAND_SAFETY_MODEL_ENABLED", "").lower()
                in {"1", "true", "yes"}
            ),
            command_safety_model=os.getenv(
                "ORCH_COMMAND_SAFETY_MODEL", cls.command_safety_model or ""
            )
            or None,
            deadlock_check_interval_seconds=float(os.getenv(
                "ORCH_DEADLOCK_INTERVAL",
                str(cls.deadlock_check_interval_seconds),
            )),
            deadlock_max_wait_seconds=float(os.getenv(
                "ORCH_DEADLOCK_MAX_WAIT",
                str(cls.deadlock_max_wait_seconds),
            )),
            message_queue_size=int(os.getenv(
                "ORCH_QUEUE_SIZE", str(cls.message_queue_size)
            )),
            log_level=os.getenv("ORCH_LOG_LEVEL", cls.log_level),
            project_id=os.getenv("PRSM_PROJECT_ID") or None,
            triage_model_shadow_enabled=(
                os.getenv("PRSM_TRIAGE_MODEL_SHADOW", "").lower()
                in {"1", "true", "yes"}
            ),
            triage_shadow_model=os.getenv(
                "PRSM_TRIAGE_SHADOW_MODEL", cls.triage_shadow_model
            ),
            telemetry_db_path=os.getenv("PRSM_TELEMETRY_DB_PATH") or None,
        )
        logger.info(
            "EngineConfig.from_env: model=%s provider=%s cwd=%s log_level=%s",
            config.default_model, config.default_provider,
            config.default_cwd, config.log_level,
        )
        return config
