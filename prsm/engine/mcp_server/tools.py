"""MCP tool implementations for inter-agent orchestration.

Each tool is an async function that captures the owning agent's ID
via closure. When Claude invokes a tool, the handler awaits the
response from the MessageRouter. Claude's execution naturally pauses
while waiting for the tool result.

TIMEOUT MODEL:
- Agent timeout: Cumulative time the agent spends *reasoning* (i.e.
  between tool calls). Tracked by ToolTimeTracker — each tool call
  pauses the agent's reasoning clock. Checked in AgentSession after
  each SDK message yield.
- Tool call timeout: Max wall-clock for any *single* tool call. Each
  handler is wrapped with asyncio.wait_for(). Default 7200s. Prevents
  a single spawn_child or ask_parent from hanging forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..config import fire_event
from ..models import (
    AgentRole,
    AgentState,
    MessageType,
    PermissionMode,
    RoutedMessage,
    SpawnRequest,
)
from ..errors import (
    MaxDepthExceededError,
    ExpertNotFoundError,
    MessageRoutingError,
    ProviderNotAvailableError,
    ToolCallTimeoutError,
)

if TYPE_CHECKING:
    from ..agent_manager import AgentManager
    from ..config import EventCallback, UserQuestionCallback
    from ..message_router import MessageRouter
    from ..expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)

# Cross-instance bash process registry:
# tool_call_id -> (agent_id, process)
_GLOBAL_ACTIVE_BASH_PROCESSES: dict[str, tuple[str, asyncio.subprocess.Process]] = {}

_COMPLEXITY_TIER_ALLOWLIST: dict[str, set[str]] = {
    "trivial": {"economy", "fast"},
    "simple": {"fast", "strong"},
    "medium": {"strong", "frontier"},
    "complex": {"strong", "frontier"},
    "frontier": {"frontier"},
}

DOCS_PROMPT_SENTINEL = "[DOCS-FIRST-ARCHITECTURE]"
DOCS_REVIEW_INSTRUCTION = (
    f"{DOCS_PROMPT_SENTINEL}\n"
    "Before exploring the codebase, review relevant architecture docs in "
    "@docs/ first (for example: @docs/architecture.md, @docs/engine.md, "
    "@docs/data-flow.md, @docs/adapters.md, @docs/shared.md, "
    "@docs/configuration.md, @docs/tui.md, @docs/vscode.md).\n"
    "Use those docs to plan targeted code exploration and avoid unnecessary "
    "broad scanning.\n"
)

ASSUMPTION_MINIMIZATION_PROMPT_SENTINEL = "[ASSUMPTION-MINIMIZATION]"
ASSUMPTION_MINIMIZATION_INSTRUCTION = (
    f"{ASSUMPTION_MINIMIZATION_PROMPT_SENTINEL}\n"
    "Make as few assumptions or interpretations as possible.\n"
    "If anything is unclear, ambiguous, or unspecified, ask clarifying "
    "questions immediately using ask_parent() or ask_user() before "
    "proceeding.\n"
)

DOCS_UPDATE_AFTER_TASK_COMPLETE_INSTRUCTION = (
    "If your implementation changed architecture, data flow, module "
    "responsibilities, integration boundaries, or runtime behavior described "
    "in @docs/, update the relevant @docs/*.md files.\n"
    "Immediately after this tool call, provide a full, detailed Markdown "
    "summary of everything you did, including concrete changes made, files "
    "touched, verification performed, and any remaining risks or follow-ups.\n"
)

_REDIRECT_OPERATORS = {
    ">",
    ">>",
    "1>",
    "1>>",
    "2>",
    "2>>",
    "&>",
    "&>>",
}
_SHELL_CONTROL_TOKENS = {"|", "||", "&&", ";"}
_SUSPICIOUS_SCRATCH_BASENAME_RE = re.compile(
    r"^\.[a-z0-9][a-z0-9_-]*\.(txt|log|out)$"
)
_BLOCKED_SCRATCH_BASENAMES = {
    ".file_index.txt",
    ".grep_agent_manager.txt",
}


def strip_injected_prompt_prefix(prompt: str) -> str:
    """Strip the docs-first / assumption-minimization prefix from a prompt.

    The ``_inject_docs_instruction`` method prepends system instructions
    delimited by a ``TASK:\\n`` marker.  This function returns only the
    user-facing task description so that display names and previews are
    meaningful.

    Handles both the full prompt (with ``TASK:\\n`` marker) and truncated
    versions (where the marker was cut off by ``[:200]`` slicing).
    """
    if not prompt:
        return prompt

    # Fast path: full prompt with TASK marker
    marker = "TASK:\n"
    idx = prompt.find(marker)
    if idx != -1:
        return prompt[idx + len(marker):]

    # Truncated prompt: the [:200] slice may have cut off the TASK marker.
    # Detect by checking for the known sentinel prefixes.
    if prompt.startswith(DOCS_PROMPT_SENTINEL):
        # The entire 200-char string is just the injected prefix —
        # return empty since the actual task was truncated away.
        return ""

    return prompt


def _text(text: str) -> dict[str, Any]:
    """Format a successful text response."""
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    """Format an error response."""
    return {
        "content": [{"type": "text", "text": f"ERROR: {text}"}],
        "is_error": True,
    }


class ToolTimeTracker:
    """Tracks cumulative time spent inside tool calls.

    The agent session reads `accumulated_tool_time` to subtract
    from wall-clock elapsed, giving "reasoning-only" time for
    the agent timeout.

    Thread-safe for single-event-loop usage (no concurrent tool
    calls per agent — the SDK serializes them).
    """

    def __init__(self) -> None:
        self.accumulated_tool_time: float = 0.0
        self._current_start: float | None = None

    def enter(self) -> None:
        """Called when a tool handler begins execution."""
        self._current_start = time.monotonic()

    def exit(self) -> None:
        """Called when a tool handler finishes execution."""
        if self._current_start is not None:
            self.accumulated_tool_time += (
                time.monotonic() - self._current_start
            )
            self._current_start = None


class OrchestrationTools:
    """Container for all orchestration tool handlers bound to one agent.

    Each instance is bound to a specific agent_id. Tool methods serve
    as MCP tool handlers. Every blocking tool is wrapped with
    asyncio.wait_for() enforcing the per-tool-call timeout.
    """

    def __init__(
        self,
        agent_id: str,
        manager: AgentManager,
        router: MessageRouter,
        expert_registry: ExpertRegistry,
        tool_call_timeout: float = 7200.0,
        time_tracker: ToolTimeTracker | None = None,
        default_model: str = "claude-opus-4-6",
        default_provider: str = "claude",
        provider_registry: object | None = None,
        peer_model: str | None = None,
        peer_provider: object | None = None,
        user_question_callback: UserQuestionCallback | None = None,
        conversation_store: object | None = None,
        peer_models: dict[str, tuple] | None = None,
        model_registry: object | None = None,
        max_parallel_children_per_call: int = 4,
        max_total_children_per_master: int = 16,
        permission_callback: Any | None = None,
        cwd: str | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._manager = manager
        self._router = router
        self._expert_registry = expert_registry
        self._tool_call_timeout = tool_call_timeout
        self.time_tracker = time_tracker or ToolTimeTracker()
        self._default_model = default_model
        self._default_provider = default_provider
        self._provider_registry = provider_registry
        self._peer_model = peer_model
        self._peer_provider = peer_provider
        self._peer_threads: dict[str, str] = {}
        self._user_question_callback = user_question_callback
        self._conversation_store = conversation_store
        self._tool_call_seq: int = 0
        self._meaningful_tool_calls: int = 0
        # Multiple peer models: {alias: (provider_instance, model_id)}
        self._peer_models = peer_models or {}
        # Guardrails to prevent runaway process spawning.
        self._max_parallel_children_per_call = max(
            1, int(max_parallel_children_per_call)
        )
        self._max_total_children_per_master = max(
            1, int(max_total_children_per_master)
        )
        # Model capability registry for intelligent model selection
        self._model_registry = model_registry
        # Model intelligence — learned rankings (may be None)
        self._model_intelligence = None

        # Permission support for run_bash
        self._permission_callback = permission_callback
        self._cwd = cwd or "."
        self._event_callback = event_callback
        self._always_allowed_commands: set[str] = set()
        self._active_bash_processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled_bash_tool_calls: set[str] = set()
        # Parent-local memory of child "types" (category/complexity/tools/etc.)
        # so similar work can reuse existing children.
        self._child_task_profiles: dict[str, dict[str, Any]] = {}
        # Child IDs explicitly registered for wait-for-message coordination.
        # Updated whenever a child (new or reused) is connected via spawn APIs.
        self._tracked_wait_children: set[str] = set()

        # Load command policy (blacklist/whitelist)
        from ...shared.services.command_policy_store import CommandPolicyStore
        policy_store = CommandPolicyStore(Path(self._cwd))
        policy_rules = policy_store.load_compiled()
        self._whitelist_patterns = list(policy_rules.whitelist)
        self._blacklist_patterns = list(policy_rules.blacklist)

    @staticmethod
    def _normalize_complexity(complexity: str | None) -> str | None:
        if complexity is None:
            return None
        normalized = str(complexity).strip().lower()
        return normalized or None

    def _allowed_tiers_for_complexity(
        self, complexity: str | None
    ) -> set[str] | None:
        normalized = self._normalize_complexity(complexity)
        if normalized is None:
            return None
        return _COMPLEXITY_TIER_ALLOWLIST.get(
            normalized,
            _COMPLEXITY_TIER_ALLOWLIST["medium"],
        )

    def _model_tier(self, model_id: str | None) -> str | None:
        if not model_id or not self._model_registry:
            return None
        cap = self._model_registry.get(model_id)
        if cap is None:
            return None
        return str(cap.tier.value)

    def _model_fits_complexity(
        self, model_id: str | None, complexity: str | None
    ) -> bool:
        allowed = self._allowed_tiers_for_complexity(complexity)
        if allowed is None:
            return True
        if self._model_registry is None:
            return True
        tier = self._model_tier(model_id)
        if tier is None:
            return False
        return tier in allowed

    def _infer_task_category(self, prompt: str) -> str:
        if self._model_registry and hasattr(self._model_registry, "_infer_category"):
            try:
                return str(self._model_registry._infer_category(prompt))
            except Exception:
                logger.exception("Failed to infer task category from prompt")
        return "general"

    @staticmethod
    def _normalize_toolset(tools: list[str] | None) -> tuple[str, ...]:
        if not tools:
            return tuple()
        normalized = [str(t).strip() for t in tools if str(t).strip()]
        normalized.sort()
        return tuple(normalized)

    @staticmethod
    def _normalize_profile_cwd(cwd: str | None) -> str:
        if not cwd:
            return ""
        cwd_value = str(cwd).strip()
        return "" if cwd_value in {".", "./"} else cwd_value

    def _build_task_profile(
        self,
        *,
        prompt: str,
        complexity: str | None,
        tools: list[str] | None,
        cwd: str | None,
        provider: str | None,
        model: str | None,
    ) -> dict[str, Any]:
        return {
            "task_category": self._infer_task_category(prompt),
            "complexity": self._normalize_complexity(complexity),
            "tools": self._normalize_toolset(tools),
            "cwd": self._normalize_profile_cwd(cwd),
            "provider": provider or "",
            "model": model or "",
            "tier": self._model_tier(model) or "",
        }

    @staticmethod
    def _profile_recency(profile: dict[str, Any]) -> datetime:
        ts = profile.get("last_used_at")
        if isinstance(ts, datetime):
            return ts
        return datetime.fromtimestamp(0, tz=timezone.utc)

    def _ensure_child_profile_store(self) -> dict[str, dict[str, Any]]:
        store = getattr(self, "_child_task_profiles", None)
        if store is None:
            store = {}
            self._child_task_profiles = store
        return store

    def _record_child_profile(
        self,
        child_id: str,
        profile: dict[str, Any],
    ) -> None:
        store = self._ensure_child_profile_store()
        stored = dict(profile)
        stored["last_used_at"] = datetime.now(timezone.utc)
        store[child_id] = stored

    def _profile_from_descriptor(self, descriptor: Any) -> dict[str, Any]:
        prompt = strip_injected_prompt_prefix(getattr(descriptor, "prompt", "") or "")
        tools = getattr(descriptor, "tools", None)
        provider = getattr(descriptor, "provider", None)
        model = getattr(descriptor, "model", None)
        cwd = self._normalize_profile_cwd(getattr(descriptor, "cwd", None))
        return self._build_task_profile(
            prompt=prompt,
            complexity=None,
            tools=tools,
            cwd=cwd,
            provider=provider,
            model=model,
        )

    def _profiles_match(
        self,
        candidate: dict[str, Any],
        requested: dict[str, Any],
        *,
        explicit_model: bool,
    ) -> bool:
        if candidate.get("task_category") != requested.get("task_category"):
            return False
        if candidate.get("tools") != requested.get("tools"):
            return False
        if candidate.get("cwd") != requested.get("cwd"):
            return False
        if requested.get("provider") and candidate.get("provider") != requested.get("provider"):
            return False
        if explicit_model and candidate.get("model") != requested.get("model"):
            return False
        requested_complexity = requested.get("complexity")
        if requested_complexity and not self._model_fits_complexity(
            candidate.get("model"),
            requested_complexity,
        ):
            return False
        return True

    def _is_terminal_child_state(self, state: Any) -> bool:
        """Return True when a child's lifecycle state is finished."""
        return state in {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.KILLED,
        }

    def _safe_get_all_descriptors(self) -> list[Any]:
        """Return all descriptors while tolerating manager API drift."""
        get_all = getattr(self._manager, "get_all_descriptors", None)
        if callable(get_all):
            try:
                descriptors = get_all()
            except Exception:
                logger.exception("Failed to read all descriptors from manager")
                descriptors = []
            else:
                return list(descriptors or [])

        all_agents = getattr(self._manager, "_agents", None)
        completed_agents = getattr(self._manager, "_completed_agents", None)
        descriptors = []
        if isinstance(all_agents, dict):
            descriptors.extend(all_agents.values())
        if isinstance(completed_agents, dict):
            descriptors.extend(completed_agents.values())

        deduped: list[Any] = []
        seen: set[str] = set()
        for descriptor in descriptors:
            agent_id = str(getattr(descriptor, "agent_id", ""))
            if not agent_id or agent_id in seen:
                continue
            seen.add(agent_id)
            deduped.append(descriptor)
        if not deduped:
            logger.warning(
                "Manager descriptor fallback returned no entries for orchestrator tools"
            )
        return deduped

    def _safe_get_descriptor(self, agent_id: str) -> Any | None:
        """Read active or completed descriptor with safe fallbacks."""
        if not agent_id:
            return None
        get_desc = getattr(self._manager, "get_descriptor", None)
        if callable(get_desc):
            try:
                descriptor = get_desc(agent_id)
            except Exception:
                logger.exception("Failed to read descriptor %s from manager", agent_id[:8])
            else:
                if descriptor is not None:
                    return descriptor

        get_completed = getattr(self._manager, "get_completed_descriptor", None)
        if callable(get_completed):
            try:
                descriptor = get_completed(agent_id)
            except Exception:
                logger.exception(
                    "Failed to read completed descriptor %s from manager",
                    agent_id[:8],
                )
            else:
                if descriptor is not None:
                    return descriptor

        completed_agents = getattr(self._manager, "_completed_agents", None)
        if isinstance(completed_agents, dict):
            return completed_agents.get(agent_id)
        return None

    def _register_wait_child(self, child_id: str) -> None:
        """Track a child ID for future wait_for_message coordination."""
        child_id = str(child_id or "")
        if child_id:
            self._tracked_wait_children.add(child_id)

    def _prune_tracked_wait_children(
        self,
        parent_id: str,
    ) -> None:
        """Drop tracked IDs that no longer belong to this parent or are terminal."""
        if not self._tracked_wait_children:
            return

        tracked = list(self._tracked_wait_children)
        all_desc = self._safe_get_all_descriptors()
        if not all_desc:
            return
        descriptors = {desc.agent_id: desc for desc in all_desc}
        for child_id in tracked:
            desc = descriptors.get(child_id)
            if desc is None:
                continue
            if desc.parent_id != parent_id or self._is_terminal_child_state(
                getattr(desc, "state", None)
            ):
                self._tracked_wait_children.discard(child_id)

    def _get_active_children_for_parent(self, parent_id: str) -> list[str]:
        """Return non-terminal child IDs for a parent."""
        parent = self._safe_get_descriptor(parent_id)
        all_desc = self._safe_get_all_descriptors()
        by_id = {desc.agent_id: desc for desc in all_desc}
        active_children: list[str] = []

        observed_children = []
        if parent is not None:
            observed_children.extend(getattr(parent, "children", []) or [])
        else:
            observed_children.extend(
                [desc.agent_id for desc in all_desc if desc.parent_id == parent_id]
            )

        seen: set[str] = set()
        for child_id in observed_children:
            cid = str(child_id)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            child_desc = by_id.get(cid)
            if child_desc is None:
                continue
            if self._is_terminal_child_state(getattr(child_desc, "state", None)):
                continue
            active_children.append(cid)

        # Include any parent-owned descriptors not referenced in parent.children.
        for desc in all_desc:
            cid = str(getattr(desc, "agent_id", ""))
            if not cid or cid in seen:
                continue
            if getattr(desc, "parent_id", None) != parent_id:
                continue
            if self._is_terminal_child_state(getattr(desc, "state", None)):
                continue
            seen.add(cid)
            active_children.append(cid)

        return active_children

    def _collect_wait_targets(self, parent_id: str) -> list[str]:
        """Get child IDs this master should wait for in wait_for_message."""
        self._prune_tracked_wait_children(parent_id)
        active_children = self._get_active_children_for_parent(parent_id)
        if not self._tracked_wait_children:
            return active_children

        tracked_waiting = [
            child_id for child_id in active_children
            if child_id in self._tracked_wait_children
        ]
        # If the tracked set is out of sync with active children (for example,
        # after a reconnect/reuse flow), fall back to all active children so
        # we continue waiting on actual running work rather than stopping early.
        if tracked_waiting:
            return tracked_waiting
        if active_children:
            return active_children
        return list(self._tracked_wait_children)

    def _children_wait_status_lines(self, child_ids: list[str]) -> list[str]:
        """Build compact status lines for a list of child IDs."""
        descriptors = {
            desc.agent_id: desc for desc in self._safe_get_all_descriptors()
        }
        lines = []
        for child_id in child_ids:
            desc = descriptors.get(child_id)
            if desc is None:
                lines.append(f"- {child_id}: completed or no longer known")
                continue
            state = desc.state
            state_val = state.value if hasattr(state, "value") else str(state)
            model = getattr(desc, "model", "")
            lines.append(f"- {child_id}: state={state_val} model={model}")
        if not lines:
            lines.append("No tracked children currently pending")
        return lines

    def _serialize_wait_message(self, msg) -> str:
        """Serialize one routed message for wait_for_message output."""
        return (
            f"type: {msg.message_type.value}\n"
            f"from: {msg.source_agent_id}\n"
            f"correlation_id: {msg.correlation_id or msg.message_id}\n"
            f"payload: {json.dumps(msg.payload, default=str)}"
        )

    async def _notify_child_reconnected(self, child_desc: Any) -> None:
        """Emit a restart-style event so UI tree can attach an existing child."""
        if not self._event_callback:
            return

        child_id = str(getattr(child_desc, "agent_id", ""))
        if not child_id:
            return

        role = getattr(child_desc, "role", None)
        await fire_event(
            self._event_callback,
            {
                "event": "agent_restarted",
                "agent_id": child_id,
                "parent_id": getattr(child_desc, "parent_id", None),
                "role": role.value if hasattr(role, "value") else str(role or "worker"),
                "model": getattr(child_desc, "model", ""),
                "prompt": strip_injected_prompt_prefix(
                    str(getattr(child_desc, "prompt", ""))
                )[:200],
            },
        )

    @staticmethod
    def _is_terminal_state(state: Any) -> bool:
        return state in {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.KILLED,
        }

    def _restart_produced_active_child(
        self,
        child_id: str,
        restarted: Any,
    ) -> bool:
        state = getattr(restarted, "state", None)
        if not self._is_terminal_state(state):
            return True

        get_descriptor = getattr(self._manager, "get_descriptor", None)
        if callable(get_descriptor):
            active = get_descriptor(child_id)
            if active is not None and not self._is_terminal_state(
                getattr(active, "state", None)
            ):
                return True
        return False

    async def _find_reusable_child(
        self,
        *,
        requested_profile: dict[str, Any],
        explicit_model: bool,
    ) -> tuple[str, Any] | None:
        get_all = getattr(self._manager, "get_all_descriptors", None)
        if not callable(get_all):
            return None
        try:
            all_desc = get_all()
        except Exception:
            logger.exception("Failed to list descriptors for child reuse")
            return None

        restartable_states = {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.KILLED,
        }
        active_states = {
            AgentState.PENDING,
            AgentState.STARTING,
            AgentState.RUNNING,
            AgentState.WAITING_FOR_PARENT,
            AgentState.WAITING_FOR_CHILD,
            AgentState.WAITING_FOR_EXPERT,
        }
        store = self._ensure_child_profile_store()
        restartable: list[Any] = []
        active: list[Any] = []

        for desc in all_desc:
            if getattr(desc, "parent_id", None) != self._agent_id:
                continue
            if getattr(desc, "role", None) != AgentRole.WORKER:
                continue
            child_id = str(getattr(desc, "agent_id", ""))
            if not child_id:
                continue
            candidate_profile = store.get(child_id)
            if candidate_profile is None:
                candidate_profile = self._profile_from_descriptor(desc)
                store[child_id] = candidate_profile
            if not self._profiles_match(
                candidate_profile,
                requested_profile,
                explicit_model=explicit_model,
            ):
                continue
            state = getattr(desc, "state", None)
            if state in restartable_states:
                restartable.append(desc)
            elif state in active_states:
                active.append(desc)

        if active:
            best_active = max(
                active,
                key=lambda d: self._profile_recency(
                    store.get(str(getattr(d, "agent_id", "")), {})
                ),
            )
            return ("active", best_active)

        if not restartable:
            return None

        if not callable(getattr(self._manager, "restart_agent", None)):
            return None
        restartable.sort(
            key=lambda d: self._profile_recency(
                store.get(str(getattr(d, "agent_id", "")), {})
            ),
            reverse=True,
        )
        for desc in restartable:
            child_id = str(getattr(desc, "agent_id", ""))
            if not child_id:
                continue
            try:
                restarted = await self._manager.restart_agent(child_id, requested_profile["restart_prompt"])
                if not self._restart_produced_active_child(child_id, restarted):
                    logger.warning(
                        "Child reuse restart returned terminal state; skipping reuse "
                        "parent=%s child=%s state=%s",
                        self._agent_id[:8],
                        child_id[:8],
                        getattr(restarted, "state", None),
                    )
                    continue
                self._record_child_profile(restarted.agent_id, requested_profile)
                return ("restarted", restarted)
            except Exception:
                logger.exception(
                    "Failed to restart matching child for reuse parent=%s child=%s",
                    self._agent_id[:8],
                    child_id[:8],
                )
                continue
        return None

    def _is_master(self) -> bool:
        """Check if this agent is a master/orchestrator (allowed to spawn)."""
        desc = self._manager.get_descriptor(self._agent_id)
        if desc is None:
            return False
        return desc.role == AgentRole.MASTER

    @staticmethod
    def _signal_process_group(
        proc: asyncio.subprocess.Process,
        sig: signal.Signals,
    ) -> bool:
        """Send a signal to the process group when available."""
        if proc.returncode is not None:
            return False
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    async def _ensure_process_group_stopped(
        proc: asyncio.subprocess.Process,
        *,
        agent_id: str,
        tool_call_id: str,
    ) -> None:
        """Escalate stop signals to guarantee process-group termination."""
        sigint_grace = float(os.getenv("PRSM_BASH_STOP_SIGINT_GRACE_SECONDS", "0.75"))
        sigterm_grace = float(os.getenv("PRSM_BASH_STOP_SIGTERM_GRACE_SECONDS", "1.0"))

        if proc.returncode is not None:
            return

        await asyncio.sleep(max(0.0, sigint_grace))
        if proc.returncode is not None:
            return

        term_sent = OrchestrationTools._signal_process_group(proc, signal.SIGTERM)
        logger.warning(
            "Bash subprocess still running after SIGINT; escalating to SIGTERM "
            "agent=%s tool_call_id=%s pid=%s sent=%s",
            agent_id[:8],
            tool_call_id[:12],
            proc.pid,
            term_sent,
        )
        if not term_sent:
            try:
                proc.terminate()
            except ProcessLookupError:
                return
            except Exception:
                pass

        await asyncio.sleep(max(0.0, sigterm_grace))
        if proc.returncode is not None:
            return

        kill_sent = OrchestrationTools._signal_process_group(proc, signal.SIGKILL)
        logger.error(
            "Bash subprocess still running after SIGTERM; escalating to SIGKILL "
            "agent=%s tool_call_id=%s pid=%s sent=%s",
            agent_id[:8],
            tool_call_id[:12],
            proc.pid,
            kill_sent,
        )
        if not kill_sent:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            except Exception:
                pass

    def get_bash_subprocess_pid(self, tool_call_id: str) -> int | None:
        """Return PID for an active bash subprocess by tool_call_id, if tracked."""
        proc = self._active_bash_processes.get(tool_call_id)
        if proc is None:
            global_entry = _GLOBAL_ACTIVE_BASH_PROCESSES.get(tool_call_id)
            if global_entry and global_entry[0] == self._agent_id:
                proc = global_entry[1]
        if proc is None:
            return None
        return proc.pid

    def get_active_bash_process_snapshot(self) -> list[dict[str, Any]]:
        """Return active bash process metadata for diagnostics/logging."""
        snapshot: list[dict[str, Any]] = []
        seen_tool_ids: set[str] = set()
        for tool_id, proc in self._active_bash_processes.items():
            snapshot.append(
                {
                    "tool_call_id": tool_id,
                    "pid": proc.pid,
                    "returncode": proc.returncode,
                    "source": "local",
                }
            )
            seen_tool_ids.add(tool_id)
        for tool_id, (agent_id, proc) in _GLOBAL_ACTIVE_BASH_PROCESSES.items():
            if agent_id != self._agent_id or tool_id in seen_tool_ids:
                continue
            snapshot.append(
                {
                    "tool_call_id": tool_id,
                    "pid": proc.pid,
                    "returncode": proc.returncode,
                    "source": "global",
                }
            )
        return snapshot

    def _get_allowed_peer_model_ids(self) -> set[str] | None:
        """Get the set of model IDs that are allowed for child spawning.

        Returns the model IDs from configured peer_models. If no peer_models
        are configured, returns None (which means all models are allowed for
        backward compatibility).
        """
        if not self._peer_models:
            return None

        allowed_ids = set()
        for peer_alias, (peer_provider, peer_model_id) in self._peer_models.items():
            allowed_ids.add(peer_model_id)
        return allowed_ids

    def _model_display_name(self, model_id: str) -> str:
        """Prefer configured aliases when presenting model names to users."""
        if not model_id:
            return model_id

        for alias, (_provider, peer_model_id) in self._peer_models.items():
            if peer_model_id == model_id:
                return alias

        if self._model_registry:
            list_aliases = getattr(self._model_registry, "list_aliases", None)
            if callable(list_aliases):
                aliases = list_aliases()
                matches = [
                    str(alias)
                    for alias, target in aliases.items()
                    if str(target) == model_id
                ]
                if matches:
                    matches.sort(key=lambda alias: (alias == model_id, len(alias), alias))
                    return matches[0]
        return model_id

    def _inject_docs_instruction(self, prompt: str) -> str:
        """Ensure child prompts include docs-first architecture guidance."""
        if (
            DOCS_PROMPT_SENTINEL in prompt
            and ASSUMPTION_MINIMIZATION_PROMPT_SENTINEL in prompt
        ):
            return prompt
        return (
            f"{DOCS_REVIEW_INSTRUCTION}"
            f"{ASSUMPTION_MINIMIZATION_INSTRUCTION}\n"
            f"TASK:\n{prompt}"
        )

    async def _with_timeout(
        self,
        coro_func, # Renamed from coro
        tool_name: str,
        tool_call_id: str,
        *coro_args: Any, # New: to pass arguments to coro_func
        **coro_kwargs: Any,
    ) -> dict[str, Any]:
        """Run a tool coroutine with the per-call timeout and time tracking.

        Enters the time tracker before the coroutine starts, exits
        after it finishes (or times out). If the coroutine exceeds
        tool_call_timeout, returns an error result to the agent.
        """
        self._tool_call_seq += 1
        call_seq = self._tool_call_seq
        started = time.monotonic()
        logger.info(
            "Tool start agent=%s seq=%s tool=%s tool_call_id=%s timeout_s=%.1f",
            self._agent_id[:8],
            call_seq,
            tool_name,
            tool_call_id[:8],
            self._tool_call_timeout,
        )
        self.time_tracker.enter()
        try:
            if self._tool_call_timeout <= 0:
                result = await coro_func(*coro_args, **coro_kwargs)
            else:
                result = await asyncio.wait_for(
                    coro_func(*coro_args, **coro_kwargs), # Call coro_func with its args
                    timeout=self._tool_call_timeout
                )
            elapsed = time.monotonic() - started
            logger.info(
                "Tool end agent=%s seq=%s tool=%s tool_call_id=%s duration_s=%.2f is_error=%s",
                self._agent_id[:8],
                call_seq,
                tool_name,
                tool_call_id[:8],
                elapsed,
                bool(result.get("is_error")),
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "Tool timeout agent=%s seq=%s tool=%s tool_call_id=%s timeout_s=%.1f",
                self._agent_id[:8],
                call_seq,
                tool_name,
                tool_call_id[:8],
                self._tool_call_timeout,
            )
            return _error(
                f"Tool call '{tool_name}' timed out after "
                f"{self._tool_call_timeout:.0f}s. The operation took "
                f"too long and was cancelled."
            )
        except asyncio.CancelledError:
            logger.warning(
                "Tool cancelled agent=%s seq=%s tool=%s tool_call_id=%s",
                self._agent_id[:8],
                call_seq,
                tool_name,
                tool_call_id[:8],
            )
            return _error(f"Tool call '{tool_name}' was cancelled.")
        except Exception:
            elapsed = time.monotonic() - started
            logger.exception(
                "Tool crash agent=%s seq=%s tool=%s tool_call_id=%s duration_s=%.2f",
                self._agent_id[:8],
                call_seq,
                tool_name,
                tool_call_id[:8],
                elapsed,
            )
            raise
        finally:
            self.time_tracker.exit()

    # ── ask_parent ─────────────────────────────────────────────

    async def ask_parent(self, question: str) -> dict[str, Any]:
        """Ask your parent agent a question and wait for the answer."""
        tool_call_id = f"ask_parent-{uuid.uuid4().hex}" # Generate tool_call_id
        return await self._with_timeout(
            self._ask_parent_impl,
            "ask_parent",
            tool_call_id,
            question,
        )

    async def _ask_parent_impl(
        self, question: str
    ) -> dict[str, Any]:
        self._meaningful_tool_calls += 1
        descriptor = self._manager.get_descriptor(self._agent_id)
        if not descriptor or not descriptor.parent_id:
            return _error("No parent agent to ask (you are the root)")

        question = str(question).strip()
        if not question:
            return _error("Question cannot be empty")

        msg = RoutedMessage(
            message_type=MessageType.QUESTION,
            source_agent_id=self._agent_id,
            target_agent_id=descriptor.parent_id,
            payload=question,
        )

        self._router.mark_waiting(self._agent_id, descriptor.parent_id)
        await self._manager.transition_agent_state(
            self._agent_id, AgentState.WAITING_FOR_PARENT,
        )

        try:
            await self._router.send(msg)

            answer = await self._router.receive(
                agent_id=self._agent_id,
                message_type_filter=MessageType.ANSWER,
                correlation_id=msg.message_id,
                timeout=self._tool_call_timeout,
            )
            return _text(str(answer.payload))

        except asyncio.TimeoutError:
            return _error(
                f"Parent did not respond within "
                f"{self._tool_call_timeout:.0f} seconds. "
                f"Proceeding without the answer."
            )
        except MessageRoutingError as e:
            return _error(f"Failed to reach parent: {e}")
        finally:
            clear_waiting = getattr(self._router, "clear_waiting", None)
            if callable(clear_waiting):
                clear_waiting(self._agent_id)
            await self._manager.transition_agent_state(
                self._agent_id, AgentState.RUNNING,
            )

    # ── spawn_child ────────────────────────────────────────────

    async def spawn_child(
        self,
        prompt: str,
        wait: bool = False,
        tools: list[str] | None = None,
        model: str | None = None,
        cwd: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
        exclude_plugins: list[str] | None = None,
        complexity: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a child agent to work on a subtask.

        Model selection priority:
        1. Explicit model parameter — always wins
        2. Smart selection via complexity + model_registry — if complexity
           is specified and a model_registry is configured
        3. Default model from engine config
        """
        raw_prompt = str(prompt)
        effective_model = model
        effective_provider: str | None = None
        model_note = ""
        complexity_normalized = self._normalize_complexity(complexity)

        # ── Resolve model aliases (e.g. "claude-sonnet" → "claude-sonnet-4-5-20250929") ──
        if effective_model and self._model_registry:
            resolved = self._model_registry.resolve_alias(effective_model)
            if resolved != effective_model:
                logger.info(
                    "Model alias resolved: %s → %s",
                    effective_model, resolved,
                )
                effective_model = resolved

        if effective_model is None and complexity and self._model_registry:
            # Smart model selection based on task complexity
            rec = self._model_registry.recommend_model(
                prompt, complexity=complexity,
            )

            # If peer_models are configured, ensure the recommendation is in the allowed list
            if rec and self._peer_models:
                allowed_peer_ids = self._get_allowed_peer_model_ids()
                if allowed_peer_ids is not None and rec.model_id not in allowed_peer_ids:
                    # Try to find an alternative from the allowed list
                    category = self._model_registry._infer_category(prompt)
                    ranked_list = self._model_registry.get_ranked_for_task(
                        category, available_only=True,
                    )
                    original_rec = rec
                    rec = None
                    for score, model in ranked_list:
                        if model.model_id in allowed_peer_ids and self._model_fits_complexity(
                            model.model_id,
                            complexity_normalized,
                        ):
                            rec = model
                            break

                    # If no alternative found, return an error
                    if rec is None:
                        available_peers = list(self._peer_models.keys())
                        return _error(
                            f"Smart model selection recommended '{original_rec.model_id}' "
                            f"(tier={original_rec.tier.value}, complexity={complexity}), but this model "
                            f"is not in the allowed peer models list. No suitable alternative found "
                            f"among the allowed models. Available peer models: {', '.join(available_peers)}. "
                            f"Either add a suitable model to your peer_models list, or explicitly "
                            f"specify a model parameter when spawning the child."
                        )

            if rec:
                effective_model = rec.model_id
                effective_provider = rec.provider
                model_note = (
                    f" (auto-selected: {rec.tier.value} tier, "
                    f"complexity={complexity})"
                )
                logger.info(
                    "Smart model selection: task=%s complexity=%s → %s (%s, provider=%s)",
                    prompt[:60],
                    complexity,
                    rec.model_id,
                    rec.tier.value,
                    rec.provider,
                )
        elif effective_model and self._model_registry:
            # Explicit model — look up its provider from the registry
            cap = self._model_registry.get(effective_model)
            if cap:
                if not cap.available:
                    return _error(
                        f"Model '{effective_model}' is not available "
                        f"(provider '{cap.provider}' not installed). "
                        f"Available models: {', '.join(m.model_id for m in self._model_registry.list_available())}"
                    )
                effective_provider = cap.provider

        if effective_model is None:
            effective_model = self._default_model

        if complexity_normalized and not self._model_fits_complexity(
            effective_model,
            complexity_normalized,
        ):
            return _error(
                f"Model '{effective_model}' does not fit complexity "
                f"'{complexity_normalized}'. Choose a model whose tier matches "
                f"that complexity or omit explicit model to use auto-selection."
            )

        # ── Validate that the model is in the allowed peer_models list ──
        if self._peer_models:
            allowed_peer_ids = self._get_allowed_peer_model_ids()
            if allowed_peer_ids is not None:
                # Resolve the effective model to its full ID
                resolved_model = self._model_registry.resolve_alias(effective_model) if self._model_registry else effective_model

                # Check if the resolved model ID is in the allowed list
                if resolved_model not in allowed_peer_ids:
                    available_peers = list(self._peer_models.keys())
                    return _error(
                        f"Model '{effective_model}' (resolved to '{resolved_model}') is not in the allowed peer models list. "
                        f"Only models configured in 'defaults.peer_models' can be used for child agents. "
                        f"Available peer models: {', '.join(available_peers)}. "
                        f"Update your prsm.yaml configuration to add this model to the peer_models list."
                    )

        tool_call_id = f"spawn_child-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._spawn_child_impl,
            "spawn_child",
            tool_call_id,
            prompt, wait, tools, effective_model, cwd,
            mcp_servers, exclude_plugins, model_note,
            effective_provider,
            complexity_normalized,
            model is not None,
            raw_prompt,
        )

    async def _spawn_child_impl(
        self,
        prompt: str,
        wait: bool,
        tools: list[str] | None,
        model: str,
        cwd: str | None,
        mcp_servers: dict[str, Any] | None = None,
        exclude_plugins: list[str] | None = None,
        model_note: str = "",
        provider: str | None = None,
        complexity: str | None = None,
        explicit_model: bool = False,
        raw_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._meaningful_tool_calls += 1
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can spawn children. "
                "Worker and expert agents should do their work directly. "
                "If the task is too large, use task_complete to report "
                "back to your parent with a recommendation to spawn "
                "additional agents."
            )
        prompt = str(prompt).strip()
        if not prompt:
            return _error("Prompt cannot be empty")
        requested_profile = self._build_task_profile(
            prompt=(raw_prompt or prompt),
            complexity=complexity,
            tools=tools,
            cwd=cwd,
            provider=provider or self._default_provider,
            model=model,
        )
        prompt = self._inject_docs_instruction(prompt)
        requested_profile["restart_prompt"] = prompt
        requested_wait = bool(wait)

        reused = await self._find_reusable_child(
            requested_profile=requested_profile,
            explicit_model=explicit_model,
        )
        if reused is not None:
            action, child_desc = reused
            child_id = child_desc.agent_id
            model_display = self._model_display_name(child_desc.model)
            if action == "active":
                self._register_wait_child(child_id)
                await self._notify_child_reconnected(child_desc)
                return _text(
                    f"Reusing active child agent (no new spawn).\\n"
                    f"child_id: {child_id}\\n"
                    f"model: {model_display}\\n"
                    "Use wait_for_message to receive its next result."
                )
            self._register_wait_child(child_id)
            return _text(
                f"Reused existing child agent via restart.\\n"
                f"child_id: {child_id}\\n"
                f"model: {model_display}\\n"
                "Use wait_for_message to receive its result."
            )

        parent = self._manager.get_descriptor(self._agent_id)
        total_spawned = len(parent.children) if parent else 0
        if total_spawned >= self._max_total_children_per_master:
            logger.warning(
                "spawn_child blocked parent=%s total_children=%d limit=%d",
                self._agent_id[:8],
                total_spawned,
                self._max_total_children_per_master,
            )
            return _error(
                "Child spawn limit reached for this master agent "
                f"({total_spawned}/{self._max_total_children_per_master}). "
                "Reuse existing children or restart with a smaller plan."
            )

        # Resolve provider: explicit > auto-detected > configured default
        effective_provider = provider or self._default_provider

        request = SpawnRequest(
            parent_id=self._agent_id,
            prompt=prompt,
            role=AgentRole.WORKER,
            tools=tools or [],
            model=model,
            permission_mode=PermissionMode.BYPASS,  # headless — no TTY for prompts
            cwd=cwd,
            mcp_servers=mcp_servers,
            exclude_plugins=exclude_plugins,
            provider=effective_provider,
        )
        logger.info(
            "spawn_child request agent=%s wait=%s model=%s tools=%d cwd=%s prompt_len=%d",
            self._agent_id[:8],
            requested_wait,
            model,
            len(request.tools),
            cwd or "<default>",
            len(prompt),
        )
        if requested_wait:
            logger.info(
                "spawn_child ignoring wait=true and forcing non-blocking launch parent=%s",
                self._agent_id[:8],
            )

        try:
            child_desc = await self._manager.spawn_agent(request)
            self._register_wait_child(child_desc.agent_id)
            self._record_child_profile(child_desc.agent_id, requested_profile)
            logger.info(
                "spawn_child spawned parent=%s child=%s",
                self._agent_id[:8],
                child_desc.agent_id[:8],
            )
        except MaxDepthExceededError as e:
            return _error(str(e))
        except ProviderNotAvailableError as e:
            return _error(str(e))
        except Exception as e:
            return _error(f"Failed to spawn child: {e}")

        wait_note = ""
        if requested_wait:
            wait_note = (
                " (note: wait=true was requested but is ignored; child launches "
                "are always non-blocking)"
            )
        logger.info(
            "spawn_child returning immediately parent=%s child=%s model=%s%s",
            self._agent_id[:8],
            child_desc.agent_id[:8],
            model,
            model_note,
        )
        model_display = self._model_display_name(model)
        return _text(
            f"Child agent spawned in background.\n"
            f"child_id: {child_desc.agent_id}\n"
            f"model: {model_display}{model_note}\n"
            f"Use wait_for_message to receive its result.{wait_note}"
        )

    # ── spawn_children_parallel ────────────────────────────────

    async def spawn_children_parallel(
        self,
        children: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Spawn multiple child agents in parallel."""
        tool_call_id = f"spawn_children_parallel-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._spawn_children_parallel_impl,
            "spawn_children_parallel",
            tool_call_id,
            children,
        )

    async def _spawn_children_parallel_impl(
        self,
        children: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can spawn children. "
                "Worker and expert agents should do their work directly. "
                "If the task is too large, use task_complete to report "
                "back to your parent with a recommendation to spawn "
                "additional agents."
            )
        if not children:
            return _error("Must specify at least one child")
        if len(children) > self._max_parallel_children_per_call:
            return _error(
                "Too many children requested in one call "
                f"({len(children)} > {self._max_parallel_children_per_call}). "
                "Split work into smaller batches."
            )

        parent = self._manager.get_descriptor(self._agent_id)
        total_spawned = len(parent.children) if parent else 0
        remaining_budget = self._max_total_children_per_master - total_spawned
        logger.info(
            "spawn_children_parallel request parent=%s children=%d",
            self._agent_id[:8],
            len(children),
        )

        child_ids: list[str] = []
        spawn_errors: list[str] = []
        for i, spec in enumerate(children):
            if isinstance(spec, str):
                spec = {"prompt": spec}

            prompt = str(spec.get("prompt", "")).strip()
            if not prompt:
                spawn_errors.append(f"Child {i}: empty prompt")
                continue
            raw_prompt = prompt
            prompt = self._inject_docs_instruction(prompt)
            complexity = self._normalize_complexity(spec.get("complexity"))

            # Resolve model: explicit > smart selection > default
            child_model = spec.get("model")
            child_provider: str | None = None

            # Resolve aliases (e.g. "claude-sonnet" → full versioned ID)
            if child_model and self._model_registry:
                resolved = self._model_registry.resolve_alias(child_model)
                if resolved != child_model:
                    logger.info(
                        "Model alias resolved (child %d): %s → %s",
                        i, child_model, resolved,
                    )
                    child_model = resolved

            if child_model is None:
                if complexity and self._model_registry:
                    rec = self._model_registry.recommend_model(
                        raw_prompt, complexity=complexity,
                    )

                    # If peer_models are configured, ensure the recommendation is in the allowed list
                    if rec and self._peer_models:
                        allowed_peer_ids = self._get_allowed_peer_model_ids()
                        if allowed_peer_ids is not None and rec.model_id not in allowed_peer_ids:
                            # Try to find an alternative from the allowed list
                            category = self._model_registry._infer_category(raw_prompt)
                            ranked_list = self._model_registry.get_ranked_for_task(
                                category, available_only=True,
                            )
                            original_rec = rec
                            rec = None
                            for score, model in ranked_list:
                                if model.model_id in allowed_peer_ids and self._model_fits_complexity(
                                    model.model_id,
                                    complexity,
                                ):
                                    rec = model
                                    break

                            # If no alternative found, add error and skip this child
                            if rec is None:
                                available_peers = list(self._peer_models.keys())
                                spawn_errors.append(
                                    f"Child {i}: Smart selection recommended '{original_rec.model_id}' "
                                    f"(complexity={complexity}), but this model is not in the allowed peer models list. "
                                    f"No suitable alternative found. Available: {', '.join(available_peers)}"
                                )
                                continue

                    if rec:
                        child_model = rec.model_id
                        child_provider = rec.provider
                        logger.info(
                            "Smart model selection child %d: complexity=%s → %s (provider=%s)",
                            i, complexity, rec.model_id, rec.provider,
                        )
            elif child_model and self._model_registry:
                # Explicit model — look up its provider and availability
                cap = self._model_registry.get(child_model)
                if cap:
                    if not cap.available:
                        spawn_errors.append(
                            f"Child {i}: model '{child_model}' is not available "
                            f"(provider '{cap.provider}' not installed)"
                        )
                        continue
                    child_provider = cap.provider
            if child_model is None:
                child_model = self._default_model

            if complexity and not self._model_fits_complexity(
                child_model,
                complexity,
            ):
                spawn_errors.append(
                    f"Child {i}: model '{child_model}' does not fit complexity "
                    f"'{complexity}'"
                )
                continue

            # ── Validate that the model is in the allowed peer_models list ──
            if self._peer_models and child_model:
                allowed_peer_ids = self._get_allowed_peer_model_ids()
                if allowed_peer_ids is not None:
                    resolved_model = self._model_registry.resolve_alias(child_model) if self._model_registry else child_model

                    if resolved_model not in allowed_peer_ids:
                        available_peers = list(self._peer_models.keys())
                        spawn_errors.append(
                            f"Child {i}: model '{child_model}' (resolved to '{resolved_model}') is not in the allowed peer models list. "
                            f"Available: {', '.join(available_peers)}"
                        )
                        continue

            requested_profile = self._build_task_profile(
                prompt=raw_prompt,
                complexity=complexity,
                tools=spec.get("tools", []),
                cwd=spec.get("cwd"),
                provider=child_provider or self._default_provider,
                model=str(child_model),
            )
            requested_profile["restart_prompt"] = prompt
            reused = await self._find_reusable_child(
                requested_profile=requested_profile,
                explicit_model=spec.get("model") is not None,
            )
            if reused is not None:
                _action, child_desc = reused
                if child_desc.agent_id not in child_ids:
                    child_ids.append(child_desc.agent_id)
                self._register_wait_child(child_desc.agent_id)
                if _action == "active":
                    await self._notify_child_reconnected(child_desc)
                continue

            if remaining_budget <= 0:
                spawn_errors.append(
                    "Child spawn budget exceeded for this master "
                    f"(remaining 0, limit {self._max_total_children_per_master})"
                )
                continue

            request = SpawnRequest(
                parent_id=self._agent_id,
                prompt=prompt,
                role=AgentRole.WORKER,
                tools=spec.get("tools", []),
                model=str(child_model),
                permission_mode=PermissionMode.BYPASS,  # headless — no TTY for prompts
                cwd=spec.get("cwd"),
                mcp_servers=spec.get("mcp_servers"),
                exclude_plugins=spec.get("exclude_plugins"),
                provider=child_provider or self._default_provider,
            )

            try:
                child_desc = await self._manager.spawn_agent(request)
                remaining_budget -= 1
                child_ids.append(child_desc.agent_id)
                self._register_wait_child(child_desc.agent_id)
                self._record_child_profile(child_desc.agent_id, requested_profile)
                logger.info(
                    "spawn_children_parallel spawned parent=%s child=%s index=%d",
                    self._agent_id[:8],
                    child_desc.agent_id[:8],
                    i,
                )
            except Exception as e:
                spawn_errors.append(f"Child {i}: {e}")
                logger.warning(
                    "spawn_children_parallel failed to spawn child %d parent=%s: %s",
                    i, self._agent_id[:8], e,
                )
                # Stop trying to spawn more, but don't kill already-spawned ones
                break

        # If no children were spawned at all, return error
        if not child_ids:
            return _error(
                f"Failed to spawn any children. "
                f"Errors: {'; '.join(spawn_errors)}"
            )

        logger.info(
            "spawn_children_parallel spawned parent=%s children=%d (non-blocking)",
            self._agent_id[:8],
            len(child_ids),
        )

        # Return immediately — do NOT block waiting for children.
        # The parent must use wait_for_message() to collect results
        # and handle ask_parent questions from children.
        parts = [
            f"Spawned {len(child_ids)} children in parallel:\n",
            *[f"- child_id: {cid}" for cid in child_ids],
            "",
            "Children are running in the background. You MUST now use "
            "wait_for_message() in a loop to collect their results.",
            "",
            "Each child will send a message when done:",
            "- type=TASK_RESULT → child completed (payload has summary)",
            "- type=QUESTION → child needs your input (use respond_to_child to answer)",
            "",
            f"Keep calling wait_for_message() until you have received "
            f"TASK_RESULT from all {len(child_ids)} children.",
            "",
            "You can also call get_children_status() at any time to check progress.",
        ]
        if spawn_errors:
            parts.append(
                f"\nSpawn errors ({len(spawn_errors)}): "
                + "; ".join(spawn_errors)
            )

        return _text("\n".join(parts))

    # ── consult_expert ─────────────────────────────────────────

    async def consult_expert(
        self,
        expert_id: str,
        question: str,
    ) -> dict[str, Any]:
        """Consult a specialist expert agent for advice."""
        tool_call_id = f"consult_expert-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._consult_expert_impl,
            "consult_expert",
            tool_call_id,
            expert_id, question,
        )

    async def _consult_expert_impl(
        self,
        expert_id: str,
        question: str,
    ) -> dict[str, Any]:
        self._meaningful_tool_calls += 1
        expert_id = str(expert_id).strip()
        question = str(question).strip()

        if not expert_id or not question:
            return _error("Both expert_id and question are required")

        try:
            profile = self._expert_registry.get(expert_id)
        except ExpertNotFoundError:
            available = ", ".join(self._expert_registry.list_ids())
            return _error(
                f"Expert '{expert_id}' not found. "
                f"Available: {available}"
            )

        expert_prompt = (
            f"{profile.system_prompt}\n\n"
            f"A colleague is consulting you with the following "
            f"question. Provide a thorough, actionable answer.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"After answering, call task_complete with your response."
        )

        # Validate expert's provider is available before spawning
        if self._provider_registry and profile.provider:
            if not self._provider_registry.is_provider_available(profile.provider):
                available = self._provider_registry.list_available()
                return _error(
                    f"Expert '{expert_id}' requires provider "
                    f"'{profile.provider}' which is not available "
                    f"(CLI not installed). "
                    f"Available providers: {', '.join(available) or 'none'}"
                )

        request = SpawnRequest(
            parent_id=self._agent_id,
            prompt=expert_prompt,
            role=AgentRole.EXPERT,
            expert_id=expert_id,
            tools=profile.tools,
            model=profile.model,
            permission_mode=PermissionMode.BYPASS,  # headless — no TTY for prompts
            cwd=profile.cwd,
            mcp_servers=profile.mcp_servers,
            provider=profile.provider,
        )

        self._router.mark_waiting(self._agent_id, "expert-pending")
        await self._manager.transition_agent_state(
            self._agent_id, AgentState.WAITING_FOR_EXPERT,
        )
        try:
            child_desc = await self._manager.spawn_agent(request)
            self._router.mark_waiting(
                self._agent_id, child_desc.agent_id
            )
            result = await self._manager.wait_for_result(
                child_desc.agent_id
            )

            return _text(
                f"Expert '{profile.name}' responded "
                f"(success={result.success}, "
                f"duration={result.duration_seconds:.1f}s):\n\n"
                f"{result.summary}"
            )
        except Exception as e:
            return _error(f"Expert consultation failed: {e}")
        finally:
            self._router.clear_waiting(self._agent_id)
            await self._manager.transition_agent_state(
                self._agent_id, AgentState.RUNNING,
            )

    # ── report_progress (non-blocking, still tracked) ──────────

    async def report_progress(
        self,
        status: str,
        percent_complete: int = 0,
    ) -> dict[str, Any]:
        """Send a non-blocking progress update to your parent."""
        tool_call_id = f"report_progress-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._report_progress_impl,
            "report_progress",
            tool_call_id,
            status, percent_complete,
        )

    async def _report_progress_impl(
        self,
        status: str,
        percent_complete: int,
    ) -> dict[str, Any]:
        self._meaningful_tool_calls += 1
        descriptor = self._manager.get_descriptor(self._agent_id)
        if not descriptor or not descriptor.parent_id:
            return _text("No parent to report to (root). Noted.")

        msg = RoutedMessage(
            message_type=MessageType.PROGRESS_UPDATE,
            source_agent_id=self._agent_id,
            target_agent_id=descriptor.parent_id,
            payload={
                "status": str(status),
                "percent_complete": int(percent_complete),
            },
        )

        try:
            await self._router.send(msg)
        except MessageRoutingError:
            pass  # Non-critical

        return _text(
            f"Progress reported: {percent_complete}% - {status}"
        )

    # ── task_complete (non-blocking, still tracked) ────────────

    async def task_complete(
        self,
        summary: str,
        artifacts: dict[str, Any] | None = None,
        steps: list[str] | None = None,
        assumptions: list[str] | None = None,
        risks: list[str] | None = None,
        rollback_plan: str | None = None,
        confidence: float | None = None,
        verification_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Signal that your assigned task is complete.

        Calling this tool will immediately terminate your agent session.
        Any code or thoughts generated after this tool call will be ignored.
        """
        tool_call_id = f"task_complete-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._task_complete_impl,
            "task_complete",
            tool_call_id,
            summary=summary,
            artifacts=artifacts,
            steps=steps,
            assumptions=assumptions,
            risks=risks,
            rollback_plan=rollback_plan,
            confidence=confidence,
            verification_results=verification_results,
        )

    async def _task_complete_impl(
        self,
        summary: str,
        artifacts: dict[str, Any] | None,
        steps: list[str] | None = None,
        assumptions: list[str] | None = None,
        risks: list[str] | None = None,
        rollback_plan: str | None = None,
        confidence: float | None = None,
        verification_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summary_text = str(summary or "").strip()
        if not summary_text:
            return _error(
                "task_complete requires a non-empty summary; "
                "provide a concise completion summary."
            )

        merged_artifacts = self._merge_task_complete_artifacts(
            artifacts=artifacts,
            steps=steps,
            assumptions=assumptions,
            risks=risks,
            rollback_plan=rollback_plan,
            confidence=confidence,
            verification_results=verification_results,
        )
        logger.info(
            "task_complete called agent=%s summary_len=%d artifacts_keys=%d",
            self._agent_id[:8],
            len(summary_text),
            len(merged_artifacts.keys()),
        )
        descriptor = self._manager.get_descriptor(self._agent_id)

        # Warn (but don't block) master agents completing without doing any tracked work.
        # Note: They might have done work using SDK tools (Bash, Read, etc.) which
        # are not tracked by self._meaningful_tool_calls.
        if (
            descriptor
            and descriptor.role == AgentRole.MASTER
            and self._meaningful_tool_calls == 0
        ):
            logger.warning(
                "Master agent %s called task_complete with 0 meaningful orchestration tool calls",
                self._agent_id[:8],
            )
            # We used to return _error() here, but that causes infinite loops if the
            # agent actually did work using non-orchestration tools (Bash, Read).
            # So we just warn and proceed.
        if descriptor:
            descriptor.result_summary = summary_text
            descriptor.result_artifacts = merged_artifacts
            if (
                self._conversation_store is not None
                and hasattr(self._conversation_store, "append_decision_report")
            ):
                self._conversation_store.append_decision_report(
                    self._agent_id,
                    {
                        "task_prompt": getattr(descriptor, "prompt", ""),
                        "decision": "accept",
                        "rationale": summary_text,
                        "artifact_ids": merged_artifacts.get("artifact_ids", []),
                        "policy_snapshot_id": merged_artifacts.get(
                            "policy_snapshot_id", ""
                        ),
                        "verification_results": merged_artifacts.get(
                            "verification_results", []
                        ),
                    },
                )

        if descriptor and descriptor.parent_id:
            payload: dict[str, Any] = {
                "summary": summary_text,
                "artifacts": merged_artifacts,
            }
            if (
                self._conversation_store is not None
                and hasattr(self._conversation_store, "get_history")
            ):
                try:
                    payload["history"] = self._conversation_store.get_history(
                        self._agent_id, detail_level="full",
                    )
                    payload["history_detail"] = "full"
                except Exception:
                    logger.exception(
                        "Failed to capture full child history for task_result "
                        "agent=%s",
                        self._agent_id[:8],
                    )
            msg = RoutedMessage(
                message_type=MessageType.TASK_RESULT,
                source_agent_id=self._agent_id,
                target_agent_id=descriptor.parent_id,
                payload=payload,
            )
            try:
                await self._router.send(msg)
            except MessageRoutingError as e:
                logger.warning(
                    "Failed to deliver TASK_RESULT to parent agent=%s parent=%s: %s",
                    self._agent_id[:8],
                    descriptor.parent_id[:8],
                    e,
                )

        return _text(
            "Task marked complete. Session will end.\n"
            f"{DOCS_UPDATE_AFTER_TASK_COMPLETE_INSTRUCTION}"
        )

    @staticmethod
    def _merge_task_complete_artifacts(
        artifacts: dict[str, Any] | None,
        steps: list[str] | None,
        assumptions: list[str] | None,
        risks: list[str] | None,
        rollback_plan: str | None,
        confidence: float | None,
        verification_results: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(artifacts or {})
        if steps:
            merged["steps"] = list(steps)
        if assumptions:
            merged["assumptions"] = list(assumptions)
        if risks:
            merged["risks"] = list(risks)
        if rollback_plan:
            merged["rollback_plan"] = str(rollback_plan)
        if confidence is not None:
            merged["confidence"] = float(confidence)
        if verification_results:
            merged["verification_results"] = list(verification_results)
        return merged

    # ── wait_for_message ───────────────────────────────────────

    async def wait_for_message(
        self,
        timeout_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Wait for the next incoming message for this agent."""
        # When timeout_seconds <= 0, the caller explicitly wants "no timeout"
        # (block indefinitely). Only cap a positive user timeout against the
        # per-tool-call timeout.
        if timeout_seconds <= 0:
            effective_timeout = 0.0
        elif self._tool_call_timeout <= 0:
            effective_timeout = timeout_seconds
        else:
            effective_timeout = min(timeout_seconds, self._tool_call_timeout)
        tool_call_id = f"wait_for_message-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._wait_for_message_impl,
            "wait_for_message",
            tool_call_id,
            effective_timeout,
        )

    async def _wait_for_message_impl(
        self,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        descriptor = self._manager.get_descriptor(self._agent_id)
        is_master = descriptor is not None and descriptor.role == AgentRole.MASTER
        waiting_state = AgentState.WAITING_FOR_CHILD
        if not is_master:
            waiting_state = AgentState.WAITING_FOR_PARENT

        # Register in the deadlock wait graph so the deadlock detector
        # can see that this agent is blocked on incoming messages.
        mark_waiting = getattr(self._router, "mark_waiting", None)
        if callable(mark_waiting):
            mark_waiting(self._agent_id, "wait_for_message")
        await self._manager.transition_agent_state(
            self._agent_id, waiting_state,
        )
        try:
            # Workers/experts continue to use the previous single-message behavior.
            if not is_master:
                msg = await self._router.receive(
                    agent_id=self._agent_id,
                    timeout=timeout_seconds,
                )
                return _text(
                    "Message received:\n"
                    f"{self._serialize_wait_message(msg)}"
                )

            wait_for = self._collect_wait_targets(self._agent_id)
            if not wait_for:
                # Fallback for masters with no currently tracked children:
                # behave like the legacy single-message wait and unblock when a
                # message is present, instead of stalling the orchestrator.
                msg = await self._router.receive(
                    agent_id=self._agent_id,
                    timeout=timeout_seconds,
                )
                return _text(
                    "Message received:\n"
                    f"{self._serialize_wait_message(msg)}"
                )

            start = time.monotonic()
            seen_messages: list[Any] = []
            pending = set(wait_for)
            # Keep tracking if a child exits without an explicit message.
            all_desc = self._safe_get_all_descriptors()
            by_id = {desc.agent_id: desc for desc in all_desc}

            poll_interval = 1.0
            while pending:
                discovered = self._collect_wait_targets(self._agent_id)
                if discovered:
                    pending.update(discovered)

                all_desc = self._safe_get_all_descriptors()
                by_id = {desc.agent_id: desc for desc in all_desc}
                for child_id in list(pending):
                    child_desc = by_id.get(child_id)
                    if child_desc is None or self._is_terminal_child_state(
                        getattr(child_desc, "state", None)
                    ):
                        pending.discard(child_id)
                        self._tracked_wait_children.discard(child_id)
                if not pending:
                    break

                if timeout_seconds > 0:
                    elapsed = time.monotonic() - start
                    remaining = timeout_seconds - elapsed
                    if remaining <= 0:
                        break
                    wait_timeout = min(poll_interval, remaining)
                else:
                    wait_timeout = poll_interval

                msg = None
                try:
                    msg = await self._router.receive(
                        agent_id=self._agent_id,
                        timeout=wait_timeout,
                    )
                except asyncio.TimeoutError:
                    # Regular polling to observe terminal child transitions.
                    msg = None

                if msg is not None:
                    seen_messages.append(msg)

                    if (
                        msg.message_type == MessageType.TASK_RESULT
                        and msg.source_agent_id in pending
                    ):
                        pending.discard(msg.source_agent_id)
                        self._tracked_wait_children.discard(msg.source_agent_id)

                    if msg.message_type == MessageType.QUESTION:
                        return _text(
                            "Child requires input while waiting:\n"
                            f"{self._serialize_wait_message(msg)}\n"
                            "Pending children after this message:\n"
                            + "\n".join(
                                self._children_wait_status_lines(
                                    sorted(pending)
                                )
                            )
                        )

            if pending:
                return _text(
                    f"No new messages within {timeout_seconds}s while waiting. "
                    f"Still waiting on {len(pending)} child(ren):\n"
                    + "\n".join(self._children_wait_status_lines(sorted(pending)))
                    + "\n\n"
                    "Call wait_for_message() again to continue waiting for results, "
                    "or call get_children_status() to inspect progress."
                )

            if not seen_messages:
                return _text(
                    f"Tracked children are now terminal: "
                    f"{', '.join(sorted(wait_for))}"
                )

            return _text(
                f"Collected {len(seen_messages)} child message(s). "
                "Pending children: none.\n\n"
                "Messages:\n"
                + "\n\n".join(
                    [f"Message {idx + 1}:\n{self._serialize_wait_message(m)}"
                     for idx, m in enumerate(seen_messages)]
                )
            )

        except asyncio.TimeoutError:
            return _text(
                f"No messages received within {timeout_seconds}s. "
                f"This is normal — your child agents are likely still working. "
                f"Call wait_for_message() again to continue waiting for results. "
                f"Do NOT start doing the work yourself. "
                f"Use get_children_status() to check if children are still running."
            )
        finally:
            clear_waiting = getattr(self._router, "clear_waiting", None)
            if callable(clear_waiting):
                clear_waiting(self._agent_id)
            await self._manager.transition_agent_state(
                self._agent_id, AgentState.RUNNING,
            )

    # ── respond_to_child (non-blocking, still tracked) ─────────

    async def respond_to_child(
        self,
        child_agent_id: str,
        correlation_id: str,
        response: str,
    ) -> dict[str, Any]:
        """Send a response to a child agent's question."""
        tool_call_id = f"respond_to_child-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._respond_to_child_impl,
            "respond_to_child",
            tool_call_id,
            child_agent_id, correlation_id, response,
        )

    async def _respond_to_child_impl(
        self,
        child_agent_id: str,
        correlation_id: str,
        response: str,
    ) -> dict[str, Any]:
        child_agent_id = str(child_agent_id).strip()
        correlation_id = str(correlation_id).strip()
        response = str(response).strip()

        if not child_agent_id or not correlation_id or not response:
            return _error(
                "child_agent_id, correlation_id, and response "
                "are all required"
            )

        msg = RoutedMessage(
            message_type=MessageType.ANSWER,
            source_agent_id=self._agent_id,
            target_agent_id=child_agent_id,
            payload=response,
            correlation_id=correlation_id,
        )

        try:
            await self._router.send(msg)
            return _text(
                f"Response delivered to child "
                f"{child_agent_id[:8]}..."
            )
        except MessageRoutingError as e:
            return _error(f"Failed to deliver response: {e}")


    # ── get_children_status ──────────────────────────────────────

    async def get_children_status(self) -> dict[str, Any]:
        """Get the current status of all children spawned by this agent.

        Returns the state of each child (running, completed, failed, etc.)
        and a summary count. Non-blocking — use this to check progress
        without waiting.
        """
        tool_call_id = f"get_children_status-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._get_children_status_impl,
            "get_children_status",
            tool_call_id,
        )

    async def _get_children_status_impl(self) -> dict[str, Any]:
        all_desc = self._safe_get_all_descriptors()
        children = [
            d for d in all_desc
            if d.parent_id == self._agent_id
        ]
        if not children:
            return _text("No children found for this agent.")

        completed = 0
        failed = 0
        running = 0
        waiting = 0
        killed = 0
        pending = 0
        starting = 0
        other = 0
        lines = []
        for child in sorted(children, key=lambda c: (str(c.created_at), c.agent_id)):
            state_val = child.state.value if hasattr(child.state, 'value') else str(child.state)
            lines.append(
                f"- {child.agent_id}: state={state_val}"
            )
            if child.state == AgentState.COMPLETED:
                completed += 1
            elif child.state == AgentState.FAILED:
                failed += 1
            elif child.state == AgentState.RUNNING:
                running += 1
            elif child.state in (
                AgentState.WAITING_FOR_PARENT,
                AgentState.WAITING_FOR_CHILD,
                AgentState.WAITING_FOR_EXPERT,
            ):
                waiting += 1
            elif child.state == AgentState.KILLED:
                killed += 1
            elif child.state == AgentState.PENDING:
                pending += 1
            elif child.state == AgentState.STARTING:
                starting += 1
            else:
                other += 1

        total = len(children)
        summary = (
            f"Children status: {completed} completed, {failed} failed, "
            f"{running} running, {waiting} waiting"
        )
        if killed:
            summary += f", {killed} killed"
        if starting:
            summary += f", {starting} starting"
        if pending:
            summary += f", {pending} pending"
        if other:
            summary += f", {other} other"

        still_active = running + waiting + starting + pending
        guidance = ""
        if still_active > 0:
            guidance = (
                "\n\n⚠️ Children are still working. Do NOT give up on them, "
                "do NOT start doing their work yourself, and do NOT kill them "
                "unless the user explicitly asks. Return to wait_for_message() "
                "to collect their results."
            )
        return _text(
            f"{summary} (total: {total})\n\n"
            + "\n".join(lines)
            + guidance
        )

    # ── restart_child ──────────────────────────────────────────

    async def restart_child(
        self,
        child_agent_id: str,
        prompt: str,
        wait: bool = False,
    ) -> dict[str, Any]:
        """Restart a completed or failed child agent with a new prompt."""
        tool_call_id = f"restart_child-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._restart_child_impl,
            "restart_child",
            tool_call_id,
            child_agent_id, prompt, wait,
        )

    async def _restart_child_impl(
        self,
        child_agent_id: str,
        prompt: str,
        wait: bool,
    ) -> dict[str, Any]:
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can restart children."
            )
        child_agent_id = str(child_agent_id).strip()
        prompt = str(prompt).strip()

        if not child_agent_id or not prompt:
            return _error("Both child_agent_id and prompt are required")
        prompt = self._inject_docs_instruction(prompt)

        # Verify the child belongs to this agent
        desc = self._manager.get_completed_descriptor(child_agent_id)
        if desc is None:
            return _error(
                f"Agent {child_agent_id[:8]}... not found in completed pool. "
                f"It may still be running or was never a child of yours."
            )
        if desc.parent_id != self._agent_id:
            return _error(
                f"Agent {child_agent_id[:8]}... is not your child."
            )

        logger.info(
            "restart_child request parent=%s child=%s wait=%s prompt_len=%d",
            self._agent_id[:8],
            child_agent_id[:8],
            wait,
            len(prompt),
        )
        requested_wait = bool(wait)
        if requested_wait:
            logger.info(
                "restart_child ignoring wait=true and forcing non-blocking launch parent=%s child=%s",
                self._agent_id[:8],
                child_agent_id[:8],
            )

        try:
            child_desc = await self._manager.restart_agent(
                child_agent_id, prompt
            )
        except Exception as e:
            return _error(f"Failed to restart child: {e}")
        if not self._restart_produced_active_child(child_agent_id, child_desc):
            state = getattr(child_desc, "state", None)
            state_val = state.value if hasattr(state, "value") else str(state)
            return _error(
                "Restart returned but child did not become active. "
                f"Current state is '{state_val}'."
            )

        wait_note = ""
        if requested_wait:
            wait_note = (
                " (note: wait=true was requested but is ignored; child launches "
                "are always non-blocking)"
            )
        return _text(
            f"Child agent restarted in background.\n"
            f"child_id: {child_desc.agent_id}\n"
            f"Use wait_for_message to receive its result.{wait_note}"
        )

    # ── consult_peer ────────────────────────────────────────────

    async def consult_peer(
        self,
        question: str,
        thread_id: str | None = None,
        peer: str | None = None,
    ) -> dict[str, Any]:
        """Consult a peer provider (e.g. Codex, Gemini, MiniMax) for a second opinion.

        Args:
            question: The question to ask the peer.
            thread_id: Optional thread ID for multi-turn follow-up.
            peer: Optional peer alias to consult a specific peer.
                  If not specified, uses the default peer provider.
                  Use 'list' to see available peers.
        """
        tool_call_id = f"consult_peer-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._consult_peer_impl,
            "consult_peer",
            tool_call_id,
            question, thread_id, peer,
        )

    async def _consult_peer_impl(
        self,
        question: str,
        thread_id: str | None,
        peer: str | None = None,
    ) -> dict[str, Any]:
        # List available peers
        if peer == "list":
            peers = []
            if self._peer_provider:
                peers.append(f"  - default ({self._peer_provider.name}, model={self._peer_model})")
            for alias, (provider, model_id) in self._peer_models.items():
                status = "available" if provider.is_available() else "unavailable"
                peers.append(f"  - {alias} ({provider.name}, model={model_id}, {status})")
            if not peers:
                return _error(
                    "No peer providers configured. Set 'defaults.peer_models' "
                    "in your YAML config to enable consult_peer."
                )
            return _text(
                "Available peer models:\n" + "\n".join(peers)
                + "\n\nUse consult_peer(question, peer='alias') to consult a specific peer."
            )

        # Resolve which peer to use
        provider = self._peer_provider
        model_id = self._peer_model

        if peer and peer in self._peer_models:
            provider, model_id = self._peer_models[peer]
        elif peer and self._peer_models:
            # Try matching by provider name too
            for alias, (p, m) in self._peer_models.items():
                if p.name == peer:
                    provider, model_id = p, m
                    break

        if provider is None:
            available_peers = list(self._peer_models.keys()) if self._peer_models else []
            hint = (
                f" Available peers: {', '.join(available_peers)}"
                if available_peers
                else " Set 'defaults.peer_models' in your YAML config."
            )
            return _error(
                f"No peer provider configured.{hint}"
            )

        question = str(question).strip()
        if not question:
            return _error("Question cannot be empty")

        try:
            result = await provider.send_message(
                question,
                model_id=model_id,
                thread_id=thread_id,
            )

            if not result.success:
                return _error(f"Peer consultation failed: {result.text}")

            response_parts = [
                f"Peer response (provider={provider.name}, model={model_id}):\n",
                result.text,
            ]

            if result.thread_id:
                response_parts.append(
                    f"\n\nthread_id: {result.thread_id}"
                    f"\n(Pass this thread_id to consult_peer for "
                    f"follow-up questions)"
                )

            # Show available peers if there are multiple
            if len(self._peer_models) > 1:
                other_peers = [
                    a for a in self._peer_models.keys()
                    if self._peer_models[a][0] != provider
                ]
                if other_peers:
                    response_parts.append(
                        f"\n\nOther available peers: {', '.join(other_peers)}"
                    )

            return _text("\n".join(response_parts))

        except Exception as exc:
            return _error(f"Peer consultation error: {exc}")

    # ── recommend_model ────────────────────────────────────────

    async def recommend_model(
        self,
        task_description: str,
        complexity: str = "medium",
    ) -> dict[str, Any]:
        """Get a model recommendation for a task from the capability registry.

        Args:
            task_description: Description of the task to find a model for.
            complexity: "trivial", "simple", "medium", "complex", or "frontier".

        Returns model_id, provider, and reasoning for the recommendation.
        """
        tool_call_id = f"recommend_model-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._recommend_model_impl,
            "recommend_model",
            tool_call_id,
            task_description, complexity,
        )

    async def _recommend_model_impl(
        self,
        task_description: str,
        complexity: str,
    ) -> dict[str, Any]:
        if self._model_registry is None:
            return _text(
                f"No model registry configured. Using default: {self._default_model}\n"
                f"Configure 'model_registry' in .prism/prsm.yaml to enable "
                f"intelligent model selection."
            )

        recommendation = self._model_registry.recommend_model(
            task_description,
            complexity=complexity,
        )
        complexity_normalized = self._normalize_complexity(complexity)

        # Filter recommendation by peer_models if configured
        if recommendation and self._peer_models:
            allowed_peer_ids = self._get_allowed_peer_model_ids()
            if allowed_peer_ids is not None and recommendation.model_id not in allowed_peer_ids:
                # Try to find an alternative from the allowed list
                category = self._model_registry._infer_category(task_description)
                ranked_list = self._model_registry.get_ranked_for_task(
                    category, available_only=True,
                )
                for score, model in ranked_list:
                    if model.model_id in allowed_peer_ids and self._model_fits_complexity(
                        model.model_id,
                        complexity_normalized,
                    ):
                        recommendation = model
                        break
                else:
                    # No allowed model found
                    available_peers = list(self._peer_models.keys())
                    return _error(
                        f"No suitable model found in the allowed peer models list. "
                        f"Available peer models: {', '.join(available_peers)}. "
                        f"Update your prsm.yaml configuration to add more models to the peer_models list."
                    )

        if recommendation and not self._model_fits_complexity(
            recommendation.model_id,
            complexity_normalized,
        ):
            return _error(
                f"Recommended model '{recommendation.model_id}' does not fit "
                f"complexity '{complexity_normalized}'."
            )

        if recommendation is None:
            return _text(
                f"No suitable model found. Using default: {self._default_model}"
            )

        category = self._model_registry._infer_category(task_description)
        score = recommendation.score_for_task(category)

        # Show the full ranked list so callers can see fallback options
        ranked_list = self._model_registry.get_ranked_for_task(
            category, available_only=True,
        )

        # Filter ranked list by peer_models if configured
        allowed_peer_ids = self._get_allowed_peer_model_ids()
        if allowed_peer_ids is not None:
            ranked_list = [
                (s, m)
                for s, m in ranked_list
                if m.model_id in allowed_peer_ids
                and self._model_fits_complexity(m.model_id, complexity_normalized)
            ]

        fallbacks = []
        for i, (s, m) in enumerate(ranked_list[1:4], 2):  # Next 3 fallbacks
            fallbacks.append(f"  {i}. {m.model_id} ({m.provider}, score={s:.0%})")
        fallback_text = "\n".join(fallbacks) if fallbacks else "  (none)"

        peer_note = ""
        if allowed_peer_ids:
            peer_note = "\nNOTE: Recommendations are filtered to only include models in 'defaults.peer_models'.\n"

        return _text(
            f"Recommended model: {recommendation.model_id}\n"
            f"Provider: {recommendation.provider}\n"
            f"Tier: {recommendation.tier.value}\n"
            f"Task category: {category}\n"
            f"Affinity score: {score:.0%}\n"
            f"Cost factor: {recommendation.cost_factor}x\n"
            f"Speed factor: {recommendation.speed_factor}x\n{peer_note}\n"
            f"Fallback options (if top choice unavailable):\n"
            f"{fallback_text}\n\n"
            f"Use model='{recommendation.model_id}' in spawn_child to use this model."
        )

    # ── get_model_rankings ──────────────────────────────────────

    async def get_model_rankings(
        self,
        task_category: str | None = None,
    ) -> dict[str, Any]:
        """Get the learned model rankings, optionally filtered by task category.

        Shows the ranked list of models per task category from background
        research. If task_category is specified, only that category is shown.
        Otherwise, all categories with their top 3 models are shown.
        """
        tool_call_id = f"get_model_rankings-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._get_model_rankings_impl,
            "get_model_rankings",
            tool_call_id,
            task_category,
        )

    async def _get_model_rankings_impl(
        self,
        task_category: str | None,
    ) -> dict[str, Any]:
        if self._model_registry is None:
            return _text("No model registry configured.")

        intel = getattr(self._model_registry, "_intelligence", None)
        if intel is None or not intel.has_rankings:
            return _text(
                "No learned model rankings available yet. "
                "The background research agent will populate these automatically. "
                "In the meantime, the system uses static affinity scores.\n\n"
                f"Static rankings summary:\n{self._model_registry.to_summary()}"
            )

        if task_category:
            # Show detailed ranking for one category
            ranked = intel.get_ranked_models(task_category)
            if not ranked:
                return _text(
                    f"No rankings found for category '{task_category}'. "
                    f"Available categories: {', '.join(intel.get_all_categories())}"
                )
            lines = [
                f"Model rankings for '{task_category}' "
                f"(updated {intel.last_updated}):",
                "",
            ]
            available_ids = {
                m.model_id for m in self._model_registry.list_available()
            }
            for i, rm in enumerate(ranked, 1):
                avail = "available" if rm.model_id in available_ids else "UNAVAILABLE"
                lines.append(
                    f"  {i}. {rm.model_id} — score={rm.score:.2f} "
                    f"[{avail}]"
                )
                if rm.reason:
                    lines.append(f"     {rm.reason}")
            return _text("\n".join(lines))

        # Show top 3 for all categories
        lines = [
            f"Model Intelligence Rankings (updated {intel.last_updated})",
            "",
        ]
        available_ids = {
            m.model_id for m in self._model_registry.list_available()
        }
        for cat in sorted(intel.get_all_categories()):
            ranked = intel.get_ranked_models(cat)
            top = ranked[:3]
            entries = []
            for rm in top:
                avail = "" if rm.model_id in available_ids else " [UNAVAIL]"
                entries.append(f"{rm.model_id}({rm.score:.2f}){avail}")
            lines.append(f"  {cat}: {' > '.join(entries)}")
        lines.append("")
        lines.append(
            "Use get_model_rankings(task_category='coding') for detailed view."
        )
        return _text("\n".join(lines))

    # ── get_child_history ───────────────────────────────────────

    async def get_child_history(
        self,
        child_agent_id: str,
        detail_level: str = "full",
    ) -> dict[str, Any]:
        """Review a child agent's conversation history."""
        tool_call_id = f"get_child_history-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._get_child_history_impl,
            "get_child_history",
            tool_call_id,
            child_agent_id, detail_level,
        )

    async def _get_child_history_impl(
        self,
        child_agent_id: str,
        detail_level: str,
    ) -> dict[str, Any]:
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can review "
                "child conversation history."
            )
        child_agent_id = str(child_agent_id).strip()
        if not child_agent_id:
            return _error("child_agent_id is required")

        detail_level = str(detail_level).strip().lower()
        if detail_level not in ("full", "summary"):
            return _error(
                "detail_level must be 'full' or 'summary'"
            )

        if not self._conversation_store:
            return _error("Conversation store not available")

        # Support prefix matching: if the agent_id is a short prefix
        # (e.g., "4925d146" from truncated output), resolve to the full ID
        resolved_id = self._conversation_store.resolve_agent_id(child_agent_id)
        if not resolved_id:
            return _text(
                f"No conversation history found for agent "
                f"{child_agent_id[:12]}."
            )

        history = self._conversation_store.get_history(
            resolved_id, detail_level
        )
        if not history:
            return _text(
                f"No conversation history found for agent "
                f"{child_agent_id[:12]}."
            )

        return _text(json.dumps(history, indent=2, default=str))

    # ── check_child_status ─────────────────────────────────────

    async def check_child_status(
        self,
        child_agent_id: str,
    ) -> dict[str, Any]:
        """Check a child agent's current status."""
        tool_call_id = f"check_child_status-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._check_child_status_impl,
            "check_child_status",
            tool_call_id,
            child_agent_id,
        )

    async def _check_child_status_impl(
        self,
        child_agent_id: str,
    ) -> dict[str, Any]:
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can check "
                "child status."
            )
        child_agent_id = str(child_agent_id).strip()
        if not child_agent_id:
            return _error("child_agent_id is required")

        # Check active agents first, then completed pool
        desc = self._manager.get_descriptor(child_agent_id)
        if desc is None:
            desc = self._manager.get_completed_descriptor(child_agent_id)
        if desc is None:
            return _error(
                f"Agent {child_agent_id[:12]} not found "
                f"(not active and not in completed pool)."
            )

        status = {
            "agent_id": desc.agent_id,
            "state": desc.state.value,
            "role": desc.role.value,
            "model": desc.model,
            "parent_id": desc.parent_id,
            "children_count": len(desc.children),
            "children_ids": [c[:12] for c in desc.children],
            "depth": desc.depth,
            "created_at": (
                desc.created_at.isoformat()
                if desc.created_at else None
            ),
            "completed_at": (
                desc.completed_at.isoformat()
                if desc.completed_at else None
            ),
            "error": desc.error,
            "prompt_preview": strip_injected_prompt_prefix(desc.prompt)[:200] if desc.prompt else "",
        }

        active_states = {
            AgentState.RUNNING, AgentState.STARTING, AgentState.PENDING,
            AgentState.WAITING_FOR_PARENT, AgentState.WAITING_FOR_CHILD,
            AgentState.WAITING_FOR_EXPERT,
        }
        guidance = ""
        if desc.state in active_states:
            guidance = (
                "\n\n⚠️ This child is still working. Do NOT give up on it, "
                "do NOT start doing its work yourself, and do NOT kill it "
                "unless the user explicitly asks. Return to wait_for_message() "
                "to collect its result."
            )

        return _text(json.dumps(status, indent=2, default=str) + guidance)

    # ── send_child_prompt ──────────────────────────────────────

    async def send_child_prompt(
        self,
        child_agent_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Send a prompt/instruction to a child agent."""
        return await self._with_timeout(
            self._send_child_prompt_impl(child_agent_id, prompt),
            "send_child_prompt",
        )

    async def _send_child_prompt_impl(
        self,
        child_agent_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        if not self._is_master():
            return _error(
                "Only the orchestrator (master) agent can send "
                "prompts to children."
            )
        child_agent_id = str(child_agent_id).strip()
        prompt = str(prompt).strip()

        if not child_agent_id or not prompt:
            return _error(
                "Both child_agent_id and prompt are required"
            )

        # Verify the child exists
        desc = self._manager.get_descriptor(child_agent_id)
        if desc is None:
            return _error(
                f"Agent {child_agent_id[:12]} not found or not active."
            )

        msg = RoutedMessage(
            message_type=MessageType.USER_PROMPT,
            source_agent_id=self._agent_id,
            target_agent_id=child_agent_id,
            payload=prompt,
        )

        try:
            await self._router.send(msg)
            return _text(
                f"Prompt delivered to child "
                f"{child_agent_id[:12]}."
            )
        except MessageRoutingError as e:
            return _error(f"Failed to deliver prompt: {e}")

    # ── ask_user ────────────────────────────────────────────────

    async def ask_user(
        self,
        question: str,
        options: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Ask the user a structured question with clickable options."""
        tool_call_id = f"ask_user-{uuid.uuid4().hex}"
        return await self._with_timeout(
            self._ask_user_impl,
            "ask_user",
            tool_call_id,
            question, options or [],
        )

    async def _ask_user_impl(
        self,
        question: str,
        options: list[dict[str, str]],
    ) -> dict[str, Any]:
        self._meaningful_tool_calls += 1
        question = str(question).strip()
        if not question:
            return _error("Question cannot be empty")

        if not self._user_question_callback:
            return _error(
                "No user question callback configured. "
                "Cannot ask the user interactively."
            )

        try:
            result = await self._user_question_callback(
                self._agent_id, question, options,
            )
            return _text(f"User responded: {result}")
        except asyncio.TimeoutError:
            return _error("User did not respond within the timeout.")
        except Exception as exc:
            return _error(f"Failed to get user input: {exc}")

    # ── run_bash: permission-checked bash execution ──

    async def run_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        tool_call_id: str | None = None, # Added tool_call_id parameter
    ) -> dict[str, Any]:
        """Execute a bash command with permission checking.

        Evaluates the command against blacklist/whitelist policies.
        Dangerous commands require explicit user approval via the
        permission callback before execution.
        """
        if tool_call_id is None:
            logger.warning(
                "run_bash called without tool_call_id. "
                "This indicates a direct call, not through AgentSession. "
                "Interrupts will not be possible."
            )
            # Generate a dummy ID for internal use if not provided,
            # though it won't enable external interruption.
            tool_call_id = f"dummy-bash-{time.time()}"

        return await self._with_timeout(
            self._run_bash_impl,
            "run_bash",
            tool_call_id,
            command, timeout, cwd, tool_call_id, # Pass arguments here
        )

    async def _run_bash_impl(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        tool_call_id: str | None = None, # Added tool_call_id parameter
    ) -> dict[str, Any]:
        command = str(command or "").strip()
        if not command:
            return _error("Command cannot be empty")

        work_dir = str(cwd or self._cwd)
        scratch_dir = self._scratch_dir_for_work_dir(work_dir)
        forbidden_targets = self._find_forbidden_redirect_targets(
            command,
            work_dir,
            scratch_dir,
        )
        if forbidden_targets:
            display_targets = ", ".join(str(p) for p in forbidden_targets[:3])
            if len(forbidden_targets) > 3:
                display_targets += ", ..."
            return _error(
                "Blocked bash command: redirecting output to workspace scratch "
                "artifacts is not allowed.\n"
                f"Targets: {display_targets}\n"
                "Use streamed stdout/stderr directly. If a temporary file is "
                "required, write it under $PRSM_SCRATCH_DIR."
            )

        # Evaluate against blacklist/whitelist policies
        needs_prompt = self._evaluate_bash_permission(
            command,
            work_dir=work_dir,
        )

        if needs_prompt:
            if not self._permission_callback:
                return _error(
                    "Command requires permission but no permission "
                    "callback is configured. Command blocked: "
                    + command[:200]
                )

            try:
                result = await self._permission_callback(
                    self._agent_id, "Bash", command[:500],
                )
                logger.info(
                    "run_bash permission result agent=%s cmd=%s result=%s",
                    self._agent_id[:8],
                    command[:80],
                    result,
                )
                if result == "allow_always":
                    self._always_allowed_commands.add(command)
                elif result not in ("allow", "allow_always"):
                    return _error(
                        "User denied the bash command. "
                        "Do NOT retry the same command."
                    )
            except Exception as exc:
                logger.warning(
                    "run_bash permission callback error: %s, blocking",
                    exc,
                )
                return _error(f"Permission check failed: {exc}")

        # Execute the command
        exec_timeout = timeout or 120

        if not self._event_callback:
            return _error(
                "Cannot stream bash output: event callback not configured."
            )

        proc: asyncio.subprocess.Process | None = None
        stdout_task: asyncio.Task[list[str]] | None = None
        stderr_task: asyncio.Task[list[str]] | None = None
        try:
            # Create process
            shell_executable = shutil.which("bash") or shutil.which("zsh") or None
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env={**os.environ, "PRSM_SCRATCH_DIR": str(scratch_dir)},
                start_new_session=True,
                executable=shell_executable,
            )
            # Register process
            if tool_call_id:
                self._active_bash_processes[tool_call_id] = proc
                _GLOBAL_ACTIVE_BASH_PROCESSES[tool_call_id] = (self._agent_id, proc)
                logger.info(
                    "Registered bash subprocess agent=%s tool_call_id=%s pid=%s timeout=%ss cwd=%s global_size=%s command=%s",
                    self._agent_id[:8],
                    tool_call_id[:12],
                    proc.pid,
                    exec_timeout,
                    work_dir,
                    len(_GLOBAL_ACTIVE_BASH_PROCESSES),
                    (command[:180] + "...") if len(command) > 180 else command,
                )

            async def _emit_bash_delta(
                delta: str,
                stream_name: str,
            ) -> None:
                await fire_event(
                    self._event_callback,
                    {
                        "event": "tool_call_delta",
                        "agent_id": self._agent_id,
                        "tool_name": "run_bash",
                        "tool_id": tool_call_id,
                        "delta": delta,
                        "stream": stream_name,
                    },
                )

            async def _read_stream_and_fire_events(
                stream: asyncio.StreamReader, prefix: str
            ) -> list[str]:
                output_chunks: list[str] = []
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    decoded_chunk = chunk.decode(errors="ignore")
                    if not decoded_chunk:
                        continue
                    output_chunks.append(decoded_chunk)
                    await _emit_bash_delta(
                        decoded_chunk,
                        "stderr" if prefix == "ERR: " else "stdout",
                    )
                return output_chunks

            await _emit_bash_delta(
                f"RUN: {command}\nCWD: {work_dir}\nSCRATCH: {scratch_dir}\n",
                "stdout",
            )

            # Read stdout and stderr concurrently, firing events
            stdout_task = asyncio.create_task(
                _read_stream_and_fire_events(proc.stdout, "OUT: ")
            )
            stderr_task = asyncio.create_task(
                _read_stream_and_fire_events(proc.stderr, "ERR: ")
            )

            # Wait for process completion with timeout even if no output streams.
            await asyncio.wait_for(proc.wait(), timeout=exec_timeout)
            stdout_lines, stderr_lines = await asyncio.gather(
                stdout_task, stderr_task
            )
            exit_code = proc.returncode

            display_command = command if len(command) <= 400 else command[:397] + "..."
            display_cwd = work_dir if len(str(work_dir)) <= 200 else str(work_dir)[:197] + "..."

            stdout_text = "".join(stdout_lines)
            stderr_text = "".join(stderr_lines)

            def _trim_output(text: str, limit: int = 12000) -> str:
                if len(text) <= limit:
                    return text
                omitted = len(text) - limit
                return f"{text[:limit]}\n... [truncated {omitted} chars]"

            if exit_code != 0:
                if (
                    tool_call_id
                    and tool_call_id in self._cancelled_bash_tool_calls
                ):
                    return _error("Bash command was cancelled.")
                full_error_output = (stderr_text or stdout_text).strip()
                if full_error_output:
                    output_detail = "Output streamed previously."
                else:
                    output_detail = f"Command failed with exit code {exit_code}."
                return _error(
                    f"Command failed (exit code {exit_code}).\n"
                    f"Command: {display_command}\n"
                    f"CWD: {display_cwd}\n"
                    f"{output_detail}"
                )
            else:
                if stdout_text.strip():
                    output_detail = "Output streamed previously."
                elif stderr_text.strip():
                    output_detail = "Output streamed previously."
                else:
                    output_detail = (
                        "No stdout/stderr captured. "
                        "The command may have redirected output."
                    )
                return _text(
                    f"Command completed successfully (exit code {exit_code}).\n"
                    f"Command: {display_command}\n"
                    f"CWD: {display_cwd}\n"
                    f"{output_detail}"
                )

        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                logger.warning(
                    "Bash command was cancelled. Terminating process %s",
                    proc.pid,
                )
                if not self._signal_process_group(proc, signal.SIGINT):
                    proc.kill()
                await proc.wait()
            return _error(f"Bash command was cancelled.")
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                if not self._signal_process_group(proc, signal.SIGTERM):
                    proc.kill()
                await proc.wait()
            for task in (stdout_task, stderr_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (stdout_task, stderr_task) if t is not None),
                return_exceptions=True,
            )
            return _error(
                f"Bash command timed out after {exec_timeout}s and was terminated."
            )
        except FileNotFoundError:
            return _error(f"Command not found: '{command.split(' ')[0]}'")
        except Exception as e:
            return _error(f"Failed to execute bash command: {e}")
        finally:
            if tool_call_id and tool_call_id in self._active_bash_processes:
                del self._active_bash_processes[tool_call_id]
                global_entry = _GLOBAL_ACTIVE_BASH_PROCESSES.get(tool_call_id)
                if global_entry and global_entry[1] is proc:
                    _GLOBAL_ACTIVE_BASH_PROCESSES.pop(tool_call_id, None)
                logger.info(
                    "Unregistered bash subprocess agent=%s tool_call_id=%s global_size=%s",
                    self._agent_id[:8],
                    tool_call_id[:12],
                    len(_GLOBAL_ACTIVE_BASH_PROCESSES),
                )
            if tool_call_id:
                self._cancelled_bash_tool_calls.discard(tool_call_id)


    def kill_bash_subprocess(self, tool_call_id: str) -> bool:
        """Kills a bash subprocess associated with a tool_call_id."""
        logger.info(
            "kill_bash_subprocess requested agent=%s tool_call_id=%s active_tool_ids=%s",
            self._agent_id[:8],
            tool_call_id[:12],
            [k[:12] for k in self._active_bash_processes.keys()],
        )
        proc = self._active_bash_processes.pop(tool_call_id, None)
        resolved_tool_call_id = tool_call_id
        if proc is None:
            global_entry = _GLOBAL_ACTIVE_BASH_PROCESSES.get(tool_call_id)
            if global_entry and global_entry[0] == self._agent_id:
                proc = global_entry[1]
                logger.info(
                    "Resolved bash subprocess via global registry agent=%s tool_call_id=%s pid=%s",
                    self._agent_id[:8],
                    tool_call_id[:12],
                    proc.pid,
                )

        if proc is None and self._active_bash_processes:
            # Fallback path: if upstream tool_call_id does not match internal
            # registration, terminate the most recently started bash process.
            resolved_tool_call_id = next(reversed(self._active_bash_processes))
            proc = self._active_bash_processes.pop(resolved_tool_call_id, None)
            logger.warning(
                "No exact tool_call_id match for bash cancel; falling back "
                "to latest active process agent=%s requested=%s resolved=%s",
                self._agent_id[:8],
                tool_call_id[:8],
                resolved_tool_call_id[:8],
            )
        if proc is None:
            # Cross-instance fallback: latest active process for this same agent.
            for candidate_tool_id, (candidate_agent_id, candidate_proc) in reversed(
                _GLOBAL_ACTIVE_BASH_PROCESSES.items()
            ):
                if candidate_agent_id != self._agent_id:
                    continue
                if candidate_proc.returncode is not None:
                    continue
                resolved_tool_call_id = candidate_tool_id
                proc = candidate_proc
                logger.warning(
                    "No exact local tool_call_id match; falling back to latest global process "
                    "agent=%s requested=%s resolved=%s",
                    self._agent_id[:8],
                    tool_call_id[:8],
                    resolved_tool_call_id[:8],
                )
                break
        if proc:
            _GLOBAL_ACTIVE_BASH_PROCESSES.pop(resolved_tool_call_id, None)
            self._cancelled_bash_tool_calls.add(resolved_tool_call_id)
            logger.warning(
                "Killing bash subprocess for agent=%s tool_call_id=%s pid=%s",
                self._agent_id[:8],
                resolved_tool_call_id[:8],
                proc.pid,
            )
            # Send SIGINT to emulate Ctrl+C in terminal sessions.
            interrupted = self._signal_process_group(proc, signal.SIGINT)
            logger.info(
                "Bash subprocess signal result agent=%s tool_call_id=%s pid=%s signal=%s interrupted=%s",
                self._agent_id[:8],
                resolved_tool_call_id[:12],
                proc.pid,
                "SIGINT",
                interrupted,
            )
            if not interrupted:
                try:
                    proc.terminate()
                    logger.info(
                        "Sent terminate() to bash subprocess agent=%s tool_call_id=%s pid=%s",
                        self._agent_id[:8],
                        resolved_tool_call_id[:12],
                        proc.pid,
                    )
                except ProcessLookupError:
                    pass
            # Ensure command actually stops: escalate if it ignores SIGINT/SIGTERM.
            try:
                asyncio.create_task(
                    self._ensure_process_group_stopped(
                        proc,
                        agent_id=self._agent_id,
                        tool_call_id=resolved_tool_call_id,
                    )
                )
            except Exception:
                logger.debug(
                    "Failed to schedule bash kill escalation watchdog agent=%s tool_call_id=%s",
                    self._agent_id[:8],
                    resolved_tool_call_id[:12],
                    exc_info=True,
                )

            # If it doesn't exit within a short time, kill it forcefully
            # This part will be handled by the AgentSession, which will await
            # the _with_timeout coroutine's asyncio.CancelledError
            
            return True
        logger.info(
            "No active bash subprocess found to kill agent=%s requested_tool_call_id=%s",
            self._agent_id[:8],
            tool_call_id[:12],
        )
        return False

    def get_latest_active_bash_tool_call_id(self) -> str | None:
        """Return the most recently started active bash tool_call_id, if any."""
        if not self._active_bash_processes:
            for tool_call_id, (agent_id, proc) in reversed(
                _GLOBAL_ACTIVE_BASH_PROCESSES.items()
            ):
                if agent_id == self._agent_id and proc.returncode is None:
                    return tool_call_id
            return None
        return next(reversed(self._active_bash_processes))

    def _scratch_dir_for_work_dir(self, work_dir: str, *, create: bool = True) -> Path:
        """Return a per-agent scratch directory under /tmp or the tmp worktree."""
        work_path = Path(work_dir).resolve()
        scratch_root = Path("/tmp") / "prsm-scratch" / self._agent_id
        # If running in a managed /tmp worktree, keep scratch local to it.
        if "/tmp/" in str(work_path):
            scratch_root = work_path / ".prsm-scratch"
        if create:
            scratch_root.mkdir(parents=True, exist_ok=True)
        return scratch_root.resolve()

    def _extract_bash_output_targets(self, command: str) -> list[str]:
        """Extract simple output redirection and tee targets."""
        try:
            tokens = shlex.split(command, posix=True)
        except Exception:
            return []

        targets: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _REDIRECT_OPERATORS:
                if i + 1 < len(tokens):
                    targets.append(tokens[i + 1])
                    i += 2
                    continue
            else:
                m = re.match(r"^(?:\d?>>?|&>>?)(.+)$", tok)
                if m and m.group(1):
                    targets.append(m.group(1))
                if tok == "tee":
                    j = i + 1
                    while j < len(tokens):
                        nxt = tokens[j]
                        if nxt in _SHELL_CONTROL_TOKENS:
                            break
                        if nxt.startswith("-"):
                            j += 1
                            continue
                        if nxt != "-":
                            targets.append(nxt)
                        j += 1
                    i = j
                    continue
            i += 1
        return targets

    @staticmethod
    def _is_suspicious_workspace_artifact_target(path: Path) -> bool:
        name = path.name.lower()
        if name.startswith(".tmp"):
            return True
        if name in _BLOCKED_SCRATCH_BASENAMES:
            return True
        return bool(_SUSPICIOUS_SCRATCH_BASENAME_RE.match(name))

    def _find_forbidden_redirect_targets(
        self,
        command: str,
        work_dir: str,
        scratch_dir: Path,
    ) -> list[Path]:
        """Return suspicious redirect targets that are outside scratch."""
        work_path = Path(work_dir).resolve()
        forbidden: list[Path] = []
        for raw_target in self._extract_bash_output_targets(command):
            if raw_target in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
                continue
            target = Path(raw_target)
            resolved = target.resolve() if target.is_absolute() else (work_path / target).resolve()
            try:
                resolved.relative_to(scratch_dir)
                continue
            except ValueError:
                pass
            if self._is_suspicious_workspace_artifact_target(resolved):
                forbidden.append(resolved)
        return forbidden

    def _evaluate_bash_permission(
        self,
        command: str,
        *,
        work_dir: str | None = None,
    ) -> bool:
        """Return True if the command requires user permission.

        Checks blacklist, whitelist, and heuristic danger patterns.
        """
        command_lower = command.lower()

        # Check if already approved via "allow always"
        if command in self._always_allowed_commands:
            return False

        # Blacklist always requires confirmation
        if any(p.search(command_lower) for p in self._blacklist_patterns):
            return True

        # Whitelist bypasses prompts
        if any(p.search(command_lower) for p in self._whitelist_patterns):
            return False

        # Heuristic danger detection
        danger_patterns = [
            r"\bsudo\b",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bchgrp\b",
            r"\bapt(-get)?\s+install\b",
            r"\bapt(-get)?\s+remove\b",
            r"\bapt(-get)?\s+upgrade\b",
            r"\bsnap\s+install\b",
            r"\bsnap\s+remove\b",
            r"\byum\s+install\b",
            r"\bdnf\s+install\b",
            r"\bpacman\s+-S\b",
            r"\bzypper\s+install\b",
            r"\bbrew\s+install\b",
            r"\bpip(?:3)?\s+install\b",
            r"\bpython(?:\d+(?:\.\d+)?)?\s+-m\s+pip\s+install\b",
            r"\buv\s+pip\s+install\b",
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
        if any(re.search(p, command_lower) for p in danger_patterns):
            return True

        effective_work_dir = work_dir or self._cwd
        scratch_dir = self._scratch_dir_for_work_dir(
            effective_work_dir,
            create=False,
        )
        if self._find_forbidden_redirect_targets(command, effective_work_dir, scratch_dir):
            return True

        return False


def create_orchestration_tools(
    agent_id: str,
    manager: AgentManager,
    router: MessageRouter,
    expert_registry: ExpertRegistry,
    tool_call_timeout: float = 7200.0,
) -> OrchestrationTools:
    """Create orchestration tools bound to a specific agent."""
    return OrchestrationTools(
        agent_id=agent_id,
        manager=manager,
        router=router,
        expert_registry=expert_registry,
        tool_call_timeout=tool_call_timeout,
    )
