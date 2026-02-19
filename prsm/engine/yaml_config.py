"""YAML configuration loader.

Loads a single YAML file that replaces env vars + Python expert files.
Backward compatible: when no YAML is provided, env vars and
--experts-file work exactly as before.

Example YAML:
    engine:
      max_agent_depth: 5
      default_model: claude-opus-4-6

    providers:
      claude:
        type: claude
      codex:
        type: codex
        command: codex
      alibaba:
        type: alibaba
        command: codex
        api_key_env: DASHSCOPE_API_KEY

    models:
      opus:
        provider: claude
        model_id: claude-opus-4-6
      codex:
        provider: codex
        model_id: gpt-5.2-codex

    defaults:
      model: opus
      cwd: /path/to/project
      peer_model: codex           # single peer (backward compat)
      peer_models: [codex, gemini-3]  # multiple peers

    model_registry:
      claude-opus-4-6:
        tier: frontier
        provider: claude
        affinities:
          architecture: 0.95
          complex-reasoning: 0.95

    experts:
      rust-systems:
        name: Rust Systems Expert
        description: "..."
        system_prompt: |
          ...
        tools: [Read, Grep, Glob, Bash]
        model: opus
        mcp_servers:
          database:
            type: http
            url: https://db.example.com/mcp

    plugins:
      filesystem:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem"]
        tags: [filesystem, code]
      github:
        type: http
        url: https://mcp.github.com/v1
        headers:
          Authorization: "Bearer ${GITHUB_TOKEN}"
        tags: [github, vcs]
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .config import EngineConfig
from .models import ExpertProfile, PermissionMode

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    type: str  # "claude", "codex", "gemini", "minimax", or "alibaba"
    api_key_env: str | None = None
    command: str | None = None  # for codex/gemini: path to CLI binary
    profile: str | None = None  # for minimax: Codex CLI profile name


@dataclass
class ModelAlias:
    """Maps a short name to a provider + model_id pair."""
    provider: str
    model_id: str
    reasoning_effort: str | None = None


@dataclass
class DefaultsConfig:
    """Default settings from YAML."""
    model: str | None = None  # alias from models section
    cwd: str | None = None
    peer_model: str | None = None  # alias used by consult_peer (primary)
    peer_models: list[str] | None = None  # multiple peer model aliases


@dataclass
class ExpertConfig:
    """Raw expert config from YAML (before alias resolution)."""
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=lambda: [
        "Read", "Grep", "Glob", "Bash",
    ])
    model: str | None = None  # alias reference
    permission_mode: str = "default"
    max_concurrent_consultations: int = 3
    cwd: str | None = None  # relative to defaults.cwd


@dataclass
class OrchestrationConfig:
    """Complete parsed YAML configuration."""
    engine: EngineConfig
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelAlias]
    defaults: DefaultsConfig
    experts: list[ExpertProfile]
    plugin_configs: dict[str, dict] = field(default_factory=dict)
    model_registry_raw: dict[str, dict] = field(default_factory=dict)
    projects: dict[str, dict] = field(default_factory=dict)


def _parse_permission_mode(value: str) -> PermissionMode:
    """Parse a permission mode string to enum."""
    mapping = {
        "default": PermissionMode.DEFAULT,
        "acceptEdits": PermissionMode.ACCEPT_EDITS,
        "accept_edits": PermissionMode.ACCEPT_EDITS,
        "bypassPermissions": PermissionMode.BYPASS,
        "bypass": PermissionMode.BYPASS,
        "plan": PermissionMode.PLAN,
    }
    return mapping.get(value, PermissionMode.DEFAULT)


# Built-in Claude family aliases — lets users write "claude-sonnet"
# instead of the full versioned model ID.  These are also registered
# in ModelRegistry.CLAUDE_FAMILY_ALIASES; keep them in sync.
_BUILTIN_CLAUDE_ALIASES: dict[str, tuple[str, str]] = {
    "claude-opus": ("claude", "claude-opus-4-6"),
    "claude-sonnet": ("claude", "claude-sonnet-4-5-20250929"),
    "claude-haiku": ("claude", "claude-3-5-haiku-20241022"),
}


def resolve_model_alias(
    alias: str,
    models: dict[str, ModelAlias],
) -> tuple[str, str]:
    """Resolve a model alias to (provider_name, model_id).

    Resolution order:
    1. YAML ``models:`` section (explicit user config)
    2. Built-in Claude family aliases (``claude-sonnet``, etc.)
    3. Falls back to treating *alias* as a raw model_id with
       the "claude" provider.
    """
    if alias in models:
        m = models[alias]
        model_id = m.model_id
        effort = (m.reasoning_effort or "").strip().lower()
        if m.provider == "codex" and effort in {"low", "medium", "high"}:
            model_id = f"{model_id}::reasoning_effort={effort}"
        return (m.provider, model_id)
    # Check built-in Claude family aliases
    if alias in _BUILTIN_CLAUDE_ALIASES:
        return _BUILTIN_CLAUDE_ALIASES[alias]
    # Not an alias — treat as raw model_id
    return ("claude", alias)


def _global_models_yaml_path() -> Path:
    """Return global PRSM model settings path (~/.prsm/models.yaml)."""
    return Path.home() / ".prsm" / "models.yaml"


def _load_models_yaml(models_path: Path, label: str = "models.yaml") -> dict:
    """Load a models YAML file from the given path.

    Returns a raw dict with optional ``models`` and ``model_registry``
    sections. If the file does not exist or cannot be parsed, returns
    an empty dict so callers can merge unconditionally.
    """
    logger.debug(
        "_load_models_yaml: checking %s (%s, exists=%s)",
        label, models_path, models_path.is_file()
    )
    if not models_path.is_file():
        logger.debug(
            "_load_models_yaml: %s not found at %s",
            label, models_path,
        )
        return {}
    try:
        with open(models_path) as f:
            data = yaml.safe_load(f) or {}
        sections = [k for k in data if k in ("models", "model_registry")]
        logger.info(
            "_load_models_yaml: loaded %s from %s (sections: %s)",
            label, models_path,
            ", ".join(sections) if sections else "empty",
        )
        return data
    except yaml.YAMLError as exc:
        logger.warning(
            "_load_models_yaml: YAML parse error in %s (%s): %s",
            label, models_path, exc
        )
        return {}
    except Exception as exc:
        logger.warning(
            "_load_models_yaml: unexpected error reading %s (%s): %s",
            label, models_path, exc, exc_info=True,
        )
        return {}


def _load_global_models_yaml() -> dict:
    """Load global models config from ``~/.prsm/models.yaml``."""
    return _load_models_yaml(
        _global_models_yaml_path(), "global ~/.prsm/models.yaml"
    )


def load_yaml_config(path: str | Path) -> OrchestrationConfig:
    """Load and parse a YAML config file.

    Also loads a sibling ``models.yaml`` from the same directory as
    *path* (typically ``.prism/models.yaml``) and the global
    ``~/.prsm/models.yaml``, merging their ``models`` and
    ``model_registry`` sections.  Precedence (highest wins):

    1. Inline sections in *path* (``prsm.yaml``)
    2. Sibling ``models.yaml`` (same directory as *path*)
    3. Global ``~/.prsm/models.yaml``

    Resolves model aliases, relative expert cwds, and constructs
    EngineConfig + ExpertProfile instances.
    """
    path = Path(path)
    logger.info(
        "load_yaml_config: attempting to load config from %s (exists=%s)",
        path, path.exists()
    )
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        logger.info("load_yaml_config: successfully read and parsed %s", path)
    except FileNotFoundError:
        logger.error(
            "load_yaml_config: config file not found at %s (absolute path: %s)",
            path, path.absolute()
        )
        raise
    except yaml.YAMLError as exc:
        logger.error(
            "load_yaml_config: YAML parse error in %s: %s",
            path, exc
        )
        raise
    except Exception as exc:
        logger.error(
            "load_yaml_config: unexpected error reading %s: %s",
            path, exc, exc_info=True
        )
        raise

    top_sections = sorted(raw.keys())
    logger.info(
        "Parsed YAML config %s — sections: %s",
        path.name, ", ".join(top_sections) if top_sections else "(empty)",
    )

    # ── Merge models from external YAML files ────────────────
    # Precedence (lowest → highest):
    #   1. Global ~/.prsm/models.yaml
    #   2. Sibling models.yaml (same dir as prsm.yaml)
    #   3. Inline sections in prsm.yaml itself
    global_models = _load_global_models_yaml()
    sibling_models_path = path.parent / "models.yaml"
    sibling_models = (
        _load_models_yaml(sibling_models_path, f"sibling {sibling_models_path}")
        if sibling_models_path.resolve() != path.resolve()
        else {}
    )

    # Merge models: global (lowest) → sibling → inline (highest)
    merged_models = {
        **(global_models.get("models") or {}),
        **(sibling_models.get("models") or {}),
        **(raw.get("models") or {}),
    }
    if merged_models:
        raw["models"] = merged_models

    # Merge model_registry: same precedence
    merged_registry = {
        **(global_models.get("model_registry") or {}),
        **(sibling_models.get("model_registry") or {}),
        **(raw.get("model_registry") or {}),
    }
    if merged_registry:
        raw["model_registry"] = merged_registry

    # ── Engine config ──────────────────────────────────────────
    engine_raw = raw.get("engine", {})
    engine = EngineConfig(
        default_model=engine_raw.get(
            "default_model", EngineConfig.default_model
        ),
        default_cwd=engine_raw.get(
            "default_cwd", EngineConfig.default_cwd
        ),
        max_agent_depth=int(engine_raw.get(
            "max_agent_depth", EngineConfig.max_agent_depth
        )),
        max_concurrent_agents=int(engine_raw.get(
            "max_concurrent_agents", EngineConfig.max_concurrent_agents
        )),
        agent_timeout_seconds=float(engine_raw.get(
            "agent_timeout_seconds", EngineConfig.agent_timeout_seconds
        )),
        tool_call_timeout_seconds=float(engine_raw.get(
            "tool_call_timeout_seconds",
            EngineConfig.tool_call_timeout_seconds,
        )),
        user_question_timeout_seconds=float(engine_raw.get(
            "user_question_timeout_seconds",
            EngineConfig.user_question_timeout_seconds,
        )),
        command_whitelist=list(engine_raw.get("command_whitelist", []) or []),
        command_blacklist=list(engine_raw.get("command_blacklist", []) or []),
        command_safety_model_enabled=bool(
            engine_raw.get(
                "command_safety_model_enabled",
                EngineConfig.command_safety_model_enabled,
            )
        ),
        command_safety_model=engine_raw.get(
            "command_safety_model", EngineConfig.command_safety_model,
        ),
        deadlock_check_interval_seconds=float(engine_raw.get(
            "deadlock_check_interval_seconds",
            EngineConfig.deadlock_check_interval_seconds,
        )),
        deadlock_max_wait_seconds=float(engine_raw.get(
            "deadlock_max_wait_seconds",
            EngineConfig.deadlock_max_wait_seconds,
        )),
        message_queue_size=int(engine_raw.get(
            "message_queue_size", EngineConfig.message_queue_size
        )),
        log_level=engine_raw.get("log_level", EngineConfig.log_level),
        triage_model_shadow_enabled=bool(
            engine_raw.get(
                "triage_model_shadow_enabled",
                EngineConfig.triage_model_shadow_enabled,
            )
        ),
        triage_shadow_model=str(
            engine_raw.get(
                "triage_shadow_model",
                EngineConfig.triage_shadow_model,
            )
        ),
        telemetry_db_path=engine_raw.get(
            "telemetry_db_path", EngineConfig.telemetry_db_path
        ),
    )
    resources_raw = raw.get("resources", {}) or {}
    if isinstance(resources_raw, dict):
        budgets = resources_raw.get("budgets", resources_raw)
        if isinstance(budgets, dict):
            engine.resource_budgets = {
                str(project_id): dict(cfg or {})
                for project_id, cfg in budgets.items()
                if isinstance(cfg, dict)
            }

    # ── Providers ──────────────────────────────────────────────
    providers_raw = raw.get("providers", {})
    providers: dict[str, ProviderConfig] = {}
    for name, cfg in providers_raw.items():
        providers[name] = ProviderConfig(
            type=cfg.get("type", name),
            api_key_env=cfg.get("api_key_env"),
            command=cfg.get("command"),
            profile=cfg.get("profile"),
        )

    # ── Models ─────────────────────────────────────────────────
    models_raw = raw.get("models", {})
    models: dict[str, ModelAlias] = {}
    for alias, cfg in models_raw.items():
        models[alias] = ModelAlias(
            provider=cfg.get("provider", "claude"),
            model_id=cfg.get("model_id", alias),
            reasoning_effort=cfg.get("reasoning_effort"),
        )

    # ── Defaults ───────────────────────────────────────────────
    defaults_raw = raw.get("defaults", {})

    # Parse peer_models — supports both single and list forms:
    #   peer_model: codex           # single (backward compat)
    #   peer_models: [codex, gemini-3]  # multiple
    peer_models_raw = defaults_raw.get("peer_models")
    peer_models: list[str] | None = None
    if isinstance(peer_models_raw, list):
        peer_models = [str(m) for m in peer_models_raw]
    elif isinstance(peer_models_raw, str):
        peer_models = [peer_models_raw]

    # If peer_models not set, fall back to single peer_model
    peer_model = defaults_raw.get("peer_model")
    if peer_models is None and peer_model:
        peer_models = [peer_model]

    defaults = DefaultsConfig(
        model=defaults_raw.get("model"),
        cwd=defaults_raw.get("cwd"),
        peer_model=peer_model,
        peer_models=peer_models,
    )

    # Apply defaults.cwd to engine if set
    if defaults.cwd:
        engine.default_cwd = defaults.cwd

    # Resolve default model alias to engine
    if defaults.model:
        provider_name, model_id = resolve_model_alias(defaults.model, models)
        engine.default_provider = provider_name
        engine.default_model = model_id

    # Resolve master/orchestrator model.
    # Any provider that supports MCP can serve as master (Claude uses
    # in-process MCP; Codex/Gemini/MiniMax use TCP bridge + proxy).
    # If no master_model is set, the default model is used if it
    # supports master mode, otherwise we fall back to Claude.
    master_alias = defaults_raw.get("master_model")
    if master_alias:
        # User explicitly configured a master model
        m_provider, m_model_id = resolve_model_alias(master_alias, models)
        engine.master_model = m_model_id
        engine.master_provider = m_provider
        logger.info(
            "Master model configured: %s (provider=%s)",
            m_model_id, m_provider,
        )
    elif defaults.model and provider_name:
        # Use the default model as master
        engine.master_model = model_id
        engine.master_provider = provider_name
    # else: no explicit master or default — keep config defaults
    # (claude-sonnet-4-5-20250929 / claude)

    # ── Experts ────────────────────────────────────────────────
    experts_raw = raw.get("experts", {})
    experts: list[ExpertProfile] = []
    base_cwd = defaults.cwd or "."

    for expert_id, cfg in experts_raw.items():
        # Resolve model alias
        model_alias = cfg.get("model", defaults.model or "claude-opus-4-6")
        _, model_id = resolve_model_alias(model_alias, models)

        # Resolve relative cwd against defaults.cwd
        expert_cwd = cfg.get("cwd")
        if expert_cwd and not os.path.isabs(expert_cwd):
            expert_cwd = os.path.join(base_cwd, expert_cwd)
        elif not expert_cwd:
            expert_cwd = None

        # Parse expert-specific MCP servers
        expert_mcp_raw = cfg.get("mcp_servers")
        expert_mcp: dict | None = None
        if expert_mcp_raw and isinstance(expert_mcp_raw, dict):
            expert_mcp = {}
            for srv_name, srv_cfg in expert_mcp_raw.items():
                srv_type = srv_cfg.get("type", "stdio")
                entry: dict = {"type": srv_type}
                if srv_type == "stdio":
                    entry["command"] = srv_cfg.get("command")
                    entry["args"] = srv_cfg.get("args", [])
                    if "env" in srv_cfg:
                        entry["env"] = srv_cfg["env"]
                elif srv_type in ("http", "sse"):
                    entry["url"] = srv_cfg.get("url")
                    if "headers" in srv_cfg:
                        entry["headers"] = srv_cfg["headers"]
                expert_mcp[srv_name] = entry

        experts.append(ExpertProfile(
            expert_id=expert_id,
            name=cfg.get("name", expert_id),
            description=cfg.get("description", ""),
            system_prompt=cfg.get("system_prompt", ""),
            tools=cfg.get("tools", ["Read", "Grep", "Glob", "Bash"]),
            model=model_id,
            permission_mode=_parse_permission_mode(
                cfg.get("permission_mode", "default")
            ),
            max_concurrent_consultations=int(
                cfg.get("max_concurrent_consultations", 3)
            ),
            cwd=expert_cwd,
            provider=resolve_model_alias(model_alias, models)[0],
            mcp_servers=expert_mcp,
            lifecycle_state=cfg.get("lifecycle_state", "active"),
            created_at=(
                datetime.fromisoformat(cfg["created_at"])
                if cfg.get("created_at")
                else None
            ),
            deprecated_at=(
                datetime.fromisoformat(cfg["deprecated_at"])
                if cfg.get("deprecated_at")
                else None
            ),
            deprecation_reason=cfg.get("deprecation_reason", ""),
            evaluation_criteria=list(cfg.get("evaluation_criteria", []) or []),
            deprecation_policy=cfg.get("deprecation_policy", ""),
            consultation_count=int(cfg.get("consultation_count", 0)),
            success_count=int(cfg.get("success_count", 0)),
            failure_count=int(cfg.get("failure_count", 0)),
            avg_duration_seconds=float(cfg.get("avg_duration_seconds", 0.0)),
            avg_confidence=float(cfg.get("avg_confidence", 0.0)),
            utility_score=float(cfg.get("utility_score", 0.0)),
        ))

    # ── Plugins ────────────────────────────────────────────────
    plugins_raw = raw.get("plugins", {})
    plugin_configs: dict[str, dict] = {}
    for name, cfg in plugins_raw.items():
        plugin_type = cfg.get("type", "stdio")
        entry = {"type": plugin_type}
        if plugin_type == "stdio":
            entry["command"] = cfg.get("command")
            entry["args"] = cfg.get("args", [])
            if "env" in cfg:
                entry["env"] = cfg["env"]
        elif plugin_type in ("http", "sse"):
            entry["url"] = cfg.get("url")
            if "headers" in cfg:
                entry["headers"] = cfg["headers"]
        if "tags" in cfg:
            entry["tags"] = cfg.get("tags", [])
        plugin_configs[name] = entry

    # ── Model Registry ────────────────────────────────────────
    model_registry_raw = raw.get("model_registry", {})
    projects_raw = raw.get("projects", {}) or {}

    logger.info(
        "Config loaded from %s: providers=[%s] models=[%s] experts=%d "
        "plugins=%d model_registry_entries=%d projects=%d default_model=%s master=%s/%s",
        path.name,
        ", ".join(sorted(providers.keys())) if providers else "none",
        ", ".join(sorted(models.keys())) if models else "none",
        len(experts),
        len(plugin_configs),
        len(model_registry_raw),
        len(projects_raw),
        engine.default_model,
        engine.master_provider,
        engine.master_model,
    )

    return OrchestrationConfig(
        engine=engine,
        providers=providers,
        models=models,
        defaults=defaults,
        experts=experts,
        plugin_configs=plugin_configs,
        model_registry_raw=model_registry_raw,
        projects=projects_raw,
    )
