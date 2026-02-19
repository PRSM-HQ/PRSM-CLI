"""Stdio MCP server for Claude Code integration.

Exposes orchestration tools to Claude Code CLI. The user's Claude
Code session IS the master agent — it gets the full CLI UI while
orchestration tools are available as MCP tools. Children run as
headless SDK query() sessions in the background.

Usage:
    # Via .mcp.json (recommended — Claude Code auto-discovers)
    # Or manually:
    python -m prsm.orchestrator.mcp_server.stdio_server
    python -m prsm.orchestrator.mcp_server.stdio_server \
        --config .prism/prsm.yaml
    python -m prsm.orchestrator.mcp_server.stdio_server \
        --experts-file examples/your_config.py \
        --cwd /path/to/project
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from ..config import EngineConfig
from ..models import AgentDescriptor, AgentRole, AgentState
from ..agent_manager import AgentManager
from ..message_router import MessageRouter
from ..expert_registry import ExpertRegistry
from ..conversation_store import ConversationStore
from ..deadlock import run_deadlock_detector
from ..model_intelligence import (
    ModelIntelligence,
    run_research_loop,
)
from .tools import OrchestrationTools

logger = logging.getLogger(__name__)

# Parsed CLI args — set in main() before server starts
_parsed_args: argparse.Namespace | None = None


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for the MCP server process."""
    parser = argparse.ArgumentParser(
        prog="prsm-orchestrator-mcp",
        description="Orchestration MCP server for Claude Code",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "YAML config file (replaces --experts-file + env vars). "
            "Also reads ORCH_CONFIG_FILE env var."
        ),
    )
    parser.add_argument(
        "--experts-file",
        default=None,
        help=(
            "Python file exporting register_experts(engine). "
            "Also reads ORCH_EXPERTS_FILE env var. "
            "Ignored when --config is provided."
        ),
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Default working directory for child agents",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


class _RegistryShim:
    """Mimics OrchestrationEngine interface for expert loading.

    Expert config files call engine.register_expert() and
    engine.set_master_prompt(). This shim forwards register_expert
    to the registry and silently ignores set_master_prompt (the
    master prompt is Claude Code's own context, not something we
    control).
    """

    def __init__(self, registry: ExpertRegistry) -> None:
        self._registry = registry

    def register_expert(self, profile) -> None:
        self._registry.register(profile)

    def set_master_prompt(self, **kwargs) -> None:
        pass  # Ignored — master prompt is Claude Code's context


def _load_experts(
    expert_registry: ExpertRegistry, path: str
) -> None:
    """Load expert profiles from a Python file.

    The file must export a register_experts(engine) function.
    Uses _RegistryShim for backward compatibility.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("experts", path)
    if spec is None or spec.loader is None:
        logger.error("Cannot load experts file: %s", path)
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    register_fn = getattr(module, "register_experts", None)
    if register_fn is None:
        logger.error(
            "%s must export a register_experts(engine) function",
            path,
        )
        return

    register_fn(_RegistryShim(expert_registry))


@asynccontextmanager
async def orchestration_lifespan(server: FastMCP):
    """Initialize all orchestration state for the server lifetime.

    Creates: MessageRouter, AgentManager, ExpertRegistry,
    OrchestrationTools (bound to virtual master), deadlock detector.
    Optionally: ProviderRegistry from YAML config.

    Yields context dict accessible via ctx.request_context.lifespan_context
    in tool handlers.
    """
    global _parsed_args
    if _parsed_args is None:
        _parsed_args = _parse_args()

    # Logging must go to stderr (stdout is the stdio transport)
    level = logging.DEBUG if _parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Determine config source: --config / ORCH_CONFIG_FILE or legacy
    config_file = (
        _parsed_args.config
        or os.getenv("ORCH_CONFIG_FILE")
    )
    if config_file:
        logger.info(
            "Config source: %s (from %s)",
            config_file,
            "--config" if _parsed_args.config else "ORCH_CONFIG_FILE env",
        )
    else:
        logger.info("No config file specified; using env vars / defaults")

    provider_registry = None
    default_model = "claude-opus-4-6"
    default_provider = "claude"
    peer_model_id = None
    peer_provider = None
    peer_models_dict: dict[str, tuple] = {}
    model_registry = None

    if config_file:
        # ── YAML config path ──────────────────────────────────
        from ..yaml_config import load_yaml_config, resolve_model_alias
        from ..providers.registry import build_provider_registry
        from ..model_registry import (
            build_default_registry,
            load_model_registry_from_yaml,
        )

        yaml_cfg = load_yaml_config(config_file)
        config = yaml_cfg.engine

        # Override cwd from CLI if provided
        if _parsed_args.cwd:
            config.default_cwd = _parsed_args.cwd

        default_model = config.default_model
        default_provider = config.default_provider

        # Build provider registry
        provider_registry = build_provider_registry(yaml_cfg.providers)

        # Resolve primary peer model (backward compat)
        if yaml_cfg.defaults.peer_model:
            peer_provider_name, peer_model_id = resolve_model_alias(
                yaml_cfg.defaults.peer_model, yaml_cfg.models
            )
            peer_provider = provider_registry.get(peer_provider_name)
            if peer_provider:
                logger.info(
                    "Primary peer provider configured: %s (model=%s)",
                    peer_provider_name,
                    peer_model_id,
                )

        # Resolve all peer models
        if yaml_cfg.defaults.peer_models:
            for peer_alias in yaml_cfg.defaults.peer_models:
                p_name, p_model_id = resolve_model_alias(
                    peer_alias, yaml_cfg.models
                )
                p_provider = provider_registry.get(p_name)
                if p_provider:
                    peer_models_dict[peer_alias] = (p_provider, p_model_id)
                    logger.info(
                        "Peer model registered: %s (%s, model=%s)",
                        peer_alias, p_name, p_model_id,
                    )
                    # Set the first peer as the default if no primary is set
                    if peer_provider is None:
                        peer_provider = p_provider
                        peer_model_id = p_model_id

        # Build model capability registry
        model_registry = build_default_registry()
        if yaml_cfg.model_registry_raw:
            yaml_models = yaml_cfg.models if hasattr(yaml_cfg, "models") else None
            model_registry = load_model_registry_from_yaml(
                yaml_cfg.model_registry_raw, model_registry,
                model_aliases=yaml_models,
            )

        # Sync model availability with actual provider availability
        provider_report = provider_registry.get_availability_report()
        changed = model_registry.sync_availability(provider_report)
        if changed:
            unavail = [m for m, ok in changed.items() if not ok]
            if unavail:
                logger.warning(
                    "Models marked unavailable (provider not installed): %s",
                    ", ".join(unavail),
                )

        # Optional startup probe (disabled by default). Running this from the
        # stdio MCP process can recurse via Claude CLI MCP startup and create
        # process storms.
        probe_on_startup = os.getenv("PRSM_PROBE_CLAUDE_MODELS_ON_STARTUP", "0").lower() in {
            "1", "true", "yes", "on",
        }
        if probe_on_startup:
            try:
                probe_changed = await model_registry.probe_claude_models()
                if probe_changed:
                    unavail_probed = [m for m, ok in probe_changed.items() if not ok]
                    if unavail_probed:
                        logger.warning(
                            "Claude models not accessible on this account: %s",
                            ", ".join(unavail_probed),
                        )
            except Exception as exc:
                logger.warning(
                    "Claude model probe failed (non-fatal): %s", exc,
                )

        available_models = model_registry.list_available()
        logger.info(
            "Model registry loaded: %d models (%d available)",
            model_registry.count,
            len(available_models),
        )

        # Load persistent model intelligence (learned rankings)
        model_intelligence = ModelIntelligence()
        model_intelligence.load()
        model_registry.set_intelligence(model_intelligence)

        # Set up expert registry from YAML
        expert_registry = ExpertRegistry()
        for profile in yaml_cfg.experts:
            expert_registry.register(profile)

        logger.info(
            "YAML config loaded: %d providers, %d models, %d experts, %d peers",
            provider_registry.count,
            len(yaml_cfg.models),
            expert_registry.count,
            len(peer_models_dict),
        )

    else:
        # ── Legacy path (env vars + --experts-file) ───────────
        config = EngineConfig.from_env()
        if _parsed_args.cwd:
            config.default_cwd = _parsed_args.cwd

        default_model = config.default_model
        default_provider = config.default_provider
        expert_registry = ExpertRegistry()
        model_intelligence = None  # No model registry in legacy path

        # Load experts from CLI arg or env var
        experts_file = (
            _parsed_args.experts_file
            or os.getenv("ORCH_EXPERTS_FILE")
        )
        if experts_file:
            _load_experts(expert_registry, experts_file)
            logger.info(
                "Loaded %d experts from %s",
                expert_registry.count,
                experts_file,
            )

    conversation_store = ConversationStore()
    router = MessageRouter(queue_maxsize=config.message_queue_size)
    manager = AgentManager(
        router=router,
        expert_registry=expert_registry,
        config=config,
        provider_registry=provider_registry,
        conversation_store=conversation_store,
    )

    # Register the virtual master (Claude Code session).
    # Fixed ID — there is exactly one master per server process.
    master_id = "master"
    master_descriptor = AgentDescriptor(
        agent_id=master_id,
        parent_id=None,
        role=AgentRole.MASTER,
        state=AgentState.RUNNING,
        prompt="[Claude Code session — external master]",
        depth=0,
        max_depth=config.max_agent_depth,
        cwd=config.default_cwd,
    )
    router.register_agent(master_descriptor)
    manager._agents[master_id] = master_descriptor

    # Create OrchestrationTools bound to master
    orch_tools = OrchestrationTools(
        agent_id=master_id,
        manager=manager,
        router=router,
        expert_registry=expert_registry,
        tool_call_timeout=config.tool_call_timeout_seconds,
        default_model=default_model,
        default_provider=default_provider,
        provider_registry=provider_registry,
        peer_model=peer_model_id,
        peer_provider=peer_provider,
        conversation_store=conversation_store,
        peer_models=peer_models_dict,
        model_registry=model_registry,
    )
    # Attach model intelligence reference for the get_model_rankings tool
    if model_intelligence is not None:
        orch_tools._model_intelligence = model_intelligence

    # Start deadlock detector as background task
    deadlock_task = asyncio.create_task(
        run_deadlock_detector(
            router=router,
            manager=manager,
            check_interval=config.deadlock_check_interval_seconds,
            max_wait_seconds=config.deadlock_max_wait_seconds,
        ),
        name="deadlock-detector",
    )

    # Start background model intelligence researcher (hidden from user).
    # Runs once on startup if rankings are stale, then every 24h.
    research_stop = asyncio.Event()
    research_task = None
    if model_registry is not None and model_intelligence is not None:
        # Build a research provider function from the default peer
        # or fall back to None (uses baseline affinities only)
        research_provider_fn = None
        if peer_provider is not None and peer_model_id is not None:
            async def _research_via_peer(prompt: str) -> str:
                result = await peer_provider.send_message(
                    prompt, model_id=peer_model_id,
                )
                if result.success:
                    return result.text
                raise RuntimeError(f"Peer research failed: {result.text}")
            research_provider_fn = _research_via_peer

        research_task = asyncio.create_task(
            run_research_loop(
                intelligence=model_intelligence,
                model_registry=model_registry,
                provider_fn=research_provider_fn,
                stop_event=research_stop,
            ),
            name="model-intelligence-researcher",
        )
        logger.info(
            "Background model intelligence researcher started "
            "(needs_research=%s)",
            model_intelligence.needs_research(),
        )

    logger.info(
        "Orchestration MCP server initialized "
        "(master_id=%s, experts=%d, cwd=%s, default_model=%s)",
        master_id,
        expert_registry.count,
        config.default_cwd,
        default_model,
    )

    try:
        yield {
            "config": config,
            "router": router,
            "manager": manager,
            "expert_registry": expert_registry,
            "orch_tools": orch_tools,
            "master_id": master_id,
            "model_intelligence": model_intelligence,
        }
    finally:
        # Graceful shutdown — stop research loop first
        research_stop.set()
        if research_task is not None:
            research_task.cancel()
            try:
                await research_task
            except asyncio.CancelledError:
                pass

        deadlock_task.cancel()
        try:
            await deadlock_task
        except asyncio.CancelledError:
            pass

        # Kill all child agents
        for desc in manager.get_all_descriptors():
            if desc.agent_id != master_id:
                await manager.kill_agent(desc.agent_id)

        # Shut down all providers
        if provider_registry:
            await provider_registry.shutdown_all()

        logger.info("Orchestration MCP server shut down")


# Create the FastMCP instance
mcp = FastMCP(
    name="prsm-orchestrator",
    instructions=(
        "Orchestration tools for spawning child agents, consulting "
        "experts, and managing inter-agent communication. You are "
        "the master orchestrator.\n\n"
        "IMPORTANT: Tasks are NOT limited to writing code. A task can "
        "be: deeply exploring and describing an existing implementation, "
        "researching how a system works, analyzing code for patterns or "
        "bugs, reviewing changes, gathering information, or any focused "
        "work that returns a summary. Delegation is optional: prefer "
        "direct execution unless there is clear benefit from parallelism, "
        "specialization, or context isolation.\n\n"
        "Use spawn_child when delegation is justified, "
        "consult_expert for domain-specific questions, "
        "consult_peer for second opinions from other AI providers, and "
        "wait_for_message + respond_to_child for interactive "
        "communication with children."
    ),
    lifespan=orchestration_lifespan,
)

# Register master-agent tools
from .master_tools import register_tools  # noqa: E402

register_tools(mcp)


def main() -> None:
    """Entry point for the MCP server."""
    global _parsed_args
    _parsed_args = _parse_args()
    # Best-effort persistent logging for debugging stdio stream closures.
    # This is independent of the lifespan logger and captures startup/runtime
    # crashes that would otherwise be invisible to users.
    try:
        log_dir = Path.home() / ".prsm" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"orchestrator-stdio-{os.getpid()}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info(
            "Starting stdio MCP server (pid=%s, argv=%s)", os.getpid(), sys.argv
        )
    except Exception:
        pass

    try:
        mcp.run(transport="stdio")
    except Exception:
        logging.getLogger(__name__).exception(
            "Fatal stdio MCP server error (pid=%s)", os.getpid()
        )
        raise


if __name__ == "__main__":
    main()
