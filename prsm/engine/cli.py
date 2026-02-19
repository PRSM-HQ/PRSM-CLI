"""CLI entry point for the orchestration engine.

Usage:
    python -m prsm.orchestrator "Build feature X with tests"
    python -m prsm.orchestrator --model claude-opus-4-6 "Complex task"
    python -m prsm.orchestrator --task-file tasks/feature.md
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .engine import OrchestrationEngine
from .config import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prsm-orchestrator",
        description=(
            "Hierarchical multi-agent orchestration for Claude Agent SDK"
        ),
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="The task to orchestrate (inline string)",
    )
    parser.add_argument(
        "--task-file", "-f",
        default=None,
        help="Read task from a file (.md, .txt, etc.)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for the master agent (default: from config)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum agent nesting depth (default: 5)",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Maximum concurrent agents (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-agent timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for agents (default: current dir)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--experts-file",
        default=None,
        help=(
            "Python file that exports a register_experts(engine) "
            "function"
        ),
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    config = EngineConfig.from_env()
    if args.max_depth is not None:
        config.max_agent_depth = args.max_depth
    if args.max_agents is not None:
        config.max_concurrent_agents = args.max_agents
    if args.timeout is not None:
        config.agent_timeout_seconds = args.timeout
    if args.cwd is not None:
        config.default_cwd = args.cwd

    # Resolve task from inline arg or file
    task = _resolve_task(args.task, args.task_file)

    engine = OrchestrationEngine(config=config)

    # Load experts from file if provided
    if args.experts_file:
        _load_experts(engine, args.experts_file)

    # Run
    try:
        result = asyncio.run(
            engine.run(
                task_definition=task,
                master_model=args.model,
            )
        )
        print("\n=== Orchestration Result ===\n")
        print(result)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        asyncio.run(engine.shutdown())
        sys.exit(1)


def _resolve_task(inline: str | None, file_path: str | None) -> str:
    """Get task from inline arg or file. Exactly one must be provided."""
    from pathlib import Path

    if inline and file_path:
        print("Error: Provide either a task string or --task-file, not both.")
        sys.exit(1)

    if file_path:
        p = Path(file_path)
        if not p.is_file():
            print(f"Error: Task file not found: {file_path}")
            sys.exit(1)
        return p.read_text(encoding="utf-8").strip()

    if inline:
        return inline

    print("Error: Provide a task string or --task-file.")
    sys.exit(1)


def _load_experts(engine: OrchestrationEngine, path: str) -> None:
    """Load expert profiles from a Python file.

    The file must export a register_experts(engine) function.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("experts", path)
    if spec is None or spec.loader is None:
        print(f"Error: Cannot load experts file: {path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    register_fn = getattr(module, "register_experts", None)
    if register_fn is None:
        print(
            f"Error: {path} must export a "
            f"register_experts(engine) function"
        )
        sys.exit(1)

    register_fn(engine)


if __name__ == "__main__":
    main()
