"""Deadlock detection for the agent wait graph.

Runs periodically to detect cycles (A waits on B, B waits on A)
and breaks them by force-failing the deepest agent in the cycle.

Uses DFS cycle detection on the directed wait graph.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .message_router import MessageRouter
    from .agent_manager import AgentManager

logger = logging.getLogger(__name__)


def detect_cycle(wait_graph: dict[str, str]) -> list[str] | None:
    """Detect a cycle in the wait graph using DFS.

    Args:
        wait_graph: Mapping of waiter_id -> target_id.

    Returns:
        List of agent IDs forming the cycle, or None.
    """
    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        if node in in_stack:
            cycle_start = path.index(node)
            return path[cycle_start:] + [node]
        if node in visited:
            return None

        visited.add(node)
        in_stack.add(node)
        path.append(node)

        next_node = wait_graph.get(node)
        if next_node is not None:
            result = dfs(next_node)
            if result is not None:
                return result

        path.pop()
        in_stack.discard(node)
        return None

    for start_node in wait_graph:
        if start_node not in visited:
            cycle = dfs(start_node)
            if cycle is not None:
                return cycle

    return None


def _select_victim(
    cycle: list[str],
    manager: AgentManager,
) -> str | None:
    """Select which agent to kill to break a deadlock.

    Strategy: kill the deepest agent (most nested child).
    Ties broken by most recent creation time.
    """
    best_id: str | None = None
    best_depth = -1

    for agent_id in cycle:
        descriptor = manager.get_descriptor(agent_id)
        if descriptor and descriptor.depth > best_depth:
            best_depth = descriptor.depth
            best_id = agent_id

    return best_id


async def run_deadlock_detector(
    router: MessageRouter,
    manager: AgentManager,
    check_interval: float = 5.0,
    max_wait_seconds: float = 120.0,
) -> None:
    """Background task that periodically checks for deadlocks.

    When a cycle is detected, the deepest agent in the cycle is
    force-failed. This unblocks its parent, which receives an
    error result.
    """
    while True:
        try:
            await asyncio.sleep(check_interval)

            wait_graph = router.get_wait_graph()
            if not wait_graph:
                continue

            cycle = detect_cycle(wait_graph)
            if cycle is not None:
                logger.error(
                    "Deadlock detected: %s",
                    " -> ".join(c[:8] for c in cycle),
                )
                victim_id = _select_victim(cycle, manager)
                if victim_id:
                    await manager.force_fail_agent(
                        victim_id,
                        f"Deadlock detected in cycle: "
                        f"{' -> '.join(c[:8] for c in cycle)}",
                    )
                    router.clear_waiting(victim_id)

        except asyncio.CancelledError:
            logger.info("Deadlock detector stopped")
            return
        except Exception:
            logger.exception("Deadlock detector error")
