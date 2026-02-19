"""MCP proxy for non-Claude master agents.

Standalone FastMCP server launched by CLI tools (Codex, Gemini, MiniMax)
as an MCP subprocess. Speaks MCP protocol on stdin/stdout (via FastMCP),
and relays tool calls to the engine's OrchBridge via TCP.

Usage:
    python -m prsm.engine.mcp_server.orch_proxy --port PORT

The CLI discovers this as an MCP server and sends tool calls over
the MCP protocol. This proxy forwards each call to the OrchBridge
TCP server running in the engine process, which dispatches to
in-process OrchestrationTools.

Tool call flow:
    CLI → MCP stdin/stdout → orch_proxy → TCP → OrchBridge → OrchestrationTools
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

logger = logging.getLogger(__name__)

# Bridge port — set from CLI args before server starts
_bridge_port: int = 0


class BridgeClient:
    """TCP client that connects to the OrchBridge in the engine process."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._req_id: int = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to the OrchBridge TCP server with retry logic.

        The bridge server may not be fully ready when this proxy is
        launched by the CLI subprocess. Retries up to 10 times with
        exponential backoff (0.2s → 0.4s → 0.8s ... capped at 2s).
        """
        max_attempts = 10
        delay = 0.2
        for attempt in range(1, max_attempts + 1):
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    "127.0.0.1", self._port,
                )
                logger.info(
                    "Connected to OrchBridge on port %d (attempt %d)",
                    self._port, attempt,
                )
                return
            except (ConnectionRefusedError, OSError) as exc:
                if attempt == max_attempts:
                    logger.error(
                        "Failed to connect to OrchBridge on port %d "
                        "after %d attempts: %s",
                        self._port, max_attempts, exc,
                    )
                    raise
                logger.debug(
                    "OrchBridge connection attempt %d/%d failed: %s, "
                    "retrying in %.1fs",
                    attempt, max_attempts, exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    async def close(self) -> None:
        """Close the TCP connection."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    async def _reset_connection(self) -> None:
        """Reset bridge stream state after a canceled/timed-out call."""
        logger.warning(
            "Resetting OrchBridge connection on port %d after interrupted call",
            self._port,
        )
        try:
            await self.close()
        except Exception:
            logger.exception("Failed closing OrchBridge connection during reset")
        await self.connect()

    async def call(self, method: str, params: dict[str, Any]) -> str:
        """Send a JSON-RPC request and return the text result.

        Returns the extracted text from the OrchestrationTools response.
        Raises ValueError for error responses (FastMCP marks these as errors).
        """
        async with self._lock:
            if self._reader is None or self._writer is None:
                raise ValueError("OrchBridge connection is not initialized")

            self._req_id += 1
            request_id = self._req_id
            request = {
                "id": request_id,
                "method": method,
                "params": params,
            }
            data = json.dumps(request).encode("utf-8") + b"\n"
            try:
                self._writer.write(data)
                await self._writer.drain()

                # Requests can timeout at the caller level. If that happens,
                # this stream may later receive the timed-out response; keep
                # reading until we see the response for this request id.
                # Each individual read has a 5-minute timeout to detect
                # stuck connections.
                _PER_READ_TIMEOUT = 300.0  # 5 minutes
                while True:
                    try:
                        line = await asyncio.wait_for(
                            self._reader.readline(),
                            timeout=_PER_READ_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "OrchBridge readline timed out after %.0fs "
                            "(port=%d, method=%s, req_id=%s)",
                            _PER_READ_TIMEOUT,
                            self._port,
                            method,
                            request_id,
                        )
                        try:
                            await self._reset_connection()
                        except Exception:
                            logger.exception(
                                "Failed to reset OrchBridge connection "
                                "after readline timeout"
                            )
                        raise
                    if not line:
                        raise ValueError("OrchBridge connection closed unexpectedly")
                    try:
                        response = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Ignoring non-JSON response from OrchBridge: %r",
                            line[:200],
                        )
                        continue
                    if not isinstance(response, dict):
                        logger.warning(
                            "Ignoring non-object OrchBridge response: %r",
                            response,
                        )
                        continue

                    response_id = response.get("id")
                    if response_id != request_id:
                        logger.warning(
                            "Discarding out-of-order OrchBridge response "
                            "(expected id=%s, got id=%s, method=%s)",
                            request_id,
                            response_id,
                            method,
                        )
                        continue
                    break
            except (asyncio.CancelledError, asyncio.TimeoutError):
                try:
                    await self._reset_connection()
                except Exception:
                    logger.exception(
                        "Failed to reset OrchBridge connection after interrupted call"
                    )
                raise

            if "error" in response:
                raise ValueError(str(response["error"]))

            result = response.get("result", {})
            return _extract_text(result)


def _extract_text(result: dict[str, Any]) -> str:
    """Convert OrchestrationTools response dict to plain string.

    OrchestrationTools returns {"content": [{"type": "text", "text": ...}]}.
    FastMCP tool functions return plain strings.
    If is_error is set, raise ValueError so FastMCP marks it as an error.
    """
    if not isinstance(result, dict) or "content" not in result:
        return str(result)
    content = result.get("content", [])
    if not content:
        return ""
    text = content[0].get("text", "")
    if result.get("is_error"):
        raise ValueError(text.removeprefix("ERROR: "))
    return text


# ── FastMCP lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def proxy_lifespan(server: FastMCP):
    """Connect to OrchBridge on startup, disconnect on shutdown."""
    client = BridgeClient(_bridge_port)
    await client.connect()
    try:
        yield {"bridge": client}
    finally:
        await client.close()


# ── FastMCP server ────────────────────────────────────────────────

mcp = FastMCP(
    name="prsm-orchestrator",
    instructions=(
        "Orchestration tools for spawning child agents, consulting "
        "experts, and managing inter-agent communication. You are "
        "the master orchestrator.\n\n"
        "Basic file tools are also available (Read/Write/Edit and aliases). "
        "Prefer these for direct file operations when needed.\n\n"
        "Use spawn_child to delegate tasks (code, research, analysis). "
        "Use complexity parameter for smart model selection. "
        "Call task_complete when your overall task is done."
    ),
    lifespan=proxy_lifespan,
)


def _bridge(ctx: Context) -> BridgeClient:
    """Get the BridgeClient from the lifespan context."""
    return ctx.request_context.lifespan_context["bridge"]


# ── Tool registrations ────────────────────────────────────────────
# Each tool forwards its call to the OrchBridge via TCP.

@mcp.tool(
    name="spawn_child",
    description=(
        "Spawn a child agent for a focused task. Tasks include coding, "
        "exploring code, researching, analyzing, reviewing — not just "
        "writing code. Child launches are always non-blocking and return "
        "immediately. The wait parameter is accepted for compatibility but "
        "ignored. "
        "Use complexity for auto "
        "model selection: trivial/simple/medium/complex/frontier."
    ),
)
async def spawn_child(
    prompt: str,
    wait: bool = False,
    tools: list[str] | None = None,
    model: str | None = None,
    cwd: str | None = None,
    complexity: str | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("spawn_child", {
        "prompt": prompt, "wait": wait, "tools": tools,
        "model": model, "cwd": cwd, "complexity": complexity,
    })


@mcp.tool(
    name="spawn_children_parallel",
    description=(
        "Spawn multiple child agents in parallel. Returns immediately "
        "with child IDs. Use wait_for_message() to collect results."
    ),
)
async def spawn_children_parallel(
    children: list[dict[str, Any]],
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("spawn_children_parallel", {
        "children": children,
    })


@mcp.tool(
    name="restart_child",
    description=(
        "Restart a completed/failed child with a new task. Reuses the "
        "child's identity, model, tools, and working directory."
    ),
)
async def restart_child(
    child_agent_id: str,
    prompt: str,
    wait: bool = False,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("restart_child", {
        "child_agent_id": child_agent_id,
        "prompt": prompt, "wait": wait,
    })


@mcp.tool(
    name="consult_expert",
    description="Consult a specialist expert agent. Use list_experts first.",
)
async def consult_expert(
    expert_id: str,
    question: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("consult_expert", {
        "expert_id": expert_id, "question": question,
    })


@mcp.tool(
    name="wait_for_message",
    description=(
        "Wait for the next message from any child agent. Returns "
        "message type, source agent ID, correlation ID, and payload. "
        "Use timeout_seconds=0 to disable timeout."
    ),
)
async def wait_for_message(
    timeout_seconds: float = 0.0,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("wait_for_message", {
        "timeout_seconds": timeout_seconds,
    })


@mcp.tool(
    name="respond_to_child",
    description=(
        "Answer a child agent's ask_parent question. Use the "
        "child_agent_id and correlation_id from wait_for_message."
    ),
)
async def respond_to_child(
    child_agent_id: str,
    correlation_id: str,
    response: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("respond_to_child", {
        "child_agent_id": child_agent_id,
        "correlation_id": correlation_id,
        "response": response,
    })


@mcp.tool(
    name="consult_peer",
    description=(
        "Consult a peer AI provider for a second opinion. Supports "
        "multi-turn via thread_id. Pass peer='list' to see available peers."
    ),
)
async def consult_peer(
    question: str,
    thread_id: str | None = None,
    peer: str | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("consult_peer", {
        "question": question, "thread_id": thread_id, "peer": peer,
    })


@mcp.tool(
    name="ask_user",
    description=(
        "Ask the user a question with clickable options. Use when "
        "you need the user to make a decision."
    ),
)
async def ask_user(
    question: str,
    options: list[dict[str, str]] | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("ask_user", {
        "question": question, "options": options or [],
    })


@mcp.tool(
    name="report_progress",
    description="Send a non-blocking progress update.",
)
async def report_progress(
    status: str,
    percent_complete: int = 0,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("report_progress", {
        "status": status, "percent_complete": percent_complete,
    })


@mcp.tool(
    name="task_complete",
    description=(
        "Signal that your overall task is complete. ALWAYS call this "
        "when you are done with your assigned work. For substantial tasks, "
        "your summary should be comprehensive and user-facing: what changed, "
        "where, verification performed, remaining risks, and recommended next steps."
    ),
)
async def task_complete(
    summary: str,
    artifacts: dict[str, Any] | None = None,
    steps: list[str] | None = None,
    assumptions: list[str] | None = None,
    risks: list[str] | None = None,
    rollback_plan: str | None = None,
    confidence: float | None = None,
    verification_results: list[dict[str, Any]] | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("task_complete", {
        "summary": summary,
        "artifacts": artifacts,
        "steps": steps,
        "assumptions": assumptions,
        "risks": risks,
        "rollback_plan": rollback_plan,
        "confidence": confidence,
        "verification_results": verification_results,
    })


@mcp.tool(
    name="get_child_history",
    description=(
        "Review a child's conversation history. Use detail_level="
        "'summary' for text + tool names, or 'full' for everything."
    ),
)
async def get_child_history(
    child_agent_id: str,
    detail_level: str = "full",
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("get_child_history", {
        "child_agent_id": child_agent_id, "detail_level": detail_level,
    })


@mcp.tool(
    name="check_child_status",
    description="Check a child agent's current state and info.",
)
async def check_child_status(
    child_agent_id: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("check_child_status", {
        "child_agent_id": child_agent_id,
    })


@mcp.tool(
    name="send_child_prompt",
    description="Send a prompt or instruction to a running child agent.",
)
async def send_child_prompt(
    child_agent_id: str,
    prompt: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("send_child_prompt", {
        "child_agent_id": child_agent_id, "prompt": prompt,
    })


@mcp.tool(
    name="get_children_status",
    description=(
        "Get status of all your children. Non-blocking — returns each "
        "child's state and a summary count."
    ),
)
async def get_children_status(ctx: Context = None) -> str:
    return await _bridge(ctx).call("get_children_status", {})


@mcp.tool(
    name="recommend_model",
    description=(
        "Get a model recommendation for a task based on complexity. "
        "Returns the optimal model from the capability registry."
    ),
)
async def recommend_model(
    task_description: str,
    complexity: str = "medium",
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("recommend_model", {
        "task_description": task_description, "complexity": complexity,
    })


@mcp.tool(
    name="Read",
    description="Read a UTF-8 text file from disk.",
)
async def read_file_pascal(
    file_path: str,
    offset: int = 0,
    limit: int | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Read", {
        "file_path": file_path,
        "offset": offset,
        "limit": limit,
    })


@mcp.tool(
    name="Write",
    description="Write UTF-8 text content to a file, creating parent directories if needed.",
)
async def write_file_pascal(
    file_path: str,
    content: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Write", {
        "file_path": file_path,
        "content": content,
    })


@mcp.tool(
    name="Edit",
    description=(
        "Edit a file by replacing old_string with new_string. "
        "Set replace_all=true to replace every occurrence."
    ),
)
async def edit_file_pascal(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Edit", {
        "file_path": file_path,
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    })


@mcp.tool(
    name="read_file",
    description="Alias of Read(file_path, offset, limit).",
)
async def read_file_alias(
    file_path: str,
    offset: int = 0,
    limit: int | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Read", {
        "file_path": file_path,
        "offset": offset,
        "limit": limit,
    })


@mcp.tool(
    name="write_file",
    description="Alias of Write(file_path, content).",
)
async def write_file_alias(
    file_path: str,
    content: str,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Write", {
        "file_path": file_path,
        "content": content,
    })


@mcp.tool(
    name="edit_file",
    description="Alias of Edit(file_path, old_string, new_string, replace_all).",
)
async def edit_file_alias(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("Edit", {
        "file_path": file_path,
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    })


@mcp.tool(
    name="run_bash",
    description=(
        "Execute a bash command with permission checking. Dangerous "
        "commands (rm, sudo, git commit, etc.) require explicit user "
        "approval before execution. You MUST use this tool for ALL "
        "shell/bash commands instead of any native Bash tool."
    ),
)
async def run_bash(
    command: str,
    timeout: int | None = None,
    cwd: str | None = None,
    ctx: Context = None,
) -> str:
    return await _bridge(ctx).call("run_bash", {
        "command": command, "timeout": timeout, "cwd": cwd,
    })


@mcp.tool(
    name="list_agents",
    description="List all active agent sessions with state and role.",
)
async def list_agents(ctx: Context = None) -> str:
    return await _bridge(ctx).call("list_agents", {})


@mcp.tool(
    name="kill_agent",
    description="Force-kill an agent and all its children.",
)
async def kill_agent(agent_id: str, ctx: Context = None) -> str:
    return await _bridge(ctx).call("kill_agent", {"agent_id": agent_id})


@mcp.tool(
    name="list_experts",
    description="List all registered expert profiles with their IDs.",
)
async def list_experts(ctx: Context = None) -> str:
    return await _bridge(ctx).call("list_experts", {})


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    """Entry point when launched by a CLI as an MCP subprocess."""
    global _bridge_port

    parser = argparse.ArgumentParser(
        prog="prsm-orch-proxy",
        description="MCP orchestration proxy for non-Claude master agents",
    )
    parser.add_argument(
        "--port", type=int, required=True,
        help="OrchBridge TCP port to connect to",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    _bridge_port = args.port

    # Logging goes to stderr (stdout is the MCP transport)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logger.info(
        "Starting orch_proxy (bridge_port=%d, pid=%d)",
        _bridge_port, __import__("os").getpid(),
    )

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
