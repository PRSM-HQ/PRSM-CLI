"""TCP bridge server for non-Claude master agents.

Runs in the engine process alongside OrchestrationTools. Accepts TCP
connections from orch_proxy.py instances and dispatches JSON-RPC tool
calls to the in-process OrchestrationTools.

Protocol: Newline-delimited JSON (JSONL) over TCP on localhost.

Request:  {"id": N, "method": "tool_name", "params": {...}}
Response: {"id": N, "result": {...}}
Error:    {"id": N, "error": "message"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..config import EventCallback, fire_event

if TYPE_CHECKING:
    from .tools import OrchestrationTools
    from ..agent_manager import AgentManager
    from ..expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)

_FILE_TOOL_ALIASES: dict[str, str] = {
    "Read": "Read",
    "read": "Read",
    "read_file": "Read",
    "file_read": "Read",
    "Write": "Write",
    "write": "Write",
    "write_file": "Write",
    "file_write": "Write",
    "Edit": "Edit",
    "edit": "Edit",
    "edit_file": "Edit",
    "file_edit": "Edit",
    "Bash": "Bash",
    "bash": "Bash",
    "run_bash": "Bash",
}


class OrchBridge:
    """TCP server bridging external MCP proxies to in-process OrchestrationTools.

    The engine starts this bridge when a non-Claude provider is used as
    the master agent. The provider's CLI is configured to launch
    orch_proxy.py as its MCP server, which connects back to this bridge.

    Tool calls flow:
        CLI → MCP protocol → orch_proxy.py → TCP → OrchBridge → OrchestrationTools
    """

    def __init__(
        self,
        orch_tools: OrchestrationTools,
        manager: AgentManager,
        expert_registry: ExpertRegistry,
        event_callback: EventCallback | None = None,
        agent_id: str = "",
    ) -> None:
        self._orch = orch_tools
        self._manager = manager
        self._registry = expert_registry
        self._event_callback = event_callback
        self._agent_id = agent_id
        self._server: asyncio.Server | None = None
        self._port: int = 0
        self._connections: set[asyncio.Task] = set()
        # Signaled when master calls task_complete
        self.task_completed = asyncio.Event()
        self.task_result: str | None = None

    @staticmethod
    def _bare_tool_name(tool_name: str) -> str:
        if not tool_name:
            return ""
        if tool_name.startswith("mcp__") and tool_name.count("__") >= 2:
            return tool_name.split("__", 2)[2]
        return tool_name

    @classmethod
    def _tool_argument_limit(cls, tool_name: str) -> int:
        bare_name = cls._normalize_method_name(tool_name)
        if bare_name in ("Write", "Edit"):
            return 50000
        if bare_name == "spawn_children_parallel":
            return 250000
        return 2000

    @classmethod
    def _normalize_method_name(cls, method: str) -> str:
        bare_name = cls._bare_tool_name(method)
        return _FILE_TOOL_ALIASES.get(bare_name, bare_name)

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> int:
        """Start TCP server on a random available port. Returns the port."""
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", 0,
        )
        addr = self._server.sockets[0].getsockname()
        self._port = addr[1]
        logger.info("OrchBridge started on port %d", self._port)
        return self._port

    async def stop(self) -> None:
        """Stop the TCP server and close all connections."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("OrchBridge stopped (port=%d)", self._port)
        for task in self._connections:
            task.cancel()
        self._connections.clear()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single proxy connection."""
        peer = writer.get_extra_info("peername")
        logger.info("OrchBridge client connected: %s", peer)
        task = asyncio.current_task()
        if task:
            self._connections.add(task)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    error_resp = json.dumps(
                        {"id": None, "error": f"Invalid JSON: {exc}"}
                    )
                    writer.write(error_resp.encode("utf-8") + b"\n")
                    await writer.drain()
                    continue

                req_id = request.get("id")
                method = request.get("method", "")
                normalized_method = self._normalize_method_name(method)
                raw_params = request.get("params", {})
                params = dict(raw_params) if isinstance(raw_params, dict) else {}

                # Generate a unique tool call ID for event correlation
                tool_id = str(uuid.uuid4())
                params.setdefault("_tool_call_id", tool_id)

                # Emit tool_call_started before dispatch
                try:
                    args_str = json.dumps(params) if params else "{}"
                except (TypeError, ValueError):
                    args_str = str(params)
                max_args = self._tool_argument_limit(normalized_method)
                await fire_event(self._event_callback, {
                    "event": "tool_call_started",
                    "agent_id": self._agent_id,
                    "tool_id": tool_id,
                    "tool_name": normalized_method,
                    "arguments": args_str[:max_args],
                })

                try:
                    result = await self._dispatch(normalized_method, params)
                    response = {"id": req_id, "result": result}

                    # Emit tool_call_completed on success
                    result_str = _extract_text(result)
                    await fire_event(self._event_callback, {
                        "event": "tool_call_completed",
                        "agent_id": self._agent_id,
                        "tool_id": tool_id,
                        "result": result_str[:500],
                        "is_error": bool(result.get("is_error")) if isinstance(result, dict) else False,
                    })
                except Exception as exc:
                    logger.exception(
                        "OrchBridge dispatch error: method=%s", normalized_method,
                    )
                    response = {"id": req_id, "error": str(exc)}

                    # Emit tool_call_completed with error info
                    await fire_event(self._event_callback, {
                        "event": "tool_call_completed",
                        "agent_id": self._agent_id,
                        "tool_id": tool_id,
                        "result": str(exc)[:500],
                        "is_error": True,
                    })

                writer.write(
                    json.dumps(response).encode("utf-8") + b"\n"
                )
                await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("OrchBridge client handler error")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            if task:
                self._connections.discard(task)
            logger.info("OrchBridge client disconnected: %s", peer)

    async def _dispatch(
        self, method: str, params: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a tool call to the appropriate handler."""
        method = self._normalize_method_name(method)
        # OrchestrationTools method dispatch
        dispatch_map: dict[str, Any] = {
            "ask_parent": lambda p: self._orch.ask_parent(
                p["question"],
            ),
            "ask_user": lambda p: self._orch.ask_user(
                p["question"], p.get("options", []),
            ),
            "spawn_child": lambda p: self._orch.spawn_child(
                prompt=p["prompt"],
                wait=p.get("wait", False),
                tools=p.get("tools"),
                model=p.get("model"),
                cwd=p.get("cwd"),
                mcp_servers=p.get("mcp_servers"),
                exclude_plugins=p.get("exclude_plugins"),
                complexity=p.get("complexity"),
            ),
            "spawn_children_parallel": lambda p: (
                self._orch.spawn_children_parallel(p["children"])
            ),
            "restart_child": lambda p: self._orch.restart_child(
                child_agent_id=p["child_agent_id"],
                prompt=p["prompt"],
                wait=p.get("wait", False),
            ),
            "consult_expert": lambda p: self._orch.consult_expert(
                p["expert_id"], p["question"],
            ),
            "report_progress": lambda p: self._orch.report_progress(
                p["status"], p.get("percent_complete", 0),
            ),
            "task_complete": lambda p: self._handle_task_complete(
                summary=p["summary"],
                artifacts=p.get("artifacts"),
                steps=p.get("steps"),
                assumptions=p.get("assumptions"),
                risks=p.get("risks"),
                rollback_plan=p.get("rollback_plan"),
                confidence=p.get("confidence"),
                verification_results=p.get("verification_results"),
            ),
            "wait_for_message": lambda p: self._orch.wait_for_message(
                p.get("timeout_seconds", 0.0),
            ),
            "respond_to_child": lambda p: self._orch.respond_to_child(
                p["child_agent_id"],
                p["correlation_id"],
                p["response"],
            ),
            "consult_peer": lambda p: self._orch.consult_peer(
                p["question"], p.get("thread_id"), p.get("peer"),
            ),
            "get_child_history": lambda p: self._orch.get_child_history(
                p["child_agent_id"], p.get("detail_level", "full"),
            ),
            "check_child_status": lambda p: self._orch.check_child_status(
                p["child_agent_id"],
            ),
            "send_child_prompt": lambda p: self._orch.send_child_prompt(
                p["child_agent_id"], p["prompt"],
            ),
            "get_children_status": lambda p: (
                self._orch.get_children_status()
            ),
            "recommend_model": lambda p: self._orch.recommend_model(
                p["task_description"], p.get("complexity", "medium"),
            ),
            "get_model_rankings": lambda p: self._orch.get_model_rankings(
                p.get("task_category"),
            ),
            "Bash": lambda p: self._orch.run_bash(
                command=p["command"],
                timeout=p.get("timeout"),
                cwd=p.get("cwd"),
                tool_call_id=p.get("tool_call_id") or p.get("_tool_call_id"),
            ),
            "run_bash": lambda p: self._orch.run_bash(
                command=p["command"],
                timeout=p.get("timeout"),
                cwd=p.get("cwd"),
                tool_call_id=p.get("tool_call_id") or p.get("_tool_call_id"),
            ),
            "Read": self._read_file,
            "Write": self._write_file,
            "Edit": self._edit_file,
        }

        # Direct manager/registry methods (not on OrchestrationTools)
        if method == "list_agents":
            return self._list_agents()
        if method == "kill_agent":
            return await self._kill_agent(params.get("agent_id", ""))
        if method == "list_experts":
            return self._list_experts()
        if method == "list_available_models":
            return self._list_available_models()

        handler = dispatch_map.get(method)
        if handler is None:
            return {
                "content": [
                    {"type": "text", "text": f"ERROR: Unknown method: {method}"}
                ],
                "is_error": True,
            }

        return await handler(params)

    def _resolve_file_path(self, params: dict[str, Any]) -> Path:
        raw_path = str(
            params.get("file_path")
            or params.get("path")
            or ""
        ).strip()
        if not raw_path:
            raise ValueError("Missing required parameter: file_path")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            descriptor = self._manager.get_descriptor(self._agent_id)
            base_cwd = descriptor.cwd if descriptor and descriptor.cwd else os.getcwd()
            path = Path(base_cwd) / path
        return path.resolve()

    async def _read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._resolve_file_path(params)
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _error(f"File not found: {params.get('file_path') or params.get('path')}")
        except OSError as exc:
            return _error(f"Failed to read file: {exc}")

        offset = int(params.get("offset", 0) or 0)
        limit_raw = params.get("limit")
        if offset or limit_raw is not None:
            lines = text.splitlines(keepends=True)
            start = max(0, offset)
            if limit_raw is None:
                end = len(lines)
            else:
                end = start + max(0, int(limit_raw))
            text = "".join(lines[start:end])

        return _text(text)

    async def _write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._resolve_file_path(params)
            content = params.get("content")
            if content is None:
                return _error("Missing required parameter: content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
            return _text(f"Wrote {path}")
        except OSError as exc:
            return _error(f"Failed to write file: {exc}")

    async def _edit_file(self, params: dict[str, Any]) -> dict[str, Any]:
        old_string = params.get("old_string")
        new_string = params.get("new_string")
        if old_string is None:
            return _error("Missing required parameter: old_string")
        if new_string is None:
            return _error("Missing required parameter: new_string")
        if old_string == "":
            return _error("old_string must be non-empty")

        replace_all = bool(params.get("replace_all", False))

        try:
            path = self._resolve_file_path(params)
            original = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _error(f"File not found: {params.get('file_path') or params.get('path')}")
        except OSError as exc:
            return _error(f"Failed to read file for edit: {exc}")

        occurrences = original.count(str(old_string))
        if occurrences == 0:
            return _error("old_string not found in file")
        if occurrences > 1 and not replace_all:
            return _error(
                f"old_string is ambiguous ({occurrences} matches). "
                "Pass replace_all=true to replace every match."
            )

        if replace_all:
            updated = original.replace(str(old_string), str(new_string))
            replaced = occurrences
        else:
            updated = original.replace(str(old_string), str(new_string), 1)
            replaced = 1

        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _error(f"Failed to write edited file: {exc}")

        return _text(f"Edited {path} ({replaced} replacement{'s' if replaced != 1 else ''})")

    async def _handle_task_complete(
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
        """Handle task_complete and signal the bridge owner."""
        summary_text = str(summary or "").strip()
        result = await self._orch.task_complete(
            summary=summary,
            artifacts=artifacts,
            steps=steps,
            assumptions=assumptions,
            risks=risks,
            rollback_plan=rollback_plan,
            confidence=confidence,
            verification_results=verification_results,
        )
        # Only terminate the provider subprocess when task_complete succeeded.
        if isinstance(result, dict) and result.get("is_error"):
            return result
        self.task_result = summary_text
        self.task_completed.set()
        return result

    def _list_agents(self) -> dict[str, Any]:
        descriptors = self._manager.get_all_descriptors()
        if not descriptors:
            return _text("No agents currently registered.")
        lines = []
        for d in descriptors:
            parent = d.parent_id[:8] + "..." if d.parent_id else "none"
            lines.append(
                f"- {d.agent_id[:12]}  "
                f"role={d.role.value:<8} "
                f"state={d.state.value:<20} "
                f"depth={d.depth} "
                f"parent={parent} "
                f"children={len(d.children)}"
            )
        return _text(
            f"Active agents ({len(descriptors)}):\n" + "\n".join(lines)
        )

    async def _kill_agent(self, agent_id: str) -> dict[str, Any]:
        desc = self._manager.get_descriptor(agent_id)
        if desc is None:
            return _error(f"Agent {agent_id} not found")
        await self._manager.kill_agent(agent_id)
        return _text(
            f"Agent {agent_id[:12]} and its children have been killed."
        )

    def _list_experts(self) -> dict[str, Any]:
        profiles = self._registry.list_profiles()
        if not profiles:
            return _text("No experts registered.")
        lines = []
        for p in profiles:
            lines.append(
                f"- **{p.expert_id}** ({p.name}): {p.description}\n"
                f"  model={p.model}, tools={p.tools}"
            )
        return _text(
            f"Available experts ({len(profiles)}):\n" + "\n".join(lines)
        )

    def _list_available_models(self) -> dict[str, Any]:
        model_registry = self._orch._model_registry
        if not model_registry:
            return _text("No model registry configured.")
        available = model_registry.list_available()
        unavailable = model_registry.list_unavailable()
        lines = []
        if available:
            lines.append(f"Available models ({len(available)}):")
            for m in available:
                top = sorted(m.affinities.items(), key=lambda x: -x[1])[:3]
                strengths = ", ".join(f"{k}" for k, v in top)
                lines.append(
                    f"  - {m.model_id} ({m.provider}, {m.tier.value}) "
                    f"— best for: {strengths}"
                )
        if unavailable:
            lines.append(f"\nUnavailable models ({len(unavailable)}):")
            for m in unavailable:
                lines.append(
                    f"  - {m.model_id} ({m.provider}) "
                    f"— provider not installed"
                )
        text = "\n".join(lines) if lines else "No models registered."
        return _text(text)


def _text(text: str) -> dict[str, Any]:
    """Format a successful text response."""
    return {"content": [{"type": "text", "text": text}]}


def _extract_text(result: Any) -> str:
    """Extract plain text from structured MCP tool result payloads."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)
        for key in ("text", "stdout", "output", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    return str(result)


def _error(text: str) -> dict[str, Any]:
    """Format an error response."""
    return {
        "content": [{"type": "text", "text": f"ERROR: {text}"}],
        "is_error": True,
    }
