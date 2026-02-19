"""OpenAI Codex CLI provider.

Uses `codex exec` for full agent tasks and `codex mcp-server`
for multi-turn conversations (via MCP tools codex() / codex-reply()).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from typing import AsyncIterator

from .base import Provider, ProviderMessage, ProviderResult

logger = logging.getLogger(__name__)
_CODEX_REASONING_SUFFIX_RE = re.compile(
    r"^(?P<model>.+?)::reasoning_effort=(?P<effort>low|medium|high)$"
)
_CANONICAL_CODEX_MODEL = "gpt-5.3-codex"
_DEFAULT_CODEX_REASONING_EFFORT = "medium"
_ALIAS_TO_REASONING = {
    "gpt-5-3-low": "low",
    "gpt-5-3-medium": "medium",
    "gpt-5-3-high": "high",
}


class CodexProvider(Provider):
    """Provider backed by the OpenAI Codex CLI.

    Dual mode:
    - run_agent(): uses `codex exec` for full single-shot tasks
    - send_message(): manages a persistent `codex mcp-server`
      subprocess for multi-turn conversations via MCP protocol

    Auth: Works with OAuth (OpenAI Plus plan) by default. If
    api_key_env is set and the env var exists, it's passed to
    the subprocess environment.
    """

    def __init__(
        self,
        command: str = "codex",
        api_key_env: str | None = None,
        default_model: str = "gpt-5.2-codex",
    ) -> None:
        self._command = self.resolve_command(command, "codex")
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._mcp_process: asyncio.subprocess.Process | None = None
        self._mcp_reader: asyncio.StreamReader | None = None
        self._mcp_writer: asyncio.StreamWriter | None = None
        self._request_id: int = 0

    @property
    def name(self) -> str:
        return "codex"

    def _build_env(self) -> dict[str, str] | None:
        """Build subprocess environment with optional API key."""
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                env = os.environ.copy()
                env["OPENAI_API_KEY"] = key
                return env
        return None

    def _resolve_model_for_auth_mode(self, model_id: str) -> str:
        """Normalize Codex aliases into canonical model ID + reasoning suffix."""
        if not model_id:
            return model_id
        normalized = model_id.strip()

        if normalized in _ALIAS_TO_REASONING:
            effort = _ALIAS_TO_REASONING[normalized]
            return f"{_CANONICAL_CODEX_MODEL}::reasoning_effort={effort}"

        if normalized in {"gpt-5-3", "codex"}:
            return (
                f"{_CANONICAL_CODEX_MODEL}"
                f"::reasoning_effort={_DEFAULT_CODEX_REASONING_EFFORT}"
            )

        base_model, effort = self._split_model_and_reasoning(normalized)
        if base_model == "gpt-5-3":
            resolved_effort = effort or _DEFAULT_CODEX_REASONING_EFFORT
            return f"{_CANONICAL_CODEX_MODEL}::reasoning_effort={resolved_effort}"
        if base_model == _CANONICAL_CODEX_MODEL and not effort:
            return (
                f"{_CANONICAL_CODEX_MODEL}"
                f"::reasoning_effort={_DEFAULT_CODEX_REASONING_EFFORT}"
            )
        return normalized

    @staticmethod
    def _split_model_and_reasoning(
        model_id: str,
    ) -> tuple[str, str | None]:
        """Extract codex reasoning effort encoded in model_id suffix."""
        match = _CODEX_REASONING_SUFFIX_RE.match(model_id)
        if match:
            return match.group("model"), match.group("effort")
        return model_id, None

    async def run_agent(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model_id: str | None = None,
        tools: list[str] | None = None,
        permission_mode: str = "default",
        cwd: str | None = None,
    ) -> AsyncIterator[ProviderMessage]:
        """Run a Codex agent via `codex exec`.

        Uses asyncio.create_subprocess_exec (array-based, no shell)
        for safe argument passing.
        """
        cmd = [self._command]
        # Disable global MCP servers — worker agents don't need them.
        # Without this, Codex loads MCP servers from ~/.codex/config.toml
        # which may fail or interfere with the orchestration.
        cmd.extend(["-c", "mcp_servers={}"])

        model = self._resolve_model_for_auth_mode(model_id or self._default_model)
        resolved_model, reasoning_effort = self._split_model_and_reasoning(model)
        cmd.extend(["-c", f'model="{resolved_model}"'])
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        cmd.append("exec")
        # Emit structured JSONL so AgentSession can reconstruct full
        # assistant + tool call history for worker/child agents.
        cmd.append("--json")
        # Avoid cross-run session accumulation and stale rollout state.
        cmd.append("--ephemeral")

        if cwd:
            cmd.extend(["-C", cwd])

        # Map permission modes
        if permission_mode == "bypassPermissions":
            cmd.append("--full-auto")

        cmd.append("-")

        try:
            # create_subprocess_exec passes args as array — no shell
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
                cwd=cwd,
            )

            if proc.stdin is not None:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()

            result_parts: list[str] = []
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                result_parts.append(text)
                yield ProviderMessage(text=text)

            await proc.wait()
            result_text = "".join(result_parts)
            stderr = await proc.stderr.read()

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace")
                yield ProviderMessage(
                    text=f"Codex failed (rc={proc.returncode}): "
                         f"{error}\n{result_text}",
                    is_result=True,
                    is_error=True,
                )
            else:
                yield ProviderMessage(
                    text=result_text,
                    is_result=True,
                )

        except FileNotFoundError:
            yield ProviderMessage(
                text=f"ERROR: '{self._command}' CLI not found. "
                     f"Install Codex CLI first.",
                is_result=True,
                is_error=True,
            )

    async def _ensure_mcp_server(self) -> bool:
        """Start the codex mcp-server subprocess if not running."""
        if self._mcp_process is not None:
            if self._mcp_process.returncode is None:
                return True
            # Process died — clean up
            self._mcp_process = None
            self._mcp_reader = None
            self._mcp_writer = None

        try:
            # create_subprocess_exec passes args as array — no shell
            proc = await asyncio.create_subprocess_exec(
                self._command, "mcp-server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            self._mcp_process = proc
            self._mcp_reader = proc.stdout
            self._mcp_writer = proc.stdin
            logger.info("Codex MCP server started (pid=%d)", proc.pid)
            return True

        except FileNotFoundError:
            logger.error(
                "'%s' CLI not found — cannot start MCP server",
                self._command,
            )
            return False
        except Exception as exc:
            logger.error("Failed to start Codex MCP server: %s", exc)
            return False

    async def _mcp_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """Call a tool on the codex mcp-server via JSON-RPC over stdio."""
        if not await self._ensure_mcp_server():
            return {"error": "Codex MCP server not available"}

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        request_id = self._request_id

        request_bytes = json.dumps(request).encode("utf-8")

        try:
            self._mcp_writer.write(request_bytes + b"\n")
            await self._mcp_writer.drain()

            # Codex MCP can emit event notifications before the tools/call
            # response; keep reading until we receive the matching request id.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 120.0
            while True:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    return {"error": "Codex MCP server response timeout"}

                line = await asyncio.wait_for(
                    self._mcp_reader.readline(),
                    timeout=timeout,
                )
                if not line:
                    return {"error": "MCP server closed connection"}

                try:
                    response = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON MCP line from codex")
                    continue

                if not isinstance(response, dict):
                    continue

                response_id = response.get("id")
                if response_id is None:
                    # Notification/event for a different lifecycle stage.
                    continue
                if response_id != request_id:
                    # Another in-flight request response; ignore.
                    continue

                if "error" in response:
                    return {"error": response["error"]}
                return response.get("result", response)

        except asyncio.TimeoutError:
            return {"error": "Codex MCP server response timeout"}
        except Exception as exc:
            return {"error": f"MCP call failed: {exc}"}

    async def send_message(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        thread_id: str | None = None,
    ) -> ProviderResult:
        """Send a message via the Codex MCP server.

        If thread_id is None, calls codex() tool (new conversation).
        If thread_id is provided, calls codex-reply() (continue).
        """
        if not self.is_available():
            return ProviderResult(
                text=f"ERROR: '{self._command}' CLI not found",
                success=False,
            )

        if thread_id is None:
            # New conversation
            result = await self._mcp_call("codex", {"prompt": prompt})
        else:
            # Continue conversation
            result = await self._mcp_call(
                "codex-reply",
                {"thread_id": thread_id, "prompt": prompt},
            )

        # Extract response from MCP result
        if "error" in result:
            return ProviderResult(
                text=str(result["error"]),
                success=False,
            )

        # Extract text and threadId from MCP tool result
        content = result.get("content", [])
        text_parts = []
        new_thread_id = thread_id

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        # Try to extract threadId from the response metadata
        if isinstance(result, dict):
            meta = result.get("metadata", {})
            if isinstance(meta, dict) and "threadId" in meta:
                new_thread_id = meta["threadId"]
            # Also check top-level for threadId
            if "threadId" in result:
                new_thread_id = result["threadId"]

        response_text = "\n".join(text_parts) if text_parts else str(result)

        return ProviderResult(
            text=response_text,
            success=True,
            thread_id=new_thread_id,
        )

    def is_available(self) -> bool:
        """Check if codex CLI is installed."""
        return shutil.which(self._command) is not None

    @property
    def supports_master(self) -> bool:
        """Codex CLI supports MCP clients via config.toml."""
        return True

    def build_master_cmd(
        self,
        prompt: str,
        bridge_port: int,
        *,
        system_prompt: str | None = None,
        model_id: str | None = None,
        cwd: str | None = None,
        plugin_mcp_servers: dict | None = None,
    ) -> tuple[list[str], dict[str, str] | None, str | None]:
        """Build codex exec command with MCP orchestration proxy.

        Configures codex to use orch_proxy.py as an MCP server via
        the -c flag, which bridges tool calls back to the engine.
        Also merges any plugin MCP servers.
        """
        model = self._resolve_model_for_auth_mode(model_id or self._default_model)
        resolved_model, reasoning_effort = self._split_model_and_reasoning(model)
        python_path = sys.executable
        proxy_module = "prsm.engine.mcp_server.orch_proxy"

        cmd = [self._command]

        # Build MCP server config — start with orchestration proxy
        mcp_servers_config = {
            "orchestrator": {
                "type": "stdio",
                "command": python_path,
                "args": ["-m", proxy_module, "--port", str(bridge_port)],
            }
        }

        # Merge plugin MCP servers
        if plugin_mcp_servers:
            for name, config in plugin_mcp_servers.items():
                if name not in mcp_servers_config:
                    mcp_servers_config[name] = config

        # Convert to TOML inline table format
        # mcp_servers={server1={...}, server2={...}}
        server_parts = []
        for name, config in mcp_servers_config.items():
            # Build inner config dict
            config_parts = []
            for key, value in config.items():
                if isinstance(value, str):
                    config_parts.append(f'{key}="{value}"')
                elif isinstance(value, list):
                    # Format list as TOML array: [item1, item2, ...]
                    formatted_items = []
                    for item in value:
                        if isinstance(item, str):
                            formatted_items.append(f'"{item}"')
                        else:
                            formatted_items.append(str(item))
                    config_parts.append(f'{key}=[{", ".join(formatted_items)}]')
                else:
                    config_parts.append(f'{key}={value}')
            server_parts.append(f'{name}={{{", ".join(config_parts)}}}')

        mcp_toml = f'mcp_servers={{{", ".join(server_parts)}}}'
        cmd.extend(["-c", mcp_toml])
        cmd.extend(["-c", f'model="{resolved_model}"'])
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])

        # Subcommand
        cmd.append("exec")

        # Full-auto mode (no interactive permission prompts)
        cmd.append("--full-auto")

        # Structured JSONL output for tool call event parsing
        cmd.append("--json")
        # Each orchestrator run should be isolated from previous Codex threads.
        cmd.append("--ephemeral")

        if cwd:
            cmd.extend(["-C", cwd])

        # Prepend system prompt to the user prompt if provided
        if system_prompt:
            stdin_prompt = (
                f"<system_instructions>\n{system_prompt}\n</system_instructions>\n\n"
                f"<user_task>\n{prompt}\n</user_task>\n\n"
                f"IMPORTANT: Read the <user_task> above carefully. You MUST use your orchestration tools "
                f"(spawn_child, ask_user, etc.) to accomplish the task. Do NOT call task_complete "
                f"until you have actually completed the work.\n\n"
                f"DOCUMENTATION-FIRST REQUIREMENTS:\n"
                f"- Before exploring code, review relevant architecture docs in @docs/ to plan targeted work.\n"
                f"- When launching agents, instruct them to review relevant @docs/ architecture files before deep exploration.\n\n"
                f"ASSUMPTION MINIMIZATION REQUIREMENTS:\n"
                f"- Make as few assumptions or interpretations as possible.\n"
                f"- If anything is unclear, ambiguous, or unspecified, ask clarifying questions immediately with ask_user() before proceeding.\n\n"
                f"COMPLETION SUMMARY REQUIREMENTS:\n"
                f"- For substantial or long tasks, provide a comprehensive final user-facing summary before task_complete.\n"
                f"- Include: what you changed/discovered, where changes were made, verification performed and gaps, remaining risks, and concrete recommended next steps.\n"
                f"- Do not end with only a brief one-line completion status."
            )
        else:
            stdin_prompt = prompt

        # Pass prompt via stdin to avoid OS argv length limits.
        cmd.append("-")
        return cmd, self._build_env(), stdin_prompt

    async def shutdown(self) -> None:
        """Kill the MCP server subprocess."""
        if self._mcp_process is not None:
            pid = self._mcp_process.pid
            try:
                self._mcp_process.terminate()
                try:
                    await asyncio.wait_for(
                        self._mcp_process.wait(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self._mcp_process.kill()
                    await self._mcp_process.wait()
                logger.info("Codex MCP server stopped (pid=%d)", pid)
            except ProcessLookupError:
                pass
            finally:
                self._mcp_process = None
                self._mcp_reader = None
                self._mcp_writer = None
