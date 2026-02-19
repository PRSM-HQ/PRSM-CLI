"""MiniMax provider using Codex CLI.

MiniMax M2.5 is accessed through the Codex CLI with model_provider=minimax.
This routes requests to the MiniMax API (https://api.minimax.io/v1).

Auth: Uses MINIMAX_API_KEY environment variable.

Setup:
  1. Install Codex CLI (https://codex.storage/)
  2. Configure ~/.codex/config.toml with MiniMax provider section
  3. export MINIMAX_API_KEY="sk-..."
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from typing import AsyncIterator

from .base import Provider, ProviderMessage, ProviderResult

logger = logging.getLogger(__name__)


class MiniMaxProvider(Provider):
    """Provider backed by Codex CLI routing to MiniMax API.

    Uses `codex exec -c model_provider=minimax` to access MiniMax models.
    The Codex CLI handles the HTTP communication with MiniMax's API.

    Auth: Requires MINIMAX_API_KEY environment variable.
    """

    def __init__(
        self,
        command: str = "codex",
        api_key_env: str | None = "MINIMAX_API_KEY",
        default_model: str = "MiniMax-M2.5",
    ) -> None:
        self._command = self.resolve_command(command, "codex")
        self._api_key_env = api_key_env or "MINIMAX_API_KEY"
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "minimax"

    def _build_env(self) -> dict[str, str] | None:
        """Build subprocess environment with MiniMax API key."""
        key = os.environ.get(self._api_key_env)
        if key:
            env = os.environ.copy()
            env["MINIMAX_API_KEY"] = key
            return env
        return None

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
        """Run a MiniMax agent via Codex CLI.

        Uses `codex exec -c model_provider=minimax` to route to MiniMax API.
        """
        model = model_id or self._default_model
        cmd = [self._command]
        # Disable global MCP servers — worker agents don't need them.
        # Without this, Codex loads MCP servers from ~/.codex/config.toml
        # which may fail or interfere with the orchestration.
        cmd.extend(["-c", "mcp_servers={}"])
        cmd.extend(["-c", "model_provider=minimax"])
        cmd.extend(["-c", f'model="{model}"'])
        cmd.append("exec")
        # Emit structured JSONL so AgentSession can reconstruct full
        # assistant + tool call history for worker/child agents.
        cmd.append("--json")
        cmd.append("--ephemeral")

        if cwd:
            cmd.extend(["-C", cwd])

        # Map permission modes
        if permission_mode == "bypassPermissions":
            cmd.append("--full-auto")

        cmd.append("-")

        try:
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
                    text=f"MiniMax via Codex failed (rc={proc.returncode}): {error}\n{result_text}",
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
                text=f"ERROR: '{self._command}' CLI not found. Install Codex CLI first.",
                is_result=True,
                is_error=True,
            )

    async def send_message(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        thread_id: str | None = None,
    ) -> ProviderResult:
        """Send a message via Codex CLI routing to MiniMax API.

        Thread/session continuity is not supported in this mode.
        """
        if not self.is_available():
            return ProviderResult(
                text=f"ERROR: '{self._command}' CLI not found or {self._api_key_env} not set",
                success=False,
            )

        model = model_id or self._default_model
        cmd = [self._command]
        # Disable global MCP servers for standalone send_message calls
        cmd.extend(["-c", "mcp_servers={}"])
        cmd.extend(["-c", "model_provider=minimax"])
        cmd.extend(["-c", f'model="{model}"'])
        cmd.append("exec")
        cmd.append("--ephemeral")
        cmd.append("-")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )

            stdout_bytes, stderr_bytes = await proc.communicate(
                prompt.encode("utf-8")
            )
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                error = stderr_bytes.decode("utf-8", errors="replace")
                return ProviderResult(
                    text=f"MiniMax via Codex failed (rc={proc.returncode}): {error}\n{stdout_text}",
                    success=False,
                )

            return ProviderResult(
                text=stdout_text.strip(),
                success=True,
            )

        except Exception as exc:
            return ProviderResult(
                text=f"MiniMax invocation failed: {exc}",
                success=False,
            )

    def _has_api_key(self) -> bool:
        """Check if the MiniMax API key is set in the environment."""
        return bool(os.environ.get(self._api_key_env))

    def is_available(self) -> bool:
        """Check if Codex CLI is installed AND MiniMax API key is set.

        Both are required for MiniMax to function:
        1. Codex CLI must be on PATH
        2. MINIMAX_API_KEY must be set in the environment
        """
        has_codex = shutil.which(self._command) is not None
        has_key = self._has_api_key()

        if has_codex and not has_key:
            logger.debug(
                "Codex CLI found but %s not set — MiniMax unavailable",
                self._api_key_env,
            )
        elif not has_codex:
            logger.debug(
                "Codex CLI not found — MiniMax unavailable",
            )

        return has_codex and has_key

    @property
    def supports_master(self) -> bool:
        """MiniMax via Codex CLI supports MCP clients."""
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
        """Build codex exec command with MCP orchestration and MiniMax routing.

        Uses Codex CLI with model_provider=minimax to route to MiniMax API.
        Also configures MCP servers via -c flag for orchestration tools.
        """
        model = model_id or self._default_model
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
        server_parts = []
        for name, config in mcp_servers_config.items():
            config_parts = []
            for key, value in config.items():
                if isinstance(value, str):
                    config_parts.append(f'{key}="{value}"')
                elif isinstance(value, list):
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
        cmd.extend(["-c", "model_provider=minimax"])
        cmd.extend(["-c", f'model="{model}"'])

        # Subcommand
        cmd.append("exec")

        # Full-auto mode (no interactive permission prompts)
        cmd.append("--full-auto")

        # Structured JSONL output for tool call event parsing
        cmd.append("--json")
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
                f"- When launching agents, instruct them to review relevant @docs/ architecture files before deep exploration.\n"
                f"\n"
                f"ASSUMPTION MINIMIZATION REQUIREMENTS:\n"
                f"- Make as few assumptions or interpretations as possible.\n"
                f"- If anything is unclear, ambiguous, or unspecified, ask clarifying questions immediately with ask_user() before proceeding.\n"
                f"\n"
                f"COMPLETION SUMMARY REQUIREMENTS:\n"
                f"- For substantial or long tasks, provide a comprehensive final user-facing summary before task_complete.\n"
                f"- Include: what you changed/discovered, where changes were made, verification performed and gaps, remaining risks, and concrete recommended next steps.\n"
                f"- Do not end with only a brief one-line completion status.\n"
                f"\n"
                f"CRITICAL FORMAT RULES:\n"
                f"- You are running inside the Codex CLI which provides tool calling natively.\n"
                f"- Do NOT output <think>...</think> tags. Think silently — your reasoning is internal.\n"
                f"- Do NOT output XML-formatted tool calls like <parameter>, </invoke>, "
                f"<tool_call>, or similar XML tags. These will NOT be executed.\n"
                f"- Do NOT hallucinate or simulate tool call results. Only use the actual tools "
                f"provided to you by the runtime.\n"
                f"- Use the native function/tool calling mechanism provided by the Codex CLI.\n"
                f"- If you find yourself outputting XML or text that looks like a tool invocation, "
                f"STOP and use the real tool calling interface instead."
            )
        else:
            stdin_prompt = prompt

        # Pass prompt via stdin to avoid OS argv length limits.
        cmd.append("-")
        return cmd, self._build_env(), stdin_prompt

    async def shutdown(self) -> None:
        """No persistent subprocess to clean up."""
        pass
