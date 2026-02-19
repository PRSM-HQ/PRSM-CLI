"""Gemini CLI provider.

Uses the `gemini` CLI for both full agent tasks and lightweight
conversations (via --resume and --output-format json).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from typing import AsyncIterator

from .base import Provider, ProviderMessage, ProviderResult

logger = logging.getLogger(__name__)


class GeminiProvider(Provider):
    """Provider backed by the Gemini CLI.

    Uses `gemini --prompt` for execution.
    Supports session resumption via --resume {session_id}.

    Auth: Uses the CLI's built-in auth (cached credentials).
    If api_key_env is set, it can be passed to the environment.
    """

    def __init__(
        self,
        command: str = "gemini",
        api_key_env: str | None = None,
        default_model: str = "gemini-2.5-flash",
    ) -> None:
        self._command = self.resolve_command(command, "gemini")
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._settings_path: str | None = None
        self._backup_orch_config: dict | None = None

    @property
    def name(self) -> str:
        return "gemini"

    def _build_env(self) -> dict[str, str] | None:
        """Build subprocess environment with optional API key."""
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                env = os.environ.copy()
                env["GEMINI_API_KEY"] = key
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
        """Run a Gemini agent via `gemini --prompt`.

        Yields ProviderMessage objects line-by-line from
        ``--output-format stream-json`` so AgentSession can parse tool
        start/complete events and assistant text.
        """
        cmd = [self._command]

        model = model_id or self._default_model
        cmd.extend(["--model", model])

        # Map permission modes
        if permission_mode == "bypassPermissions":
            cmd.append("--yolo")
        elif permission_mode in ("plan", "auto_edit", "default"):
            cmd.extend(["--approval-mode", permission_mode])

        # Use --flag=value syntax for --prompt and --output-format to
        # prevent yargs from misinterpreting the prompt text or format
        # value as positional query arguments.  With separate argv entries
        # (["--prompt", value]), certain yargs configurations or CLI
        # versions can treat the value as a positional when it coexists
        # with array flags like --allowed-tools, triggering:
        #   "Cannot use both a positional prompt and --prompt flag together"
        # Combining flag and value into a single argv entry with "="
        # eliminates the ambiguity entirely.
        cmd.append(f"--prompt={prompt}")
        cmd.append("--output-format=stream-json")

        if tools:
            cmd.extend(["--allowed-tools"] + tools)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
                cwd=cwd,
            )

            # We'll read stdout line by line to simulate streaming
            # though gemini CLI might buffer.
            result_parts = []
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                result_parts.append(text)
                yield ProviderMessage(text=text)

            await proc.wait()
            stdout_all = "".join(result_parts)

            if proc.returncode != 0:
                stderr_bytes = await proc.stderr.read()
                error = stderr_bytes.decode("utf-8", errors="replace")
                yield ProviderMessage(
                    text=f"Gemini failed (rc={proc.returncode}): {error}",
                    is_result=True,
                    is_error=True,
                )
            else:
                yield ProviderMessage(
                    text=stdout_all,
                    is_result=True,
                )

        except FileNotFoundError:
            yield ProviderMessage(
                text=f"ERROR: '{self._command}' CLI not found.",
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
        """Send a message via the Gemini CLI with JSON output.

        Uses --resume {thread_id} to continue conversations.
        """
        if not self.is_available():
            return ProviderResult(
                text=f"ERROR: '{self._command}' CLI not found",
                success=False,
            )

        cmd = [self._command]
        if thread_id:
            cmd.extend(["--resume", thread_id])

        model = model_id or self._default_model
        cmd.extend(["--model", model])
        cmd.append(f"--prompt={prompt}")
        cmd.append("--output-format=json")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )

            stdout_bytes, stderr_bytes = await proc.communicate()
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                error = stderr_bytes.decode("utf-8", errors="replace")
                return ProviderResult(
                    text=f"Gemini failed (rc={proc.returncode}): {error}\n{stdout_text}",
                    success=False,
                )

            # Parse JSON output
            # The CLI might output "Loaded cached credentials..." etc before the JSON
            # so we find the first '{'
            json_start = stdout_text.find("{")
            if json_start == -1:
                return ProviderResult(
                    text=f"Gemini output was not JSON: {stdout_text}",
                    success=False,
                )

            json_text = stdout_text[json_start:]
            try:
                data = json.loads(json_text)
                return ProviderResult(
                    text=data.get("response", ""),
                    success=True,
                    thread_id=data.get("session_id"),
                    metadata=data.get("stats", {}),
                )
            except json.JSONDecodeError:
                return ProviderResult(
                    text=f"Failed to parse Gemini JSON: {json_text}",
                    success=False,
                )

        except Exception as exc:
            return ProviderResult(
                text=f"Gemini invocation failed: {exc}",
                success=False,
            )

    def is_available(self) -> bool:
        """Check if gemini CLI is installed."""
        return shutil.which(self._command) is not None

    @property
    def supports_master(self) -> bool:
        """Gemini CLI supports MCP clients via settings.json."""
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
        """Build gemini command with MCP orchestration proxy.

        Injects the orchestrator MCP server config into the project-level
        .gemini/settings.json file. Gemini CLI reads MCP config from:
        1. Project-level: {cwd}/.gemini/settings.json
        2. Global: ~/.gemini/settings.json

        We use the project-level settings, preserving any existing config
        and injecting our orchestrator server entry plus any plugin MCP
        servers. The caller should call cleanup_master_settings() after
        the session ends to restore the original file.
        """
        model = model_id or self._default_model
        python_path = sys.executable
        proxy_module = "prsm.engine.mcp_server.orch_proxy"

        # Inject MCP config into project .gemini/settings.json
        # Use abspath so cleanup works regardless of working directory
        effective_cwd = os.path.abspath(cwd or ".")
        gemini_dir = os.path.join(effective_cwd, ".gemini")
        settings_path = os.path.join(gemini_dir, "settings.json")

        # Read existing settings if any
        existing_settings = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    existing_settings = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Preserve existing MCP servers, add our orchestrator and plugins
        mcp_servers = existing_settings.get("mcpServers", {})
        # Save backup of existing orchestrator config if any
        self._backup_orch_config = mcp_servers.get("orchestrator")

        mcp_servers["orchestrator"] = {
            "command": python_path,
            "args": [
                "-m", proxy_module,
                "--port", str(bridge_port),
            ],
        }

        # Merge plugin MCP servers
        if plugin_mcp_servers:
            for name, config in plugin_mcp_servers.items():
                if name not in mcp_servers or name == "orchestrator":
                    # Don't overwrite existing plugins, but do add new ones
                    # orchestrator always uses our config
                    if name != "orchestrator":
                        mcp_servers[name] = config

        existing_settings["mcpServers"] = mcp_servers

        os.makedirs(gemini_dir, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(existing_settings, f, indent=2)

        self._settings_path = settings_path

        cmd = [self._command]
        cmd.extend(["--model", model])
        cmd.append("--yolo")  # Full-auto mode

        # Structured stream-JSON output for tool call event parsing
        cmd.append("--output-format=stream-json")

        if system_prompt:
            full_prompt = (
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
            full_prompt = prompt

        # Use --prompt=value to prevent yargs positional/flag conflict
        cmd.append(f"--prompt={full_prompt}")

        return cmd, self._build_env(), None

    def cleanup_master_settings(self) -> None:
        """Restore the .gemini/settings.json after a master session.

        Removes the orchestrator MCP server entry we injected, or
        restores the backup if one existed before.
        """
        if not self._settings_path or not os.path.exists(self._settings_path):
            return

        try:
            with open(self._settings_path, "r") as f:
                settings = json.load(f)

            mcp_servers = settings.get("mcpServers", {})
            if self._backup_orch_config is not None:
                # Restore the previous orchestrator config
                mcp_servers["orchestrator"] = self._backup_orch_config
            else:
                # Remove the orchestrator entry we added
                mcp_servers.pop("orchestrator", None)

            settings["mcpServers"] = mcp_servers
            with open(self._settings_path, "w") as f:
                json.dump(settings, f, indent=2)

            logger.info("Restored .gemini/settings.json")
        except Exception as exc:
            logger.warning("Failed to cleanup .gemini/settings.json: %s", exc)
        finally:
            self._settings_path = None
            self._backup_orch_config = None

    async def shutdown(self) -> None:
        """Clean up injected MCP settings."""
        self.cleanup_master_settings()
