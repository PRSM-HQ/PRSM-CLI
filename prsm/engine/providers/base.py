"""Abstract base for execution providers.

Each provider wraps a different agent runtime (Claude Agent SDK,
OpenAI Codex CLI, etc.). The orchestration engine calls
run_agent() for full agent tasks and send_message() for
lightweight conversations (e.g. consult_peer).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
import logging
import shutil
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    """Result from a provider invocation."""
    text: str
    success: bool = True
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderMessage:
    """A streaming message from an agent execution."""
    text: str = ""
    is_result: bool = False
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """Abstract provider interface.

    Implementations wrap a specific agent runtime:
    - ClaudeProvider: Claude Agent SDK (query())
    - CodexProvider: OpenAI Codex CLI (codex exec / codex mcp-server)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short provider name (e.g. 'claude', 'codex')."""

    @abc.abstractmethod
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
        """Run a full agent session.

        Yields ProviderMessage objects as the agent works.
        The final message should have is_result=True.
        """
        yield  # pragma: no cover

    @abc.abstractmethod
    async def send_message(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        thread_id: str | None = None,
    ) -> ProviderResult:
        """Send a message for a lightweight conversation.

        If thread_id is None, starts a new conversation.
        If thread_id is provided, continues an existing one.

        Returns ProviderResult with response text and thread_id
        for follow-up calls.
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this provider's runtime is available.

        Only checks if the CLI tool is installed — does NOT
        require an API key (OAuth handles auth).
        """

    def resolve_command(self, command: str, fallback: str | None = None) -> str:
        """Resolve a provider binary by preferring explicit command, then fallback.

        The command may point to a CLI that is not on PATH when using
        custom wrappers or tests. In that case, keep the raw value so
        callers can surface the configured command in error messages.
        """
        if command:
            if shutil.which(command):
                return command
            if fallback and shutil.which(fallback):
                logger.debug(
                    "Command %s not found; falling back to %s for provider %s",
                    command, fallback, self.name if hasattr(self, "name") else "<unknown>",
                )
                return fallback
            if command:
                return command
        if fallback:
            return fallback
        return command

    @property
    def supports_master(self) -> bool:
        """Whether this provider can serve as a master/orchestrator agent.

        Master agents need MCP tool support (spawn_child, task_complete,
        etc.). Providers that support this return True and implement
        build_master_cmd() which returns a CLI command configured with
        an MCP server pointing to the orchestration proxy.

        Default: False. Override in providers that support MCP clients.
        """
        return False

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
        """Build CLI command for master agent mode with MCP orchestration.

        Returns (command_list, env_dict_or_None, stdin_payload_or_None).

        The command should include MCP server config pointing to
        orch_proxy.py --port <bridge_port>, which bridges tool calls
        back to the engine's in-process OrchestrationTools.

        Additionally, it should include any plugin_mcp_servers that
        provide additional tools/capabilities to the agent.

        Only called when supports_master is True.
        """
        raise NotImplementedError(
            f"{self.name} provider does not support master agent mode"
        )

    async def shutdown(self) -> None:
        """Clean up resources (e.g. kill subprocess).

        Default no-op. Override in providers that manage
        long-lived subprocesses.
        """
        return None
