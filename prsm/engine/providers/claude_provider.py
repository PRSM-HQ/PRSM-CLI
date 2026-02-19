"""Claude Agent SDK provider.

Wraps claude_agent_sdk.query() for full agent sessions and
lightweight SDK calls for conversations.
"""
from __future__ import annotations

import logging
import shutil
from typing import AsyncIterator

from .base import Provider, ProviderMessage, ProviderResult

logger = logging.getLogger(__name__)


class ClaudeProvider(Provider):
    """Provider backed by the Claude Agent SDK.

    Uses claude_agent_sdk.query() for run_agent() and a lightweight
    SDK call for send_message().

    Auth: Works with OAuth (Claude Max plan) by default. If
    api_key_env is set and the env var exists, the SDK will use it.
    """

    def __init__(
        self,
        api_key_env: str | None = None,
        default_model: str = "claude-opus-4-6",
        command: str = "claude",
    ) -> None:
        self._command = self.resolve_command(command, "claude")
        self._api_key_env = api_key_env
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "claude"

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
        """Run a Claude agent session via the SDK."""
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions
        except ImportError:
            yield ProviderMessage(
                text="ERROR: claude_agent_sdk not installed",
                is_result=True,
                is_error=True,
            )
            return

        options = ClaudeAgentOptions(
            system_prompt=system_prompt or "",
            allowed_tools=tools or [],
            permission_mode=permission_mode,
            cwd=cwd or ".",
            model=model_id or self._default_model,
        )

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "result"):
                result_text = message.result
                yield ProviderMessage(
                    text=result_text,
                    is_result=True,
                )
            else:
                yield ProviderMessage(text=str(message))

        if not result_text:
            yield ProviderMessage(text="", is_result=True)

    async def send_message(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        thread_id: str | None = None,
    ) -> ProviderResult:
        """Send a lightweight message via the SDK (no tools)."""
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions
        except ImportError:
            return ProviderResult(
                text="ERROR: claude_agent_sdk not installed",
                success=False,
            )

        options = ClaudeAgentOptions(
            system_prompt="",
            allowed_tools=[],
            permission_mode="plan",
            model=model_id or self._default_model,
        )

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "result"):
                result_text = message.result

        return ProviderResult(
            text=result_text,
            success=True,
            # Claude SDK doesn't have native thread continuity,
            # so thread_id is not applicable
            thread_id=None,
        )

    def is_available(self) -> bool:
        """Check if claude CLI is installed."""
        return shutil.which("claude") is not None
