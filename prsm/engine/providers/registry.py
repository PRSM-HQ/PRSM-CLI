"""Provider registry — maps provider names to Provider instances."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Provider

if TYPE_CHECKING:
    from ..yaml_config import ProviderConfig

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry of available execution providers.

    Maps short names (e.g. 'claude', 'codex') to Provider instances.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, name: str, provider: Provider) -> None:
        """Register a provider by name."""
        self._providers[name] = provider
        logger.info(
            "Provider registered: %s (available=%s)",
            name,
            provider.is_available(),
        )

    def get(self, name: str) -> Provider | None:
        """Get a provider by name, or None if not registered."""
        return self._providers.get(name)

    def get_or_raise(self, name: str) -> Provider:
        """Get a provider by name, raising KeyError if not found."""
        provider = self._providers.get(name)
        if provider is None:
            available = ", ".join(self._providers.keys())
            raise KeyError(
                f"Provider '{name}' not found. "
                f"Available: {available or 'none'}"
            )
        return provider

    def list_names(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers.keys())

    def list_available(self) -> list[str]:
        """Return names of providers whose runtime is installed."""
        return [
            name for name, p in self._providers.items()
            if p.is_available()
        ]

    def list_unavailable(self) -> list[str]:
        """Return names of providers whose runtime is NOT installed."""
        return [
            name for name, p in self._providers.items()
            if not p.is_available()
        ]

    def get_availability_report(self) -> dict[str, bool]:
        """Return a mapping of provider name → is_available for all providers."""
        return {
            name: p.is_available()
            for name, p in self._providers.items()
        }

    def validate(self) -> dict[str, bool]:
        """Validate all registered providers and log their availability.

        Checks is_available() on each provider and logs warnings for
        providers that are not available (CLI not installed, etc.).

        Returns:
            Dict mapping provider name → available (bool).
        """
        report = self.get_availability_report()
        available = [n for n, ok in report.items() if ok]
        unavailable = [n for n, ok in report.items() if not ok]

        if available:
            logger.info(
                "Available providers: %s", ", ".join(available)
            )
        if unavailable:
            logger.warning(
                "Unavailable providers (CLI not installed): %s",
                ", ".join(unavailable),
            )
        if not available:
            logger.error(
                "No providers are available! Agents cannot be spawned."
            )

        return report

    def is_provider_available(self, name: str) -> bool:
        """Check if a specific provider is registered AND available."""
        provider = self._providers.get(name)
        if provider is None:
            return False
        return provider.is_available()

    async def shutdown_all(self) -> None:
        """Shut down all registered providers."""
        for name, provider in self._providers.items():
            try:
                await provider.shutdown()
            except Exception as exc:
                logger.error(
                    "Error shutting down provider '%s': %s",
                    name, exc,
                )

    @property
    def count(self) -> int:
        return len(self._providers)


def build_provider_registry(
    provider_configs: dict[str, ProviderConfig] | None = None,
) -> ProviderRegistry:
    """Build a ProviderRegistry from YAML-sourced provider configs.

    If no configs provided, creates a default registry with just
    the Claude provider.

    After building, validates all providers and logs their
    availability status.
    """
    from .claude_provider import ClaudeProvider
    from .codex_provider import CodexProvider
    from .gemini_provider import GeminiProvider
    from .minimax_provider import MiniMaxProvider
    from .alibaba_provider import AlibabaProvider

    registry = ProviderRegistry()

    if not provider_configs:
        # Default: register Claude only
        registry.register("claude", ClaudeProvider())
        registry.validate()
        return registry

    for name, cfg in provider_configs.items():
        if cfg.type == "claude":
            provider = ClaudeProvider(
                api_key_env=cfg.api_key_env,
                command=cfg.command or "claude",
            )
            registry.register(name, provider)

        elif cfg.type == "codex":
            provider = CodexProvider(
                command=cfg.command or "codex",
                api_key_env=cfg.api_key_env,
            )
            registry.register(name, provider)

        elif cfg.type == "gemini":
            provider = GeminiProvider(
                command=cfg.command or "gemini",
                api_key_env=cfg.api_key_env,
            )
            registry.register(name, provider)

        elif cfg.type == "minimax":
            provider = MiniMaxProvider(
                command=cfg.command,
                api_key_env=cfg.api_key_env or "MINIMAX_API_KEY",
            )
            registry.register(name, provider)

        elif cfg.type == "alibaba":
            provider = AlibabaProvider(
                command=cfg.command or "codex",
                api_key_env=cfg.api_key_env or "DASHSCOPE_API_KEY",
                model_provider=cfg.profile or "alibaba",
            )
            registry.register(name, provider)

        else:
            logger.warning(
                "Unknown provider type '%s' for '%s' — skipping",
                cfg.type, name,
            )

    # Validate all providers and log availability
    registry.validate()

    return registry
