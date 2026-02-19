"""Provider abstraction for multi-engine orchestration."""
from .base import Provider, ProviderResult
from .registry import ProviderRegistry, build_provider_registry
from .claude_provider import ClaudeProvider
from .codex_provider import CodexProvider
from .gemini_provider import GeminiProvider
from .minimax_provider import MiniMaxProvider
from .alibaba_provider import AlibabaProvider

__all__ = [
    "Provider",
    "ProviderResult",
    "ProviderRegistry",
    "build_provider_registry",
    "ClaudeProvider",
    "CodexProvider",
    "GeminiProvider",
    "MiniMaxProvider",
    "AlibabaProvider",
]
