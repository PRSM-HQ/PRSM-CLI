"""Claude Orchestrator — Hierarchical multi-agent orchestration for Claude Agent SDK."""
from .models import (
    AgentDescriptor,
    AgentResult,
    AgentRole,
    AgentState,
    EgoDecisionReport,
    ExpertProfile,
    ExpertOutput,
    MessageType,
    PermissionMode,
    RoutedMessage,
    SpawnRequest,
    VerificationResult,
)
from .config import EngineConfig
from .resource_manager import CircuitBreakerState, ResourceBudget, ResourceManager
from .errors import (
    AgentSpawnError,
    AgentTimeoutError,
    DeadlockDetectedError,
    ExpertNotFoundError,
    MaxDepthExceededError,
    MessageRoutingError,
    ModelNotAvailableError,
    OrchestrationError,
    ProviderNotAvailableError,
    ToolCallTimeoutError,
)

__all__ = [
    # Core engine (lazy import to avoid circular deps)
    "OrchestrationEngine",
    # Models
    "AgentDescriptor",
    "AgentResult",
    "AgentRole",
    "AgentState",
    "EgoDecisionReport",
    "ExpertProfile",
    "ExpertOutput",
    "MessageType",
    "PermissionMode",
    "RoutedMessage",
    "SpawnRequest",
    "VerificationResult",
    # Config
    "EngineConfig",
    "ResourceBudget",
    "CircuitBreakerState",
    "ResourceManager",
    "TelemetryCollector",
    "ShadowTriageModel",
    # YAML config (lazy import)
    "OrchestrationConfig",
    "load_yaml_config",
    # Providers (lazy import)
    "Provider",
    "ProviderRegistry",
    "ClaudeProvider",
    "CodexProvider",
    "MiniMaxProvider",
    # Model registry (lazy import)
    "ModelRegistry",
    "ModelCapability",
    "build_default_registry",
    # Model intelligence (lazy import)
    "ModelIntelligence",
    # Multi-project runtime primitives (lazy import)
    "ProjectRegistry",
    "ProjectRuntime",
    "CrossProjectEventBroker",
    "CrossProjectEvent",
    # Errors
    "AgentSpawnError",
    "AgentTimeoutError",
    "DeadlockDetectedError",
    "ExpertNotFoundError",
    "MaxDepthExceededError",
    "MessageRoutingError",
    "ModelNotAvailableError",
    "OrchestrationError",
    "ProviderNotAvailableError",
    "ToolCallTimeoutError",
]


def __getattr__(name: str):
    if name == "OrchestrationEngine":
        from .engine import OrchestrationEngine
        return OrchestrationEngine
    if name == "OrchestrationConfig":
        from .yaml_config import OrchestrationConfig
        return OrchestrationConfig
    if name == "load_yaml_config":
        from .yaml_config import load_yaml_config
        return load_yaml_config
    if name == "Provider":
        from .providers.base import Provider
        return Provider
    if name == "ProviderRegistry":
        from .providers.registry import ProviderRegistry
        return ProviderRegistry
    if name == "ClaudeProvider":
        from .providers.claude_provider import ClaudeProvider
        return ClaudeProvider
    if name == "CodexProvider":
        from .providers.codex_provider import CodexProvider
        return CodexProvider
    if name == "MiniMaxProvider":
        from .providers.minimax_provider import MiniMaxProvider
        return MiniMaxProvider
    if name == "ModelRegistry":
        from .model_registry import ModelRegistry
        return ModelRegistry
    if name == "ModelCapability":
        from .model_registry import ModelCapability
        return ModelCapability
    if name == "build_default_registry":
        from .model_registry import build_default_registry
        return build_default_registry
    if name == "ModelIntelligence":
        from .model_intelligence import ModelIntelligence
        return ModelIntelligence
    if name == "ProjectRegistry":
        from .project_registry import ProjectRegistry
        return ProjectRegistry
    if name == "ProjectRuntime":
        from .project_registry import ProjectRuntime
        return ProjectRuntime
    if name == "CrossProjectEventBroker":
        from .project_registry import CrossProjectEventBroker
        return CrossProjectEventBroker
    if name == "CrossProjectEvent":
        from .project_registry import CrossProjectEvent
        return CrossProjectEvent
    if name == "TelemetryCollector":
        from .telemetry import TelemetryCollector
        return TelemetryCollector
    if name == "ShadowTriageModel":
        from .shadow_triage import ShadowTriageModel
        return ShadowTriageModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
