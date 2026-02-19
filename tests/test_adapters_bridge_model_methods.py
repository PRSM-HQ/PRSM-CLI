from __future__ import annotations

from unittest.mock import MagicMock
import pytest

# Test the canonical OrchestratorBridge from orchestrator.py
from prsm.adapters.orchestrator import OrchestratorBridge
from prsm.engine.model_registry import ModelCapability, ModelRegistry, ModelTier

class MockModelCapability:
    def __init__(self, model_id, available=True, tier="frontier", provider="claude"):
        self.model_id = model_id
        self.available = available
        self.tier = tier
        self.provider = provider

class MockModelRegistry:
    def __init__(self):
        self._models = {
            "claude-opus-4-6": MockModelCapability("claude-opus-4-6", tier="frontier"),
            "claude-sonnet": MockModelCapability("claude-sonnet", tier="strong"),
            "gemini-pro": MockModelCapability("gemini-pro", provider="gemini", tier="frontier"),
            "unavailable-model": MockModelCapability("unavailable-model", available=False, tier="frontier"),
        }

    def list_models(self):
        return list(self._models.values())

    def resolve_alias(self, model_id):
        return model_id

    def is_model_available(self, model_id):
        if model_id == "unavailable-model":
            return False
        return True

    def get(self, model_id):
        return self._models.get(model_id)

    def list_aliases(self):
        return {}

class MockEngineConfig:
    def __init__(self):
        self.default_model = "claude-opus-4-6"
        self.default_provider = "claude"
        self.master_model = "claude-opus-4-6"
        self.master_provider = "claude"
        self.model_registry = MockModelRegistry()

class MockEngine:
    def __init__(self):
        self._config = MockEngineConfig()

def test_bridge_model_methods():
    bridge = OrchestratorBridge()
    # Inject mock engine manually since we skip configure()
    bridge._engine = MockEngine()

    # Test current_model — uses master_model in orchestrator.py
    assert bridge.current_model == "claude-opus-4-6"

    # Test get_available_models
    models = bridge.get_available_models()
    assert len(models) == 4

    # Verify structure matches what UI expects
    opus = next(m for m in models if m["model_id"] == "claude-opus-4-6")
    assert opus["is_current"] is True
    assert opus["available"] is True
    assert opus["tier"] == "frontier"

    sonnet = next(m for m in models if m["model_id"] == "claude-sonnet")
    assert sonnet["is_current"] is False
    assert sonnet["available"] is True

    unavailable = next(m for m in models if m["model_id"] == "unavailable-model")
    assert unavailable["available"] is False

    # Test set_model — returns (resolved_id, provider) tuple in orchestrator.py
    resolved, provider = bridge.set_model("claude-sonnet")
    assert resolved == "claude-sonnet"
    assert provider == "claude"
    assert bridge.current_model == "claude-sonnet"
    assert bridge._engine._config.master_model == "claude-sonnet"
    assert bridge._engine._config.master_provider == "claude"

    # Test set_model (Different provider)
    resolved, provider = bridge.set_model("gemini-pro")
    assert resolved == "gemini-pro"
    assert provider == "gemini"
    assert bridge.current_model == "gemini-pro"
    assert bridge._engine._config.master_model == "gemini-pro"
    assert bridge._engine._config.master_provider == "gemini"

    # Check is_current updates
    models_new = bridge.get_available_models()
    gemini_new = next(m for m in models_new if m["model_id"] == "gemini-pro")
    assert gemini_new["is_current"] is True

    # Test setting unavailable model (should allow it)
    resolved, provider = bridge.set_model("unavailable-model")
    assert resolved == "unavailable-model"
    assert bridge.current_model == "unavailable-model"
    assert bridge._engine._config.master_model == "unavailable-model"


def test_bridge_set_model_runtime_variant_provider_codex():
    bridge = OrchestratorBridge()

    registry = ModelRegistry()
    registry.register(ModelCapability(
        model_id="gpt-5-3::reasoning_effort=high",
        provider="codex",
        tier=ModelTier.FRONTIER,
        cost_factor=1.0,
        speed_factor=1.0,
        max_context=200_000,
        affinities={},
    ))

    engine = MagicMock()
    engine._config = MagicMock()
    engine._config.master_model = "claude-opus-4-6"
    engine._config.master_provider = "claude"
    engine._config.model_registry = registry
    bridge._engine = engine

    resolved, provider = bridge.set_model("gpt-5-3::reasoning_effort=high")
    assert resolved == "gpt-5-3::reasoning_effort=high"
    assert provider == "codex"
    assert bridge._engine._config.master_model == "gpt-5-3::reasoning_effort=high"
    assert bridge._engine._config.master_provider == "codex"


def test_bridge_get_available_models_uses_alias_display_name():
    bridge = OrchestratorBridge()

    registry = ModelRegistry()
    registry.register(ModelCapability(
        model_id="gpt-5-3::reasoning_effort=medium",
        provider="codex",
        tier=ModelTier.STRONG,
        cost_factor=1.0,
        speed_factor=1.0,
        max_context=200_000,
        affinities={},
    ))
    registry.register_alias(
        "gpt-5-3-medium",
        "gpt-5-3::reasoning_effort=medium",
    )

    engine = MagicMock()
    engine._config = MagicMock()
    engine._config.master_model = "gpt-5-3::reasoning_effort=medium"
    engine._config.master_provider = "codex"
    engine._config.model_registry = registry
    bridge._engine = engine

    models = bridge.get_available_models()
    entry = next(
        m for m in models
        if m["model_id"] == "gpt-5-3::reasoning_effort=medium"
    )
    assert entry["display_name"] == "gpt-5-3-medium"


def test_current_model_display_prefers_preferred_alias():
    bridge = OrchestratorBridge()
    bridge._preferred_model_alias = "spark"

    registry = ModelRegistry()
    registry.register_alias("spark", "gpt-5-3::reasoning_effort=medium")

    engine = MagicMock()
    engine._config = MagicMock()
    engine._config.master_model = "gpt-5-3::reasoning_effort=medium"
    engine._config.model_registry = registry
    bridge._engine = engine

    assert bridge.current_model_display == "spark"


def test_get_model_display_name_matches_alias_on_normalized_model_id():
    bridge = OrchestratorBridge()

    registry = ModelRegistry()
    registry.register_alias("spark", "gpt-5-3::reasoning_effort=medium")

    engine = MagicMock()
    engine._config = MagicMock()
    engine._config.model_registry = registry
    bridge._engine = engine

    assert bridge.get_model_display_name("gpt-5-3::reasoning_effort=high") == "spark"


def test_map_agent_uses_model_alias_display_name():
    bridge = OrchestratorBridge()

    registry = ModelRegistry()
    registry.register_alias("gpt-5-3-high", "gpt-5-3::reasoning_effort=high")

    engine = MagicMock()
    engine._config = MagicMock()
    engine._config.model_registry = registry
    bridge._engine = engine

    node = bridge.map_agent(
        agent_id="child-1",
        parent_id="master-1",
        role="worker",
        model="gpt-5-3::reasoning_effort=high",
        prompt="Implement change",
    )
    assert node.model == "gpt-5-3-high"
