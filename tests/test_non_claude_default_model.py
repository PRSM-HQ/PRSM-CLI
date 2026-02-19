"""Tests for non-Claude default model handling.

Verifies that PRSM works correctly when:
1. Claude CLI is NOT installed but another provider (Codex/Gemini) is
2. A non-Claude model is set as default in YAML config
3. The model selector shows available non-Claude models
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from prsm.adapters.orchestrator import OrchestratorBridge
from prsm.engine.model_registry import ModelCapability, ModelRegistry, ModelTier


# ── OrchestratorBridge.available ──────────────────────────────────


class TestBridgeAvailableWithoutClaude:
    """The available property should work with any installed provider CLI."""

    @patch("prsm.adapters.orchestrator.shutil.which")
    def test_available_with_only_codex(self, mock_which):
        """When only codex CLI is installed, available should be True."""
        def which_side_effect(name):
            return "/usr/bin/codex" if name == "codex" else None
        mock_which.side_effect = which_side_effect

        bridge = OrchestratorBridge()
        assert bridge.available is True

    @patch("prsm.adapters.orchestrator.shutil.which")
    def test_available_with_only_gemini(self, mock_which):
        """When only gemini CLI is installed, available should be True."""
        def which_side_effect(name):
            return "/usr/bin/gemini" if name == "gemini" else None
        mock_which.side_effect = which_side_effect

        bridge = OrchestratorBridge()
        assert bridge.available is True

    @patch("prsm.adapters.orchestrator.shutil.which")
    def test_available_with_only_claude(self, mock_which):
        """When only claude CLI is installed, available should be True (backward compat)."""
        def which_side_effect(name):
            return "/usr/bin/claude" if name == "claude" else None
        mock_which.side_effect = which_side_effect

        bridge = OrchestratorBridge()
        assert bridge.available is True

    @patch("prsm.adapters.orchestrator.shutil.which")
    def test_not_available_with_no_providers(self, mock_which):
        """When no provider CLI is installed, available should be False."""
        mock_which.return_value = None

        bridge = OrchestratorBridge()
        assert bridge.available is False

    @patch("prsm.adapters.orchestrator.shutil.which")
    def test_available_with_multiple_providers(self, mock_which):
        """When multiple provider CLIs are installed, available should be True."""
        def which_side_effect(name):
            if name in ("codex", "gemini"):
                return f"/usr/bin/{name}"
            return None
        mock_which.side_effect = which_side_effect

        bridge = OrchestratorBridge()
        assert bridge.available is True


# ── current_model fallback ────────────────────────────────────────


class TestCurrentModelFallback:
    """current_model should respect YAML-configured default, not hardcode Claude."""

    def test_current_model_no_engine_no_config(self):
        """Without engine and no config, falls back to claude-opus-4-6."""
        bridge = OrchestratorBridge()
        assert bridge.current_model == "claude-opus-4-6"

    def test_current_model_no_engine_with_configured_default(self):
        """Without engine but with a configured default, uses that default."""
        bridge = OrchestratorBridge()
        bridge._configured_default_model = "gpt-5-3"
        assert bridge.current_model == "gpt-5-3"

    def test_current_model_with_engine_ignores_configured_default(self):
        """With a running engine, uses engine's master_model, not the fallback."""
        bridge = OrchestratorBridge()
        bridge._configured_default_model = "gpt-5-3"

        engine = MagicMock()
        engine._config = MagicMock()
        engine._config.master_model = "claude-opus-4-6"
        bridge._engine = engine

        assert bridge.current_model == "claude-opus-4-6"


# ── Model selector with non-Claude models ─────────────────────────


class TestModelSelectorNonClaude:
    """Model selector should show available non-Claude models."""

    def test_get_available_models_shows_codex_models(self):
        """When Codex is available, its models should appear as available."""
        bridge = OrchestratorBridge()

        registry = ModelRegistry()
        # Claude model — unavailable
        registry.register(ModelCapability(
            model_id="claude-opus-4-6",
            provider="claude",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        # Codex model — available
        registry.register(ModelCapability(
            model_id="gpt-5-3",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        # Sync availability: Claude unavailable, Codex available
        registry.sync_availability({"claude": False, "codex": True})

        engine = MagicMock()
        engine._config = MagicMock()
        engine._config.master_model = "gpt-5-3"
        engine._config.model_registry = registry
        bridge._engine = engine

        models = bridge.get_available_models()
        assert len(models) == 2

        # Codex model should be available and current
        codex_model = next(m for m in models if m["model_id"] == "gpt-5-3")
        assert codex_model["available"] is True
        assert codex_model["is_current"] is True
        assert codex_model["provider"] == "codex"

        # Claude model should be unavailable
        claude_model = next(m for m in models if m["model_id"] == "claude-opus-4-6")
        assert claude_model["available"] is False

    def test_get_available_models_sorted_available_first(self):
        """Available models should sort before unavailable ones."""
        bridge = OrchestratorBridge()

        registry = ModelRegistry()
        registry.register(ModelCapability(
            model_id="claude-opus-4-6",
            provider="claude",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        registry.register(ModelCapability(
            model_id="gpt-5-3",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        registry.sync_availability({"claude": False, "codex": True})

        engine = MagicMock()
        engine._config = MagicMock()
        engine._config.master_model = "gpt-5-3"
        engine._config.model_registry = registry
        bridge._engine = engine

        models = bridge.get_available_models()
        # Available models should come first
        assert models[0]["available"] is True
        assert models[0]["model_id"] == "gpt-5-3"


# ── Provider registry availability ────────────────────────────────


class TestProviderRegistryAvailability:
    """Provider availability should correctly reflect CLI installation state."""

    def test_sync_availability_marks_claude_unavailable(self):
        """When Claude provider reports unavailable, Claude models are marked unavailable."""
        registry = ModelRegistry()
        registry.register(ModelCapability(
            model_id="claude-opus-4-6",
            provider="claude",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        registry.register(ModelCapability(
            model_id="gpt-5-3",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))

        # Before sync, all models default to available=True
        assert registry.get("claude-opus-4-6").available is True
        assert registry.get("gpt-5-3").available is True

        # Sync: Claude unavailable, Codex available
        changed = registry.sync_availability({"claude": False, "codex": True})

        assert registry.get("claude-opus-4-6").available is False
        assert registry.get("gpt-5-3").available is True

    def test_list_available_returns_only_available_models(self):
        """list_available() should only return models whose provider is available."""
        registry = ModelRegistry()
        registry.register(ModelCapability(
            model_id="claude-opus-4-6",
            provider="claude",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        registry.register(ModelCapability(
            model_id="gpt-5-3",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        registry.sync_availability({"claude": False, "codex": True})

        available = registry.list_available()
        model_ids = [m.model_id for m in available]
        assert "gpt-5-3" in model_ids
        assert "claude-opus-4-6" not in model_ids


# ── YAML config default model resolution ──────────────────────────


class TestYamlConfigDefaultModel:
    """YAML-configured default model should be respected throughout the stack."""

    def test_configured_default_model_persists_across_bridge_lifecycle(self):
        """_configured_default_model set during configure() persists."""
        bridge = OrchestratorBridge()
        assert bridge._configured_default_model is None
        assert bridge.current_model == "claude-opus-4-6"

        # Simulate what configure() does when YAML sets master_model
        bridge._configured_default_model = "gpt-5-3"
        assert bridge.current_model == "gpt-5-3"

    def test_engine_master_model_overrides_configured_default(self):
        """When engine is running, its master_model takes priority."""
        bridge = OrchestratorBridge()
        bridge._configured_default_model = "gpt-5-3"

        engine = MagicMock()
        engine._config = MagicMock()
        engine._config.master_model = "gemini-3-pro-preview"
        bridge._engine = engine

        # Engine's model wins over configured default
        assert bridge.current_model == "gemini-3-pro-preview"
