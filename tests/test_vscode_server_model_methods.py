"""Tests for VSCode server model selection endpoints.

Tests the following endpoints in prsm/vscode/server.py:
- GET /sessions/{id}/models - Get available models for a session
- POST /sessions/{id}/model - Set the default model for a session
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from prsm.vscode.server import PrsmServer, SessionState
from prsm.shared.models.session import Session


class MockModelCapability:
    """Mock ModelCapability for testing."""

    def __init__(self, model_id, available=True, tier="frontier", provider="claude"):
        self.model_id = model_id
        self.available = available
        self.tier = tier
        self.provider = provider


class MockModelRegistry:
    """Mock ModelRegistry that provides test model data."""

    def list_models(self):
        return [
            MockModelCapability("claude-opus-4-6", tier="frontier"),
            MockModelCapability("claude-sonnet-4-5-20250929", tier="strong"),
            MockModelCapability("gemini-pro", provider="gemini", tier="frontier"),
            MockModelCapability("unavailable-model", available=False, tier="economy"),
        ]

    def resolve_alias(self, model_id):
        """Resolve model aliases to their full IDs."""
        aliases = {
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-5-20250929",
        }
        return aliases.get(model_id, model_id)

    def is_model_available(self, model_id):
        """Check if a model is available."""
        if model_id == "unavailable-model":
            return False
        return True

    def get(self, model_id):
        """Get model capability by ID."""
        resolved = self.resolve_alias(model_id)
        for cap in self.list_models():
            if cap.model_id == resolved:
                return cap
        return None


class MockEngineConfig:
    """Mock EngineConfig for testing."""

    def __init__(self):
        self.default_model = "claude-opus-4-6"
        self.default_provider = "claude"
        self.master_model = "claude-sonnet-4-5-20250929"
        self.master_provider = "claude"
        self.model_registry = MockModelRegistry()


class MockEngine:
    """Mock OrchestrationEngine for testing."""

    def __init__(self):
        self._config = MockEngineConfig()


class MockBridge:
    """Mock OrchestratorBridge that implements model methods."""

    def __init__(self):
        self._engine = MockEngine()

    @property
    def current_model(self):
        """Return the current model ID."""
        return self._engine._config.default_model

    def get_available_models(self):
        """Get list of available models with metadata."""
        if self._engine is None or self._engine._config is None:
            return []

        registry = self._engine._config.model_registry
        models = registry.list_models()
        current = self.current_model

        result = []
        for cap in models:
            result.append({
                "model_id": cap.model_id,
                "provider": cap.provider,
                "tier": cap.tier,
                "available": cap.available,
                "is_current": cap.model_id == current,
            })
        return result

    def set_model(self, model_id: str) -> tuple[str, str]:
        """Set the default model for this session. Returns (resolved_id, provider)."""
        if self._engine is None or self._engine._config is None:
            return model_id, "claude"

        registry = self._engine._config.model_registry
        resolved_id = registry.resolve_alias(model_id)
        provider = "claude"

        # Update the config
        self._engine._config.default_model = resolved_id
        self._engine._config.master_model = resolved_id

        # Update provider if we can determine it
        cap = registry.get(resolved_id)
        if cap:
            provider = cap.provider
            self._engine._config.default_provider = cap.provider
            self._engine._config.master_provider = cap.provider

        return resolved_id, provider


class TestVSCodeServerModelMethods(AioHTTPTestCase):
    """Test VSCode server model selection endpoints."""

    async def get_application(self):
        """Create a test server instance."""
        # Use a temporary directory for the server
        self.tmpdir = tempfile.mkdtemp()

        # Patch SessionPersistence to avoid disk I/O
        with patch("prsm.vscode.server.SessionPersistence"):
            self.server = PrsmServer(
                cwd=self.tmpdir,
                model="claude-opus-4-6",
            )

            # Create a test session with our mock bridge
            self.session_id = "test-session-123"
            self.mock_bridge = MockBridge()

            # Create a minimal session state
            session = Session(
                agents={},
                messages={},
                active_agent_id=None,
                name="Test Session",
                created_at=None,
                forked_from=None,
                worktree=None,
            )

            state = SessionState(
                session_id=self.session_id,
                name="Test Session",
                project_id="test-project",
                bridge=self.mock_bridge,
                session=session,
            )

            # Inject the session into the server
            self.server._sessions[self.session_id] = state

            return self.server._app

    async def test_get_models_success(self):
        """Test GET /sessions/{id}/models returns available models."""
        resp = await self.client.get(f"/sessions/{self.session_id}/models")
        assert resp.status == 200

        data = await resp.json()
        assert "models" in data
        models = data["models"]

        # Should have 4 models from our mock
        assert len(models) == 4

        # Find specific models and verify structure
        opus = next((m for m in models if m["model_id"] == "claude-opus-4-6"), None)
        assert opus is not None
        assert opus["provider"] == "claude"
        assert opus["tier"] == "frontier"
        assert opus["available"] is True
        assert opus["is_current"] is True  # This is the current model

        sonnet = next((m for m in models if m["model_id"] == "claude-sonnet-4-5-20250929"), None)
        assert sonnet is not None
        assert sonnet["is_current"] is False
        assert sonnet["available"] is True

        gemini = next((m for m in models if m["model_id"] == "gemini-pro"), None)
        assert gemini is not None
        assert gemini["provider"] == "gemini"
        assert gemini["tier"] == "frontier"
        assert gemini["available"] is True

        unavailable = next((m for m in models if m["model_id"] == "unavailable-model"), None)
        assert unavailable is not None
        assert unavailable["available"] is False
        assert unavailable["tier"] == "economy"

    async def test_get_models_session_not_found(self):
        """Test GET /sessions/{id}/models with invalid session ID."""
        resp = await self.client.get("/sessions/nonexistent-session/models")
        assert resp.status == 404

        data = await resp.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    async def test_set_model_success(self):
        """Test POST /sessions/{id}/model successfully sets the model."""
        # Verify initial model
        assert self.mock_bridge.current_model == "claude-opus-4-6"

        # Set to a different model
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": "claude-sonnet-4-5-20250929"}
        )
        assert resp.status == 200

        data = await resp.json()
        assert data["status"] == "ok"
        assert data["model_id"] == "claude-sonnet-4-5-20250929"
        assert data["provider"] == "claude"
        assert data["old_model"] == "claude-opus-4-6"

        # Verify the bridge was updated
        assert self.mock_bridge.current_model == "claude-sonnet-4-5-20250929"
        assert self.mock_bridge._engine._config.master_model == "claude-sonnet-4-5-20250929"

    async def test_set_model_with_alias(self):
        """Test POST /sessions/{id}/model with an alias resolves correctly."""
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": "sonnet"}
        )
        assert resp.status == 200

        data = await resp.json()
        assert data["status"] == "ok"
        # Should resolve the alias
        assert data["model_id"] == "claude-sonnet-4-5-20250929"
        assert data["provider"] == "claude"
        assert self.mock_bridge.current_model == "claude-sonnet-4-5-20250929"

    async def test_set_model_different_provider(self):
        """Test POST /sessions/{id}/model updates provider when changing to different provider."""
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": "gemini-pro"}
        )
        assert resp.status == 200

        data = await resp.json()
        assert data["status"] == "ok"
        assert data["model_id"] == "gemini-pro"
        assert data["provider"] == "gemini"

        # Verify provider was updated
        assert self.mock_bridge._engine._config.default_provider == "gemini"
        assert self.mock_bridge._engine._config.master_provider == "gemini"

    async def test_set_model_missing_model_id(self):
        """Test POST /sessions/{id}/model without model_id returns 400."""
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={}
        )
        assert resp.status == 400

        data = await resp.json()
        assert "error" in data
        assert "required" in data["error"].lower()

    async def test_set_model_empty_model_id(self):
        """Test POST /sessions/{id}/model with empty model_id returns 400."""
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": ""}
        )
        assert resp.status == 400

        data = await resp.json()
        assert "error" in data

    async def test_set_model_whitespace_model_id(self):
        """Test POST /sessions/{id}/model with whitespace-only model_id returns 400."""
        resp = await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": "   "}
        )
        assert resp.status == 400

        data = await resp.json()
        assert "error" in data

    async def test_set_model_session_not_found(self):
        """Test POST /sessions/{id}/model with invalid session ID."""
        resp = await self.client.post(
            "/sessions/nonexistent-session/model",
            json={"model_id": "claude-sonnet-4-5-20250929"}
        )
        assert resp.status == 404

        data = await resp.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    async def test_set_model_updates_is_current_flag(self):
        """Test that setting a model updates the is_current flag in get_available_models."""
        # Set to sonnet
        await self.client.post(
            f"/sessions/{self.session_id}/model",
            json={"model_id": "claude-sonnet-4-5-20250929"}
        )

        # Get models and verify is_current updated
        resp = await self.client.get(f"/sessions/{self.session_id}/models")
        data = await resp.json()
        models = data["models"]

        opus = next((m for m in models if m["model_id"] == "claude-opus-4-6"), None)
        assert opus["is_current"] is False

        sonnet = next((m for m in models if m["model_id"] == "claude-sonnet-4-5-20250929"), None)
        assert sonnet["is_current"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
