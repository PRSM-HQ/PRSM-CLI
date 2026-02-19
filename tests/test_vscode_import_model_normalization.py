from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.shared.models.session import Session
from prsm.vscode.server import PrsmServer, SessionState


class _FakeModelRegistry:
    def resolve_alias_with_provider(self, name: str) -> tuple[str, str | None]:
        aliases = {
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-5-20250929",
        }
        resolved = aliases.get(name, name)
        if resolved.startswith("claude-"):
            return resolved, "claude"
        if resolved.startswith("gpt-"):
            return resolved, "codex"
        return resolved, None


class _FakeBridge:
    def __init__(self, current_model: str = "claude-opus-4-6") -> None:
        self._current_model = current_model
        self._engine = SimpleNamespace(
            _config=SimpleNamespace(
                master_provider="claude",
                default_provider="claude",
                model_registry=_FakeModelRegistry(),
            )
        )

    @property
    def current_model(self) -> str:
        return self._current_model


def _build_server() -> PrsmServer:
    server = object.__new__(PrsmServer)
    server._cwd = str(Path.cwd())
    server._yaml_config = SimpleNamespace(
        engine=SimpleNamespace(default_provider="claude")
    )
    return server


def _build_state(bridge: _FakeBridge, session: Session | None = None) -> SessionState:
    return SessionState(
        session_id="sess-1",
        name="Imported Session",
        project_id="proj-1",
        bridge=bridge,  # type: ignore[arg-type]
        session=session or Session(),
    )


def test_normalize_placeholder_model_uses_runnable_current_model() -> None:
    server = _build_server()
    state = _build_state(_FakeBridge(current_model="claude-opus-4-6"))

    model, provider = server._normalize_agent_model_for_restart(
        state=state,
        model="imported:codex",
        provider="codex",
    )

    assert model == "claude-opus-4-6"
    assert provider == "claude"


def test_reconstitute_descriptor_normalizes_imported_placeholder_model() -> None:
    server = _build_server()
    session = Session()
    session.add_agent(
        AgentNode(
            id="root",
            name="Imported Codex",
            state=AgentState.COMPLETED,
            role=AgentRole.MASTER,
            model="imported:codex",
            provider="codex",
        )
    )
    state = _build_state(_FakeBridge(current_model="claude-opus-4-6"), session)

    desc = server._reconstitute_descriptor(state, "root")

    assert desc is not None
    assert desc.model == "claude-opus-4-6"
    assert desc.provider == "claude"
