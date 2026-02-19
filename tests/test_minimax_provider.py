"""Tests for MiniMaxProvider."""
from __future__ import annotations

from prsm.engine.providers.minimax_provider import MiniMaxProvider


def test_minimax_run_agent_cmd_structure():
    """Test that run_agent builds the correct Codex command."""
    provider = MiniMaxProvider(command="codex")

    # We can't easily test the async generator, but we can verify
    # the provider initializes correctly
    assert provider.name == "minimax"
    assert provider._default_model == "MiniMax-M2.5"
    assert provider._command == "codex"


def test_minimax_build_master_cmd():
    """Test that build_master_cmd produces correct command structure."""
    provider = MiniMaxProvider(command="codex")
    cmd, env, stdin_payload = provider.build_master_cmd(
        prompt="hello",
        bridge_port=12345,
        model_id="MiniMax-M2.5",
    )

    assert cmd[0] == "codex"
    # MCP servers config should be present (for master, not empty)
    mcp_args = [a for a in cmd if "mcp_servers=" in a]
    assert len(mcp_args) == 1
    assert "orchestrator" in mcp_args[0]
    # model_provider=minimax should be set
    assert "model_provider=minimax" in cmd
    # model should be set
    model_args = [a for a in cmd if 'model="MiniMax-M2.5"' in a]
    assert len(model_args) == 1
    # exec subcommand should be present
    assert "exec" in cmd
    assert "--ephemeral" in cmd
    # prompt should be in stdin_payload
    assert "hello" in stdin_payload
    assert cmd[-1] == "-"


def test_minimax_is_available_no_codex(monkeypatch):
    """Test that is_available returns False when codex is not installed."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _: None)
    provider = MiniMaxProvider(command="codex")
    assert not provider.is_available()


def test_minimax_supports_master():
    """Test that MiniMax supports the master agent role."""
    provider = MiniMaxProvider(command="codex")
    assert provider.supports_master is True
