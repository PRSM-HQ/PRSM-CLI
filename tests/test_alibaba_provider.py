"""Tests for AlibabaProvider."""
from __future__ import annotations

from prsm.engine.providers.alibaba_provider import AlibabaProvider


def test_alibaba_build_master_cmd_uses_ephemeral_sessions() -> None:
    provider = AlibabaProvider(command="codex")
    cmd, _, stdin_payload = provider.build_master_cmd(
        prompt="hello",
        bridge_port=12345,
        model_id="qwen3.5-plus",
    )

    assert cmd[0] == "codex"
    assert "exec" in cmd
    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert stdin_payload is not None
    assert "hello" in stdin_payload
