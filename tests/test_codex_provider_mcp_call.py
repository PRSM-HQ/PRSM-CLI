import json

import pytest

from prsm.engine.providers.codex_provider import CodexProvider


class _FakeReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


@pytest.mark.asyncio
async def test_mcp_call_ignores_event_notifications_until_matching_response() -> None:
    provider = CodexProvider()
    provider._mcp_reader = _FakeReader(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "codex/event",
                    "params": {"msg": {"type": "session_configured"}},
                }
            )
            + "\n",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "Fix flaky auth tests"}]
                    },
                }
            )
            + "\n",
        ]
    )
    provider._mcp_writer = _FakeWriter()

    async def _ready() -> bool:
        return True

    provider._ensure_mcp_server = _ready  # type: ignore[method-assign]

    result = await provider._mcp_call("codex", {"prompt": "hello"})

    assert result == {
        "content": [{"type": "text", "text": "Fix flaky auth tests"}]
    }
    assert provider._mcp_writer.writes


def test_codex_build_master_cmd_uses_ephemeral_sessions() -> None:
    provider = CodexProvider(command="codex")
    cmd, _, _ = provider.build_master_cmd("hello", bridge_port=12345)

    assert "exec" in cmd
    assert "--json" in cmd
    assert "--ephemeral" in cmd
