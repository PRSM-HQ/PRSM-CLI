from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock

import pytest


def _install_fastmcp_stub() -> None:
    if "mcp.server.fastmcp" in sys.modules:
        return

    class _FakeFastMCP:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def tool(self, *args, **kwargs):
            del args, kwargs

            def _decorator(fn):
                return fn

            return _decorator

    class _FakeContext:
        request_context = types.SimpleNamespace(lifespan_context={})

    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    fastmcp_module.Context = _FakeContext
    sys.modules["mcp"] = mcp_module
    sys.modules["mcp.server"] = server_module
    sys.modules["mcp.server.fastmcp"] = fastmcp_module


_install_fastmcp_stub()

from prsm.engine.mcp_server.orch_proxy import BridgeClient


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


class _CancelingReader:
    async def readline(self) -> bytes:
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_bridge_client_call_discards_stale_response_ids() -> None:
    client = BridgeClient(port=12345)
    client._req_id = 1
    client._reader = _FakeReader([
        json.dumps(
            {
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "stale response"}]
                },
            }
        ) + "\n",
        json.dumps(
            {
                "id": 2,
                "result": {
                    "content": [{"type": "text", "text": "fresh response"}]
                },
            }
        ) + "\n",
    ])
    writer = _FakeWriter()
    client._writer = writer

    result = await client.call("wait_for_message", {"timeout_seconds": 30})

    assert result == "fresh response"
    assert writer.writes
    request = json.loads(writer.writes[0].decode("utf-8"))
    assert request["id"] == 2


@pytest.mark.asyncio
async def test_bridge_client_call_ignores_noise_until_matching_id() -> None:
    client = BridgeClient(port=12345)
    client._reader = _FakeReader([
        "not-json\n",
        json.dumps({"event": "bridge-ready"}) + "\n",
        json.dumps(
            {
                "id": 1,
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        ) + "\n",
    ])
    client._writer = _FakeWriter()

    result = await client.call("check_child_status", {"child_agent_id": "abc"})

    assert result == "ok"


@pytest.mark.asyncio
async def test_bridge_client_call_resets_connection_on_cancel() -> None:
    client = BridgeClient(port=12345)
    client._reader = _CancelingReader()
    client._writer = _FakeWriter()

    close_mock = AsyncMock()
    connect_mock = AsyncMock()
    client.close = close_mock  # type: ignore[method-assign]
    client.connect = connect_mock  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await client.call("wait_for_message", {"timeout_seconds": 30})

    close_mock.assert_awaited_once()
    connect_mock.assert_awaited_once()
