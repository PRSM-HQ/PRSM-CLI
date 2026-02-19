"""Tests for GeminiProvider."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from prsm.engine.providers.gemini_provider import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_name():
    provider = GeminiProvider()
    assert provider.name == "gemini"


@pytest.mark.asyncio
async def test_gemini_provider_is_available():
    provider = GeminiProvider()
    with patch("shutil.which", return_value="/usr/bin/gemini"):
        assert provider.is_available() is True
    with patch("shutil.which", return_value=None):
        assert provider.is_available() is False


@pytest.mark.asyncio
async def test_gemini_provider_send_message_success():
    provider = GeminiProvider(command="gemini")
    
    mock_response = {
        "session_id": "test-session-id",
        "response": "Hello from Gemini!",
        "stats": {"tokens": 100}
    }
    stdout = b"Loaded credentials...\n" + json.dumps(mock_response).encode("utf-8")
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (stdout, b"")
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await provider.send_message("hi")
        
        assert result.success is True
        assert result.text == "Hello from Gemini!"
        assert result.thread_id == "test-session-id"
        assert result.metadata == {"tokens": 100}


@pytest.mark.asyncio
async def test_gemini_provider_send_message_resume():
    provider = GeminiProvider(command="gemini")
    
    mock_response = {
        "session_id": "test-session-id",
        "response": "Resumed response",
    }
    stdout = json.dumps(mock_response).encode("utf-8")
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (stdout, b"")
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await provider.send_message("hi again", thread_id="test-session-id")
        
        assert result.success is True
        assert result.text == "Resumed response"
        
        # Verify --resume was passed
        args, kwargs = mock_exec.call_args
        assert "--resume" in args
        assert "test-session-id" in args


@pytest.mark.asyncio
async def test_gemini_provider_run_agent_streaming():
    provider = GeminiProvider(command="gemini")
    
    mock_proc = AsyncMock()
    mock_proc.stdout.readline.side_effect = [
        b"Thought: I should list files.\n",
        b"Action: list_files\n",
        b"Final Result: done\n",
        b""
    ]
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        messages = []
        async for msg in provider.run_agent("do something"):
            messages.append(msg)
            
        assert len(messages) >= 3
        assert any("Thought" in m.text for m in messages)
        assert any("Final Result" in m.text for m in messages)
        assert messages[-1].is_result is True
