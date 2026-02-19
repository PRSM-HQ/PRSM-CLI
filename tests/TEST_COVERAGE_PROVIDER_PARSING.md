# Provider Parsing Test Coverage

This document summarizes the test coverage for provider-specific parsing logic in `test_provider_parsing.py`.

## Test Summary

**Total Tests:** 35
**Status:** ✅ All passing

---

## 1. Codex JSONL Parsing (`_process_codex_jsonl`)

### Tool Call Detection (item.started)
- ✅ `command_execution` → emits `tool_call_started` with `tool_name="Bash"`
- ✅ `file_edit` → emits `tool_call_started` with `tool_name="Edit"`
- ✅ `file_write` → emits `tool_call_started` with `tool_name="Write"`
- ✅ `file_read` → emits `tool_call_started` with `tool_name="Read"`
- ✅ `mcp_tool_call` → emits `tool_call_started` with MCP tool name
- ✅ MCP tool call with dict arguments → serializes to JSON

### Tool Completion (item.completed)
- ✅ `agent_message` → returns text for display
- ✅ `command_execution` → emits `tool_call_completed`
- ✅ Failed status → sets `is_error=True`

### Event Suppression
- ✅ `turn.completed` → suppressed (returns None)
- ✅ Other events → suppressed

### Error Handling
- ✅ `error` event → returns formatted error message
- ✅ Non-JSON line → returns as-is (plain text passthrough)
- ✅ Empty line → returns None

---

## 2. Gemini stream-JSON Parsing (`_process_gemini_stream_json`)

### Message Handling
- ✅ `message` with `role=assistant` → returns text for display
- ✅ Non-JSON line → suppressed (returns None)

### Tool Call Detection (tool_use)
- ✅ `tool_use` → emits `tool_call_started` with mapped tool name
- ✅ Tool name mapping for all standard tools:
  - `run_shell_command` → `Bash`
  - `read_file` → `Read`
  - `write_file` → `Write`
  - `edit_file` → `Edit`
  - `list_directory` → `Glob`
  - `search_files` → `Grep`
  - `web_search` → `WebSearch`
- ✅ Unmapped tool name → uses raw name

### Tool Completion (tool_result)
- ✅ `tool_result` → emits `tool_call_completed`
- ✅ Non-success status → sets `is_error=True`

### Event Suppression
- ✅ `init` event → suppressed (returns None)
- ✅ `result` event → suppressed (returns None)
- ✅ Empty line → returns None

---

## 3. CodexProvider Command Building

### `build_master_cmd` Method
- ✅ Includes `--json` flag for structured JSONL output
- ✅ Includes `-c` flag with MCP server config (orchestrator pointing to orch_proxy)
- ✅ MCP config includes bridge port
- ✅ Wraps prompts in XML tags (`<system_instructions>`, `<user_task>`)

### Provider Properties
- ✅ `name` property returns `"codex"`

---

## 4. MiniMaxProvider Command Building

### `build_master_cmd` Method
- ✅ Includes `--json` flag for structured JSONL output
- ✅ Uses `model_provider=minimax` in `-c` flag
- ✅ Includes MCP server config pointing to orch_proxy
- ✅ Wraps prompts in XML tags

### Provider Properties
- ✅ `name` property returns `"minimax"`

---

## 5. GeminiProvider Command Building

### `build_master_cmd` Method
- ✅ Includes `--output-format stream-json` flag
- ✅ Creates temporary settings directory with MCP config
- ✅ Sets `GEMINI_HOME` environment variable to temp directory
- ✅ Wraps prompts in XML tags

### Provider Properties
- ✅ `name` property returns `"gemini"`

---

## Test Infrastructure

### MockAgentSession
- Minimal mock setup for testing parsing methods
- Provides required attributes: `agent_id`, `_event_callback`, `_log_conversation`
- Uses `AsyncMock` for event callbacks to verify async behavior

### Test Approach
- Unit tests for individual parsing functions
- Direct method calls on `AgentSession` class methods
- Mock event callbacks to verify correct events emitted
- Tests both success and error paths
- Edge case coverage (empty lines, non-JSON, unmapped names)

---

## Coverage Notes

This test suite verifies:

1. **Event emission**: All parsing methods correctly emit `tool_call_started` and `tool_call_completed` events with proper parameters
2. **Display text extraction**: Methods correctly identify and return text that should be displayed vs. suppressed
3. **Tool name mapping**: Gemini's tool names are correctly mapped to standard PRSM tool names
4. **Command construction**: Provider `build_master_cmd` methods generate correct CLI commands with all required flags
5. **Error handling**: Failed tool calls and malformed input are handled gracefully

## Running the Tests

```bash
.venv/bin/python -m pytest tests/test_provider_parsing.py -v
```

All tests use `pytest-asyncio` for async function testing and `unittest.mock.AsyncMock` for async callback verification.
