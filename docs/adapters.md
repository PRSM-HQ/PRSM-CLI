# Adapters Layer

The adapters layer (`prsm/adapters/`) bridges the engine to the UI frontends. It translates raw engine callbacks into typed events, manages permission/question resolution via Futures, tracks file changes, and provides UI-friendly display helpers.

## OrchestratorBridge (`orchestrator.py`)

The central adapter connecting the engine to TUI/VSCode. This is the single most important adapter class.

### Configuration
`configure(model, cwd, experts, plugin_mcp_servers, plugin_manager, project_dir, yaml_config)` creates an `EngineConfig` with three callbacks wired:
- `event_callback` → EventBus (for real-time UI updates)
- `permission_callback` → Future-based handler (blocks engine until user decides)
- `user_question_callback` → Future-based handler (blocks engine until user answers)

Also:
- Loads persisted tool permissions via `PermissionStore(project_dir)`
- Creates a `CommandPolicyStore(cwd)` for bash command whitelist/blacklist
- Builds a `ProviderRegistry` from YAML config or defaults
- Builds a `ModelRegistry` (with YAML overrides, provider availability sync, deferred Claude model probing)
- Loads `ModelIntelligence` for learned model rankings
- Resolves `peer_models` from YAML `defaults.peer_models` — maps aliases to `(provider_instance, model_id)` tuples and sets on `EngineConfig.peer_models` to restrict child agent model selection
- Creates the `OrchestrationEngine` with plugin_mcp_servers, plugin_manager, provider_registry, and initial_allowed_tools

### Running Orchestration
- `run(prompt)` — Calls `engine.run()`. On first run, performs deferred Claude model probing (async).
- `run_continuation(prompt, master_agent_id)` — Restarts a completed master agent with a follow-up prompt for seamless conversation continuation.

### Interrupt & Shutdown
- `shutdown()` — Shuts down the engine, cancels all pending futures, closes event bus.
- `interrupt()` — Interrupts current orchestration while keeping the master agent restartable. Returns the master agent ID or None. Does not close the event bus (pending agent_killed events may still arrive).

### Properties
- `available` — Whether the orchestrator can run (requires at least one provider CLI on PATH — `claude`, `codex`, or `gemini` — plus the engine importable). No longer hard-gates on the `claude` CLI alone.
- `running` — Whether an orchestration is currently in progress.
- `last_master_id` — ID of the most recently completed master agent, if any.

### Model Selection
- `current_model` — The currently configured default model ID. When the engine is running, returns `engine._config.master_model`. Otherwise falls back to `_configured_default_model` (set from YAML config during `configure()`), then finally to `"claude-opus-4-6"` as a last resort.
- `get_available_models()` — Returns a list of dicts with model details (id, provider, tier, available, is_current) for UI selection.
- `set_model(model_id)` — Updates the default model for new agents. Validates availability via the ModelRegistry.

### Permission Resolution (Future Pattern)

```
Engine agent calls a tool requiring permission
  → permission_callback fires
    → OrchestratorBridge._handle_permission()
      → Creates asyncio.Future
      → Stores in _permission_futures[request_id]
      → Emits PermissionRequest event to EventBus
      → await Future (blocks the engine agent, 300s timeout)

TUI/VSCode receives PermissionRequest event
  → Shows permission UI to user
  → User clicks Allow/Deny/Always
  → bridge.resolve_permission(request_id, result)
    → Sets Future result
    → Engine unblocks and proceeds
```

Permission results and persistence:
- `allow` — Allow once
- `allow_project` — If terminal tool (bash/shell), persists a command-class-aware pattern to `CommandPolicyStore` whitelist (strict for destructive commands, broader for exploratory commands); otherwise persists tool name to project-level `PermissionStore`
- `allow_global` — Persists tool name to global `PermissionStore`
- `deny_project` — If terminal tool, persists an exact command pattern to `CommandPolicyStore` blacklist
- `deny` — Deny once

Terminal tool detection: `_is_terminal_tool()` checks if the bare tool name (after stripping MCP prefixes via `__` split) matches bash/shell/terminal.

The same Future pattern applies to `UserQuestionRequest` for agent-to-user questions (with 7200s timeout).

### Agent Mapping
`map_agent()` converts engine agent descriptor strings to UI `AgentNode`. Maintains `agent_map: dict[str, AgentNode]` for the current session. Display name derivation via `_agent_name(role, prompt)`:
- Master → "Orchestrator"
- Expert → first 40 chars of prompt (or "Expert" if empty)
- Reviewer → first 50 chars + "..." if truncated (or "Reviewer" if empty)
- Worker → first 50 chars + "..." if truncated (or "Worker" if empty)

`map_state()` converts engine state strings to `AgentState` enum values.

### Cleanup
`cancel_agent_futures(agent_id)` cancels orphaned Futures when agents die/fail/are killed to prevent memory leaks. Tracks agent-to-request-id mappings via `_agent_permission_requests` and `_agent_question_requests`.

## EventBus (`event_bus.py`)

Async queue bridging engine callbacks to UI event consumers.

- `make_callback()` — Returns the async `_callback` function to pass to `EngineConfig.event_callback`.
- `_callback(data: dict)` — Converts dicts to typed events, puts on queue.
- `consume()` → `AsyncIterator[OrchestratorEvent]` — Yields events as they arrive.
- `emit(event)` — Manually emit events (for bridge-generated events like PermissionRequest).
- `close()` — Stop the consumer loop permanently.
- `reset()` — Drain leftover events and re-open for a new orchestration run.

Queue: maxsize=5000, 30s timeout backpressure, 0.5s polling interval.

## Event Types (`events.py`)

19 typed event dataclasses:

| Event | When Fired |
|-------|------------|
| `EngineStarted` | Orchestration begins |
| `EngineFinished` | Orchestration ends (includes success, summary, error, duration_seconds) |
| `AgentSpawned` | New agent created |
| `AgentStateChanged` | Agent state transition |
| `AgentRestarted` | Agent restarted with new prompt |
| `AgentKilled` | Agent force-killed |
| `StreamChunk` | LLM streaming token |
| `ToolCallStarted` | Tool execution begins |
| `ToolCallCompleted` | Tool execution ends |
| `AgentResult` | Agent finished |
| `PermissionRequest` | Tool needs user approval (includes message_index) |
| `UserQuestionRequest` | Agent asks user a question (includes options list) |
| `Thinking` | Extended thinking content |
| `UserPrompt` | User prompt logged |
| `ContextWindowUsage` | Per-turn token usage (input, cached_input, output, total, max_context, percent_used) |
| `MessageRouted` | Inter-agent message sent |
| `FileChanged` | File modification detected (includes pre_tool_content, added/removed ranges) |
| `SnapshotCreated` | Snapshot created |
| `SnapshotRestored` | Snapshot restored |

Conversion functions:
- `dict_to_event(data)` — Engine dict → typed event (maps "event" key to "event_type" field, filters to valid dataclass fields)
- `event_to_dict(event)` — Typed event → dict (maps "event_type" back to "event" key for JSON/SSE transmission)

## Agent Adapter (`agent_adapter.py`)

UI display helpers without losing engine state fidelity.

- `STATE_ICONS` — Maps 9 AgentState values to (icon, color) tuples
- `STATE_DISPLAY` — Full display strings with icon + state + context: "▶ Running", "⏳ Waiting (child)", "⏳ Waiting (parent)", "⏳ Waiting (expert)", etc.
- `ROLE_DISPLAY` — Emoji-prefixed labels: 👑 Orchestrator, ⛏ Worker, 🎓 Expert, 🔍 Reviewer
- `STALE_STATES` — Transient states (`RUNNING`, `WAITING_FOR_PARENT`, `WAITING_FOR_CHILD`, `WAITING_FOR_EXPERT`, `STARTING`) reset to COMPLETED on session restore
- `parse_state()` — Backward-compatible parsing: tries direct enum match, then legacy mapping ("idle"→PENDING, "waiting"→WAITING_FOR_CHILD, "error"→FAILED), fallback PENDING
- `parse_role()` — Backward-compatible parsing: tries direct enum match, then legacy mapping ("orchestrator"→MASTER), fallback WORKER

### AgentAdapter Class
`AgentAdapter.to_ui_node(descriptor)` converts engine `AgentDescriptor` to UI `AgentNode` with full fidelity. Derives display name via `_derive_name()` with role-specific truncation (same char limits as bridge `_agent_name()`).

## File Tracker (`file_tracker.py`)

Detects and records file modifications from agent Write/Edit tool calls.

### FileChangeRecord
```python
@dataclass
class FileChangeRecord:
    file_path: str
    agent_id: str
    change_type: str           # "create" | "modify"
    tool_call_id: str
    tool_name: str
    message_index: int
    old_content: str | None    # For Edit: the old_string; for Write modify: full pre-tool content
    new_content: str | None    # For Edit: the new_string; for Write: full new file content
    pre_tool_content: str | None  # Full file content before the tool ran
    added_ranges: list         # [{"startLine": int, "endLine": int}]
    removed_ranges: list       # [{"startLine": int, "endLine": int}]
    timestamp: str             # ISO format
    status: str                # "pending" | "accepted" | "rejected"
```

Includes `to_dict()` and `from_dict()` for JSON serialization.

### FileChangeTracker Workflow
1. `capture_pre_tool(tool_id, tool_name, arguments)` — Called when Write/Edit starts. Parses arguments (JSON or Python repr fallback), reads file content before modification.
2. `track_change(agent_id, tool_id, message_index)` — Called when Write/Edit completes. Reads new content, computes line-level diffs. For Edit: finds old_string position and computes precise line ranges. For Write: uses `_compute_line_ranges()` for full-file diff.
3. `persist(changes_dir)` — Saves records as individual `{tool_call_id}.json` files via `atomic_write_text()`. Deletes stale on-disk files not in current in-memory state.
4. `load(changes_dir)` — Loads records from `{tool_call_id}.json` files, deduplicates by tool_call_id.
5. `clear()` — Resets all internal state.

## Permission Store (`permission_store.py`)

Persists tool permission decisions at two scopes:
- **Global:** `~/.prsm/allowed_tools.json` (all projects)
- **Project:** `~/.prsm/projects/{ID}/allowed_tools.json` (per-project)

Methods:
- `load()` — Returns merged set of allowed tools (global + project)
- `add_project(tool_name)` — Adds to project-level list (falls back to global if no project context)
- `add_global(tool_name)` — Adds to global list

Storage format: JSON array of tool name strings, sorted.

## Session Naming (`session_naming.py`)

Generates descriptive 3-7 word session titles from user prompts using a lightweight LLM call (`claude-sonnet-4-5-20250929`, 15s timeout). Truncates user message to 500 chars, limits output to 60 chars. Returns None on failure (non-blocking). Uses `ClaudeProvider.send_message()` directly.

Note: Identical implementation exists in `prsm/shared/services/session_naming.py`.
