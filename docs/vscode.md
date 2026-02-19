# VSCode Integration

The VSCode layer (`prsm/vscode/`) provides an HTTP+SSE server that enables the VS Code extension to interact with the PRSM orchestration engine.

## PrsmServer (`server.py`)

A multi-session `aiohttp` web server with real-time event streaming. ~2300 lines of parallel implementation to the TUI — same engine, same events, different transport.

### Architecture

```
VS Code Extension
    ↕ HTTP + SSE
PrsmServer (aiohttp)
    ↕ OrchestratorBridge (per session)
OrchestrationEngine
```

### Session Management

Each session gets its own:
- `OrchestratorBridge` — Engine connection
- `Session` — State container
- `FileChangeTracker` — File change detection
- Event consumer task — Background event processing
- Per-agent inject/queue state — For prompt injection delivery modes

`_sessions: dict[str, SessionState]` — Hot sessions with active bridges.
`_session_index: dict[str, dict]` — Metadata index of persisted sessions for lazy loading.

The VS Code extension also persists open chat tab state (`session` and `agent` webviews) in workspace state. After reconnect/startup, once session trees are rehydrated from the server, those chat tabs are reopened with `preserveFocus` so editor focus is not stolen.

### SSE Event Streaming

`GET /events` — Server-Sent Events endpoint for real-time updates.
- Fan-out to all connected clients via `_sse_queues` (max 5000 queued events per client)
- 30s keepalive heartbeat
- Initial `connected` event with list of session IDs
- All orchestrator events are broadcast as SSE messages

### REST API

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Health | `GET /health` | Server health check |
| SSE | `GET /events` | Real-time event stream |
| Sessions | `GET /sessions` | List all sessions (hot + indexed) |
| | `POST /sessions` | Create new session |
| | `POST /sessions/{id}/fork` | Fork a session |
| | `DELETE /sessions/{id}` | Delete a session |
| | `PATCH /sessions/{id}` | Rename session |
| | `POST /sessions/{id}/model` | Set session model |
| | `GET /sessions/{id}/models` | List available models |
| Agents | `GET /sessions/{id}/agents` | List agents in session |
| | `GET /sessions/{id}/agents/{aid}/messages` | Get agent messages |
| Transcript Import | `GET /import/sessions` | List importable sessions from external providers |
| | `GET /import/preview` | Preview a provider session transcript |
| | `POST /sessions/{id}/import` | Import provider transcript into existing PRSM session |
| Slash Commands | `POST /sessions/{id}/command` | Execute slash commands in VS Code chat path |
| Orchestration | `POST /sessions/{id}/run` | Start orchestration run |
| | `POST /sessions/{id}/resolve-permission` | Resolve tool permission |
| | `POST /sessions/{id}/resolve-question` | Resolve user question |
| Agent Control | `POST /sessions/{id}/agents/{aid}/message` | Send message to agent |
| | `POST /sessions/{id}/kill-agent` | Kill an agent |
| | `POST /sessions/{id}/shutdown` | Shutdown session |
| | `POST /sessions/{id}/cancel-latest-tool-call` | Cancel the latest pending tool call |
| Prompt Injection | `POST /sessions/{id}/stop-after-tool` | Stop orchestration after current tool |
| | `POST /sessions/{id}/agents/{aid}/inject-prompt` | Inject prompt to agent (mode: interrupt/inject/queue) |
| Persistence | `POST /sessions/{id}/save` | Save session |
| | `GET /sessions/restore` | List restorable sessions |
| | `POST /sessions/restore/{name}` | Restore a session |
| Snapshots | `GET /sessions/{id}/snapshots` | List snapshots |
| | `POST /sessions/{id}/snapshots` | Create snapshot |
| | `POST /sessions/{id}/snapshots/{snap}/restore` | Restore snapshot |
| | `POST /sessions/{id}/snapshots/{snap}/fork` | Fork session from snapshot |
| | `DELETE /sessions/{id}/snapshots/{snap}` | Delete snapshot |
| File Changes | `GET /sessions/{id}/file-changes` | List file changes |
| | `POST /sessions/{id}/file-changes/{tool_call_id}/accept` | Accept file change |
| | `POST /sessions/{id}/file-changes/{tool_call_id}/reject` | Reject file change |
| | `POST /sessions/{id}/file-changes/accept-all` | Accept all file changes |
| | `POST /sessions/{id}/file-changes/reject-all` | Reject all file changes |

See [file-change-tracking.md](file-change-tracking.md) for the full design of the file change tracking and workspace sync system.
| Files | `GET /files/complete` | File autocomplete |
| History | `GET /api/sessions/{id}/agents/{aid}/history` | Agent conversation history |
| | `GET /api/sessions/{id}/agents/{aid}/tool-rationale/{tool_call_id}` | Tool call rationale |
| Config | `GET /config` | Get current YAML config (providers, models, registry) |
| | `PUT /config` | Update YAML config |
| | `GET /config/detect-providers` | Detect available providers and models |

### Prompt Injection (3 Modes)

Per-agent delivery via `POST /sessions/{id}/agents/{aid}/inject-prompt` with `mode` field:
1. **Interrupt** — Kill agent, add user message, rebuild conversation context, restart agent immediately
2. **Inject** — Set per-agent flag in `_inject_after_tool_agents`; on next `ToolCallCompleted` for that agent: kill + restart with injected prompt
3. **Queue** — Append to `_agent_prompt_queue`; deferred restart on `AgentResult` after agent completes

Auto-snapshots are created before each inject/message operation.

### Auto-Features
- **Auto-snapshot** — Creates snapshot before each orchestration run and before inject/message operations
- **Auto-name** — LLM-generated session titles (only for default-named sessions)
- **30s autosave** — Background periodic save loop (also triggers inactive session unloading)
- **Lazy session indexing** — On startup, builds metadata index from persisted sessions without loading bridges; sessions are loaded on first access
- **Inactivity unloading** — Idle sessions are saved and unloaded after configurable timeout (default 15 min, set via `PRSM_SESSION_INACTIVITY_MINUTES`)

### Plan Writing
`_write_plan_chunk()` appends master agent stream chunks to `plans/{session_name}.md` in the working directory.

### Provider/Model Registries
On init, builds `ProviderRegistry` and `ModelRegistry` from YAML config. These are the same registries the engine uses at runtime. Optional background probe of Claude models on startup (disabled by default, enable via `PRSM_PROBE_CLAUDE_MODELS_ON_STARTUP=1`).

## VS Code Extension (`extension/`)

TypeScript extension that connects to the PrsmServer:
- Provides sidebar panels for agent tree, conversation, file changes
- Sends HTTP requests for orchestration commands
- Receives real-time updates via SSE
- Supports slash command UX in chat tabs:
  - Submitting `/` (or unknown slash command) opens a QuickPick command selector
  - `/import` opens guided import selection: provider -> session (recency-sorted) -> import depth
  - Session selection labels show title/summary, turn count, updated time, and source ID
- Built with esbuild, packaged with @vscode/vsce

## Setup

### Starting the Server

```bash
# Via PRSM CLI
prsm --server --config .prism/prsm.yaml

# Or via the build script
./start_prsm_server.sh
```

### Extension Installation

The `install.sh` and `start_prsm_server.sh` scripts handle:
1. Building the TypeScript extension with esbuild
2. Packaging as VSIX
3. Installing into VS Code
4. Symlinking to `.vscode-server/extensions/` for snap-based VS Code

### Important Paths
- VS Code (snap) loads extensions from `~/.vscode-server/extensions/` NOT `~/.vscode/extensions/`
- `start_prsm_server.sh` handles the sync automatically
- Server log: `~/.prsm/logs/prsm-server.log`
