# Engine Layer

The engine (`prsm/engine/`) is the backend orchestration core. It manages the complete lifecycle of hierarchical multi-agent workflows — spawning agents, routing messages between them, detecting deadlocks, and coordinating with AI providers. It has zero frontend dependencies.

## OrchestrationEngine (`engine.py`)

The top-level coordinator that wires together all engine subsystems.

**Key method: `run(task_definition, master_model, master_tools)`**
1. Starts the deadlock detector as a background asyncio task
2. Builds a master prompt from template + expert list + user's task
3. Auto-detects provider from `model_registry` if available; falls back to same-tier model when the requested model's provider is unavailable
4. Spawns a master agent with `AgentRole.MASTER`, `PermissionMode.BYPASS`, and read-only tools (`Read`, `Glob`, `Grep`)
5. Waits for the master agent to complete
6. Returns the final summary

**Additional method: `run_continuation(task_definition, master_agent_id)`**
Restarts an existing completed master agent with a follow-up prompt. Preserves identity (UUID, tree position, model, tools) so the TUI conversation view continues seamlessly.

The master agent delegates file-modifying work to child agents — it only reads and reasons.

## Agent Lifecycle (`agent_manager.py`)

`AgentManager` is the central registry for all active agents. Passes `EngineConfig.peer_models` to each `AgentSession` it creates.

### Spawning
`spawn_agent(SpawnRequest)` validates depth and concurrency limits, creates an `AgentDescriptor`, registers with the message router, resolves the provider and plugins, creates an `AgentSession` (with `peer_models`), and launches it as an `asyncio.Task`.

**Limits enforced:**
- `max_agent_depth` (default 5) — Maximum nesting depth
- `max_concurrent_agents` (default 10) — Maximum active agents

### Killing
`kill_agent(agent_id)` recursively kills all children (cascading kill), captures partial output, sets KILLED state, and resolves the agent's result future.

### Restarting
`restart_agent(agent_id, new_prompt)` reuses the agent's identity (ID, parent, role, model, tools, cwd) but resets state and runs a fresh session.

### Plugin Resolution
`_resolve_agent_plugins(request)` resolves the effective MCP server plugins for an agent. Resolution order:
1. If `request.mcp_servers` is explicitly set, merge on top of global plugins
2. If `plugin_manager` is available, auto-match by prompt/role relevance
3. Otherwise, use all global plugins
4. Apply `exclude_plugins` removals last

## Agent Sessions (`agent_session.py`)

`AgentSession` wraps a single Claude Agent SDK `query()` call with orchestration tools, permission checking, retry logic, and timeout management. Receives `peer_models` from `AgentManager` and passes it to `build_agent_mcp_config()` so child agent model selection is restricted to the configured peer models.

### Execution Flow
1. If a non-Claude provider is set, delegates to `_run_with_provider()` (worker) or `_run_with_provider_mcp()` (master with MCP bridge)
2. For Claude: builds per-agent MCP server via `build_agent_mcp_config()`
3. Merges plugin MCP servers
4. Builds `PreToolUse` hooks for bash command policy enforcement via `_build_bash_permission_hooks()`
5. Constructs `ClaudeAgentOptions` with system prompt, tools, `can_use_tool` callback, and hooks
6. Uses `AsyncIterable` prompt input (not string) to avoid premature stdin close
7. Streams `query()` results, emitting events for thinking, text, tool calls, and tool results. Note: if `task_complete` is called, the loop breaks immediately to prevent further output.
8. Tracks reasoning time against agent timeout (subtracts `ToolTimeTracker.accumulated_tool_time`)
9. Logs everything to `ConversationStore`

### Permission Checking (`_check_permission`)
Conforms to `claude_agent_sdk` `can_use_tool` callback signature. Processing order:

1. **BashRepeatGuard** — Blocks identical bash commands repeated >3 times (applies to ALL agents, even bypass)
2. **AskUserQuestion interception** — Routes SDK's `AskUserQuestion`/`request_user_input` through PRSM's UI (applies to ALL agents, even bypass)
3. **Bypass mode** → auto-allow everything else (master agents run in BYPASS)
4. *(Steps 4–8 apply only to non-bypass agents:)*
5. **Orchestration tools** → always auto-allowed (matches `_ORCHESTRATION_TOOLS` set or `mcp__orchestrator__` prefix)
6. **Control tools** (ExitPlanMode, update_plan) → auto-allowed
7. **"Allow Always" tools** → from shared mutable set across all agents
8. **Non-terminal tools** → auto-allowed
9. **Terminal command policy** — evaluates `CommandPolicyStore` whitelist/blacklist; safe commands auto-allowed, dangerous commands prompt user via `permission_callback`

### Bash Command Hooks (`_build_bash_permission_hooks`)
`PreToolUse` hooks that intercept bash commands before the SDK's permission mode evaluation. These hooks enforce `CommandPolicyStore` policy (workspace `.prism` whitelist/blacklist) and route dangerous commands through PRSM's permission UI. Because hooks run before permission mode, they work even in `bypassPermissions` mode.

### Timeout Model (Dual Clock)
- **Agent timeout** (default 7200s): Cumulative *reasoning* time — excludes tool execution time
- **Tool call timeout** (default 7200s): Maximum wall-clock time for any single tool call

### Transport Resilience
- Retries up to 5 times on transient transport failures with exponential backoff
- Circuit breaker: after 3 consecutive failures, pauses for 20s cooldown

## Message Router (`message_router.py`)

Routes messages between agents using per-agent `asyncio.Queue`.

- `register_agent()` / `unregister_agent()` — Creates/destroys per-agent queues
- `send(RoutedMessage)` — Pushes to target agent's queue
- `receive(agent_id, timeout, filter)` — Blocks until matching message arrives
- `mark_waiting()` / `get_wait_graph()` — Maintains directed wait graph for deadlock detection

## Deadlock Detection (`deadlock.py`)

Background task that runs every 5 seconds:
1. Reads the wait graph from `MessageRouter`
2. DFS cycle detection on the directed graph (waiter → target)
3. If a cycle is found, selects the deepest agent as victim
4. Force-fails the victim, clearing the deadlock

## State Machine (`lifecycle.py`)

```
PENDING → STARTING → RUNNING → COMPLETED
                      ↓ ↑
              WAITING_FOR_{PARENT,CHILD,EXPERT}
                      ↓
                    FAILED
Any state → KILLED (forced termination)
COMPLETED/FAILED → STARTING (restart via validate_transition)
KILLED → PENDING (restart via AgentManager.restart_agent, bypasses state machine)
```

`validate_transition()` enforces valid state changes and raises `ValueError` on invalid ones. Note: `AgentManager.restart_agent()` directly resets KILLED agents to PENDING without going through `validate_transition`, since the formal state machine marks KILLED as terminal.

## Data Models (`models.py`)

Single source of truth for all engine types, avoiding circular imports.

**Enums:**
- `AgentState` — 9 states: PENDING, STARTING, RUNNING, WAITING_FOR_PARENT, WAITING_FOR_CHILD, WAITING_FOR_EXPERT, COMPLETED, FAILED, KILLED
- `AgentRole` — MASTER, WORKER, EXPERT, REVIEWER
- `MessageType` — QUESTION, ANSWER, PROGRESS_UPDATE, TASK_RESULT, EXPERT_REQUEST/RESPONSE, SPAWN_REQUEST, KILL_SIGNAL, USER_PROMPT
- `PermissionMode` — DEFAULT, ACCEPT_EDITS, BYPASS, PLAN, DELEGATE

**Dataclasses:**
- `AgentDescriptor` — Complete agent description (ID, parent, role, state, prompt, tools, model, etc.)
- `RoutedMessage` — Message in transit (ID, type, source/target agents, payload, correlation_id)
- `SpawnRequest` — Spawn parameters (parent_id, prompt, role, tools, model, mcp_servers, etc.)
- `AgentResult` — Final result (agent_id, success, summary, artifacts, error, duration)
- `ExpertProfile` — Expert configuration (ID, name, system_prompt, tools, model, etc.)

## Error Hierarchy (`errors.py`)

```
OrchestrationError (base)
├── AgentSpawnError
├── AgentTimeoutError
├── MessageRoutingError
├── DeadlockDetectedError
├── MaxDepthExceededError
├── ExpertNotFoundError
├── ToolCallTimeoutError
├── ProviderNotAvailableError
└── ModelNotAvailableError
```

## Conversation Store (`conversation_store.py`)

Per-agent structured conversation log for reviewing child history.

- Entry types: TEXT, THINKING, TOOL_CALL, TOOL_RESULT, USER_MESSAGE
- `get_history(agent_id, detail_level)` — "full" (everything) or "summary" (text + tool names)
- `resolve_agent_id()` — Prefix matching for truncated IDs

## Rationale Extractor (`rationale_extractor.py`)

Analyzes agent conversation history around tool calls to extract WHY changes were made.

- Looks backward from a tool call up to 10 entries
- Extracts thinking/text blocks with rationale keywords (fix, improve, prevent, etc.)
- Scores candidates: "fix" +3, "bug" +3, "improve" +2, etc.
- Used for commit messages and the "View Context" feature in file change widgets

## YAML Configuration (`yaml_config.py`)

`load_yaml_config(path)` parses sections:
- `engine:` → EngineConfig (depth, concurrency, timeouts, command policy)
- `providers:` → Provider configs (claude, codex, gemini, minimax, alibaba)
- `models:` → Model aliases (short name → provider + model_id + optional reasoning_effort)
- `defaults:` → Default model, cwd, peer_model/peer_models, master_model
- `experts:` → Expert profiles with model alias resolution and per-expert MCP servers
- `plugins:` → MCP server configs with tags for auto-matching
- `model_registry:` → Per-model tier and task affinity overrides

Also loads a sibling `models.yaml` from the same directory (typically `.prism/models.yaml`) and merges its `models` and `model_registry` sections; the main config takes precedence on conflicts.

## MCP Server (`mcp_server/`)

### In-Process Server (`server.py`)
Factory that creates per-agent MCP servers. Each agent gets its own `OrchestrationTools` instance bound to its ID. No subprocess, no HTTP — purely in-process via `McpSdkServerConfig`. Receives `peer_models` from `AgentSession` to enforce child agent model restrictions.

### 17 Orchestration Tools (`tools.py`)

**Communication:**
- `ask_parent(question)` — Child asks parent, blocks for answer
- `respond_to_child(child_id, correlation_id, response)` — Parent answers child
- `wait_for_message(timeout)` — Receives next message from queue (default `timeout=0`, which disables timeout and blocks until a message arrives)
- `send_child_prompt(child_id, prompt)` — Sends prompt to child

**Agent Management:**
- `spawn_child(prompt, wait, tools, model, cwd, mcp_servers, complexity)` — Reuses matching children when possible (prefers active child reuse, then restart of completed/failed child) and only spawns a fresh child when no suitable match exists. `complexity` enables auto model selection and enforces tier-fit checks for the chosen model. When `peer_models` is configured, validates requested model against the allowed set
- `spawn_children_parallel(children)` — Spawns/reuses multiple children simultaneously using the same reuse and complexity-fit rules as `spawn_child`
- `restart_child(child_id, prompt, wait)` — Restarts a completed/failed child

**Expertise:**
- `consult_expert(expert_id, question)` — Spawns expert agent, blocks for result
- `consult_peer(question, thread_id, peer)` — Calls peer provider (e.g., Codex, Gemini, MiniMax) for second opinion; `peer='list'` to discover available peers

**Lifecycle:**
- `report_progress(status, percent)` — Non-blocking progress update to parent
- `task_complete(summary, artifacts)` — Signals task completion; `summary` must be non-empty (blank summaries are rejected)
- `ask_user(question, options)` — Asks the user directly with clickable options

**Observability:**
- `get_child_history(child_id, detail_level)` — Review child's conversation
- `check_child_status(child_id)` — Check child's state and metadata
- `get_children_status()` — Summary of all children's states

**Model Intelligence:**
- `recommend_model(task_description, complexity)` — Query the model capability registry for optimal model selection. When `peer_models` is configured, only considers models in the allowed set

**Shell:**
- `run_bash(command, timeout, cwd)` — Execute a bash command with permission checking

### Stdio Server (`stdio_server.py`)
For Claude Code integration. The user's Claude Code session IS the master agent. Exposes a subset of tools (no ask_parent, report_progress, task_complete — those are child-only).

### Non-Claude Master Agent Bridge (`orch_bridge.py` + `orch_proxy.py`)
Non-Claude providers (Codex, Gemini, MiniMax) use a TCP bridge + MCP proxy pattern for master agent orchestration:
- **`orch_bridge.py`** — TCP server running in the engine process. Accepts JSONL connections from `orch_proxy` and dispatches to in-process `OrchestrationTools`. Protocol: newline-delimited JSON-RPC over localhost TCP.
- **`orch_proxy.py`** — Standalone FastMCP server launched by the CLI tool as an MCP subprocess. Speaks MCP protocol on stdin/stdout and relays tool calls to `OrchBridge` via TCP.
- **Flow:** `CLI → MCP stdin/stdout → orch_proxy → TCP → OrchBridge → OrchestrationTools`

`agent_session.py._run_with_provider_mcp()` manages the bridge lifecycle for master agents on non-Claude providers.

## Providers (`providers/`)

Abstract `Provider` interface with four implementations:

| Provider | Backend | `run_agent()` | `send_message()` | `supports_master` |
|----------|---------|---------------|-------------------|--------------------|
| Claude | Claude Agent SDK | `query()` with in-process MCP | Lightweight no-tools query | N/A (in-process) |
| Codex | OpenAI Codex CLI | `codex exec` subprocess | Persistent `codex mcp-server` via JSON-RPC | Yes |
| Gemini | Google Gemini CLI | `gemini --prompt` subprocess | `--resume` for continuity | Yes |
| MiniMax | Codex CLI + MiniMax API | `codex exec -c model_provider=minimax` | Via Codex MCP server | Yes |
| Alibaba | Codex CLI + Alibaba Model Studio API | `codex exec -c model_provider=alibaba` | Via Codex MCP server | Yes |

All non-Claude providers implement `build_master_cmd()` which returns a CLI command configured with an MCP server pointing to `orch_proxy.py`.

### Codex-based Provider Session Isolation

For providers that run through the Codex CLI (`codex`, `minimax`, `alibaba`), PRSM now runs `codex exec` with `--ephemeral` for both worker and master agent execution paths.

Reason:
- Prevents stale local rollout/session state from interfering with new orchestration runs.
- Avoids noisy Codex stderr spam like `state db missing rollout path for thread ...` from dominating failure output.
- Keeps each orchestration run isolated so context growth does not inherit hidden prior state.

Scope:
- Applies to `run_agent()` and `build_master_cmd()` execution paths for Codex-backed providers.
- Does **not** remove intentional thread continuity for peer-chat style flows that use `send_message()` + `thread_id` via `codex mcp-server`.

Operational note:
- True model context-window exhaustion can still happen on very large prompts/history for any provider. The `--ephemeral` change removes Codex session-state coupling, not model token limits.
- PRSM also filters known Codex rollout-path noise lines from surfaced stderr so master failures remain readable.

### Timeout & Liveness Safeguards

The engine includes several safeguards to prevent indefinite hangs:

- **`wait_for_result()` / `wait_for_multiple()`** (`agent_manager.py`): Wrapped with `asyncio.wait_for(timeout=agent_timeout_seconds)`. On timeout, returns a failed `AgentResult` with descriptive error text instead of blocking forever.
- **`BridgeClient.call()` readline** (`orch_proxy.py`): Each `readline()` inside the response-reading loop is wrapped with a 5-minute per-read timeout. On timeout, the connection is reset and the error is re-raised.
- **`ask_user` tool** (`tools.py`): Now uses `_with_timeout()` like all other tools, preventing indefinite blocking if the user never responds.
- **`wait_for_message` deadlock tracking** (`tools.py`): Now calls `router.mark_waiting()` / `clear_waiting()` so the deadlock detector can detect cycles involving `wait_for_message`.
- **Shutdown concurrency guard** (`engine.py`): `shutdown()` is protected by an `asyncio.Lock` to prevent overlapping shutdown calls from causing duplicate `engine_finished` events.

### TUI Liveness & Progress Display

- **ThinkingIndicator elapsed time** (`tui/widgets/thinking.py`): Shows running elapsed time (e.g. "Thinking... (32s)") on every thinking animation tick.
- **StatusBar elapsed time** (`tui/widgets/status_bar.py`): Shows engine elapsed time in the status bar while streaming (e.g. "streaming (1m 14s)").
- **Liveness watchdog** (`tui/handlers/event_processor.py`): If no engine events are received for 30+ seconds, logs a warning to the tool log and updates the ThinkingIndicator with a "no events" warning.
- **Engine duration on finish**: `_handle_engine_finished` now displays total engine run duration in the tool log.
- **Event drain window**: The `_run_orchestration` finally block waits 0.5s (up from 0.1s) to drain remaining events before cancelling the event consumer.

`ProviderRegistry` maps provider names to instances, built from YAML config via `build_provider_registry()`.

## Model Registry (`model_registry.py`)

Maps model IDs to `ModelCapability` instances with tier (`FRONTIER`, `STRONG`, `FAST`, `ECONOMY`), task affinities (scores per `TaskCategory`), cost/speed factors, and max context window. Used by `spawn_child(complexity=...)` to auto-select the optimal model for each subtask. Populated from built-in defaults, YAML `model_registry:` section, and CLI availability probing.

**12 task categories:** architecture, complex-reasoning, coding, code-review, planning, tool-use, exploration, simple-tasks, general, agentic, search, documentation.

## Model Intelligence (`model_intelligence.py`)

Persistent, learned model rankings per task category. A background research loop (`run_research_loop`) runs daily, scoring models against task categories using static affinities and optional LLM-based research. Rankings are persisted to `~/.prsm/model_intelligence.json` and survive restarts. The `ModelRegistry` consults these rankings as the primary scoring source, falling back to static affinities.

## Model Discovery (`model_discovery.py`)

Automatic CLI tool updating and model discovery that runs at startup (both TUI and VSCode server).

**Three-phase pipeline:**

1. **CLI Updates** (parallel, best-effort): Runs `claude update`, `npm update -g @openai/codex`, and `npm update -g @google/gemini-cli` to keep provider CLIs current. Controlled by `PRSM_UPDATE_CLIS` env var (default `"1"`).

2. **Model Discovery** (parallel, best-effort): Probes each installed provider to enumerate available models:
   - **Codex**: Reads `~/.codex/models_cache.json` (auto-refreshed if stale >24h). Parses model slugs, descriptions, reasoning levels, and visibility. Only includes `visibility: "list"` models.
   - **Gemini**: Locates `@google/gemini-cli-core/dist/src/config/models.js` under the global npm root. Extracts model IDs via regex (VALID_GEMINI_MODELS Set, named constants, or quoted strings). Skips non-chat models (e.g. embedding).
   - **Claude**: Resolves the `claude` binary, runs `strings` on it, and extracts model IDs matching `claude-(opus|sonnet|haiku)-\d[\w.-]*`. Filters out legacy (3.x), Bedrock/Vertex (`-v1`), and alternate-format (`@YYYYMMDD`) variants.

3. **YAML Merge**: Writes newly discovered models into `~/.prsm/models.yaml` without overwriting user customizations. Adds both short aliases (in `models:`) and capability metadata with default affinities (in `model_registry:`). Tier is inferred from model ID patterns (opus→frontier, sonnet→strong, haiku→fast, etc.).

**Integration:**
- **VSCode server** (`PrsmServer.start`): Launches `_discover_models_background()` as an `asyncio.Task` after server startup. If models.yaml is updated, reloads YAML config and rebuilds registries.
- **TUI** (`OrchestratorBridge.run`): Launches discovery as a deferred background task alongside the existing model probe.

**Environment variables:**
- `PRSM_MODEL_DISCOVERY` — Enable/disable discovery (default `"1"`)
- `PRSM_UPDATE_CLIS` — Enable/disable CLI auto-updates (default `"1"`)

**Key design decisions:**
- All operations are best-effort: failures are logged but never crash the caller
- User-customized entries (cost_factor, speed_factor, custom affinities) are never overwritten
- Affinity scores are rounded to 4 decimal places to avoid floating-point display artefacts
