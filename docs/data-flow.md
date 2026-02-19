# Data Flow

This document traces how data flows through the PRSM system for the key operations.

## User Prompt → Agent Execution → Result

### TUI Path
```
1. User types prompt in InputBar
2. InputBar resolves @references (file content injection)
3. InputBar posts Submitted event to MainScreen
4. MainScreen._run_orchestration(prompt) starts as @work background worker:
   a. Auto-names the session via LLM
   b. Resets EventBus
   c. Starts event consumer task (EventProcessor.consume_events())
   d. await bridge.run(prompt)
5. OrchestratorBridge.run() calls engine.run(task_definition)
6. Engine auto-detects provider from model_registry; spawns master agent (MASTER role, BYPASS mode, read-only tools)
7. AgentSession.run() dispatches to Claude SDK query() or provider subprocess
8. Events flow: Engine → event_callback → EventBus → EventProcessor → TUI widgets
9. Master spawns children, consults experts, waits for results
10. Master calls task_complete() with synthesis
11. engine.run() returns summary to bridge
12. bridge.run() returns to MainScreen
13. MainScreen cleans up, finalizes streams
```

### VSCode Path
```
1. VS Code extension sends POST /{session_id}/run {prompt}
2. PrsmServer starts _run_orchestration_task(session_id, prompt)
3. Same engine flow as TUI (steps 5-11)
4. Events broadcast via SSE to all connected clients
5. Extension receives SSE events and updates UI
```

### Non-Claude Master Path (Codex/Gemini/MiniMax)
```
1. Engine detects non-Claude provider for master agent
2. AgentSession._run_with_provider_mcp() starts OrchBridge (TCP server)
3. Provider.build_master_cmd() builds CLI command with MCP config pointing to orch_proxy.py
4. CLI subprocess launches → discovers orch_proxy as MCP server
5. Agent calls orchestration tools → MCP stdin/stdout → orch_proxy → TCP → OrchBridge → OrchestrationTools
6. OrchestrationTools dispatches to AgentManager (spawn, message routing, etc.)
7. CLI subprocess exits → agent session waits for any active children → completes
```

### Claude Code MCP Path
```
1. User's Claude Code session calls spawn_child() MCP tool
2. stdio_server's OrchestrationTools.spawn_child() creates SpawnRequest
3. AgentManager spawns child AgentSession
4. Child runs as headless SDK query()
5. Results flow back through MCP tool responses
```

## Event Flow: Engine → UI

```
OrchestrationEngine
  ├── event_callback (fires on every engine event)
  │   └── EventBus._callback(data: dict)
  │       ├── dict_to_event(data) → typed OrchestratorEvent
  │       └── queue.put(event)
  │
  ├── permission_callback (fires when tool needs approval)
  │   └── OrchestratorBridge._handle_permission()
  │       ├── Creates asyncio.Future
  │       ├── Emits PermissionRequest → EventBus queue
  │       └── await Future (blocks engine agent)
  │
  └── user_question_callback (fires when agent asks user)
      └── OrchestratorBridge._handle_user_question()
          ├── Creates asyncio.Future
          ├── Emits UserQuestionRequest → EventBus queue
          └── await Future (blocks engine agent)

EventBus Queue
  ├── TUI: EventProcessor.consume_events()
  │   └── Dispatches to: tree updates, stream widgets,
  │       tool call widgets, permission modals, question widgets
  │
  └── VSCode: PrsmServer._consume_session_events()
      └── Broadcasts via SSE to all connected VS Code clients
```

## Permission Resolution

Two layers: `_check_permission` (can_use_tool callback) and `_build_bash_permission_hooks` (PreToolUse hooks). Both run before the SDK's built-in permission mode.

```
Layer 1 — PreToolUse hooks (bash commands only, runs even in bypass mode):
  1. Hook intercepts bash tool calls before SDK permission evaluation
  2. Checks CommandPolicyStore whitelist → auto-allow
  3. Checks CommandPolicyStore blacklist → route to permission_callback
  4. Non-matching commands → pass through to SDK

Layer 2 — can_use_tool callback (_check_permission):
  1. BashRepeatGuard → auto-block commands repeated >3 times (all modes)
  2. AskUserQuestion interception → route SDK questions through PRSM UI (all modes)
  3. Bypass mode → auto-allow everything else (master agents)
  4. Orchestration tools → auto-allow
  5. Control tools → auto-allow
  6. "Allow Always" set → auto-allow
  7. Non-terminal tools → auto-allow
  8. Terminal command policy evaluation → safe auto-allow, dangerous → permission_callback

When permission_callback fires:
  a. Bridge._handle_permission() creates Future with UUID request_id
  b. Emits PermissionRequest event
  c. await Future (engine blocks)
  d. UI receives PermissionRequest (TUI: modal, VSCode: SSE broadcast)
  e. User clicks Allow/Deny/Always
  f. bridge.resolve_permission(request_id, result):
     - "allow_project"/"allow_global" → persist + return "allow_always"
     - Sets Future result
  g. Future resolves → engine unblocks → tool executes or is denied
```

## Inter-Agent Messaging

```
Child calls ask_parent(question):
  1. OrchestrationTools.ask_parent() creates RoutedMessage(QUESTION)
  2. MessageRouter.send() pushes to parent's asyncio.Queue
  3. marks child as waiting in wait graph
  4. child blocks on MessageRouter.receive() with correlation_id filter

Parent calls wait_for_message():
  1. OrchestrationTools.wait_for_message()
  2. MessageRouter.receive() pops from parent's queue
  3. Returns QUESTION message to parent

Parent calls respond_to_child(child_id, correlation_id, answer):
  1. Creates RoutedMessage(ANSWER) with matching correlation_id
  2. MessageRouter.send() pushes to child's queue
  3. Child's receive() finds matching message
  4. ask_parent() returns the answer
```

## Interactive Child Runbook (Required Pattern)

Use this pattern for any child that may call `ask_parent()` or run longer than a short tool timeout:

1. Spawn non-blocking: `spawn_child(..., wait=false)`
2. Receive messages in a loop with `wait_for_message(timeout_seconds=0)` (0 disables timeout)
3. If child asks a question, answer with `respond_to_child(...)`
4. Continue until you receive `task_result`

Do not use `spawn_child(wait=true)` for interactive work. A blocking parent call can time out while the child is still running, which creates false failure handling and duplicate retries.

## Session Persistence

```
Save:
  1. Session → JSON (agents, messages, metadata)
  2. FileChangeTracker → individual {tool_call_id}.json files
  3. Written to ~/.prsm/sessions/{repo_identity}/{session_id}.json
  4. .active_session marker updated

Restore:
  1. Load JSON from disk
  2. Deserialize agents (parse_state/parse_role for backward compat)
  3. Reset stale states (RUNNING → COMPLETED)
  4. Deserialize messages and tool calls
  5. Load file changes from disk
  6. Rebuild UI: add agents to tree, configure bridge

Auto-resume:
  1. On launch (no --new flag), find most recent session file
  2. Load and restore
```

## Snapshot Workflow

```
Create snapshot:
  1. Capture session state (serialize Session)
  2. Capture git diff HEAD → working_tree.patch
  3. Copy untracked files to untracked/
  4. Persist FileChangeTracker records to file-changes/
  5. Save meta.json (id, session, description, timestamp, branch)

Restore snapshot:
  1. git checkout HEAD -- . (reset working tree)
  2. Clean untracked files
  3. Apply working_tree.patch
  4. Restore untracked files from backup
  5. Deserialize session state
  6. Load file change records
```

## File Change Tracking

```
1. EventProcessor receives ToolCallStarted (Write/Edit)
2. FileChangeTracker.capture_pre_tool() reads file before modification
3. Engine executes the tool (agent writes/edits file)
4. EventProcessor receives ToolCallCompleted
5. FileChangeTracker.track_change() reads new content, computes diff
6. FileChanged event emitted to EventBus
7. UI renders FileChangeWidget:
   - Collapsed: file path + line stats
   - Expanded: diff preview + View Context / View Agent buttons
8. "View Context" → RationaleExtractor → FileContextScreen
9. "View Agent" → Navigate to agent in tree + highlight message
```
