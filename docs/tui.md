# TUI Layer

The TUI (`prsm/tui/`) is a Textual-based terminal UI that provides a rich interactive interface for managing agent orchestration workflows.

## PrsmApp (`app.py`)

Thin `App` subclass that pushes `MainScreen` and provides global keybindings:

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit (auto-saves session) |
| `Ctrl+C` | Cancel running orchestration |
| `Ctrl+N` | New session |
| `Ctrl+S` | Save session |
| `Ctrl+T` | Focus agent tree |
| `Ctrl+E` | Focus input |
| `F1` | Toggle tool log panel |
| `F2` | Open settings menu |
| `Escape` | Cancel / blur focus |

## MainScreen (`screens/main.py`)

The primary workspace screen (~1250 lines). Ties everything together.

### Layout
```
Header
├── Horizontal#workspace
│   ├── AgentTree#agent-tree (left)
│   ├── Vertical#main-pane (center)
│   │   ├── ConversationView#conversation
│   │   └── ToolLog#tool-log
│   └── AgentContextPanel#agent-context-panel (right)
├── InputBar#input-bar
└── StatusBar#status-bar
```

### Handler Delegation
MainScreen delegates to 3 handler objects to keep the main screen focused on layout and wiring:
- **CommandHandler** — Slash command dispatch
- **EventProcessor** — Orchestrator event consumption
- **SessionManager** — Session lifecycle management

### Orchestration Flow
1. User types prompt in InputBar → `on_input_bar_submitted()`
2. If agents are running: shows `DeliveryModeScreen` (interrupt/inject/queue)
3. If idle: shows temporary user message + thinking indicator, then `_run_orchestration()`
4. `_run_orchestration()` is a `@work(exclusive=True)` background worker:
   - Creates auto-snapshot before run
   - Auto-names the session
   - Resets event bus
   - Starts event consumer task
   - `await bridge.run(prompt)`
   - Cleans up on completion

### On Mount
- Loads project memory (`MEMORY.md` from `~/.prsm/projects/{ID}/memory/`)
- Loads `PRSM.md` project instructions from cwd
- Initializes plugin manager
- Dispatches CLI args via `SessionManager.handle_cli_args()`

### Demo Mode
When Claude CLI is not available, simulates streaming responses with random delays, tool calls, and markdown content. Shows prominent demo banner and warning badge in status bar.

## Handlers

### CommandHandler (`handlers/command_handler.py`)

Slash command dispatcher:
- `/help` — Shows available commands
- `/new` — Starts a fresh session
- `/session SESSION_UUID` — Save current + load session by UUID
- `/sessions` — Lists all saved sessions
- `/fork [NAME]` — Forks current session
- `/save [NAME]` — Saves session with optional name
- `/import {list|preview|run}` — Transcript portability from Codex/Claude sessions
  - `/import list [all|codex|claude]`
  - `/import preview PROVIDER SOURCE_ID`
  - `/import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]`
  - If `--max-turns` is omitted, TUI prompts for import depth (200/500/full)
- `/plugin {add|add-json|list|remove|tag}` — Plugin management (stdio and remote HTTP/SSE)
- `/policy [list|add-allow|add-block|remove]` — Bash command whitelist/blacklist management
- `/settings` — Opens the settings menu modal
- `/worktree {list|create|remove|switch}` — Git worktree management

### EventProcessor (`handlers/event_processor.py`)

Consumes orchestrator events from EventBus and updates the TUI. The core event loop.

**Event dispatching:** `consume_events()` iterates `bridge.event_bus.consume()` and handles each event type:
- `AgentSpawned` → Map agent, add to tree, set thinking, store pending user prompt for root
- `AgentRestarted` → Update or re-create agent state, set thinking
- `AgentStateChanged` → Update tree icon/state, manage thinking for terminal vs running states
- `StreamChunk` → Write to Markdown streaming widget (active agent) or buffer (inactive)
- `ToolCallStarted` → Capture pre-tool file content (Write/Edit), finalize stream, add ToolCallWidget
- `ToolCallCompleted` → Update tool result, track file changes, persist diffs, check inject prompts
- `AgentResult` → Finalize stream, flush buffered chunks
- `AgentKilled` → Update tree state, cancel pending futures
- `FileChanged` → Show file change widget
- `PermissionRequest` → Push PermissionScreen modal
- `UserQuestionRequest` → Mount QuestionWidget
- `EngineFinished` → Clear all thinking, update status bar, suppress error during interrupt

**Thinking management:** Per-agent thinking state tracked in `_thinking_agents` set. Mount/remove `ThinkingIndicator` based on active agent's thinking state.

**Stream management:** Mounts `Markdown` widgets with `get_stream()` for token-by-token rendering. Non-active agent chunks are buffered and flushed on agent switch.

**File tracking:** Owns a `FileChangeTracker` instance. Captures pre-tool content for Write/Edit tools, computes diffs on completion, persists file change records to disk for crash recovery.

### SessionManager (`handlers/session_manager.py`)

Session lifecycle management:
- `handle_cli_args()` — Dispatches `--fork`, `--fork-snapshot`, `--resume`, `--new`, or auto-resume
- `start_fresh_session()` — Auto-discovers `.prism/prsm.yaml` (or legacy `prsm.yaml`), configures bridge with plugins and YAML config, sets up live or demo mode
- `_auto_discover_yaml_config()` — Discovers and loads YAML config from `.prism/prsm.yaml`, `prsm.yaml`, or `--config` CLI flag. Mirrors the config discovery logic used by the server in `app.py`.
- `restore_session()` — Rebuilds UI: add agents to tree, sort by activity, restore file changes
- `save_session()` — Saves session + persists file changes
- `auto_name_session(prompt)` — Generates name via LLM
- `create_auto_snapshot()` — Snapshots session + working tree before each prompt
- `truncate_session_to()` — Truncates message history for resend from earlier point
- `restore_snapshot_files()` — Reverts working tree to snapshot state
- `rebuild_ui_after_resend()` — Clears and re-renders tree + conversation after truncation

## Screens

### Permission Screen (`screens/permission.py`)
Modal dialog for tool permission requests. Returns: "allow", "allow_project", "allow_global", "deny", "deny_project", or "view_agent".

### Delivery Mode Screen (`screens/delivery_mode.py`)
Shown when user sends a prompt while agents are running. Hotkeys: `i`/`j`/`q`/`Esc`. Arrow key navigation between buttons.
- **Interrupt** — Cancel current task and replace with this prompt (red)
- **Inject** — Finish current tool call, then process this prompt (yellow)
- **Queue** — Run this prompt after the current task completes (blue)

### Settings Screen (`screens/settings.py`)
Modal hub for managing PRSM preferences:
- Bash command whitelist/blacklist (add/remove regex patterns)
- File revert on resend preference (ask/always/never)
- Model & provider config editor (edits `.prism/prsm.yaml` and `~/.prsm/models.yaml`)
 - Engine setting: user question timeout (`engine.user_question_timeout_seconds`, set `0` to disable timeout)
  - Default model, peer models, providers, models, and model registry as YAML

### Kill Confirm Screen (`screens/kill_confirm.py`)
Modal confirmation before killing/removing an agent. Shows agent name, ID, and current state. Hotkey `y` to confirm.

### Resend Confirm Screen (`screens/resend_confirm.py`)
Modal for resending a previous prompt. Shows file change count and prompt preview. Three options:
- **Revert** — Restore working tree to snapshot state (hotkey `r`)
- **Keep** — Keep current files, rewind conversation only (hotkey `k`)
- **Cancel** — Dismiss

### File Context Screen (`screens/file_context.py`)
Modal showing why an agent made a file change: file path, tool name, agent name, and extracted rationale.

### Model Selector Screen (`screens/model_selector.py`)
Modal dialog for selecting the AI model for the session. Shows available models grouped by provider with tier badges (Frontier/Strong/Fast/Economy).

### Command Policy Screen (`screens/command_policy.py`)
Interactive modal for managing bash command whitelist and blacklist regex patterns. Stored in `<workspace>/.prism/command_whitelist.txt` and `command_blacklist.txt`.

### Worktree Screens (`screens/worktree.py`)
4 modals: WorktreeListScreen, WorktreeCreateScreen, WorktreeRemoveScreen, WorktreeSwitchScreen.

### Agent Context Menu (`screens/agent_context_menu.py`)
Right-click popup: "View Context" and "Kill Agent". Dismisses on click outside or Escape.

## Widgets

### AgentTree (`widgets/agent_tree.py`)
Hierarchical `Tree[AgentNode]` widget showing all orchestrated agents.
- Custom `render_label()` — State icon (colored) + name + [role] tag + relative time
- `sort_by_activity()` — Reorders nodes by `last_active` timestamp
- Right-click → context menu, `i` → view context, `Delete` → kill agent (with confirmation)

### ConversationView (`widgets/conversation.py`)
Scrollable per-agent message display with streaming support.
- `MessageWidget` — Renders user/assistant/system messages with timestamp and role badge. User messages are clickable to trigger resend (`ResendRequested` event)
- `ToolCallWidget` — Collapsible tool call: collapsed=one-line, expanded=full args/result
- `buffer_stream_chunk()` / `flush_stream_buffer()` — Buffers chunks for non-active agents
- `_smart_scroll()` — Only auto-scrolls if user is near the bottom

### InputBar (`widgets/input_bar.py`)
Prompt input with submit handling, history navigation, and @file completion.
- Enter → submit, Shift+Enter → newline
- Up/Down → prompt history with draft preservation
- `@` → triggers FileCompleter for file/directory autocomplete
- `🔧 Model` button → opens Model Selector Screen
- Slash commands parsed before send

### FileCompleter (`widgets/file_completer.py`)
Floating autocomplete dropdown for @file references. Top 10 matches, up/down navigation, tab/enter to confirm. Directory selection re-triggers for drill-down.

### AgentContextPanel (`widgets/agent_context_panel.py`)
Docked right-side panel with comprehensive agent details:
1. Metadata — Name, role, state, model, ID, parent, children
2. Hierarchy — Ancestor chain + children with grandchild counts
3. Task Prompt — Agent's prompt preview
4. Conversation — Full message history with highlighting support

### FileChangeWidget (`widgets/file_change.py`)
Collapsible file modification widget:
- Collapsed: file path + +N/-M line stats
- Expanded: colored diff preview + "View Context" / "View Agent" buttons

### QuestionWidget (`widgets/question.py`)
Agent-to-user questions with clickable option buttons. Click locks to prevent double-answering.

### StatusBar (`widgets/status_bar.py`)
Bottom bar: agent name, model, token count, connection status. Demo mode shows warning badge with "Simulated responses" message.

### ThinkingIndicator (`widgets/thinking.py`)
Animated indicator cycling through verbs (Thinking, Analyzing, Pondering...) with dot animation.

### ToolLog (`widgets/tool_log.py`)
Collapsible `RichLog` panel for tool calls. Toggle with F1.
