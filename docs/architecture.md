# PRSM CLI Architecture

## Overview

PRSM is a hierarchical multi-agent orchestration system built on the Claude Agent SDK. It provides a rich TUI (terminal UI) and an HTTP+SSE server for VS Code integration, both sharing the same engine core.

The codebase has 5 packages with strict dependency direction:

```
engine/ ← adapters/ ← shared/ ← tui/
                               ← vscode/
```

- **engine/** — Backend orchestration: agent lifecycle, message routing, deadlock detection, MCP tools, providers. Zero frontend dependencies.
- **adapters/** — Bridge layer connecting engine to UI frontends: event bus, permission resolution, file change tracking.
- **shared/** — Models, services, and utilities used by both TUI and VSCode frontends.
- **tui/** — Textual-based terminal UI frontend.
- **vscode/** — HTTP+SSE server for the VS Code extension.

TUI and VSCode never import from each other.

## Package Layout

```
prsm/
├── app.py                    # Entry point — dispatches to TUI or server mode
├── engine/                   # Backend orchestration engine
│   ├── engine.py             # OrchestrationEngine — top-level coordinator
│   ├── agent_manager.py      # Agent lifecycle: spawn, track, kill, restart
│   ├── agent_session.py      # Claude Agent SDK wrapper per agent
│   ├── message_router.py     # Async queue-based inter-agent messaging
│   ├── config.py             # EngineConfig from ORCH_* env vars
│   ├── context.py            # Master agent prompt template & context
│   ├── models.py             # Core enums & dataclasses (single source of truth)
│   ├── lifecycle.py          # Agent state machine & transition validation
│   ├── deadlock.py           # DFS cycle detection on wait graph
│   ├── errors.py             # Exception hierarchy
│   ├── conversation_store.py # Per-agent conversation log
│   ├── expert_registry.py    # Domain expert profile registry
│   ├── rationale_extractor.py# Extract WHY from agent reasoning
│   ├── yaml_config.py        # YAML config loader (engine, providers, models, experts)
│   ├── cli.py                # Headless CLI entry point
│   ├── model_registry.py    # Model capability registry (tiers, affinities)
│   ├── model_intelligence.py# Learned task→model rankings (persisted JSON)
│   ├── model_discovery.py   # CLI auto-update & model discovery at startup
│   ├── mcp_server/           # MCP tool implementations
│   │   ├── server.py         # Per-agent in-process MCP server factory
│   │   ├── tools.py          # 17 orchestration tool implementations
│   │   ├── master_tools.py   # Tools exposed to Claude Code (stdio MCP)
│   │   ├── stdio_server.py   # Stdio transport for Claude Code integration
│   │   ├── orch_bridge.py    # TCP bridge for non-Claude master agents
│   │   └── orch_proxy.py     # MCP proxy subprocess for non-Claude masters
│   └── providers/            # AI provider backends
│       ├── base.py           # Abstract Provider interface
│       ├── registry.py       # ProviderRegistry factory
│       ├── claude_provider.py# Claude Agent SDK integration
│       ├── codex_provider.py # OpenAI Codex CLI integration
│       ├── gemini_provider.py# Google Gemini CLI integration
│       ├── minimax_provider.py# MiniMax via Codex CLI integration
│       └── alibaba_provider.py# Alibaba Model Studio via Codex CLI integration
├── adapters/                 # Engine-to-UI bridge layer
│   ├── orchestrator.py       # OrchestratorBridge — main adapter
│   ├── event_bus.py          # Async event queue
│   ├── events.py             # 17 typed event dataclasses + serialization
│   ├── agent_adapter.py      # UI display helpers (icons, colors, labels)
│   ├── file_tracker.py       # File change detection & diff computation
│   ├── permission_store.py   # Persisted tool permission decisions
│   └── session_naming.py     # LLM-powered session auto-naming
├── shared/                   # Shared models, services, utilities
│   ├── models/
│   │   ├── agent.py          # AgentNode — UI agent representation
│   │   ├── message.py        # Message, ToolCall, MessageRole
│   │   └── session.py        # Session state container
│   ├── services/
│   │   ├── persistence.py    # JSON session save/load with worktree awareness
│   │   ├── plugins.py        # MCP server plugin management
│   │   ├── project.py        # Per-project directory management
│   │   ├── project_memory.py # MEMORY.md per project
│   │   ├── session_naming.py # Session auto-naming (shared copy)
│   │   ├── snapshot.py       # Session + working tree snapshot/restore
│   │   ├── command_policy_store.py # Command whitelist/blacklist policy
│   │   ├── durable_write.py  # Atomic file writes with backup
│   │   ├── preferences.py    # User preference storage
│   │   └── process_cleanup.py# Orphan process cleanup
│   ├── formatters/
│   │   └── tool_call.py      # Tool call display formatting
│   ├── commands.py           # Slash command parser
│   └── file_utils.py         # @reference resolution, FileIndex, tree outline
├── tui/                      # Textual terminal UI
│   ├── app.py                # PrsmApp — Textual App subclass
│   ├── handlers/
│   │   ├── command_handler.py# Slash command dispatcher
│   │   ├── event_processor.py# Orchestrator event consumer
│   │   └── session_manager.py# Session lifecycle management
│   ├── screens/
│   │   ├── main.py           # MainScreen — primary workspace
│   │   ├── permission.py     # Tool permission modal
│   │   ├── delivery_mode.py  # Interrupt/inject/queue modal
│   │   ├── file_context.py   # File change rationale modal
│   │   ├── worktree.py       # Git worktree management modals
│   │   └── agent_context_menu.py
│   ├── widgets/
│   │   ├── agent_tree.py     # Hierarchical agent tree
│   │   ├── conversation.py   # Streaming markdown conversation
│   │   ├── input_bar.py      # Prompt input with history + @ completion
│   │   ├── file_completer.py # @file autocomplete dropdown
│   │   ├── agent_context_panel.py  # Agent detail panel
│   │   ├── file_change.py    # File modification diff widget
│   │   ├── question.py       # User question with clickable options
│   │   ├── status_bar.py     # Agent/model/token status
│   │   ├── thinking.py       # Animated thinking indicator
│   │   └── tool_log.py       # Tool call log panel
│   └── styles/
│       ├── app.tcss          # Main Textual CSS
│       └── modal.tcss        # Modal dialog CSS
└── vscode/                   # VS Code HTTP+SSE server
    ├── server.py             # PrsmServer — aiohttp multi-session server
    └── extension/            # TypeScript VS Code extension
```

## Entry Point

`prsm/app.py` → `main()` is the single entry point registered in `pyproject.toml` as the `prsm` console script.

It dispatches based on CLI flags:
- `prsm --server` → Starts `PrsmServer` (HTTP+SSE for VS Code)
- `prsm --list` → Lists saved sessions and exits
- `prsm` (default) → Starts `PrsmApp` (Textual TUI)

Server mode includes:
- SDK preflight health check (opt-out via `PRSM_SDK_PREFLIGHT=0`)
- Rotating file logging to `~/.prsm/logs/prsm-server.log`
- Auto-discovery of `.prism/prsm.yaml` (or legacy `prsm.yaml`) in the working directory

## Three Modes of Operation

### 1. TUI Mode (default)
User interacts through a terminal UI built with Textual. The TUI manages a single session with one orchestration bridge.

### 2. VS Code Server Mode (`--server`)
HTTP+SSE server that supports multiple concurrent sessions. The VS Code extension connects to this server.

### 3. Claude Code MCP Mode (`.mcp.json`)
The engine runs as a stdio MCP server for Claude Code. The user's Claude Code session IS the master agent — it gets orchestration tools (spawn_child, consult_expert, etc.) as MCP tools. Configured via `.mcp.json` in the project root.
