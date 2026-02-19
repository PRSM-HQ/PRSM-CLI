# PRSM

```bash
curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash
```

A VS Code extension for multi-agent AI orchestration. PRSM gives you a rich interactive interface for managing hierarchical agent workflows — spawn child agents, consult domain experts, stream responses in real-time, and visualize your entire agent tree in the sidebar.

Built on an HTTP+SSE server with a TypeScript VS Code extension frontend.

## Features

- **Agent tree visualization** — Live sidebar view of your agent hierarchy (master → workers → experts)
- **Multi-session support** — Run multiple orchestration sessions in parallel, each with its own agent tree
- **Streaming markdown** — Real-time agent responses rendered as rich markdown in editor tabs
- **@ file references** — Attach file/directory context to prompts with autocomplete
- **Permission modals** — Approve or deny tool execution requests from agents
- **Model switching** — Change models on the fly from the status bar
- **Session management** — Save, resume, fork, and list sessions
- **Multi-provider** — Claude, Codex, and Gemini backends via their respective CLIs
- **MCP plugin system** — Register stdio or HTTP MCP servers for agent tool access

## Requirements

- Python 3.12+
- VS Code
- Node.js + npm
- At least one AI provider CLI: [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenAI Codex](https://github.com/openai/codex), or [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## Installation

The one-liner above downloads the latest release and installs everything:

1. Downloads and extracts PRSM to `~/.prsm/app/`
2. Creates a Python virtualenv and installs dependencies
3. Links the `prsm` command to `~/.local/bin/`
4. Detects and configures installed AI provider CLIs
5. Installs the VS Code extension

### From source

```bash
git clone https://github.com/PRSM-HQ/PRSM-CLI.git prsm-cli
cd prsm-cli
./install.sh          # install
./install.sh --dev    # with dev dependencies
```

### Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash -s -- --uninstall
```

## Usage

Open any project in VS Code. The PRSM sidebar panel appears in the activity bar. The server starts automatically when the extension activates.

### Starting a session

1. Open the PRSM sidebar (click the PRSM icon in the activity bar)
2. Click **New Session** to create an orchestration session
3. Type your prompt in the chat tab and send it
4. Watch the agent tree populate as the master agent spawns workers and experts

### Server mode

The extension connects to the PRSM HTTP+SSE server. To rebuild the extension and restart the server after code changes:

```bash
./start_prsm_server.sh
```

### @ file references

Type `@` followed by a file or directory path to attach context to your prompt. An autocomplete dropdown appears as you type.

- **File reference** (`@src/auth.py`): Injects the file's content into the prompt
- **Directory reference** (`@prsm/models/`): Injects a tree outline of the directory structure
- Selecting a directory re-triggers completion so you can navigate deeper

Files over 100KB are truncated. Binary files are skipped. Paths outside the project root are blocked.

### Slash commands

Type these in the chat input:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/new` | Start a fresh session |
| `/session SESSION_UUID` | Save current + load session by UUID |
| `/sessions` | List all saved sessions |
| `/fork [NAME]` | Fork current session |
| `/save [NAME]` | Save session with optional name |
| `/plugin add NAME CMD [ARGS...]` | Add a stdio MCP server plugin |
| `/plugin add NAME --type http --url URL` | Add an HTTP MCP server |
| `/plugin add-json NAME '{...}'` | Add plugin from JSON config |
| `/plugin list` | List all registered plugins |
| `/plugin remove NAME` | Remove a plugin |
| `/plugin tag NAME tag1 tag2` | Tag a plugin for auto-matching |

## Architecture

PRSM has 5 packages with strict dependency direction:

```
engine/ ← adapters/ ← shared/ ← vscode/
```

```
prsm/
├── app.py              # Entry point — dispatches to server mode
├── engine/             # Backend orchestration (zero frontend deps)
│   ├── engine.py       # OrchestrationEngine — top-level coordinator
│   ├── agent_manager.py# Agent lifecycle: spawn, track, kill, restart
│   ├── agent_session.py# Claude Agent SDK wrapper per agent
│   ├── message_router.py# Async queue-based inter-agent messaging
│   ├── mcp_server/     # 15 orchestration MCP tools
│   └── providers/      # Claude, Codex, Gemini backends
├── adapters/           # Engine-to-UI bridge layer
│   ├── orchestrator.py # OrchestratorBridge — Future-based permission/question resolution
│   ├── event_bus.py    # Async event queue (engine → UI)
│   ├── events.py       # 17 typed event dataclasses
│   └── file_tracker.py # File change detection & diff computation
├── shared/             # Models, services, utilities
│   ├── models/         # AgentNode, Message, Session
│   ├── services/       # Persistence, plugins, project, snapshot
│   └── file_utils.py   # @reference resolution, FileIndex
└── vscode/             # HTTP+SSE server for VS Code extension
    ├── server.py       # Multi-session aiohttp server (40+ REST endpoints)
    └── extension/      # TypeScript VS Code extension
```

### How it works

1. You type a prompt in the VS Code chat panel
2. The extension sends it to the **HTTP+SSE server**, which routes it to the **orchestration engine**
3. The **master agent** can spawn child agents, consult domain experts, and use tools
4. All agent activity streams back via SSE as markdown
5. The **agent tree** in the sidebar shows the live hierarchy — click a node to view that agent's output
6. **Permission requests** appear as modal dialogs when agents want to execute tools

### Agent hierarchy

```
Master (orchestrator)
├── Worker (implementation task)
│   ├── Worker (subtask A)
│   └── Worker (subtask B)
├── Expert (rust-systems)
└── Expert (code-reviewer)
```

Agents communicate via async message queues. The engine includes deadlock detection — if agents form a circular wait, the deepest agent in the cycle is killed automatically.

## Configuration

PRSM is configured via `.prism/prsm.yaml` in your project root. See [docs/configuration.md](docs/configuration.md) for the full reference.

### Minimal config

```yaml
defaults:
  model: opus

models:
  opus:
    provider: claude
    model_id: claude-opus-4-6
```

## Development

```bash
# Install with dev deps
./install.sh --dev

# Run tests
.venv/bin/python -m pytest tests/ -v

# Rebuild extension and restart server
./start_prsm_server.sh
```

## Documentation

Detailed documentation is in the [`docs/`](docs/) directory:

- **[Architecture](docs/architecture.md)** — Package layout, entry points, modes of operation
- **[Engine](docs/engine.md)** — Orchestration engine, agent lifecycle, MCP tools, providers
- **[Adapters](docs/adapters.md)** — Event bus, bridge, permission resolution, file tracking
- **[Shared](docs/shared.md)** — Models, persistence, plugins, project management, snapshots
- **[VSCode](docs/vscode.md)** — HTTP+SSE server, REST API, VS Code extension
- **[Data Flow](docs/data-flow.md)** — End-to-end traces: prompts, events, permissions, messaging
- **[Configuration](docs/configuration.md)** — YAML config, env vars, plugins, project storage

## License

MIT
