# Configuration

PRSM is configured via a YAML file, environment variables, and JSON plugin configs.

## YAML Configuration (`.prism/prsm.yaml`)

The primary config file. Place in `.prism/prsm.yaml` within your project root (auto-discovered), or pass with `--config`. Legacy `prsm.yaml` in the project root is also supported as a fallback.

Global model settings are loaded from `~/.prsm/models.yaml` when present. It may contain `models` and `model_registry` sections. Values in the main `prsm.yaml` take precedence on key conflicts.

### Engine Settings

```yaml
engine:
  default_model: claude-opus-4-6     # Default model ID for agents (can be changed in TUI)
  default_cwd: .                     # Default working directory
  max_agent_depth: 5                 # Max nesting depth for child agents
  max_concurrent_agents: 10          # Max agents running simultaneously
  agent_timeout_seconds: 7200        # Max reasoning time per agent (excludes tool time, 0 disables timeout)
  tool_call_timeout_seconds: 7200    # Max wall-clock per tool call (0 disables timeout)
  user_question_timeout_seconds: 0    # Max wait for ask_user/request_user_input replies (0 disables timeout)
  deadlock_check_interval_seconds: 5
  deadlock_max_wait_seconds: 120
  message_queue_size: 1000
  log_level: INFO
  command_whitelist: []              # Regex patterns for auto-allowed bash commands
  command_blacklist: []              # Regex patterns for always-denied bash commands
  command_safety_model_enabled: false
  command_safety_model: null         # Optional model for secondary command screening
```

### Package Install Permission Prompts

To force interactive approval prompts for package-manager installs (instead of silent execution attempts), add patterns to `engine.command_blacklist` in `.prism/prsm.yaml`:

```yaml
engine:
  command_blacklist:
    - '(^|\\s)sudo\\s+apt(\\s|$)'
    - '(^|\\s)apt(-get)?\\s+install(\\s|$)'
    - '(^|\\s)snap\\s+install(\\s|$)'
    - '(^|\\s)(pip|pip3)\\s+install(\\s|$)'
    - '(^|\\s)python(\\d+(\\.\\d+)?)?\\s+-m\\s+pip\\s+install(\\s|$)'
```

Note: this only controls PRSM's command approval callback. Host runtime policy (approval mode, sandbox level, network, and sudo availability) must also permit escalated installs.

### Providers

Define AI provider backends. Supported types: `claude`, `codex`, `gemini`, `minimax`, `alibaba`.

```yaml
providers:
  claude:
    type: claude
  codex:
    type: codex
    command: codex              # Path to Codex CLI binary
  gemini:
    type: gemini
    command: gemini             # Path to Gemini CLI binary
  minimax:
    type: minimax
    profile: minimax            # Codex CLI profile name for MiniMax
  alibaba:
    type: alibaba
    command: codex              # Path to Codex CLI binary
    profile: alibaba            # Codex model_provider value (default: alibaba)
    api_key_env: DASHSCOPE_API_KEY
```

All providers support an optional `api_key_env` field to specify the environment variable holding the API key.

For Codex-backed execution providers (`codex`, `minimax`, `alibaba`), PRSM runs `codex exec` with `--ephemeral` by default for orchestrator/agent execution. This avoids stale local Codex rollout state affecting new runs.

### Model Aliases

Short names that map to a provider + model ID. The optional `reasoning_effort` field (`low`, `medium`, `high`) is appended to Codex model IDs at runtime.

```yaml
models:
  opus:
    provider: claude
    model_id: claude-opus-4-6
  sonnet:
    provider: claude
    model_id: claude-sonnet-4-5-20250929
  codex:
    provider: codex
    model_id: gpt-5.2-codex
    reasoning_effort: medium     # Optional (Codex only)
  gemini-3:
    provider: gemini
    model_id: gemini-3-pro-preview
  gemini-3-flash:
    provider: gemini
    model_id: gemini-3-flash-preview
```

Built-in Claude family aliases are always available (no YAML needed):
- `claude-opus` → `claude-opus-4-6`
- `claude-sonnet` → `claude-sonnet-4-5-20250929`
- `claude-haiku` → `claude-3-5-haiku-20241022`

### Defaults

```yaml
defaults:
  model: opus                  # Default model alias for new agents
  master_model: sonnet         # Orchestrator model alias (any MCP-capable provider)
  peer_model: codex            # Model for consult_peer (single, backward compat)
  peer_models: [codex, gemini-3]  # Multiple peers (overrides peer_model)
  cwd: /path/to/project        # Override default working directory
```

When `peer_models` is set, child agents spawned via `spawn_child`, `spawn_children_parallel`, and `recommend_model` are restricted to only use models in this list. Models not in the peer list cannot be selected for child agents. Expert agents (via `consult_expert`) are not subject to this restriction — they use their own explicitly configured models.

If `master_model` is not set, the `model` default is used when its provider supports master mode; otherwise falls back to `claude-sonnet-4-5-20250929`.

### Model Registry

Maps model IDs to capability tiers and task affinities for intelligent child-agent model selection. Can also live in `~/.prsm/models.yaml`.

```yaml
model_registry:
  claude-opus-4-6:
    tier: frontier
    provider: claude
    affinities:
      architecture: 0.95
      complex-reasoning: 0.95
```

### Experts

Domain specialists that agents can consult via `consult_expert()`:

```yaml
experts:
  rust-systems:
    name: Rust Systems Expert
    description: Tokio, crossbeam, database integration
    system_prompt: |
      You are a Rust systems programming expert.
      Focus on async runtimes and concurrency.
    tools: [Read, Grep, Glob, Bash]
    model: opus                 # Alias reference
    permission_mode: default    # default | acceptEdits | bypassPermissions | bypass | plan | delegate
    max_concurrent_consultations: 3
    cwd: src/                   # Relative to defaults.cwd
    mcp_servers:                # Expert-specific MCP servers
      browser:
        type: stdio
        command: npx
        args: ["-y", "@anthropic/mcp-server-puppeteer"]
```

### Plugins

External MCP servers available to agents:

```yaml
plugins:
  # stdio: local subprocess
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    env:                        # Optional env vars for subprocess
      HOME: /tmp
    tags: [filesystem, code]

  # http: remote streamable HTTP
  github:
    type: http
    url: https://mcp.github.com/v1
    headers:
      Authorization: "Bearer ${GITHUB_TOKEN}"
    tags: [github, vcs]

  # sse: Server-Sent Events
  database:
    type: sse
    url: http://localhost:3001/sse
    tags: [database, sql]
```

## Environment Variables

### Engine (`ORCH_*`)

All engine settings can be set via `ORCH_*` environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_DEFAULT_MODEL` | `claude-opus-4-6` | Default model for agents |
| `ORCH_DEFAULT_PROVIDER` | `claude` | Default provider for agents |
| `ORCH_DEFAULT_CWD` | `.` | Default working directory |
| `ORCH_MAX_DEPTH` | `5` | Max agent nesting depth |
| `ORCH_MAX_AGENTS` | `10` | Max concurrent agents |
| `ORCH_AGENT_TIMEOUT` | `7200` | Agent reasoning timeout (seconds, 0 disables timeout) |
| `ORCH_TOOL_TIMEOUT` | `7200` | Tool call timeout (seconds, 0 disables timeout) |
| `ORCH_USER_QUESTION_TIMEOUT` | `0` | User question timeout (seconds, 0 disables timeout) |
| `ORCH_DEADLOCK_INTERVAL` | `5` | Deadlock check interval (seconds) |
| `ORCH_DEADLOCK_MAX_WAIT` | `120` | Max wait before deadlock (seconds) |
| `ORCH_QUEUE_SIZE` | `1000` | Message queue size |
| `ORCH_LOG_LEVEL` | `INFO` | Engine log level |
| `ORCH_COMMAND_SAFETY_MODEL_ENABLED` | `false` | Enable secondary command safety model |
| `ORCH_COMMAND_SAFETY_MODEL` | *(none)* | Model ID for command safety screening |
| `ORCH_CONFIG_FILE` | *(none)* | YAML config path (alternative to `--config`) |

### Application (`PRSM_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PRSM_MODEL` | `claude-opus-4-6` | Override model for preflight check |
| `PRSM_CLAUDE_CLI_PATH` | `claude` | Path to Claude CLI binary |
| `PRSM_SDK_PREFLIGHT` | `0` | Enable SDK preflight check on startup |
| `PRSM_SDK_PREFLIGHT_STRICT` | `0` | Fail on preflight error |
| `PRSM_SDK_PREFLIGHT_TIMEOUT_SECONDS` | `15` | Preflight timeout |
| `PRSM_LOG_LEVEL` | `INFO` | Server log level |
| `PRSM_SESSION_INACTIVITY_MINUTES` | `15` | Auto-cleanup idle sessions (server mode) |
| `PRSM_CLEANUP_STALE_CLAUDE` | `0` | Clean up stale Claude processes on startup |
| `PRSM_PROBE_CLAUDE_MODELS_ON_STARTUP` | `0` | Probe available models on startup |

### Transport resilience (`PRSM_TRANSPORT_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PRSM_TRANSPORT_BREAKER_THRESHOLD` | `3` | Failures before circuit breaker opens |
| `PRSM_TRANSPORT_BREAKER_COOLDOWN_SECONDS` | `20.0` | Cooldown after breaker trips |
| `PRSM_TRANSPORT_RETRY_ATTEMPTS` | `5` | Max retry attempts per transport call |
| `PRSM_TRANSPORT_RETRY_BASE_DELAY_SECONDS` | `0.75` | Base delay between retries |
| `PRSM_TRANSPORT_RETRY_MAX_DELAY_SECONDS` | `5.0` | Max delay between retries |

## Plugin Config Files

Plugins can also be defined in JSON files (Claude CLI `.mcp.json` format):

### `.mcp.json` (workspace-level, preferred)
```json
{
  "mcpServers": {
    "orchestrator": {
      "type": "stdio",
      "command": ".venv/bin/python",
      "args": ["-m", "prsm.engine.mcp_server.stdio_server", "--config", ".prism/prsm.yaml"]
    }
  }
}
```

### `.prsm.json` (workspace-level, legacy fallback)
Same format as `.mcp.json`. If both exist, `.mcp.json` wins for duplicate names.

### `~/.prsm/projects/{ID}/plugins.json` (project-level)
User-specific plugin configs. Higher precedence than workspace-level.

### Load order / precedence
1. `plugins.json` (project-level) overrides workspace-level
2. `.mcp.json` overrides `.prsm.json`
3. YAML `plugins:` section merged with JSON configs

## Plugin Auto-Matching

Plugins tagged with keywords are automatically assigned to agents based on role:

- **Master agents** → No plugins (they delegate to children)
- **Worker agents** → All plugins available
- **Expert/reviewer agents** → Untagged plugins + plugins whose tags match prompt keywords

Tag plugins via YAML config or `/plugin tag NAME tag1 tag2`.

## Project Storage

Per-project data is stored at `~/.prsm/projects/{repo_identity}/`:

```
~/.prsm/
├── models.yaml                # Global model aliases + model_registry
├── allowed_tools.json          # Global tool permissions
├── preferences.json            # User preferences (file revert behavior, etc.)
├── model_intelligence.json     # Learned model→task rankings
├── logs/
│   └── prsm-server.log         # Server log
├── sessions/
│   └── {repo_identity}/        # Per-repo session storage
│       ├── {session_id}.json
│       └── .active_session
└── projects/
    └── {repo_identity}/
        ├── allowed_tools.json  # Project tool permissions
        ├── plugins.json        # Project plugins
        ├── memory/
        │   └── MEMORY.md       # Project memory
        └── snapshots/
            └── {snapshot_id}/
                ├── meta.json
                ├── session.json
                ├── working_tree.patch
                ├── untracked/
                └── file-changes/
```

`repo_identity` is based on git common-dir (stable across worktrees) or path hash for non-git directories.
