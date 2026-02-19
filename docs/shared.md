# Shared Layer

The shared layer (`prsm/shared/`) provides models, services, formatters, and utilities used by both TUI and VSCode frontends.

## Models

### AgentNode (`models/agent.py`)

The UI representation of an agent in the tree hierarchy. Uses engine `AgentState` (9 states) and `AgentRole` (4 roles) directly — no duplicate enums.

```python
@dataclass
class AgentNode:
    id: str
    name: str
    state: AgentState = AgentState.PENDING
    role: AgentRole | None = None
    model: str = "claude-opus-4-6"
    parent_id: str | None = None
    children_ids: list[str]
    prompt_preview: str = ""
    created_at: datetime | None = None
    completed_at: datetime | None = None
    last_active: datetime | None = None
    error: str | None = None
    tools: list[str] | None = None
    depth: int = 0
    cwd: str | None = None
    permission_mode: str | None = None
    provider: str = "claude"
```

### Message (`models/message.py`)

```python
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    result: str | None = None
    success: bool = True

@dataclass
class Message:
    role: MessageRole
    content: str
    agent_id: str
    snapshot_id: str | None = None  # Snapshot captured before this user prompt
    id: str                         # Auto-generated 8-char UUID prefix
    timestamp: datetime             # UTC-aware
    tool_calls: list[ToolCall]
    streaming: bool = False
```

### Session (`models/session.py`)

Primary state container for the entire UI.

```python
@dataclass
class WorktreeMetadata:
    root: str                       # Absolute worktree path
    branch: str | None = None
    common_dir: str | None = None   # Git common dir

@dataclass
class Session:
    agents: dict[str, AgentNode]
    messages: dict[str, list[Message]]  # Per-agent message history
    active_agent_id: str | None
    name: str | None
    created_at: datetime | None
    forked_from: str | None
    worktree: WorktreeMetadata | None
```

Methods: `add_agent()`, `remove_agent()`, `set_active()`, `get_active_agent()`, `add_message()`, `get_messages()`, `get_active_messages()`, `clear_messages()`, `fork(new_name, new_worktree)` (deep copy).

Property: `message_count` — total messages across all agents.

Helpers: `format_forked_name(name)` — prepends "(Forked) " prefix. `is_default_session_name(name)` — True for "Session N" pattern names.

## Services

### Session Persistence (`services/persistence.py`)

JSON file serialization with git worktree awareness. Class: `SessionPersistence`.

**Storage layout:** `~/.prsm/sessions/{repo_identity}/{session_id}.json`

`repo_identity` is git-common-dir-based (stable across worktrees) or path-based for non-git dirs.

**Key operations:**
- `save(session, name, session_id)` — Serializes to JSON v1.3 format. Tags with worktree metadata. Writes `.active_session` marker in project_dir. Uses `atomic_write_text()`.
- `load(name)` — Deserializes + resets stale states (RUNNING/WAITING/STARTING → COMPLETED on restore). Warns on worktree mismatch.
- `load_with_meta(name)` — Returns `(Session, metadata_dict)` with saved_at, session_id, name.
- `auto_resume()` — Loads most recently modified session (by file mtime).
- `list_sessions()` — Alphabetically sorted session names.
- `list_sessions_by_mtime()` — Most recently modified first.
- `list_sessions_detailed(all_worktrees)` — Lists with metadata (name, session_id, saved_at, forked_from, agent_count, message_count, worktree_root, branch). Filterable by current worktree.
- `fork(session, new_name)` — Deep-copy session with new identity.
- `delete(name)` — Removes session file.

Includes legacy migration from `~/.prsm/projects/{ID}/sessions/` and basename-based `~/.prsm/sessions/{dirname}/` layouts.

Properties: `project_dir`, `workspace`.

### Plugin Manager (`services/plugins.py`)

Manages external MCP server plugins with three transport types: stdio, http, sse.

**Config sources (in load order):**
1. Project-level: `~/.prsm/projects/{ID}/plugins.json`
2. Workspace: `.prsm.json` then `.mcp.json` (`.mcp.json` loaded last, overrides same-name entries from `.prsm.json`)

**PluginConfig fields:** name, type, command, args, env, url, headers, tags.

**Operations:**
- `add(name, command, args)` — Register stdio plugin
- `add_remote(name, transport_type, url, headers, tags)` — Register http/sse plugin
- `add_json(name, config)` — Register from raw JSON config dict (same format as `.mcp.json` entries)
- `remove(name)` — Unregister
- `list_plugins()` — Return all registered `PluginConfig` objects
- `get_mcp_server_configs()` — Build full MCP server config dict for agent sessions
- `get_plugins_for_agent(prompt, role)` → `dict[str, Any]` — Tag-matched MCP configs using PluginMatcher

**PluginMatcher rules:**
- Master → no plugins
- Worker → all plugins
- Expert/Reviewer → untagged + tag-matched (tag found as substring in prompt)

### Project Manager (`services/project.py`)

Maps working directories to project storage paths with full git worktree awareness.

**Data models:**
- `WorktreeInfo` — path, head, branch, detached, bare, locked, lock_reason
- `RepositoryContext` — is_git_repo, is_worktree, repo_identity, worktree_root, branch, common_dir, all_worktrees

**Key methods (all static):**
- `get_project_dir(cwd)` → `~/.prsm/projects/{repo_identity}/`
- `get_repo_identity(cwd)` — Git-common-dir-based stable identity
- `get_memory_path(project_dir)` → `{project_dir}/memory/MEMORY.md`
- `get_sessions_dir(project_dir)` → `{project_dir}/sessions/`
- `get_repository_context(cwd)` — Complete git state (identity, branch, worktrees, common_dir, etc.)
- Git helpers: `get_git_branch()`, `is_git_repo()`, `is_worktree()`, `get_git_common_dir()`, `get_git_dir()`, `get_worktree_root()`, `list_worktrees()`, `create_worktree()`, `remove_worktree()`

### Snapshot Service (`services/snapshot.py`)

Captures and restores combined session state + working tree state.

**Storage:** `~/.prsm/projects/{ID}/snapshots/{snapshot_id}/` containing:
- `meta.json` — Metadata (snapshot_id, session_name, session_id, parent_snapshot_id, agent_id, agent_name, parent_agent_id, description, timestamp, git_branch)
- `session.json` — Serialized session state
- `working_tree.patch` — `git diff HEAD` output
- `untracked/` — Copies of untracked files
- `file-changes/` — FileChangeTracker records

**Operations:**
- `create(session, session_name, description, file_tracker, session_id, parent_snapshot_id, agent_id, agent_name, parent_agent_id)` — Atomic 5-step capture using staging directory + `os.replace()`
- `restore(snapshot_id)` → `(Session, FileChangeTracker)` — 5-step restore: reset tracked files, clean post-snapshot untracked files, apply patch, restore untracked, deserialize session + file changes
- `load_session(snapshot_id)` → `(Session, FileChangeTracker)` — Load snapshot state without restoring files
- `list_snapshots()` — All snapshots with metadata
- `list_snapshots_by_session(session_id)` — Filter by session
- `group_snapshots_by_session()` — Group all snapshots by session_id
- `delete(snapshot_id)` — Remove snapshot directory
- `get_meta(snapshot_id)` — Read metadata only

### Project Memory (`services/project_memory.py`)

Manages persistent per-project notes (like Claude Code's MEMORY.md).

```python
class ProjectMemory:
    def __init__(self, memory_path: Path): ...
    def load(self) -> str: ...
    def save(self, content: str): ...
    def exists(self) -> bool: ...
```

### User Preferences (`services/preferences.py`)

Persistent per-user settings stored at `~/.prsm/preferences.json`.

```python
@dataclass
class UserPreferences:
    file_revert_on_resend: str = "ask"  # "ask" | "always" | "never"
    enable_nsfw_thinking_verbs: bool = True
    custom_thinking_verbs: list[str] = field(default_factory=list)
```

Methods: `save(path)`, `load(path)` (class method, returns defaults on failure), `validate()`.

### Command Policy Store (`services/command_policy_store.py`)

Workspace bash command whitelist/blacklist persistence. Stores regex patterns in:
- `<workspace>/.prism/command_whitelist.txt`
- `<workspace>/.prism/command_blacklist.txt`

Default blacklist patterns block `rm` and `git commit`.

Methods: `ensure_files()`, `load_compiled()` → `CommandPolicyRules`, `read_whitelist()`, `read_blacklist()`, `add_whitelist_pattern()`, `add_blacklist_pattern()`, `remove_whitelist_pattern()`, `remove_blacklist_pattern()`, `build_command_pattern()`.

`build_command_pattern()` is used by permission persistence to derive regex rules from terminal commands:
- Allow rules: command-class-aware templates (destructive commands are strict; exploratory commands are generalized)
- Deny rules: exact-command regexes

### Process Cleanup (`services/process_cleanup.py`)

Best-effort cleanup for stale runtime subprocesses (orphaned helpers/providers).

`cleanup_stale_runtime_processes(current_pid, include_claude, log)` — Kills processes matching managed signatures (orch_proxy, codex, gemini) that are orphaned (parent PID 1 or missing) and have no PRSM server ancestry.

### Durable Write (`services/durable_write.py`)

Crash-safe file writing utilities.

- `atomic_write_text(path, content, encoding)` — Write via temp file + `os.replace()` + fsync of file and parent directory.
- `_fsync_dir(dir_path)` — Best-effort directory fsync for rename/unlink metadata.

### Session Naming (`services/session_naming.py`)

Identical to `prsm/adapters/session_naming.py`. See adapters docs.

## Formatters

### Tool Call Formatter (`formatters/tool_call.py`)

Registry-based tool call formatting system producing structured intermediate representations (IR) for rendering.

**IR types:**
- `Section(kind, title, content)` — Typed content section. Kinds: `diff`, `code`, `path`, `checklist`, `kv`, `plain`, `progress`, `agent_prompt`, `transcript`, `result_block`
- `FormattedToolCall(icon, label, summary, file_path, sections)` — Structured representation of a formatted tool call

**Entry point:** `format_tool_call(name, arguments, result, success)` → `FormattedToolCall`. Dispatches to registered formatters, strips MCP prefixes (`mcp__orchestrator__` → bare name).

**Registered formatters:** Edit, Bash, Read, Write, Glob, Grep, Task, TodoWrite, WebFetch, WebSearch, NotebookEdit, Skill, plus orchestration tools: task_complete, spawn_child, spawn_children_parallel, restart_child, ask_parent, ask_user, wait_for_message, respond_to_child, consult_expert, consult_peer, report_progress, get_child_history, check_child_status, send_child_prompt, get_children_status, recommend_model.

**Rendering:** `render_collapsed_rich(fmt, status)` and `render_expanded_rich(fmt, status, timestamp)` convert IR to Rich markup for the TUI.

**Helper:** `parse_args(arguments)` — Parses tool arguments (JSON → ast.literal_eval → `{"_raw": ...}` fallback).

## Utilities

### Slash Commands (`commands.py`)

`parse_command(text)` parses `/command args` input. Returns `ParsedCommand(name, args, raw)` or None.

Commands: help, new, session, sessions, fork, save, plugin, policy, settings, worktree.

### File Utilities (`file_utils.py`)

Shared file operations for `@reference` resolution.

**FileEntry** — Data model for file/directory in completion lists: `path`, `is_dir`, `size`.

**FileIndex** — Fast cached project file scanner:
- Uses `os.scandir()`, respects `.gitignore` and SKIP_DIRS
- Max depth: 4, cache TTL: 30s, top 50 results
- Two-tier matching: prefix first, then substring (case-insensitive)

**`resolve_references(text, cwd)`** — Parses `@path` tokens:
- Files → content (truncated at 100KB via `MAX_FILE_SIZE`)
- Directories → tree outline
- Security: skips paths outside cwd, detects binaries (null bytes in first 1KB)
- Ignores `@` inside backtick regions (inline and fenced code blocks)

**`build_tree_outline(dir_path, max_depth)`** — ASCII directory tree for directory @references. Default max_depth=3, respects SKIP_DIRS.

**`format_size(size)`** — Human-readable file size (B/KB/MB).
