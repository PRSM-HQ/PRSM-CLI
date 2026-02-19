# File Change Tracking & Management

Agent file edits are isolated in per-session git worktrees and automatically synced to the workspace as patches. This document covers the full lifecycle: detection, normalization, auto-sync, accept/reject, persistence, and session restore.

---

## Overview

```
Agent (writes to worktree)
    ↓ ToolCallCompleted event
_on_tool_call_completed
    ↓ detect changes (3 paths)
FileChangeRecord created
    ↓ normalize path (worktree → workspace)
_normalize_file_change_record_path
    ↓ apply delta immediately
_sync_file_to_workspace → _apply_diff_to_workspace_file
    ↓ broadcast SSE
file_changed event → VS Code extension
    ↓ user can explicitly accept/reject via UI
_handle_accept_change / _handle_reject_change
```

Each session's agents write into an isolated git worktree at `/tmp/prsm-wt-{session_id}`. Changes are detected after every tool call, normalized to workspace-rooted paths, and applied to the workspace immediately as diff patches. The accept/reject UI gives the user visibility and a manual override, but the workspace is already updated.

---

## Data Model

### `FileChangeRecord` (`prsm/adapters/file_tracker.py`)

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | `str` | Absolute workspace-rooted path. Always normalized away from worktree paths. |
| `agent_id` | `str` | UUID of the agent that made the change. `"unknown"` for reconciled records. |
| `change_type` | `str` | `"create"`, `"modify"`, or `"delete"`. |
| `tool_call_id` | `str` | The tool call that caused the change. Multi-file Bash changes get `:N` suffixes (`tool_id:0`, `tool_id:1`). Reconciled records: `reconcile-{session_id[:8]}-N`. |
| `tool_name` | `str` | Canonical tool name after alias normalization (`write_file` → `Write`, `mcp__x__edit` → `Edit`, etc.). |
| `message_index` | `int` | Index of the message in the agent's history containing this tool call. Used to order cascaded rejections. |
| `old_content` | `str \| None` | For Edit: the `old_string` being replaced. For Write modify: full pre-tool file content. `None` for creates. |
| `new_content` | `str \| None` | Full file content **after** the tool ran. Used for workspace sync and revert fallback. |
| `pre_tool_content` | `str \| None` | Full file content **before** the tool ran. Primary base for patch application and revert. |
| `added_ranges` | `list[dict]` | `[{"startLine": int, "endLine": int}]` — lines added, for UI decorations. |
| `removed_ranges` | `list[dict]` | `[{"startLine": int, "endLine": int}]` — lines removed, for UI decorations. |
| `timestamp` | `str` | ISO 8601 creation time. |
| `status` | `str` | `"pending"` → `"accepted"` or `"rejected"`. |

### `FileChangeTracker` (in-memory store)

```python
file_changes: dict[str, list[FileChangeRecord]]
# key: absolute workspace-rooted file path
# value: ordered list of records for that file
```

Also holds transient pre-tool state keyed by `tool_id`: `_pre_tool_content`, `_tool_call_args`, `_tool_call_names`, `_tool_call_cwds`, `_pre_tool_snapshots`.

---

## Change Detection

Three paths run in sequence. The first to produce records wins for a given tool call.

### Path 1 — Snapshot-Based (Primary)

Handled by `FileChangeTracker` in `file_tracker.py`. This is the normal path for Edit and Write tools.

**Pre-tool capture** (`capture_pre_tool`, called from `_on_tool_call_started`):
- Parses tool arguments (JSON → `ast.literal_eval` → raw fallback)
- Normalizes tool name via `normalize_tool_name()`
- For Bash: `_infer_file_tool_from_bash_command()` parses shell commands to detect `sed -i` → Edit, `tee`/echo redirects → Write. Handles chained commands (`&&`, `||`, `;`), pipes, and shell wrappers (`bash -c`, `zsh -lc`)
- Runs `git status --porcelain --untracked-files=all` in the agent cwd and stores path+hash snapshot
- Reads target file content into `_pre_tool_content`

**Post-tool tracking** (`track_changes`, called from `_on_tool_call_completed`):
- Takes a new `git status` snapshot
- Compares pre/post hashes for all candidate paths — creates a `FileChangeRecord` for each file whose hash changed
- `change_type` derived from hash presence: new only → create, old only → delete, both changed → modify
- Line ranges computed by `_compute_line_ranges()`: linear scan for first/last differing line
- If snapshot diff produces nothing: `_fallback_record_from_args()` reads the file and compares with stored pre-tool content (skips Read and Bash tools explicitly)

### Path 2 — Worktree File Change Fallback

`_worktree_file_change_fallback`, activated when Path 1 returns empty and `state._worktree_path` is set.

Handles the event timing race where `ToolCallStarted` is processed after the tool already ran (pre-snapshot = post-snapshot = already modified state).

- Scans session message history backwards to find tool arguments for the given `tool_call_id`
- Reads current file content from the worktree
- Gets HEAD content via `git show HEAD:{rel_path}` as the base
- Pre-content: uses the last existing record's `new_content` for sequential edits; falls back to HEAD content
- Deduplication: skips if any record with this `tool_call_id` already exists (across all file paths)
- Handles Edit and Write directly; delegates other tools to Path 3

### Path 3 — Git Diff Fallback

`_worktree_diff_fallback`, used for Bash and other multi-file tools.

- Runs `git diff HEAD --name-only` + `git ls-files --others --exclude-standard` in the worktree
- Builds a dedup set containing **both** workspace and worktree forms of every already-tracked path — prevents creating duplicate records when workspace-keyed records already exist for worktree-path files
- Filters `.git/` paths and workspace scratch files
- Creates one record per untracked changed file; assigns synthetic `:N` suffixes to `tool_call_id` for multi-file changes

---

## Path Normalization

All records must carry workspace-rooted absolute paths. Worktree paths (`/private/tmp/prsm-wt-{session_id}/...`) are mapped to workspace paths at every ingestion point.

### Key Functions

**`_canonicalize_file_path_for_state(state, file_path) → str`**
Core mapper. Resolves to absolute path, then:
1. If under `workspace_root` → return as-is
2. If under `state._worktree_path` → strip prefix, prepend workspace root
3. If under `session.worktree.root` (legacy, for dead worktrees) → same strip/prepend
4. Otherwise → return as-is (logs warning at call site)

**`_normalize_file_change_record_path(state, record)`**
Applies canonicalization to a record. If the path changed:
- Removes the record from the old key in `file_changes`
- Inserts it under the new canonical key (deduplicating by `tool_call_id`)
- Logs a warning if the result is still outside workspace root

**`_normalize_loaded_file_change_paths(state) → bool`**
Batch normalizer run at load time (before worktree reconnect). Returns `True` if anything changed, triggering a re-persist.

**`_on_tool_call_completed`**
Normalizes every new record inline immediately after creation.

**`_reconcile_worktree_changes`**
Normalizes inline after creating each reconcile record (worktree is live at this point, so mapping always succeeds).

### macOS `/private/tmp` Handling

`Path.resolve()` on macOS expands `/tmp` → `/private/tmp`. All path comparison code uses `.resolve()` consistently so both forms match.

---

## Auto-Sync to Workspace

Every new `FileChangeRecord` is immediately synced to the workspace upon creation. No user action is required. This happens in two places:

- `_on_tool_call_completed` (line ~1228): after normalization, before broadcasting the SSE event
- `_reconcile_worktree_changes` (line ~3440): after normalization of each reconcile record

### `_sync_file_to_workspace(state, record)`

1. **Resolve workspace path**
   - Worktree alive → `_worktree_to_workspace_path(state, record.file_path)`
   - Worktree gone, was a worktree session → `_resolve_workspace_path_from_record(state, record)` (strips `/tmp/prsm-wt-*` prefix, handles `/private/tmp` variant)
   - Worktree gone, never was a worktree session → return (file is already in workspace)

2. **Delete**: `ws_path.unlink(missing_ok=True)`

3. **Create / file doesn't exist**: direct write of `new_content`

4. **Modify**: delegate to `_apply_diff_to_workspace_file`

### `_apply_diff_to_workspace_file(ws_path, pre_content, new_content, ...)`

Applies only the delta from this session's change, preserving concurrent edits from other sessions.

```
pre_content  →  new_content      (what this session changed)
     ↓
diff(pre_content, new_content)   (the patch)
     ↓
apply patch to current workspace file
```

**No divergence** (`current_workspace == pre_content`): direct write of `new_content`. This is the fast path and covers the common single-session case.

**Diverged workspace**: `difflib.unified_diff(pre_content, new_content)` → temp `.patch` file → `patch -u ws_path patch_file`. If `patch` fails (non-zero exit): falls back to direct write with warning. If `patch` binary not found: same fallback.

This is critical for multi-session scenarios. If two sessions both edit the same file starting from HEAD, accepting the second session after the first must not clobber the first session's changes — only the second session's delta is applied.

---

## Accept / Reject Flow

The explicit accept/reject UI exists for user visibility and manual override. The workspace is already updated by the time the user acts.

### Accept (`POST /sessions/{id}/file-changes/{tool_call_id}/accept`)

1. Find record by `tool_call_id`
2. Set `status = "accepted"`
3. Call `_sync_file_to_workspace` (idempotent — divergence check writes nothing if workspace already matches)
4. Broadcast `file_change_status` SSE: `{"status": "accepted", "file_path": ..., "tool_call_id": ...}`
5. Persist, then `_maybe_cleanup_empty_worktree`

### Reject (`POST /sessions/{id}/file-changes/{tool_call_id}/reject`)

1. Find record
2. `_revert_workspace_file`:
   - `create` → delete file
   - `modify` / `delete` → write `pre_tool_content` (or `old_content`) back to the workspace file
3. Set `status = "rejected"`
4. **Cascade**: all later pending records for the same file (higher `message_index`) are also set to `"rejected"` — reverting a change invalidates the base for everything that followed
5. Broadcast `file_change_status` for the target and all cascaded records
6. Persist, maybe clean up worktree

### Bulk Accept-All (`POST /sessions/{id}/file-changes/accept-all`)

Iterates all pending records across all files, accepts and syncs each.

### Bulk Reject-All (`POST /sessions/{id}/file-changes/reject-all`)

For each file, finds the earliest pending record, reverts to its `pre_tool_content`, then marks all pending records for that file as rejected.

### Client-Side (TypeScript `FileChangeTracker`)

The extension maintains an in-memory `Map<string, TrackedFileChange[]>` mirror of server state.

- **Accept**: marks accepted locally, fires `acceptFileChange()` HTTP call (fire-and-forget)
- **Reject**: performs the revert directly in VS Code, then fires `rejectFileChange()`:
  - Edit tool: finds `oldContent` in the current open document via `WorkspaceEdit`, replaces with... falls back to writing `preToolContent` via `fs.writeFileSync` if the find-and-replace fails
  - Write tool: writes `preToolContent` or `oldContent` via `fs.writeFileSync`
- **Scope levels**: individual change, per-file, per-session, global — each with accept and reject variants
- `onDidChange` event emitter fires on every mutation, triggering all UI updates (decorations, tree view, CodeLens)

---

## Persistence

Records are stored as individual JSON files:

```
~/.prsm/projects/{repo_identity}/sessions/{session_id}/file-changes/{tool_call_id}.json
```

Each file is one serialized `FileChangeRecord`.

`_persist_file_changes(state)`:
- Calls `file_tracker.persist(changes_dir)` which writes all in-memory records and **removes stale on-disk files** — any `.json` file whose stem is not a known `tool_call_id` is deleted
- Calls `_save_session(state)` to persist session metadata

Persist is triggered after:
- New changes detected in `_on_tool_call_completed`
- Accept or reject operations
- Reconciliation finding untracked changes
- Path normalization changing any record

---

## Session Restore & Reconcile

On server startup, sessions are lazy-loaded from disk. When a session is first accessed, `_load_file_changes` runs:

1. `file_tracker.load(changes_dir)` — reads all `{tool_call_id}.json` files, deduplicates by `tool_call_id`
2. `_normalize_loaded_file_change_paths(state)` — fixes legacy/worktree paths; re-persists if anything changed
3. `_reconnect_session_worktree(state)` — restores the live worktree link

### `_reconnect_session_worktree`

- Checks `session.worktree.root` and `/tmp/prsm-wt-{session_id}` for an existing worktree directory
- Verifies with `git rev-parse --is-inside-work-tree`
- Sets `state._worktree_path`
- Calls `_reconcile_worktree_changes`
- Calls `_maybe_cleanup_empty_worktree` to prune if already clean

### `_reconcile_worktree_changes`

Creates records for worktree changes that happened during a server downtime (the agent wrote to the worktree but no `ToolCallCompleted` event was processed):

1. `git diff HEAD --name-only` + `git ls-files --others --exclude-standard` in the worktree
2. Dedup set with both workspace and worktree forms of all known paths
3. For each untracked changed file: create record with `agent_id="unknown"`, HEAD content as `old_content` / `pre_tool_content`, current worktree content as `new_content`
4. Normalize path inline (worktree is alive, mapping always succeeds)
5. Sync to workspace immediately
6. Persist if any records were created

### Worktree Lifecycle

- **Created**: `_setup_session_worktree` — `git worktree add /tmp/prsm-wt-{session_id}`; bridge reconfigured to use worktree cwd
- **Pruned when empty**: `_maybe_cleanup_empty_worktree` — if no pending changes remain after accept-all or reconcile, removes the worktree
- **Orphan cleanup**: `_prune_orphaned_worktrees` at server startup — removes any `/tmp/prsm-wt-*` directory not belonging to a known session

### Frontend Restore

`FileChangeTracker.restoreFromServer()` bulk-loads records from `GET /sessions/{id}/file-changes`. Deduplicates by `tool_call_id`, preserves status (pending/accepted/rejected), resolves agent display names via optional resolver callback.

---

## Filtering

Both server and client filter out workspace scratch files at every ingestion point:

- Paths containing any segment starting with `.tmp`
- `.file_index.txt`
- `.grep_*.txt`
- `.pytest_*.txt`
- `.diff_*.txt`
- `.compile_*.txt`
- `*.rej` and `*.orig` — patch(1) artifacts from diverged-workspace diff application

Server: `_is_filtered_workspace_artifact_path()` — used in all three detection paths and reconciliation.
Client: `isIgnoredTmpPath()` in `FileChangeTracker`.

Additionally, `_apply_diff_to_workspace_file` proactively cleans up `.rej` and `.orig` files after running `patch -u`, preventing them from persisting in the workspace or worktree.

---

## Concurrency & Locking

All file change operations on the server are protected by `state._file_sync_lock` (an `asyncio.Lock`), preventing races between concurrent tool completions, accept/reject requests, and reconciliation.

```python
async with state._file_sync_lock:
    records = state.file_tracker.track_changes(...)
    for record in records:
        self._normalize_file_change_record_path(state, record)
        self._sync_file_to_workspace(state, record)
        ...
```

---

## Edge Cases

**Event timing race**: The snapshot-based tracker can miss changes if `ToolCallStarted` arrives after the tool already ran (pre-snapshot equals post-snapshot). Paths 2 and 3 exist to catch these.

**Bash tool inference**: `_infer_file_tool_from_bash_command()` parses shell commands to classify `sed -i` as Edit and echo/tee redirects as Write, enabling pre-tool capture even for shell-based file edits.

**Cascade rejection**: Rejecting a change automatically rejects all later pending changes to the same file (`message_index` ordering), because reverting invalidates the base those later changes were applied on.

**Patch failure fallback**: If `patch -u` returns non-zero or the `patch` binary is absent, `_apply_diff_to_workspace_file` falls back to a direct write of `new_content`. The workspace gets the change but concurrent edits from other sessions may be lost for that file.

**Dead worktree path recovery**: When a session's worktree has been deleted but records still reference its path, `_resolve_workspace_path_from_record` strips the `/tmp/prsm-wt-*` (or `/private/tmp/prsm-wt-*`) prefix and resolves relative to the workspace root.

**Multi-file tool calls**: A single Bash tool call that modifies N files produces N records with synthetic `tool_call_id:0` … `tool_call_id:N-1` suffixes to keep them unique in persistence.

**Cold session normalization**: Sessions not loaded into memory can still have their file change paths normalized for the REST API response via `_normalize_cold_file_change_paths()`, which uses `_display_file_path_for_session_id()` for best-effort `/tmp/{session_id}/...` → workspace mapping.

---

## Key Files

| File | Role |
|------|------|
| `prsm/adapters/file_tracker.py` | `FileChangeRecord` data model, `FileChangeTracker` with snapshot-based detection |
| `prsm/vscode/server.py` | All server-side tracking, normalization, sync, accept/reject, persist, restore logic |
| `prsm/vscode/extension/src/tracking/fileChangeTracker.ts` | Frontend in-memory store, UI event driver, client-side revert |
