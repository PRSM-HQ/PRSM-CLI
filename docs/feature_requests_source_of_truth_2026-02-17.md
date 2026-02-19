# Feature Requests Source of Truth (2026-02-17)

This document treats user requests captured in session artifacts as the canonical requirements baseline.

## Canonical Requests

1. File changes pane must show edits from worktrees (including Codex-style string path args for `Write`/`Edit`).
2. VS Code editor must show inline red/green diff, with `Accept` / `Reject` / `Inspect`; `Inspect` must jump to the originating tool call in chat.
3. Chat header content for orchestrator/session should use task metadata (title/description), not generic or clipped prompt defaults.
4. When tracked file-change records are missing, server must fall back to snapshot `working_tree.patch`.
5. Patch-fallback file-change entries must be actionable via normal accept/reject flow.
6. File changes tree should render as a rooted tree with compact-folder behavior aligned to VS Code explorer settings.
7. File-change paths from managed worktrees/tmp paths must be remapped to workspace paths consistently (keys and payload).
8. Opening chat from agent tree should target the currently selected tab group (not force left pane).
9. Snapshot tree should assign unassigned snapshots to an orchestrator bucket and flatten snapshot listing under each agent.
10. Snapshot agent labels should include UUID in the tree display.
11. Session default-name handling should correctly detect canonical default names (including forked variants).

## Implementation Gap Audit

| Request | Status | Evidence | Notes / Gap |
|---|---|---|---|
| 1 | Implemented | `prsm/adapters/file_tracker.py`, `tests/test_file_tracker_tool_aliases.py` | Tests cover plain-string and JSON-string path args for `Write`/`Edit`. |
| 2 | Implemented | `prsm/vscode/extension/src/extension.ts`, `prsm/vscode/extension/src/tracking/changeLensProvider.ts`, `prsm/vscode/extension/src/tracking/decorationProvider.ts`, `prsm/vscode/extension/package.json`, `prsm/vscode/extension/media/webview/main.js` | Inspect commands and tool-call scroll wiring are present. |
| 3 | Implemented | `prsm/vscode/extension/media/webview/main.js`, `prsm/shared/services/session_naming.py`, `prsm/vscode/server.py`, `prsm/shared/services/persistence.py`, `tests/test_session_persistence_metadata.py` | Session summary metadata now persists through save/load/index and is returned in session APIs/events used by orchestrator header rendering. |
| 4 | Implemented | `prsm/vscode/server.py`, `tests/test_vscode_file_change_fallback.py` | Patch parser and fallback payload generation exist. |
| 5 | Implemented | `prsm/vscode/server.py`, `tests/test_vscode_file_change_fallback.py` | `_ensure_patch_fallback_records` hydrates tracker records for action endpoints. |
| 6 | Implemented | `prsm/vscode/extension/src/views/fileChangesTreeProvider.ts` | Rooted directory tree and `explorer.compactFolders` compaction are present. |
| 7 | Implemented | `prsm/vscode/server.py`, `prsm/vscode/extension/src/tracking/fileChangeTracker.ts`, `tests/test_vscode_file_change_fallback.py` | Server now canonicalizes live file-change record paths to workspace-root-consistent keys/payloads; restore path canonicalization remains in place. |
| 8 | Implemented | `prsm/vscode/extension/src/views/agentWebviewManager.ts` | Uses `getPreferredViewColumn()` for both agent and orchestrator tabs. |
| 9 | Implemented | `prsm/vscode/extension/src/views/snapshotTreeProvider.ts` | Unassigned snapshots bucket to resolved orchestrator; snapshot nodes are flat under each agent. |
| 10 | Implemented | `prsm/vscode/extension/src/views/snapshotTreeProvider.ts` | `formatAgentLabel()` includes `(<agentId>)`. |
| 11 | Implemented | `prsm/shared/models/session.py`, `tests/test_session_default_names.py` | Canonical default names and forked default names are covered. |

## Verification Notes

- Synchronous/unit coverage for requests 1, 4, 5, 11 is present in tests.
- Async/TUI parity tests in this environment did not run due missing pytest async plugin support (`pytest.mark.asyncio` unrecognized), so those paths were validated by static code inspection only.

## Next Minimal Actions

1. Add an integration test that exercises `session summary -> save -> server restart -> list/restore -> webview header` end-to-end.
2. Add a worktree-originated live `file_changed` SSE integration test validating canonical workspace paths in both map keys and payloads.
