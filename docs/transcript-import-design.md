# Transcript Import Design (Cross-Provider -> PRSM)

## Goal

Import transcript history from locally installed provider sessions (Codex, Claude first) into a PRSM `Session`, so users can continue work in PRSM with prior conversation context.

This is transcript portability, not full runtime portability. We preserve visible chat/tool history, not provider-internal hidden state.

## Non-goals (v1)

- Rehydrating provider-native thread/session internals.
- Replaying tool side effects automatically.
- Perfect semantic equivalence across providers.

## Why This Is Feasible

PRSM already persists sessions as provider-agnostic message history (`Session`, `Message`, `ToolCall`) and agent graph metadata. Codex and Claude both store local JSONL session artifacts that can be parsed into a normalized event stream.

## Source Discovery (v1)

Providers and paths to index:

- Codex:
  - `~/.codex/sessions/**/rollout-*.jsonl` (primary rich transcript)
  - `~/.codex/history.jsonl` (fallback index metadata)
- Claude:
  - `~/.claude/projects/**/*.jsonl` (primary rich transcript)
  - `~/.claude/history.jsonl` and `~/.claude/sessions-index.json` (index metadata)

Discovery should not depend on CLI availability; import works from disk artifacts.

## Proposed Architecture

Add a new import service package:

- `prsm/shared/services/transcript_import/models.py`
  - Canonical import records (`ImportSession`, `ImportTurn`, `ImportToolUse`).
- `prsm/shared/services/transcript_import/providers/base.py`
  - `TranscriptProviderAdapter` interface.
- `prsm/shared/services/transcript_import/providers/codex.py`
  - Codex file discovery + parsing.
- `prsm/shared/services/transcript_import/providers/claude.py`
  - Claude file discovery + parsing.
- `prsm/shared/services/transcript_import/normalize.py`
  - Convert provider events to canonical turns.
- `prsm/shared/services/transcript_import/service.py`
  - Public APIs for list/preview/import into `Session`.

### Canonical Model (minimal)

- `ImportSession`
  - `provider`: `codex|claude|...`
  - `source_session_id`: provider-native ID
  - `source_path`: transcript file path
  - `title`
  - `started_at`, `updated_at`
  - `turn_count`
- `ImportTurn`
  - `role`: `user|assistant|system|tool`
  - `content`: text
  - `timestamp`
  - `tool_calls`: list of `ImportToolUse`
  - `meta`: optional provider raw IDs/types

## Mapping Into PRSM

Import target is a single-agent PRSM session in v1:

- Create root `AgentNode`:
  - `id="root"`, role `MASTER`, state `COMPLETED`, provider tag in name/model.
  - If a placeholder model marker like `imported:codex` is present, normalize
    to the session's currently configured runnable model before restart/run.
- Map turns to `Session.messages["root"]`:
  - `user` -> `MessageRole.USER`
  - `assistant` -> `MessageRole.ASSISTANT`
  - `system` -> `MessageRole.SYSTEM`
  - tool results -> `MessageRole.TOOL` or attach to prior assistant `tool_calls` as possible.
- Preserve timestamps where available.
- Store provenance in session metadata file:
  - `imported_from`: provider, source session ID/path, imported_at, importer version.

`SessionPersistence` can write this as additional JSON keys without breaking old readers.

## UX Surface

### CLI

Extend `prsm/app.py` with non-TUI import commands:

- `prsm --import-list [--provider codex|claude|all] [--limit N]`
- `prsm --import-preview --provider P --source-id ID`
- `prsm --import-run --provider P --source-id ID [--name NAME] [--open]`

`--open` writes and marks as active session for immediate resume.

### TUI Slash Command

Add `/import` command family:

- `/import list [provider]`
- `/import preview PROVIDER SOURCE_ID`
- `/import run PROVIDER SOURCE_ID [SESSION_NAME]`

Integrate in:
- `prsm/shared/commands.py`
- `prsm/tui/handlers/command_handler.py`

## Context-Window Safety

Imported transcripts can be large. Add import modes:

- `full`: import all turns.
- `recent`: import last N turns.
- `summarized`: import recent N turns + generated summary message for earlier turns.

Defaults for safety:

- TUI default: `recent` (e.g., 200 turns).
- CLI can opt into `full`.

Also store raw parsed transcript as artifact for later re-summarization instead of forcing full replay into active context.

## Idempotency and Dedup

Compute stable import key:

- `sha256(provider + source_session_id + source_path + last_timestamp + turn_count)`

Store key in imported session metadata. On re-import:

- If same key exists, warn and skip unless `--force`.

## Error Handling

- Corrupt JSONL lines: skip with warning counter.
- Unknown event types: keep in `meta.unmapped_count`; continue.
- Missing timestamps: fall back to file mtime ordering.
- Partial tool payloads: capture text fallback, keep raw metadata.

## Testing Plan

Unit tests:

- Provider discovery returns expected files from fixtures.
- Codex parser maps `response_item`/event streams into canonical turns.
- Claude parser maps `user`/`assistant` content blocks and ignores non-chat events.
- Normalizer preserves order and timestamps.
- Importer writes valid PRSM sessions loadable by `SessionPersistence.load`.
- Dedup key behavior and `--force`.

Integration tests:

- End-to-end `import-run` creates session, appears in `/sessions`, can resume.
- Large transcript imports respect `recent`/`summarized` mode limits.

## Incremental Rollout

1. Ship internal import service + fixture tests (no UI yet).
2. Add CLI `--import-list/preview/run`.
3. Add `/import` slash commands.
4. Add optional summarization mode and artifact retention.
5. Add Gemini/other adapters once stable sources are identified.

## Effort Estimate

- v1 (Codex + Claude list/preview/run + tests): medium, ~2-4 days.
- With summarized mode + polish + migration metadata: +1-2 days.

## Key Limitation to Communicate

This preserves transcript context, not hidden provider memory internals. It is sufficient for "continue the conversation with context", but not for exact hidden-state continuation semantics.
