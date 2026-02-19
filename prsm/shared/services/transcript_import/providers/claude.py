"""Claude transcript adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import ImportSessionSummary, ImportToolUse, ImportTranscript, ImportTurn
from ..normalize import coerce_text, extract_text_from_blocks, parse_timestamp
from .base import TranscriptProviderAdapter

_PROJECT_GLOB = "projects/**/*.jsonl"


class ClaudeTranscriptAdapter(TranscriptProviderAdapter):
    """Import adapter for local Claude JSONL sessions."""

    provider_name = "claude"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.home() / ".claude")

    def list_sessions(self, *, limit: int | None = None) -> list[ImportSessionSummary]:
        summaries: list[ImportSessionSummary] = []
        for path in self._iter_session_files():
            summary, _, _ = self._parse_session_file(path, include_turns=False)
            summaries.append(summary)
        summaries.sort(
            key=lambda s: s.updated_at or s.started_at or parse_timestamp("1970-01-01T00:00:00Z"),
            reverse=True,
        )
        if limit is not None and limit > 0:
            return summaries[:limit]
        return summaries

    def load_session(self, source_session_id: str) -> ImportTranscript:
        for path in self._iter_session_files():
            summary, _, _ = self._parse_session_file(path, include_turns=False)
            if (
                summary.source_session_id == source_session_id
                or summary.source_path.stem == source_session_id
            ):
                full_summary, turns, warnings = self._parse_session_file(path, include_turns=True)
                return ImportTranscript(summary=full_summary, turns=turns, warnings=warnings)
        raise FileNotFoundError(source_session_id)

    def _iter_session_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        files = sorted(self.root.glob(_PROJECT_GLOB))
        return [path for path in files if "subagents" not in path.parts]

    def _parse_session_file(
        self,
        path: Path,
        *,
        include_turns: bool,
    ) -> tuple[ImportSessionSummary, list[ImportTurn], list[str]]:
        source_session_id: str | None = None
        title: str | None = None
        started_at = None
        updated_at = None
        turn_count = 0
        warnings: list[str] = []
        turns: list[ImportTurn] = []

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            warnings.append(f"failed to read {path}: {exc}")
            lines = []

        for line_no, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                warnings.append(f"{path.name}:{line_no}: invalid json")
                continue
            if not isinstance(row, dict):
                continue

            session_id = row.get("sessionId")
            if isinstance(session_id, str) and session_id:
                source_session_id = session_id

            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp:
                if started_at is None or timestamp < started_at:
                    started_at = timestamp
                if updated_at is None or timestamp > updated_at:
                    updated_at = timestamp

            row_type = str(row.get("type") or "").lower()
            message = row.get("message") if isinstance(row.get("message"), dict) else {}

            if row_type == "user":
                content = extract_text_from_blocks(message.get("content"), include_thinking=False)
                if not content:
                    continue
                if not title:
                    title = content.strip().splitlines()[0][:120]
                turn_count += 1
                if include_turns:
                    turns.append(
                        ImportTurn(
                            role="user",
                            content=content,
                            timestamp=timestamp,
                            metadata={"event_type": row_type},
                        )
                    )
                continue

            if row_type == "assistant":
                raw_blocks = message.get("content")
                tool_calls = self._extract_tool_calls(raw_blocks)
                text = extract_text_from_blocks(raw_blocks, include_thinking=False)
                if text.strip() == "(no content)" and not tool_calls:
                    continue
                turn_count += 1
                if include_turns:
                    turns.append(
                        ImportTurn(
                            role="assistant",
                            content="" if text.strip() == "(no content)" else text,
                            timestamp=timestamp,
                            tool_calls=tool_calls,
                            metadata={"event_type": row_type, "message_id": message.get("id")},
                        )
                    )

        source_session_id = source_session_id or path.stem
        if not title:
            title = f"Claude import {source_session_id[:8]}"
        summary = ImportSessionSummary(
            provider=self.provider_name,
            source_session_id=source_session_id,
            source_path=path,
            title=title,
            started_at=started_at,
            updated_at=updated_at,
            turn_count=turn_count,
            metadata={"warning_count": len(warnings)},
        )
        return summary, turns, warnings

    def _extract_tool_calls(self, blocks: Any) -> list[ImportToolUse]:
        if not isinstance(blocks, list):
            return []
        tool_calls: list[ImportToolUse] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type != "tool_use":
                continue
            tool_calls.append(
                ImportToolUse(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or "tool"),
                    arguments=coerce_text(block.get("input")),
                )
            )
        return tool_calls

