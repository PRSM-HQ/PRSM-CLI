"""Gemini transcript adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import ImportSessionSummary, ImportToolUse, ImportTranscript, ImportTurn
from ..normalize import coerce_text, extract_text_from_blocks, parse_timestamp
from .base import TranscriptProviderAdapter

logger = logging.getLogger(__name__)
_SESSION_GLOB = "tmp/**/chats/session-*.json"


class GeminiTranscriptAdapter(TranscriptProviderAdapter):
    """Import adapter for local Gemini CLI JSON sessions."""

    provider_name = "gemini"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.home() / ".gemini")

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
        return sorted(self.root.glob(_SESSION_GLOB))

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
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            warnings.append(f"failed to read {path}: {exc}")
            raw_text = "{}"

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path.name}: invalid json: {exc}")
            data = {}

        if not isinstance(data, dict):
            warnings.append(f"{path.name}: expected top-level object")
            data = {}

        # Extract session-level metadata.
        session_id = data.get("sessionId")
        if isinstance(session_id, str) and session_id:
            source_session_id = session_id

        start_time = parse_timestamp(data.get("startTime"))
        if start_time:
            started_at = start_time

        last_updated = parse_timestamp(data.get("lastUpdated"))
        if last_updated:
            updated_at = last_updated

        # Parse messages.
        messages = data.get("messages")
        if not isinstance(messages, list):
            messages = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_type = str(msg.get("type") or "").lower()
            # Only count user/assistant turns; skip info/error.
            if msg_type not in {"user", "assistant"}:
                continue

            timestamp = parse_timestamp(msg.get("timestamp"))
            if timestamp:
                if started_at is None or timestamp < started_at:
                    started_at = timestamp
                if updated_at is None or timestamp > updated_at:
                    updated_at = timestamp

            content = extract_text_from_blocks(msg.get("content"), include_thinking=False)

            if not title and msg_type == "user" and content.strip():
                title = content.strip().splitlines()[0][:120]

            turn_count += 1
            if include_turns:
                tool_calls = self._extract_tool_calls(msg.get("content")) if msg_type == "assistant" else []
                turns.append(
                    ImportTurn(
                        role=msg_type,
                        content=content,
                        timestamp=timestamp,
                        tool_calls=tool_calls,
                        metadata={"event_type": msg_type, "message_id": msg.get("id")},
                    )
                )

        source_session_id = source_session_id or path.stem
        if not title:
            title = f"Gemini import {source_session_id[:8]}"
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

    def _extract_tool_calls(self, content: Any) -> list[ImportToolUse]:
        """Extract tool calls from assistant message content.

        Gemini may embed tool_use blocks as JSON objects within content.
        Handles both structured block lists and JSON-encoded strings.
        """
        if content is None:
            return []

        # If content is a string, try parsing as JSON to find tool blocks.
        if isinstance(content, str):
            return self._parse_tool_calls_from_text(content)

        # If content is a list of blocks, scan for tool_use entries.
        if isinstance(content, list):
            tool_calls: list[ImportToolUse] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "").lower()
                if block_type == "tool_use":
                    tool_calls.append(
                        ImportToolUse(
                            id=str(block.get("id") or ""),
                            name=str(block.get("name") or "tool"),
                            arguments=coerce_text(block.get("input")),
                        )
                    )
            return tool_calls

        return []

    @staticmethod
    def _parse_tool_calls_from_text(text: str) -> list[ImportToolUse]:
        """Best-effort extraction of tool calls from a JSON string."""
        stripped = text.strip()
        if not stripped or not stripped.startswith(("{", "[")):
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []

        blocks: list[dict] = []
        if isinstance(parsed, dict):
            blocks = [parsed]
        elif isinstance(parsed, list):
            blocks = [b for b in parsed if isinstance(b, dict)]
        else:
            return []

        tool_calls: list[ImportToolUse] = []
        for block in blocks:
            block_type = str(block.get("type") or "").lower()
            if block_type == "tool_use":
                tool_calls.append(
                    ImportToolUse(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or "tool"),
                        arguments=coerce_text(block.get("input")),
                    )
                )
        return tool_calls
