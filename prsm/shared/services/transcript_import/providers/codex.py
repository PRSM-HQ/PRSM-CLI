"""Codex transcript adapter."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..models import ImportSessionSummary, ImportToolUse, ImportTranscript, ImportTurn
from ..normalize import coerce_text, extract_text_from_blocks, parse_timestamp
from .base import TranscriptProviderAdapter

logger = logging.getLogger(__name__)
_ROLLOUT_GLOB = "sessions/**/rollout-*.jsonl"
_ID_TAIL_RE = re.compile(r"([0-9a-f]{8,}(?:-[0-9a-f]{4,}){2,})$", flags=re.IGNORECASE)


class CodexTranscriptAdapter(TranscriptProviderAdapter):
    """Import adapter for local Codex rollout JSONL sessions."""

    provider_name = "codex"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.home() / ".codex")

    def list_sessions(self, *, limit: int | None = None) -> list[ImportSessionSummary]:
        summaries: list[ImportSessionSummary] = []
        for path in self._iter_rollout_files():
            summary, _, _ = self._parse_rollout(path, include_turns=False)
            summaries.append(summary)
        summaries.sort(
            key=lambda s: s.updated_at or s.started_at or parse_timestamp("1970-01-01T00:00:00Z"),
            reverse=True,
        )
        if limit is not None and limit > 0:
            return summaries[:limit]
        return summaries

    def load_session(self, source_session_id: str) -> ImportTranscript:
        for path in self._iter_rollout_files():
            summary, _, _ = self._parse_rollout(path, include_turns=False)
            if (
                summary.source_session_id == source_session_id
                or summary.source_path.stem == source_session_id
            ):
                full_summary, turns, warnings = self._parse_rollout(path, include_turns=True)
                return ImportTranscript(summary=full_summary, turns=turns, warnings=warnings)
        raise FileNotFoundError(source_session_id)

    def _iter_rollout_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob(_ROLLOUT_GLOB))

    def _parse_rollout(
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
        turns: list[ImportTurn] = []
        warnings: list[str] = []
        pending_tool_calls: dict[str, ImportToolUse] = {}

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

            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp:
                if started_at is None or timestamp < started_at:
                    started_at = timestamp
                if updated_at is None or timestamp > updated_at:
                    updated_at = timestamp

            row_type = str(row.get("type") or "")
            payload = row.get("payload")
            payload = payload if isinstance(payload, dict) else {}

            if row_type == "session_meta":
                session_id = payload.get("id")
                if isinstance(session_id, str) and session_id:
                    source_session_id = session_id
                meta_timestamp = parse_timestamp(payload.get("timestamp"))
                if meta_timestamp and (started_at is None or meta_timestamp < started_at):
                    started_at = meta_timestamp
                continue

            if row_type != "response_item":
                continue

            payload_type = str(payload.get("type") or "")
            if payload_type == "message":
                role = str(payload.get("role") or "").lower()
                if role not in {"user", "assistant", "system"}:
                    continue
                content = extract_text_from_blocks(payload.get("content"), include_thinking=False)
                if role == "assistant" and content.strip() == "(no content)":
                    content = ""
                if not title and role == "user" and content.strip():
                    title = content.strip().splitlines()[0][:120]

                turn_count += 1
                if include_turns:
                    turns.append(
                        ImportTurn(
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            metadata={"event_type": "message"},
                        )
                    )
                continue

            if payload_type == "custom_tool_call":
                turn_count += 1
                if not include_turns:
                    continue
                call_id = str(payload.get("call_id") or f"tool-{line_no}")
                tool = ImportToolUse(
                    id=call_id,
                    name=str(payload.get("name") or "tool"),
                    arguments=coerce_text(payload.get("input")),
                    metadata={"status": payload.get("status")},
                )
                pending_tool_calls[call_id] = tool
                turns.append(
                    ImportTurn(
                        role="assistant",
                        content="",
                        timestamp=timestamp,
                        tool_calls=[tool],
                        metadata={"event_type": "custom_tool_call"},
                    )
                )
                continue

            if payload_type == "custom_tool_call_output" and include_turns:
                call_id = str(payload.get("call_id") or "")
                output = coerce_text(payload.get("output"))
                tool = pending_tool_calls.get(call_id)
                if tool:
                    tool.result = output
                    tool.success = self._infer_tool_success(output, default=tool.success)
                else:
                    turn_count += 1
                    turns.append(
                        ImportTurn(
                            role="tool",
                            content=output,
                            timestamp=timestamp,
                            metadata={"event_type": "custom_tool_call_output", "call_id": call_id},
                        )
                    )

        source_session_id = source_session_id or self._session_id_from_path(path)
        if not title:
            title = f"Codex import {source_session_id[:8]}"
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

    def _session_id_from_path(self, path: Path) -> str:
        match = _ID_TAIL_RE.search(path.stem)
        if match:
            return match.group(1)
        logger.debug("Could not infer Codex session id from filename: %s", path)
        return path.stem

    @staticmethod
    def _infer_tool_success(output: str, *, default: bool) -> bool:
        text = output.strip()
        if not text:
            return default
        try:
            payload = json.loads(text)
        except Exception:
            normalized = text.replace(" ", "").lower()
            if '"exit_code":0' in normalized or '"is_error":false' in normalized:
                return True
            if '"exit_code":' in normalized and '"exit_code":0' not in normalized:
                return False
            if '"is_error":true' in normalized:
                return False
            return default
        if not isinstance(payload, dict):
            return default
        meta = payload.get("metadata")
        if isinstance(meta, dict) and "exit_code" in meta:
            return meta.get("exit_code") == 0
        if "is_error" in payload:
            return payload.get("is_error") is False
        return default
