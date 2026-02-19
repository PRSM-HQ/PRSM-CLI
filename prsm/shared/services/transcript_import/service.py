"""High-level transcript import service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from prsm.engine.models import AgentRole, AgentState
from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import Message, MessageRole, ToolCall
from prsm.shared.models.session import Session

from .models import (
    ImportSessionSummary,
    ImportTranscript,
    ImportTurn,
    SessionImportResult,
)
from .providers.base import TranscriptProviderAdapter
from .providers.claude import ClaudeTranscriptAdapter
from .providers.codex import CodexTranscriptAdapter
from .providers.prsm import PrsmTranscriptAdapter


class TranscriptImportService:
    """Discover provider sessions and convert them into PRSM sessions."""

    def __init__(
        self,
        *,
        codex_root: Path | None = None,
        claude_root: Path | None = None,
        prsm_root: Path | None = None,
        adapters: dict[str, TranscriptProviderAdapter] | None = None,
    ) -> None:
        if adapters is not None:
            self._adapters = {name.lower(): adapter for name, adapter in adapters.items()}
            return
        self._adapters = {
            "codex": CodexTranscriptAdapter(root=codex_root),
            "claude": ClaudeTranscriptAdapter(root=claude_root),
            "prsm": PrsmTranscriptAdapter(root=prsm_root),
        }

    def list_sessions(
        self,
        *,
        provider: str = "all",
        limit: int | None = None,
    ) -> list[ImportSessionSummary]:
        provider_key = provider.lower().strip()
        if provider_key == "all":
            summaries: list[ImportSessionSummary] = []
            for adapter in self._adapters.values():
                summaries.extend(adapter.list_sessions(limit=None))
            summaries.sort(
                key=lambda s: s.updated_at or s.started_at or datetime.fromtimestamp(0, tz=timezone.utc),
                reverse=True,
            )
            if limit is not None and limit > 0:
                return summaries[:limit]
            return summaries

        adapter = self._get_adapter(provider_key)
        return adapter.list_sessions(limit=limit)

    def load_transcript(self, provider: str, source_session_id: str) -> ImportTranscript:
        adapter = self._get_adapter(provider)
        return adapter.load_session(source_session_id)

    def import_to_session(
        self,
        provider: str,
        source_session_id: str,
        *,
        session_name: str | None = None,
        max_turns: int | None = None,
    ) -> SessionImportResult:
        transcript = self.load_transcript(provider, source_session_id)
        turns = transcript.turns
        dropped_turns = 0
        if max_turns is not None and max_turns > 0 and len(turns) > max_turns:
            dropped_turns = len(turns) - max_turns
            turns = turns[-max_turns:]

        session = self._build_session(transcript.summary, turns, session_name=session_name)
        metadata = {
            "imported_from": {
                "provider": transcript.summary.provider,
                "source_session_id": transcript.summary.source_session_id,
                "source_path": str(transcript.summary.source_path),
                "imported_at": datetime.now(timezone.utc).isoformat(),
            },
            "warnings": list(transcript.warnings),
        }
        return SessionImportResult(
            session=session,
            source=transcript.summary,
            imported_turns=len(turns),
            dropped_turns=dropped_turns,
            metadata=metadata,
        )

    def import_all_sessions(
        self,
        provider: str,
        *,
        max_turns: int | None = None,
    ) -> list[SessionImportResult]:
        """Import all sessions from a given provider.

        Returns a list of SessionImportResult for each successfully imported
        session. Failed imports are skipped with warnings.
        """
        summaries = self.list_sessions(provider=provider)
        results: list[SessionImportResult] = []
        for summary in summaries:
            try:
                result = self.import_to_session(
                    summary.provider,
                    summary.source_session_id,
                    session_name=summary.title,
                    max_turns=max_turns,
                )
                results.append(result)
            except Exception:
                pass  # skip failed imports silently
        return results

    def _build_session(
        self,
        summary: ImportSessionSummary,
        turns: list[ImportTurn],
        *,
        session_name: str | None,
    ) -> Session:
        session = Session()
        if summary.started_at is not None:
            session.created_at = summary.started_at
        session.name = session_name or summary.title or f"Imported {summary.provider} session"

        root = AgentNode(
            id="root",
            name=f"Imported {summary.provider.title()}",
            state=AgentState.COMPLETED,
            role=AgentRole.MASTER,
            model=f"imported:{summary.provider}",
            provider=summary.provider,
        )
        session.add_agent(root)
        session.set_active(root.id)

        for idx, turn in enumerate(turns, start=1):
            role = self._map_role(turn.role)
            tool_calls = [
                ToolCall(
                    id=tool.id or f"tool-{idx}",
                    name=tool.name,
                    arguments=tool.arguments,
                    result=tool.result,
                    success=tool.success,
                )
                for tool in turn.tool_calls
            ]
            timestamp = turn.timestamp or datetime.now(timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            message = Message(
                role=role,
                content=turn.content,
                agent_id=root.id,
                timestamp=timestamp,
                tool_calls=tool_calls,
            )
            session.messages[root.id].append(message)

        return session

    def _get_adapter(self, provider: str) -> TranscriptProviderAdapter:
        provider_key = provider.lower().strip()
        adapter = self._adapters.get(provider_key)
        if adapter is None:
            available = ", ".join(sorted(self._adapters.keys())) or "none"
            raise KeyError(f"Unknown provider '{provider}'. Available: {available}")
        return adapter

    @staticmethod
    def _map_role(role: str) -> MessageRole:
        normalized = role.lower().strip()
        if normalized == "user":
            return MessageRole.USER
        if normalized == "assistant":
            return MessageRole.ASSISTANT
        if normalized == "tool":
            return MessageRole.TOOL
        return MessageRole.SYSTEM

    @staticmethod
    def session_import_metadata(result: SessionImportResult) -> dict:
        """Return a JSON-serializable metadata payload for persistence."""
        source = result.source
        payload = {
            "provider": source.provider,
            "source_session_id": source.source_session_id,
            "source_path": str(source.source_path),
            "title": source.title,
            "started_at": source.started_at.isoformat() if source.started_at else None,
            "updated_at": source.updated_at.isoformat() if source.updated_at else None,
            "turn_count": source.turn_count,
            "metadata": source.metadata,
            "imported_turns": result.imported_turns,
            "dropped_turns": result.dropped_turns,
            "warnings": result.metadata.get("warnings", []),
        }
        return payload
