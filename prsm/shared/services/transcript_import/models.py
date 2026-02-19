"""Canonical transcript import models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from prsm.shared.models.session import Session


@dataclass
class ImportSessionSummary:
    """Provider-native session metadata discovered on disk."""

    provider: str
    source_session_id: str
    source_path: Path
    title: str | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportToolUse:
    """Normalized provider tool call."""

    id: str
    name: str
    arguments: str
    result: str | None = None
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportTurn:
    """Normalized conversation turn."""

    role: str
    content: str
    timestamp: datetime | None = None
    tool_calls: list[ImportToolUse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportTranscript:
    """Full transcript for one provider session."""

    summary: ImportSessionSummary
    turns: list[ImportTurn]
    warnings: list[str] = field(default_factory=list)


@dataclass
class SessionImportResult:
    """Result of converting a provider transcript into a PRSM session."""

    session: Session
    source: ImportSessionSummary
    imported_turns: int
    dropped_turns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

