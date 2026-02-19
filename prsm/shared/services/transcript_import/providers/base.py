"""Base interfaces for transcript import provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ImportSessionSummary, ImportTranscript


class TranscriptProviderAdapter(ABC):
    """Adapter that discovers and parses provider transcript artifacts."""

    provider_name: str

    @abstractmethod
    def list_sessions(self, *, limit: int | None = None) -> list[ImportSessionSummary]:
        """List importable sessions for this provider."""

    @abstractmethod
    def load_session(self, source_session_id: str) -> ImportTranscript:
        """Load full transcript turns for a discovered provider session."""

