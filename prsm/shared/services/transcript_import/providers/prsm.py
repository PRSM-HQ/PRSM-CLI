"""PRSM session import adapter.

Imports sessions from exported ~/.prsm directories or ZIP archives containing
PRSM session JSON files.  Supports both:

  1. Bare directories with the standard ``~/.prsm/sessions/{repo}/`` layout.
  2. ZIP archives (``.prsm`` or ``.zip`` files) containing that same layout
     (i.e. a zipped ``~/.prsm/`` directory tree).

When importing from a ZIP archive, the adapter extracts it to a temporary
directory before scanning for session JSON files.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import ImportSessionSummary, ImportToolUse, ImportTranscript, ImportTurn
from ..normalize import parse_timestamp
from .base import TranscriptProviderAdapter

logger = logging.getLogger(__name__)

# Glob patterns for finding session JSON files within a .prsm directory tree
_SESSION_GLOBS = [
    "sessions/**/*.json",    # Standard layout: sessions/{repo}/{id}.json
    "sessions/*.json",       # Flat layout: sessions/{id}.json
    "**/*.json",             # Fallback: any JSON under the root
]


class PrsmTranscriptAdapter(TranscriptProviderAdapter):
    """Import adapter for PRSM session JSON files from directories or ZIP archives."""

    provider_name = "prsm"

    def __init__(self, root: Path | None = None) -> None:
        """Initialize with optional root path.

        Args:
            root: Path to a .prsm directory or a ZIP archive.
                  Defaults to ``~/.prsm``.
        """
        self.root = root or (Path.home() / ".prsm")
        self._temp_dirs: list[str] = []

    def list_sessions(self, *, limit: int | None = None) -> list[ImportSessionSummary]:
        summaries: list[ImportSessionSummary] = []
        scan_root = self._resolve_root()
        if scan_root is None:
            return summaries

        for path in self._iter_session_files(scan_root):
            try:
                summary = self._parse_session_summary(path)
                if summary is not None:
                    summaries.append(summary)
            except Exception as exc:
                logger.debug("Skipping %s: %s", path, exc)

        summaries.sort(
            key=lambda s: s.updated_at or s.started_at or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )
        if limit is not None and limit > 0:
            return summaries[:limit]
        return summaries

    def load_session(self, source_session_id: str) -> ImportTranscript:
        scan_root = self._resolve_root()
        if scan_root is None:
            raise FileNotFoundError(source_session_id)

        for path in self._iter_session_files(scan_root):
            try:
                summary = self._parse_session_summary(path)
                if summary is None:
                    continue
                if (
                    summary.source_session_id == source_session_id
                    or summary.source_path.stem == source_session_id
                ):
                    return self._parse_full_session(path, summary)
            except Exception:
                continue

        raise FileNotFoundError(source_session_id)

    def _resolve_root(self) -> Path | None:
        """Resolve the root to a scannable directory.

        If self.root is a ZIP archive, extract it to a temp directory.
        If it's a directory, return it directly.
        """
        if not self.root.exists():
            return None

        if self.root.is_dir():
            return self.root

        # Check if it's a ZIP file
        if self.root.is_file() and self._is_zip_file(self.root):
            return self._extract_zip(self.root)

        return None

    def _is_zip_file(self, path: Path) -> bool:
        """Check if a file is a ZIP archive."""
        try:
            return zipfile.is_zipfile(str(path))
        except Exception:
            return False

    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a ZIP archive to a temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="prsm-import-")
        self._temp_dirs.append(temp_dir)
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(temp_dir)
            logger.info("Extracted ZIP archive %s to %s", zip_path, temp_dir)

            # The ZIP might contain a top-level directory (e.g., .prsm/)
            # or the contents directly. Look for sessions/ directory.
            extracted = Path(temp_dir)
            candidates = [
                extracted,                          # Contents directly
                extracted / ".prsm",                # .prsm/ at top level
                extracted / "prsm",                 # prsm/ at top level
            ]
            # Also check for a single top-level directory
            top_items = list(extracted.iterdir())
            if len(top_items) == 1 and top_items[0].is_dir():
                candidates.insert(1, top_items[0])

            for candidate in candidates:
                if (candidate / "sessions").is_dir():
                    return candidate

            # If no sessions/ found, return the extracted root for fallback glob
            return extracted
        except Exception as exc:
            logger.error("Failed to extract ZIP %s: %s", zip_path, exc)
            self._cleanup_temp(temp_dir)
            raise

    def _cleanup_temp(self, temp_dir: str) -> None:
        """Clean up a temporary extraction directory."""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up all temporary extraction directories."""
        for td in self._temp_dirs:
            self._cleanup_temp(td)
        self._temp_dirs.clear()

    def __del__(self) -> None:
        self.cleanup()

    def _iter_session_files(self, scan_root: Path) -> list[Path]:
        """Find all session JSON files under the scan root."""
        seen: set[str] = set()
        files: list[Path] = []

        for glob_pattern in _SESSION_GLOBS:
            for path in sorted(scan_root.glob(glob_pattern)):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                # Skip non-JSON and metadata files
                if path.suffix != ".json":
                    continue
                if path.name.startswith("."):
                    continue
                if path.name in {"allowed_tools.json", "plugins.json",
                                 "preferences.json", "model_intelligence.json",
                                 "models.yaml"}:
                    continue
                # Quick validation: must contain session-like keys
                if self._looks_like_session(path):
                    seen.add(resolved)
                    files.append(path)

        return files

    def _looks_like_session(self, path: Path) -> bool:
        """Quick heuristic check if a JSON file looks like a PRSM session."""
        try:
            # Read just the first 4KB to check for key markers
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(4096)
            return '"agents"' in head and '"messages"' in head
        except Exception:
            return False

    def _parse_session_summary(self, path: Path) -> ImportSessionSummary | None:
        """Parse a PRSM session JSON file into a summary."""
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
            return None

        if not isinstance(data, dict):
            return None

        session_id = str(data.get("session_id") or path.stem)
        name = data.get("name") or path.stem
        saved_at = parse_timestamp(data.get("saved_at"))
        created_at_str = data.get("created_at")
        created_at = parse_timestamp(created_at_str) if created_at_str else None

        # Count turns from messages
        messages = data.get("messages", {})
        turn_count = 0
        if isinstance(messages, dict):
            for msgs in messages.values():
                if isinstance(msgs, list):
                    turn_count += len(msgs)

        agents = data.get("agents", {})
        agent_count = len(agents) if isinstance(agents, dict) else 0

        return ImportSessionSummary(
            provider=self.provider_name,
            source_session_id=session_id,
            source_path=path,
            title=name,
            started_at=created_at,
            updated_at=saved_at,
            turn_count=turn_count,
            metadata={
                "agent_count": agent_count,
                "version": data.get("version"),
                "workspace": data.get("workspace"),
                "forked_from": data.get("forked_from"),
                "summary": data.get("summary"),
            },
        )

    def _parse_full_session(
        self, path: Path, summary: ImportSessionSummary
    ) -> ImportTranscript:
        """Parse a PRSM session JSON into full ImportTranscript turns."""
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        turns: list[ImportTurn] = []
        warnings: list[str] = []

        messages = data.get("messages", {})
        if not isinstance(messages, dict):
            warnings.append("No messages found in session")
            return ImportTranscript(summary=summary, turns=turns, warnings=warnings)

        # Collect all messages across all agents, sorted by timestamp
        all_msgs: list[tuple[str, dict]] = []
        for agent_id, msgs in messages.items():
            if not isinstance(msgs, list):
                continue
            for msg_data in msgs:
                if not isinstance(msg_data, dict):
                    continue
                all_msgs.append((agent_id, msg_data))

        # Sort by timestamp
        all_msgs.sort(
            key=lambda x: x[1].get("timestamp", "1970-01-01T00:00:00Z")
        )

        for agent_id, msg_data in all_msgs:
            role = str(msg_data.get("role", "")).lower()
            if role not in {"user", "assistant", "system", "tool"}:
                continue
            content = msg_data.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            timestamp = parse_timestamp(msg_data.get("timestamp"))

            tool_calls: list[ImportToolUse] = []
            for tc in msg_data.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                tool_calls.append(
                    ImportToolUse(
                        id=str(tc.get("id", "")),
                        name=str(tc.get("name", "tool")),
                        arguments=str(tc.get("arguments", "")),
                        result=tc.get("result"),
                        success=tc.get("success", True),
                    )
                )

            turns.append(
                ImportTurn(
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    tool_calls=tool_calls,
                    metadata={
                        "agent_id": agent_id,
                        "message_id": msg_data.get("id"),
                        "snapshot_id": msg_data.get("snapshot_id"),
                    },
                )
            )

        return ImportTranscript(summary=summary, turns=turns, warnings=warnings)
