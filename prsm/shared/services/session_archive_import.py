"""Session archive import service — import .prsm archives into ~/.prsm/.

Imports session data from archive files (tar.gz, tar, zip) that were
created by SessionExportService or manually assembled with the correct
directory layout.

Expected archive layout (mirrors ~/.prsm/):
    manifest.json (optional)
    sessions/{repo_identity}/{session_id}.json
    projects/{repo_identity}/allowed_tools.json
    projects/{repo_identity}/plugins.json
    projects/{repo_identity}/memory/MEMORY.md
    projects/{repo_identity}/snapshots/{snapshot_id}/...
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImportArchiveResult:
    """Result of an archive import operation."""
    success: bool = True
    sessions_imported: int = 0
    sessions_skipped: int = 0
    files_imported: int = 0
    files_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    manifest: dict[str, Any] | None = None


class SessionArchiveImportService:
    """Import session archives into the local ~/.prsm/ directory."""

    # Only allow extraction into these top-level directories
    _ALLOWED_PREFIXES = ("sessions/", "projects/")
    _ALLOWED_ROOT_FILES = ("manifest.json",)

    def __init__(self, prsm_dir: Path | None = None) -> None:
        self._prsm_dir = prsm_dir or (Path.home() / ".prsm")

    def import_archive(
        self,
        archive_path: Path,
        *,
        conflict_mode: str = "skip",  # "skip", "overwrite", "rename"
    ) -> ImportArchiveResult:
        """Import a session archive into ~/.prsm/.

        Args:
            archive_path: Path to the archive file.
            conflict_mode: How to handle existing files:
                - "skip": Skip files that already exist.
                - "overwrite": Replace existing files.
                - "rename": Rename imported files with a suffix.

        Returns:
            ImportArchiveResult with import statistics.
        """
        if not archive_path.exists():
            return ImportArchiveResult(
                success=False,
                error=f"Archive file not found: {archive_path}",
            )

        archive_name = archive_path.name.lower()
        try:
            if archive_name.endswith(".zip"):
                return self._import_zip(archive_path, conflict_mode=conflict_mode)
            elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
                return self._import_tar(archive_path, conflict_mode=conflict_mode)
            elif archive_name.endswith(".tar"):
                return self._import_tar(
                    archive_path, conflict_mode=conflict_mode, compression=""
                )
            elif archive_name.endswith(".tar.bz2"):
                return self._import_tar(
                    archive_path, conflict_mode=conflict_mode, compression="bz2"
                )
            else:
                return ImportArchiveResult(
                    success=False,
                    error=(
                        f"Unsupported archive format: {archive_path.suffix}. "
                        "Supported: .tar.gz, .tgz, .tar, .tar.bz2, .zip"
                    ),
                )
        except Exception as exc:
            logger.error("Archive import failed: %s", exc, exc_info=True)
            return ImportArchiveResult(
                success=False,
                error=f"Import failed: {exc}",
            )

    def preview_archive(
        self,
        archive_path: Path,
    ) -> ImportArchiveResult:
        """Preview what an archive contains without importing."""
        if not archive_path.exists():
            return ImportArchiveResult(
                success=False,
                error=f"Archive file not found: {archive_path}",
            )

        archive_name = archive_path.name.lower()
        try:
            if archive_name.endswith(".zip"):
                return self._preview_zip(archive_path)
            elif (
                archive_name.endswith(".tar.gz")
                or archive_name.endswith(".tgz")
                or archive_name.endswith(".tar")
                or archive_name.endswith(".tar.bz2")
            ):
                return self._preview_tar(archive_path)
            else:
                return ImportArchiveResult(
                    success=False,
                    error=f"Unsupported archive format: {archive_path.suffix}",
                )
        except Exception as exc:
            return ImportArchiveResult(
                success=False,
                error=f"Preview failed: {exc}",
            )

    def _is_safe_path(self, member_name: str) -> bool:
        """Check that an archive member path is safe to extract."""
        # Reject absolute paths
        if member_name.startswith("/") or member_name.startswith("\\"):
            return False
        # Reject path traversal
        if ".." in member_name.split("/"):
            return False
        # Must be in allowed directories or be a root-level allowed file
        if member_name in self._ALLOWED_ROOT_FILES:
            return True
        return any(member_name.startswith(prefix) for prefix in self._ALLOWED_PREFIXES)

    def _resolve_dest(
        self,
        archive_name: str,
        conflict_mode: str,
    ) -> tuple[Path | None, bool]:
        """Resolve destination path, handling conflicts.

        Returns:
            (dest_path, is_skipped) — dest_path is None if skipped.
        """
        # Skip manifest — we only read it, never write to disk
        if archive_name == "manifest.json":
            return None, True

        dest = self._prsm_dir / archive_name
        if not dest.exists():
            return dest, False

        if conflict_mode == "skip":
            return None, True
        elif conflict_mode == "overwrite":
            return dest, False
        elif conflict_mode == "rename":
            stem = dest.stem
            suffix = dest.suffix
            parent = dest.parent
            counter = 1
            while True:
                new_name = f"{stem}_imported_{counter}{suffix}"
                new_dest = parent / new_name
                if not new_dest.exists():
                    return new_dest, False
                counter += 1
        else:
            return dest, False

    def _import_tar(
        self,
        archive_path: Path,
        *,
        conflict_mode: str = "skip",
        compression: str = "gz",
    ) -> ImportArchiveResult:
        """Import from a tar archive."""
        mode = f"r:{compression}" if compression else "r:"
        result = ImportArchiveResult()
        session_ids_seen: set[str] = set()

        with tarfile.open(archive_path, mode) as tar:
            # First pass: read manifest if present
            try:
                manifest_member = tar.getmember("manifest.json")
                f = tar.extractfile(manifest_member)
                if f:
                    result.manifest = json.loads(f.read().decode("utf-8"))
            except (KeyError, json.JSONDecodeError):
                pass  # No manifest or invalid — that's OK

            # Second pass: extract files
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not self._is_safe_path(member.name):
                    result.warnings.append(f"Skipped unsafe path: {member.name}")
                    continue

                dest, skipped = self._resolve_dest(member.name, conflict_mode)
                if skipped or dest is None:
                    result.files_skipped += 1
                    if member.name.startswith("sessions/") and member.name.endswith(".json"):
                        result.sessions_skipped += 1
                    continue

                # Extract
                dest.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted:
                    dest.write_bytes(extracted.read())
                    result.files_imported += 1
                    if member.name.startswith("sessions/") and member.name.endswith(".json"):
                        result.sessions_imported += 1
                        session_ids_seen.add(Path(member.name).stem)

        logger.info(
            "Imported %d sessions (%d files) from %s",
            result.sessions_imported,
            result.files_imported,
            archive_path,
        )
        return result

    def _import_zip(
        self,
        archive_path: Path,
        *,
        conflict_mode: str = "skip",
    ) -> ImportArchiveResult:
        """Import from a zip archive."""
        result = ImportArchiveResult()

        with zipfile.ZipFile(archive_path, "r") as zf:
            # Read manifest if present
            try:
                with zf.open("manifest.json") as f:
                    result.manifest = json.loads(f.read().decode("utf-8"))
            except (KeyError, json.JSONDecodeError):
                pass

            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not self._is_safe_path(info.filename):
                    result.warnings.append(f"Skipped unsafe path: {info.filename}")
                    continue

                dest, skipped = self._resolve_dest(info.filename, conflict_mode)
                if skipped or dest is None:
                    result.files_skipped += 1
                    if info.filename.startswith("sessions/") and info.filename.endswith(".json"):
                        result.sessions_skipped += 1
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info.filename))
                result.files_imported += 1
                if info.filename.startswith("sessions/") and info.filename.endswith(".json"):
                    result.sessions_imported += 1

        logger.info(
            "Imported %d sessions (%d files) from %s",
            result.sessions_imported,
            result.files_imported,
            archive_path,
        )
        return result

    def _preview_tar(self, archive_path: Path) -> ImportArchiveResult:
        """Preview contents of a tar archive."""
        result = ImportArchiveResult()
        try:
            mode = "r:gz" if archive_path.name.endswith((".gz", ".tgz")) else "r:"
            if archive_path.name.endswith(".bz2"):
                mode = "r:bz2"
            with tarfile.open(archive_path, mode) as tar:
                try:
                    manifest_member = tar.getmember("manifest.json")
                    f = tar.extractfile(manifest_member)
                    if f:
                        result.manifest = json.loads(f.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    pass

                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if member.name.startswith("sessions/") and member.name.endswith(".json"):
                        result.sessions_imported += 1
                    result.files_imported += 1
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result

    def _preview_zip(self, archive_path: Path) -> ImportArchiveResult:
        """Preview contents of a zip archive."""
        result = ImportArchiveResult()
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                try:
                    with zf.open("manifest.json") as f:
                        result.manifest = json.loads(f.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    pass

                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.filename.startswith("sessions/") and info.filename.endswith(".json"):
                        result.sessions_imported += 1
                    result.files_imported += 1
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result
