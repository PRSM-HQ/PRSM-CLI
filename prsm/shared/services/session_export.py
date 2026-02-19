"""Session export service — bundle session data into portable archives.

Exports session JSON files, snapshots, project configs, and memory
into a .tar.gz or .zip archive for backup or transfer between machines.

Archive layout mirrors ~/.prsm/ structure:
    sessions/{repo_identity}/{session_id}.json
    projects/{repo_identity}/allowed_tools.json
    projects/{repo_identity}/plugins.json
    projects/{repo_identity}/memory/MEMORY.md
    projects/{repo_identity}/snapshots/{snapshot_id}/...
    manifest.json
"""
from __future__ import annotations

import json
import logging
import os
import tarfile
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExportManifest:
    """Metadata describing an exported archive."""
    export_version: str = "1.0"
    created_at: str = ""
    repo_identity: str = ""
    session_count: int = 0
    session_ids: list[str] = field(default_factory=list)
    includes_snapshots: bool = False
    includes_plugins: bool = False
    includes_memory: bool = False
    includes_permissions: bool = False
    source_machine: str = ""
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_version": self.export_version,
            "created_at": self.created_at,
            "repo_identity": self.repo_identity,
            "session_count": self.session_count,
            "session_ids": self.session_ids,
            "includes_snapshots": self.includes_snapshots,
            "includes_plugins": self.includes_plugins,
            "includes_memory": self.includes_memory,
            "includes_permissions": self.includes_permissions,
            "source_machine": self.source_machine,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportManifest:
        return cls(
            export_version=data.get("export_version", "1.0"),
            created_at=data.get("created_at", ""),
            repo_identity=data.get("repo_identity", ""),
            session_count=data.get("session_count", 0),
            session_ids=data.get("session_ids", []),
            includes_snapshots=data.get("includes_snapshots", False),
            includes_plugins=data.get("includes_plugins", False),
            includes_memory=data.get("includes_memory", False),
            includes_permissions=data.get("includes_permissions", False),
            source_machine=data.get("source_machine", ""),
            files=data.get("files", []),
        )


@dataclass
class ExportResult:
    """Result of a session export operation."""
    archive_path: Path
    manifest: ExportManifest
    success: bool = True
    error: str | None = None


class SessionExportService:
    """Bundle session data into portable archive files."""

    def __init__(self, prsm_dir: Path | None = None) -> None:
        self._prsm_dir = prsm_dir or (Path.home() / ".prsm")

    def export_session(
        self,
        repo_identity: str,
        session_id: str,
        output_path: Path,
        *,
        include_snapshots: bool = True,
        include_plugins: bool = True,
        include_memory: bool = True,
        include_permissions: bool = True,
        archive_format: str = "tar.gz",
    ) -> ExportResult:
        """Export a single session to an archive file.

        Args:
            repo_identity: The repository identity string.
            session_id: The session ID to export.
            output_path: Where to write the archive.
            include_snapshots: Include snapshot directories.
            include_plugins: Include plugins.json.
            include_memory: Include MEMORY.md.
            include_permissions: Include allowed_tools.json.
            archive_format: "tar.gz" or "zip".
        """
        return self._export(
            repo_identity=repo_identity,
            session_ids=[session_id],
            output_path=output_path,
            include_snapshots=include_snapshots,
            include_plugins=include_plugins,
            include_memory=include_memory,
            include_permissions=include_permissions,
            archive_format=archive_format,
        )

    def export_all_sessions(
        self,
        repo_identity: str,
        output_path: Path,
        *,
        include_snapshots: bool = True,
        include_plugins: bool = True,
        include_memory: bool = True,
        include_permissions: bool = True,
        archive_format: str = "tar.gz",
    ) -> ExportResult:
        """Export all sessions for a repo identity to an archive file."""
        sessions_dir = self._prsm_dir / "sessions" / repo_identity
        if not sessions_dir.exists():
            return ExportResult(
                archive_path=output_path,
                manifest=ExportManifest(),
                success=False,
                error=f"No sessions found for repo identity: {repo_identity}",
            )

        session_ids = [
            p.stem for p in sessions_dir.glob("*.json")
        ]
        if not session_ids:
            return ExportResult(
                archive_path=output_path,
                manifest=ExportManifest(),
                success=False,
                error=f"No session files found in {sessions_dir}",
            )

        return self._export(
            repo_identity=repo_identity,
            session_ids=session_ids,
            output_path=output_path,
            include_snapshots=include_snapshots,
            include_plugins=include_plugins,
            include_memory=include_memory,
            include_permissions=include_permissions,
            archive_format=archive_format,
        )

    def _export(
        self,
        repo_identity: str,
        session_ids: list[str],
        output_path: Path,
        *,
        include_snapshots: bool,
        include_plugins: bool,
        include_memory: bool,
        include_permissions: bool,
        archive_format: str,
    ) -> ExportResult:
        """Core export logic."""
        try:
            files_to_archive: list[tuple[Path, str]] = []  # (abs_path, archive_name)
            manifest = ExportManifest(
                created_at=datetime.now(timezone.utc).isoformat(),
                repo_identity=repo_identity,
                source_machine=os.uname().nodename if hasattr(os, "uname") else "unknown",
            )

            # 1. Session JSON files
            sessions_dir = self._prsm_dir / "sessions" / repo_identity
            for sid in session_ids:
                session_file = sessions_dir / f"{sid}.json"
                if session_file.exists():
                    archive_name = f"sessions/{repo_identity}/{sid}.json"
                    files_to_archive.append((session_file, archive_name))
                    manifest.session_ids.append(sid)
                else:
                    logger.warning("Session file not found: %s", session_file)

            manifest.session_count = len(manifest.session_ids)
            if manifest.session_count == 0:
                return ExportResult(
                    archive_path=output_path,
                    manifest=manifest,
                    success=False,
                    error="No session files found to export.",
                )

            # 2. Project-level files
            project_dir = self._prsm_dir / "projects" / repo_identity

            if include_permissions:
                perms_file = project_dir / "allowed_tools.json"
                if perms_file.exists():
                    archive_name = f"projects/{repo_identity}/allowed_tools.json"
                    files_to_archive.append((perms_file, archive_name))
                    manifest.includes_permissions = True

            if include_plugins:
                plugins_file = project_dir / "plugins.json"
                if plugins_file.exists():
                    archive_name = f"projects/{repo_identity}/plugins.json"
                    files_to_archive.append((plugins_file, archive_name))
                    manifest.includes_plugins = True

            if include_memory:
                memory_file = project_dir / "memory" / "MEMORY.md"
                if memory_file.exists():
                    archive_name = f"projects/{repo_identity}/memory/MEMORY.md"
                    files_to_archive.append((memory_file, archive_name))
                    manifest.includes_memory = True

            if include_snapshots:
                snapshots_dir = project_dir / "snapshots"
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if not snapshot_dir.is_dir():
                            continue
                        # Only include snapshots belonging to exported sessions
                        meta_path = snapshot_dir / "meta.json"
                        if meta_path.exists():
                            try:
                                meta = json.loads(meta_path.read_text())
                                snap_session_id = str(meta.get("session_id", ""))
                                if snap_session_id and snap_session_id not in manifest.session_ids:
                                    continue
                            except Exception:
                                pass  # Include if we can't determine session

                        for root, _dirs, filenames in os.walk(snapshot_dir):
                            for fname in filenames:
                                abs_path = Path(root) / fname
                                rel = abs_path.relative_to(self._prsm_dir)
                                files_to_archive.append((abs_path, str(rel)))
                    if any(
                        name.startswith(f"projects/{repo_identity}/snapshots/")
                        for _, name in files_to_archive
                    ):
                        manifest.includes_snapshots = True

            # Record all files in manifest
            manifest.files = [name for _, name in files_to_archive]

            # 3. Write archive
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if archive_format == "zip":
                self._write_zip(output_path, files_to_archive, manifest)
            else:
                self._write_tar_gz(output_path, files_to_archive, manifest)

            logger.info(
                "Exported %d sessions (%d files) to %s",
                manifest.session_count,
                len(files_to_archive),
                output_path,
            )
            return ExportResult(
                archive_path=output_path,
                manifest=manifest,
                success=True,
            )

        except Exception as exc:
            logger.error("Export failed: %s", exc, exc_info=True)
            return ExportResult(
                archive_path=output_path,
                manifest=ExportManifest(),
                success=False,
                error=str(exc),
            )

    def _write_tar_gz(
        self,
        output_path: Path,
        files: list[tuple[Path, str]],
        manifest: ExportManifest,
    ) -> None:
        with tarfile.open(output_path, "w:gz") as tar:
            for abs_path, archive_name in files:
                tar.add(str(abs_path), arcname=archive_name)

            # Add manifest
            manifest_bytes = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
            import io
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))

    def _write_zip(
        self,
        output_path: Path,
        files: list[tuple[Path, str]],
        manifest: ExportManifest,
    ) -> None:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for abs_path, archive_name in files:
                zf.write(str(abs_path), archive_name)

            # Add manifest
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            zf.writestr("manifest.json", manifest_json)
