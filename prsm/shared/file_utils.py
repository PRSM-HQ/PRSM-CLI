"""File utilities — @ reference resolution and file indexing.

Shared between TUI and VSCode frontends. No Textual dependency.

Provides:
- FileEntry: file/directory data model
- FileAttachment: resolved @reference data model
- FileIndex: fast cached project file scanner
- build_tree_outline: directory tree string generator
- resolve_references: parse @path tokens and load content
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── Data models ──────────────────────────────────────────────

@dataclass
class FileEntry:
    """A single file or directory in the completion list."""

    path: str  # Relative to cwd (e.g. "src/prsm/widgets/")
    is_dir: bool
    size: int | None  # None for directories


@dataclass
class FileAttachment:
    """A resolved @reference."""

    path: str  # Relative path as typed by user
    content: str  # File contents or tree outline
    is_directory: bool
    truncated: bool = False


# ── FileIndex ────────────────────────────────────────────────

SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "target", "build", "dist", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".eggs",
}

MAX_FILE_SIZE: int = 100 * 1024  # 100KB


class FileIndex:
    """Fast file lookup with gitignore filtering and caching."""

    MAX_DEPTH: int = 4
    CACHE_TTL: float = 30.0

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd
        self._entries: list[FileEntry] = []
        self._last_scan: float = 0.0
        self._gitignore_patterns: list[str] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Parse .gitignore from cwd into glob patterns."""
        gi_path = self._cwd / ".gitignore"
        if not gi_path.exists():
            return
        try:
            for line in gi_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self._gitignore_patterns.append(line)
        except OSError:
            pass

    def _is_ignored(self, rel_path: str, is_dir: bool) -> bool:
        """Check if a path matches gitignore or SKIP_DIRS."""
        parts = rel_path.replace("\\", "/").split("/")
        for part in parts:
            if part in SKIP_DIRS:
                return True

        for pattern in self._gitignore_patterns:
            clean = pattern.rstrip("/")
            # Directory-only pattern (trailing /)
            if pattern.endswith("/") and is_dir:
                if fnmatch.fnmatch(parts[-1], clean):
                    return True
                if fnmatch.fnmatch(rel_path, clean):
                    return True
            # Match against basename
            if fnmatch.fnmatch(parts[-1], clean):
                return True
            # Match against full relative path
            if fnmatch.fnmatch(rel_path, clean):
                return True

        return False

    def _scan(self) -> list[FileEntry]:
        """Walk cwd up to MAX_DEPTH using os.scandir()."""
        entries: list[FileEntry] = []
        self._scan_dir(self._cwd, "", 0, entries)
        # Sort: dirs first, then alphabetical within each group
        entries.sort(key=lambda e: (not e.is_dir, e.path.lower()))
        return entries

    def _scan_dir(
        self,
        abs_path: Path,
        rel_prefix: str,
        depth: int,
        entries: list[FileEntry],
    ) -> None:
        """Recursively scan a directory."""
        if depth >= self.MAX_DEPTH:
            return
        try:
            with os.scandir(abs_path) as it:
                for entry in it:
                    rel = f"{rel_prefix}{entry.name}" if rel_prefix else entry.name
                    is_dir = entry.is_dir(follow_symlinks=False)

                    if self._is_ignored(rel, is_dir):
                        continue

                    if is_dir:
                        entries.append(FileEntry(
                            path=rel + "/", is_dir=True, size=None,
                        ))
                        self._scan_dir(
                            abs_path / entry.name,
                            rel + "/",
                            depth + 1,
                            entries,
                        )
                    else:
                        try:
                            size = entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            size = 0
                        entries.append(FileEntry(
                            path=rel, is_dir=False, size=size,
                        ))
        except OSError:
            pass

    def _ensure_fresh(self) -> None:
        """Refresh cache if stale."""
        now = time.monotonic()
        if now - self._last_scan > self.CACHE_TTL or not self._entries:
            self._entries = self._scan()
            self._last_scan = now

    def search(self, prefix: str) -> list[FileEntry]:
        """Return entries matching prefix or containing it as substring.

        Two-tier matching: path-prefix matches first, then substring
        matches anywhere in the path. Both are case-insensitive.
        """
        self._ensure_fresh()
        if not prefix:
            return self._entries[:50]
        prefix_lower = prefix.lower()
        prefix_matches = []
        substring_matches = []
        for e in self._entries:
            path_lower = e.path.lower()
            if path_lower.startswith(prefix_lower):
                prefix_matches.append(e)
            elif prefix_lower in path_lower:
                substring_matches.append(e)
        return (prefix_matches + substring_matches)[:50]


# ── Tree outline ─────────────────────────────────────────────

def build_tree_outline(
    dir_path: Path,
    max_depth: int | None = None,
    _prefix: str = "",
    _depth: int = 0,
) -> str:
    """Generate a visual directory tree string.

    Uses ASCII tree characters for terminal safety.
    Respects SKIP_DIRS. Caps at max_depth levels when provided.
    """
    if max_depth is not None and _depth >= max_depth:
        return ""

    try:
        items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return ""

    # Filter out ignored entries
    filtered = []
    for item in items:
        if item.name in SKIP_DIRS:
            continue
        filtered.append(item)

    lines: list[str] = []
    for i, item in enumerate(filtered):
        is_last = i == len(filtered) - 1
        connector = "|-- " if not is_last else "+-- "
        continuation = "|   " if not is_last else "    "

        if item.is_dir():
            lines.append(f"{_prefix}{connector}{item.name}/")
            subtree = build_tree_outline(
                item,
                max_depth=max_depth,
                _prefix=_prefix + continuation,
                _depth=_depth + 1,
            )
            if subtree:
                lines.append(subtree)
        else:
            lines.append(f"{_prefix}{connector}{item.name}")

    return "\n".join(lines)


# ── Reference resolution ────────────────────────────────────

# Pattern: @ followed by path-like characters (word chars, dots, slashes, hyphens)
# Must be preceded by whitespace or start of string to avoid matching emails
_REF_PATTERN = re.compile(r"(?:^|(?<=\s))@([\w./\-]+)", re.MULTILINE)

# Backtick regions to mask before searching for @references
_BACKTICK_BLOCK = re.compile(r"```[\s\S]*?```")
_BACKTICK_INLINE = re.compile(r"`[^`]*`")


def resolve_references(
    text: str,
    cwd: Path,
) -> tuple[str, list[FileAttachment]]:
    """Parse all @path references from text and resolve them.

    Returns (original_text, list_of_attachments).

    Parsing rules:
    - @path starts after '@' and extends until whitespace or end-of-text
    - '@' inside backtick-delimited regions is ignored
    - Paths are resolved relative to cwd
    - Missing paths produce no attachment
    - Binary files (null bytes in first 1KB) are skipped
    """
    if "@" not in text:
        return (text, [])

    # Build a set of character positions inside backtick regions
    masked: set[int] = set()
    for m in _BACKTICK_BLOCK.finditer(text):
        masked.update(range(m.start(), m.end()))
    for m in _BACKTICK_INLINE.finditer(text):
        masked.update(range(m.start(), m.end()))

    attachments: list[FileAttachment] = []
    seen_paths: set[str] = set()

    for m in _REF_PATTERN.finditer(text):
        if m.start() in masked:
            continue

        ref_path = m.group(1)
        # Normalize: strip trailing dots that aren't part of extensions
        ref_path = ref_path.rstrip(".")

        if ref_path in seen_paths:
            continue
        seen_paths.add(ref_path)

        resolved = (cwd / ref_path).resolve()

        # Security: skip paths outside cwd
        try:
            resolved.relative_to(cwd.resolve())
        except ValueError:
            continue

        if resolved.is_file():
            att = _read_file_attachment(ref_path, resolved)
            if att:
                attachments.append(att)
        elif resolved.is_dir():
            outline = build_tree_outline(resolved)
            attachments.append(FileAttachment(
                path=ref_path,
                content=outline,
                is_directory=True,
            ))

    return (text, attachments)


def _read_file_attachment(
    ref_path: str,
    resolved: Path,
) -> FileAttachment | None:
    """Read a file and return an attachment, or None if binary."""
    try:
        # Check for binary content
        with open(resolved, "rb") as f:
            head = f.read(1024)
        if b"\x00" in head:
            return None  # Binary file

        content = resolved.read_text(errors="replace")
        truncated = False
        if len(content) > MAX_FILE_SIZE:
            content = content[:MAX_FILE_SIZE]
            truncated = True

        return FileAttachment(
            path=ref_path,
            content=content,
            is_directory=False,
            truncated=truncated,
        )
    except OSError:
        return None


def format_size(size: int) -> str:
    """Format a file size for display."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size // 1024}KB"
    else:
        return f"{size // (1024 * 1024)}MB"
