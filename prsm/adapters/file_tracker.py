"""Shared file change tracker for detecting file modifications by agent tools.

Captures file content before Write/Edit tool calls and computes diffs
after completion. Used by both the TUI EventProcessor and VSCode server.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prsm.shared.services.durable_write import _fsync_dir, atomic_write_text

logger = logging.getLogger(__name__)

_TOOL_NAME_ALIASES: dict[str, str] = {
    "read": "Read",
    "read_file": "Read",
    "file_read": "Read",
    "write": "Write",
    "write_file": "Write",
    "file_write": "Write",
    "edit": "Edit",
    "edit_file": "Edit",
    "file_edit": "Edit",
    "bash": "Bash",
    "run_bash": "Bash",
    "run_shell_command": "Bash",
    "glob": "Glob",
    "list_directory": "Glob",
    "grep": "Grep",
    "search_files": "Grep",
}


def normalize_tool_name(tool_name: str) -> str:
    """Normalize provider-specific tool aliases to canonical names."""
    if not tool_name:
        return ""
    bare_name = tool_name
    if bare_name.startswith("mcp__") and bare_name.count("__") >= 2:
        bare_name = bare_name.split("__", 2)[2]
    return _TOOL_NAME_ALIASES.get(bare_name.lower(), bare_name)


def _parse_sed_substitution(expr: str) -> tuple[str, str] | None:
    """Parse sed substitution expression like s/old/new/g."""
    expr = str(expr or "").strip().strip("'\"")
    if len(expr) < 4 or not expr.startswith("s"):
        return None
    delim = expr[1]
    if not delim or delim.isalnum():
        return None
    i = 2
    old: list[str] = []
    new: list[str] = []
    target = old
    escaped = False
    while i < len(expr):
        ch = expr[i]
        i += 1
        if escaped:
            target.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == delim:
            if target is old:
                target = new
                continue
            break
        target.append(ch)
    if target is old:
        return None
    return ("".join(old), "".join(new))


def _extract_last_non_flag(parts: list[str]) -> str:
    for token in reversed(parts):
        if not token.startswith("-"):
            return token
    return ""


def _extract_redirect_target(command: str) -> str:
    m = re.search(r"(?:^|[\s])>>?\s*([^\s|;&]+)", str(command or ""))
    if not m:
        return ""
    return m.group(1).strip("'\"")


def _split_shell_segments(command: str) -> list[str]:
    """Split shell command on operators while respecting quotes."""
    if not command:
        return []
    segments: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    while i < len(command):
        ch = command[i]
        nxt = command[i + 1] if i + 1 < len(command) else ""
        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            current.append(ch)
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            if ch == ";" or (ch == "|" and nxt == "|") or (ch == "&" and nxt == "&"):
                seg = "".join(current).strip()
                if seg:
                    segments.append(seg)
                current = []
                i += 2 if ch in {"|", "&"} and nxt == ch else 1
                continue
            if ch == "|":
                seg = "".join(current).strip()
                if seg:
                    segments.append(seg)
                current = []
                i += 1
                continue
        current.append(ch)
        i += 1
    seg = "".join(current).strip()
    if seg:
        segments.append(seg)
    return segments


def _resolve_path(path_str: str, cwd: Path | None) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    base = cwd or Path.cwd()
    return str((base / p).resolve())


def _is_ignored_tmp_path(file_path: str) -> bool:
    """Return True for temporary .tmp* paths we never track as file changes."""
    if not file_path:
        return False
    try:
        parts = Path(file_path).parts
    except Exception:
        return False
    return any(part.startswith(".tmp") for part in parts if part and part != ".")


def _is_ignored_artifact_name(file_path: str) -> bool:
    """Return True for known local scratch artifacts that should never be tracked."""
    name = Path(file_path).name
    if not name:
        return False
    if name in {".file_index.txt"}:
        return True
    if re.fullmatch(r"\.grep_.*\.txt", name):
        return True
    if re.fullmatch(r"\.pytest_.*\.txt", name):
        return True
    if re.fullmatch(r"\.diff_.*\.txt", name):
        return True
    if re.fullmatch(r"\.compile_.*\.txt", name):
        return True
    # Filter patch artifacts (.rej / .orig) left by `patch(1)`.
    if name.endswith(".rej") or name.endswith(".orig"):
        return True
    return False


def _is_ignored_tracking_path(file_path: str) -> bool:
    return _is_ignored_tmp_path(file_path) or _is_ignored_artifact_name(file_path)


def _infer_file_tool_from_bash_command(
    command: str, cwd: Path | None = None,
) -> tuple[str, dict] | None:
    """Infer Write/Edit semantics from Bash command payloads."""
    if not command:
        return None

    # Handle chained commands (cd ... && sed -i ..., etc.)
    segments = _split_shell_segments(command)
    if len(segments) > 1:
        effective_cwd = cwd
        for seg in segments:
            try:
                tokens = shlex.split(seg)
            except ValueError:
                tokens = []
            if len(tokens) >= 2 and tokens[0] == "cd":
                target = tokens[1].strip("'\"")
                effective_cwd = Path(_resolve_path(target, effective_cwd))
                continue
            inferred = _infer_file_tool_from_bash_command(seg, effective_cwd)
            if inferred:
                return inferred
        return None

    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None

    prog = os.path.basename(parts[0])

    # Shell wrappers: zsh -lc "..."
    if prog in {"bash", "sh", "zsh"}:
        for i, token in enumerate(parts[1:], start=1):
            if token in {"-c", "-lc"} and i + 1 < len(parts):
                return _infer_file_tool_from_bash_command(parts[i + 1], cwd)

    if prog == "sed":
        is_inplace = any(t == "-i" or t.startswith("-i") for t in parts[1:])
        if is_inplace:
            expr_token = ""
            file_path = ""
            for token in parts[1:]:
                if token.startswith("-"):
                    continue
                if _parse_sed_substitution(token):
                    expr_token = token
                    continue
                file_path = token
            parsed = _parse_sed_substitution(expr_token)
            if parsed and file_path:
                old_string, new_string = parsed
                return ("Edit", {"file_path": _resolve_path(file_path, cwd), "old_string": old_string, "new_string": new_string})
            if file_path:
                return ("Write", {"file_path": _resolve_path(file_path, cwd)})

    if prog == "tee":
        file_path = _extract_last_non_flag(parts[1:])
        if file_path:
            return ("Write", {"file_path": _resolve_path(file_path, cwd)})

    redirect_target = _extract_redirect_target(command)
    if redirect_target and prog in {"echo", "printf", "cat"}:
        return ("Write", {"file_path": _resolve_path(redirect_target, cwd)})

    return None


@dataclass
class FileChangeRecord:
    """Tracks a single file modification made by an agent."""
    file_path: str
    agent_id: str
    change_type: str  # "create", "modify", "delete"
    tool_call_id: str
    tool_name: str
    message_index: int
    old_content: str | None = None
    new_content: str | None = None  # Full file content after the tool ran
    pre_tool_content: str | None = None  # Full file content before the tool ran
    added_ranges: list = field(default_factory=list)
    removed_ranges: list = field(default_factory=list)
    timestamp: str = ""
    status: str = "pending"  # "pending", "accepted", "rejected"

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "file_path": self.file_path,
            "agent_id": self.agent_id,
            "change_type": self.change_type,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "message_index": self.message_index,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "pre_tool_content": self.pre_tool_content,
            "added_ranges": self.added_ranges,
            "removed_ranges": self.removed_ranges,
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileChangeRecord:
        """Deserialize from a dict."""
        return cls(
            file_path=data["file_path"],
            agent_id=data["agent_id"],
            change_type=data["change_type"],
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            message_index=data.get("message_index", 0),
            old_content=data.get("old_content"),
            new_content=data.get("new_content"),
            pre_tool_content=data.get("pre_tool_content"),
            added_ranges=data.get("added_ranges", []),
            removed_ranges=data.get("removed_ranges", []),
            timestamp=data.get("timestamp", ""),
            status=data.get("status", "pending"),
        )


class FileChangeTracker:
    """Tracks file modifications made by agents during orchestration.

    Usage:
        tracker = FileChangeTracker()

        # When a Write/Edit tool starts:
        tracker.capture_pre_tool(tool_id, tool_name, arguments)

        # When a Write/Edit tool completes (non-error):
        record = tracker.track_change(agent_id, tool_id)
    """

    def __init__(self) -> None:
        self.file_changes: dict[str, list[FileChangeRecord]] = {}
        self._pre_tool_content: dict[str, str | None] = {}
        self._tool_call_args: dict[str, dict] = {}
        self._tool_call_names: dict[str, str] = {}
        self._tool_call_cwds: dict[str, Path | None] = {}
        self._pre_tool_snapshots: dict[str, dict[str, tuple[str | None, str | None]]] = {}

    def clear(self) -> None:
        self.file_changes.clear()
        self._pre_tool_content.clear()
        self._tool_call_args.clear()
        self._tool_call_names.clear()
        self._tool_call_cwds.clear()
        self._pre_tool_snapshots.clear()

    @staticmethod
    def _status_path_set(root: Path) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return set()
        if proc.returncode != 0:
            return set()
        paths: set[str] = set()
        for raw_line in proc.stdout.splitlines():
            line = raw_line.rstrip()
            if len(line) < 4:
                continue
            path_part = line[3:]
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1]
            rel_path = path_part.strip()
            if not rel_path or rel_path.startswith(".git/"):
                continue
            if _is_ignored_tracking_path(rel_path):
                continue
            paths.add(rel_path)
        return paths

    @staticmethod
    def _safe_read_text(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    @staticmethod
    def _content_hash(content: str | None) -> str | None:
        if content is None:
            return None
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def _snapshot_status_files(self, root: Path | None) -> dict[str, tuple[str | None, str | None]]:
        if root is None:
            return {}
        paths = self._status_path_set(root)
        snapshot: dict[str, tuple[str | None, str | None]] = {}
        for rel_path in paths:
            file_path = (root / rel_path).resolve()
            content = self._safe_read_text(file_path)
            snapshot[rel_path] = (self._content_hash(content), content)
        return snapshot

    def persist(self, changes_dir: Path) -> None:
        """Persist all file change records to individual JSON files.

        Args:
            changes_dir: Directory to write {tool_call_id}.json files into.
        """
        try:
            changes_dir.mkdir(parents=True, exist_ok=True)
            expected_ids: set[str] = set()
            for records in self.file_changes.values():
                for r in records:
                    expected_ids.add(r.tool_call_id)
                    change_file = changes_dir / f"{r.tool_call_id}.json"
                    atomic_write_text(
                        change_file,
                        json.dumps(r.to_dict()),
                    )
            # Keep on-disk state in sync when records are removed in-memory
            for existing in changes_dir.glob("*.json"):
                if existing.stem not in expected_ids:
                    try:
                        existing.unlink()
                    except OSError:
                        logger.debug("Failed to delete stale file change %s", existing)
            _fsync_dir(changes_dir)
        except Exception:
            logger.debug("Failed to persist file changes to %s", changes_dir)

    def load(self, changes_dir: Path) -> None:
        """Load persisted file change records from disk, merging into current state.

        Deduplicates by tool_call_id to avoid loading the same record twice.

        Args:
            changes_dir: Directory containing {tool_call_id}.json files.
        """
        try:
            if not changes_dir.exists():
                return
            for change_file in changes_dir.glob("*.json"):
                try:
                    data = json.loads(change_file.read_text(encoding="utf-8"))
                    record = FileChangeRecord.from_dict(data)
                    fp = record.file_path
                    if fp not in self.file_changes:
                        self.file_changes[fp] = []
                    # Avoid duplicates
                    existing_ids = {
                        r.tool_call_id for r in self.file_changes[fp]
                    }
                    if record.tool_call_id not in existing_ids:
                        self.file_changes[fp].append(record)
                except Exception:
                    logger.debug("Failed to load file change from %s", change_file)
        except Exception:
            logger.debug("Failed to load file changes from %s", changes_dir)

    def capture_pre_tool(
        self,
        tool_id: str,
        tool_name: str,
        arguments: str | dict,
        cwd: Path | str | None = None,
    ) -> None:
        """Capture pre-tool filesystem snapshot for robust post-tool deltas.

        Args:
            tool_id: The tool call ID.
            tool_name: The tool name (Write, Edit, etc.).
            arguments: Tool arguments as a JSON string or dict.
        """
        try:
            if isinstance(arguments, dict):
                args = arguments
            elif isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict):
                        args = parsed
                    elif isinstance(parsed, str):
                        # Some providers pass a plain file path as JSON string.
                        args = {"file_path": parsed}
                    else:
                        args = {"_raw": arguments}
                except (json.JSONDecodeError, ValueError):
                    # Fallback: arguments may be Python repr (str(dict)) with single quotes
                    try:
                        parsed = ast.literal_eval(arguments)
                        if isinstance(parsed, dict):
                            args = parsed
                        elif isinstance(parsed, str):
                            args = {"file_path": parsed}
                        else:
                            args = {"_raw": arguments}
                    except (ValueError, SyntaxError):
                        # Some providers emit raw command string for Bash.
                        args = {"_raw": arguments}
            else:
                return

            if not isinstance(args, dict):
                return

            normalized_tool_name = normalize_tool_name(tool_name)
            default_cwd: Path | None = None
            if isinstance(cwd, Path):
                default_cwd = cwd.resolve()
            elif isinstance(cwd, str) and cwd.strip():
                default_cwd = Path(cwd.strip()).resolve()

            inferred_name = normalized_tool_name
            self._tool_call_cwds[tool_id] = default_cwd
            self._pre_tool_snapshots[tool_id] = self._snapshot_status_files(default_cwd)

            inferred_args = dict(args)
            if normalized_tool_name == "Bash":
                command = str(args.get("command", args.get("_raw", "")))
                inferred = _infer_file_tool_from_bash_command(command, default_cwd)
                if inferred:
                    inferred_name, inferred_args = inferred
            self._tool_call_names[tool_id] = inferred_name

            file_path = inferred_args.get("file_path", "")
            if not file_path and inferred_name != "Bash":
                raw_arg = inferred_args.get("_raw", "")
                if isinstance(raw_arg, str) and raw_arg.strip():
                    file_path = raw_arg.strip().strip("'\"")
                    inferred_args["file_path"] = file_path

            if file_path and not Path(file_path).is_absolute():
                cwd_value = inferred_args.get("cwd")
                cwd_path = Path(cwd_value).resolve() if isinstance(cwd_value, str) and cwd_value else default_cwd
                resolved = _resolve_path(file_path, cwd_path)
                inferred_args["file_path"] = resolved
                file_path = resolved

            self._tool_call_args[tool_id] = inferred_args
            if file_path:
                self._pre_tool_content[tool_id] = self._safe_read_text(Path(file_path))
            logger.debug(
                "capture_pre_tool: snapshot for tool %s cwd=%s tracked_paths=%d",
                tool_id[:12],
                str(default_cwd) if default_cwd else "<none>",
                len(self._pre_tool_snapshots[tool_id]),
            )
        except Exception:
            logger.debug("capture_pre_tool: unexpected error for tool %s", tool_id[:12], exc_info=True)

    def track_change(
        self,
        agent_id: str,
        tool_id: str,
        message_index: int = 0,
    ) -> FileChangeRecord | None:
        records = self.track_changes(agent_id, tool_id, message_index=message_index)
        return records[0] if records else None

    def _fallback_record_from_args(
        self,
        *,
        tool_id: str,
        tool_name: str,
        agent_id: str,
        message_index: int,
        args: dict,
        pre_content: str | None,
    ) -> list[FileChangeRecord]:
        # Bash fallback based on raw args is unsafe and can misclassify command
        # strings as file paths. Bash changes should come from snapshot diffs.
        if tool_name in {"Read", "Bash"}:
            return []
        file_path = str(args.get("file_path") or "")
        if not file_path or _is_ignored_tracking_path(file_path):
            return []
        path = Path(file_path)
        if path.exists():
            change_type = "modify" if pre_content is not None else "create"
            new_content = self._safe_read_text(path)
        else:
            change_type = "delete"
            new_content = None
        if change_type == "modify" and pre_content == new_content:
            return []
        record = FileChangeRecord(
            file_path=str(path),
            agent_id=agent_id,
            change_type=change_type,
            tool_call_id=tool_id,
            tool_name=tool_name,
            message_index=message_index,
            old_content=pre_content,
            new_content=new_content,
            pre_tool_content=pre_content,
            added_ranges=[],
            removed_ranges=[],
            timestamp=datetime.now().isoformat(),
        )
        self.file_changes.setdefault(record.file_path, []).append(record)
        return [record]

    def track_changes(
        self,
        agent_id: str,
        tool_id: str,
        message_index: int = 0,
    ) -> list[FileChangeRecord]:
        """Track all file modifications produced by a completed tool call.

        Uses pre/post git-status snapshots under the tool's cwd to detect
        changed paths robustly across any tool type.
        """
        args = self._tool_call_args.pop(tool_id, {})
        pre_content = self._pre_tool_content.pop(tool_id, None)
        tool_name = normalize_tool_name(self._tool_call_names.pop(tool_id, ""))
        cwd = self._tool_call_cwds.pop(tool_id, None)
        pre_snapshot = self._pre_tool_snapshots.pop(tool_id, {})
        post_snapshot = self._snapshot_status_files(cwd)

        records: list[FileChangeRecord] = []
        candidate_paths = sorted(set(pre_snapshot.keys()) | set(post_snapshot.keys()))
        for idx, rel_path in enumerate(candidate_paths):
            if _is_ignored_tracking_path(rel_path):
                continue
            old_hash, old_content = pre_snapshot.get(rel_path, (None, None))
            new_hash, new_content = post_snapshot.get(rel_path, (None, None))
            if old_hash == new_hash:
                continue

            if old_hash is None and new_hash is not None:
                change_type = "create"
            elif old_hash is not None and new_hash is None:
                change_type = "delete"
            else:
                change_type = "modify"

            added_ranges: list[dict] = []
            removed_ranges: list[dict] = []
            if change_type == "create" and new_content is not None:
                line_count = new_content.count("\n") + 1
                added_ranges = [{"startLine": 0, "endLine": line_count - 1}]
            elif change_type == "delete" and old_content is not None:
                line_count = old_content.count("\n") + 1
                removed_ranges = [{"startLine": 0, "endLine": line_count - 1}]
            elif old_content is not None and new_content is not None:
                added_ranges, removed_ranges = _compute_line_ranges(
                    old_content.split("\n"),
                    new_content.split("\n"),
                )

            abs_path = str(((cwd or Path.cwd()) / rel_path).resolve())
            record_tool_id = tool_id if len(candidate_paths) == 1 else f"{tool_id}:{idx}"
            record = FileChangeRecord(
                file_path=abs_path,
                agent_id=agent_id,
                change_type=change_type,
                tool_call_id=record_tool_id,
                tool_name=tool_name or "Write",
                message_index=message_index,
                old_content=old_content,
                new_content=new_content,
                pre_tool_content=old_content,
                added_ranges=added_ranges,
                removed_ranges=removed_ranges,
                timestamp=datetime.now().isoformat(),
            )
            self.file_changes.setdefault(record.file_path, []).append(record)
            records.append(record)

        if records:
            return records
        return self._fallback_record_from_args(
            tool_id=tool_id,
            tool_name=tool_name or "Write",
            agent_id=agent_id,
            message_index=message_index,
            args=args,
            pre_content=pre_content,
        )


def _compute_line_ranges(
    old_lines: list[str], new_lines: list[str]
) -> tuple[list[dict], list[dict]]:
    """Compute added/removed line ranges from old vs new line lists."""
    max_len = max(len(old_lines), len(new_lines))
    first_diff = -1
    for i in range(max_len):
        old = old_lines[i] if i < len(old_lines) else None
        new = new_lines[i] if i < len(new_lines) else None
        if old != new:
            first_diff = i
            break

    if first_diff == -1:
        return [], []

    last_diff_old = len(old_lines) - 1
    last_diff_new = len(new_lines) - 1
    while last_diff_old > first_diff and last_diff_new > first_diff:
        if old_lines[last_diff_old] == new_lines[last_diff_new]:
            last_diff_old -= 1
            last_diff_new -= 1
        else:
            break

    added: list[dict] = []
    removed: list[dict] = []
    if last_diff_old >= first_diff:
        removed.append({"startLine": first_diff, "endLine": last_diff_old})
    if last_diff_new >= first_diff:
        added.append({"startLine": first_diff, "endLine": last_diff_new})
    return added, removed
