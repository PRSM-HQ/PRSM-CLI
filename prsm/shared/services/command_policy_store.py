"""Workspace command whitelist/blacklist persistence.

Rules are stored as one regex pattern per line in:
- <workspace>/.prism/command_whitelist.txt
- <workspace>/.prism/command_blacklist.txt
"""
from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PRISM_DIRNAME = ".prism"
WHITELIST_FILENAME = "command_whitelist.txt"
BLACKLIST_FILENAME = "command_blacklist.txt"
DEFAULT_BLACKLIST_PATTERNS: tuple[str, ...] = (
    r"(?:^|\&\&|\|\||;|&|\|)\s*rm(?:\s|$)",
    r"(?:^|\&\&|\|\||;|&|\|)\s*git\s+commit(?:\s|$)",
)


@dataclass
class CommandPolicyRules:
    """Compiled command policy regex rules."""

    whitelist: list[re.Pattern[str]]
    blacklist: list[re.Pattern[str]]


class CommandPolicyStore:
    """Reads and writes workspace command policy files."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._prism_dir = self._workspace_dir / PRISM_DIRNAME
        self._whitelist_path = self._prism_dir / WHITELIST_FILENAME
        self._blacklist_path = self._prism_dir / BLACKLIST_FILENAME

    @property
    def prism_dir(self) -> Path:
        return self._prism_dir

    @property
    def whitelist_path(self) -> Path:
        return self._whitelist_path

    @property
    def blacklist_path(self) -> Path:
        return self._blacklist_path

    def ensure_files(self) -> None:
        """Create policy files if missing."""
        self._prism_dir.mkdir(parents=True, exist_ok=True)
        if not self._whitelist_path.exists():
            self._whitelist_path.write_text("", encoding="utf-8")
        if not self._blacklist_path.exists():
            self._blacklist_path.write_text("", encoding="utf-8")

    def load_compiled(self) -> CommandPolicyRules:
        """Load and compile regex lists from workspace policy files."""
        whitelist_patterns = self._read_patterns(self._whitelist_path)
        blacklist_patterns = list(DEFAULT_BLACKLIST_PATTERNS)
        blacklist_patterns.extend(self._read_patterns(self._blacklist_path))
        logger.debug(
            "Command policy loaded: whitelist=%d patterns from %s, "
            "blacklist=%d patterns (%d default + %d custom) from %s",
            len(whitelist_patterns), self._whitelist_path,
            len(blacklist_patterns), len(DEFAULT_BLACKLIST_PATTERNS),
            len(blacklist_patterns) - len(DEFAULT_BLACKLIST_PATTERNS),
            self._blacklist_path,
        )
        return CommandPolicyRules(
            whitelist=self._compile_patterns(whitelist_patterns),
            blacklist=self._compile_patterns(blacklist_patterns),
        )

    def read_whitelist(self) -> list[str]:
        """Return raw whitelist patterns (non-compiled)."""
        return self._read_patterns(self._whitelist_path)

    def read_blacklist(self) -> list[str]:
        """Return raw blacklist patterns (non-compiled)."""
        return self._read_patterns(self._blacklist_path)

    def add_whitelist_pattern(self, pattern: str) -> None:
        self._add_pattern(self._whitelist_path, pattern)

    def add_blacklist_pattern(self, pattern: str) -> None:
        self._add_pattern(self._blacklist_path, pattern)

    def remove_whitelist_pattern(self, pattern: str) -> bool:
        """Remove a pattern from the whitelist. Returns True if removed."""
        return self._remove_pattern(self._whitelist_path, pattern)

    def remove_blacklist_pattern(self, pattern: str) -> bool:
        """Remove a pattern from the blacklist. Returns True if removed."""
        return self._remove_pattern(self._blacklist_path, pattern)

    @staticmethod
    def build_command_pattern(command: str, *, allow: bool) -> str:
        """Build a regex pattern for a command persistence rule.

        For allow rules, uses command-class-aware templating:
        - destructive commands: strict path-shape matching
        - exploratory commands: broader argument wildcards
        For deny rules, keeps exact matching to avoid broad blocks.
        """
        cleaned = str(command or "").strip()
        if not cleaned:
            return ""
        if not allow:
            return CommandPolicyStore._exact_pattern(cleaned)

        try:
            tokens = shlex.split(cleaned)
        except ValueError:
            return CommandPolicyStore._exact_pattern(cleaned)
        if not tokens:
            return CommandPolicyStore._exact_pattern(cleaned)

        command_class = CommandPolicyStore._classify_command(tokens)
        token_patterns = CommandPolicyStore._build_token_patterns(
            tokens, command_class=command_class,
        )
        if not token_patterns:
            return CommandPolicyStore._exact_pattern(cleaned)
        body = r"\s+".join(token_patterns)
        return rf"^\s*{body}\s*$"

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
            except re.error:
                logger.warning("Invalid command policy regex ignored: %s", pattern)
        return compiled

    @staticmethod
    def _read_patterns(path: Path) -> list[str]:
        if not path.exists():
            return []
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            lines.append(entry)
        return lines

    def _add_pattern(self, path: Path, pattern: str) -> None:
        cleaned = pattern.strip()
        if not cleaned:
            return
        self.ensure_files()
        entries = self._read_patterns(path)
        if cleaned in entries:
            return
        entries.append(cleaned)
        path.write_text("\n".join(entries) + "\n", encoding="utf-8")

    def _remove_pattern(self, path: Path, pattern: str) -> bool:
        """Remove a pattern from a policy file. Returns True if found and removed."""
        cleaned = pattern.strip()
        if not cleaned:
            return False
        entries = self._read_patterns(path)
        if cleaned not in entries:
            return False
        entries.remove(cleaned)
        if entries:
            path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
        return True

    @staticmethod
    def _exact_pattern(command: str) -> str:
        return f"^{re.escape(command.strip())}$"

    @staticmethod
    def _classify_command(tokens: list[str]) -> str:
        exe = Path(tokens[0]).name.lower()
        sub = ""
        for token in tokens[1:]:
            if not token.startswith("-"):
                sub = token.lower()
                break

        destructive_executables = {"rm", "rmdir", "unlink", "shred", "srm"}
        if exe in destructive_executables:
            return "destructive"
        if exe == "git" and sub in {"clean", "rm", "reset"}:
            return "destructive"
        if exe == "find" and any(t.lower() == "-delete" for t in tokens[1:]):
            return "destructive"
        if exe == "kubectl" and sub == "delete":
            return "destructive"
        if exe in {"docker", "podman"} and sub in {"rm", "rmi"}:
            return "destructive"
        if exe == "aws" and len(tokens) >= 3:
            if tokens[1].lower() == "s3" and tokens[2].lower() in {"rm", "rb"}:
                return "destructive"

        exploratory_executables = {
            "ls", "grep", "rg", "find", "cat", "head", "tail", "wc", "sort",
            "uniq", "cut", "awk", "sed", "tree", "du", "df", "which",
            "whereis", "type",
        }
        if exe in exploratory_executables:
            return "exploratory"
        if exe == "git" and sub in {
            "status", "log", "show", "diff", "grep", "ls-files", "branch",
            "remote", "rev-parse",
        }:
            return "exploratory"
        return "default"

    @staticmethod
    def _build_token_patterns(tokens: list[str], *, command_class: str) -> list[str]:
        patterns: list[str] = []
        exe = Path(tokens[0]).name
        patterns.append(re.escape(exe))

        subcommand_index: int | None = None
        if exe.lower() in {"git", "aws", "docker", "podman", "kubectl"}:
            for i in range(1, len(tokens)):
                if not tokens[i].startswith("-"):
                    subcommand_index = i
                    break

        for i, token in enumerate(tokens[1:], start=1):
            if i == subcommand_index:
                patterns.append(re.escape(token))
                continue
            if token.startswith("-"):
                patterns.append(re.escape(token))
                continue

            if command_class == "exploratory":
                patterns.append(r"[^\s]+")
                continue

            if command_class == "destructive":
                if CommandPolicyStore._looks_like_path(token):
                    patterns.append(
                        CommandPolicyStore._strict_path_shape_pattern(token),
                    )
                elif CommandPolicyStore._looks_variable(token):
                    patterns.append(r"[^\s]+")
                else:
                    patterns.append(re.escape(token))
                continue

            # default class
            if CommandPolicyStore._looks_variable(token):
                patterns.append(r"[^\s]+")
            else:
                patterns.append(re.escape(token))
        return patterns

    @staticmethod
    def _looks_like_path(token: str) -> bool:
        if "/" in token:
            return True
        if token.startswith(("./", "../", "~/", "/")):
            return True
        if token in {".", "..", "~"}:
            return True
        return token.endswith((".py", ".sh", ".txt", ".json", ".yaml", ".yml"))

    @staticmethod
    def _strict_path_shape_pattern(token: str) -> str:
        if token in {".", "..", "~"}:
            return re.escape(token)

        prefix = ""
        rest = token
        if token.startswith("~/"):
            prefix = "~/"
            rest = token[2:]
        elif token.startswith("./"):
            prefix = "./"
            rest = token[2:]
        elif token.startswith("../"):
            prefix = "../"
            rest = token[3:]
        elif token.startswith("/"):
            prefix = "/"
            rest = token[1:]

        parts = [part for part in rest.split("/") if part]
        if not parts:
            return re.escape(token)

        root = re.escape(parts[0])
        if prefix:
            base = re.escape(prefix) + root
        else:
            base = root
        return base + r"(?:/[^\s]+)*"

    @staticmethod
    def _looks_variable(token: str) -> bool:
        lowered = token.lower()
        if re.fullmatch(r"\d+", token):
            return True
        if re.fullmatch(r"[0-9a-f]{7,64}", lowered):
            return True
        if re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            lowered,
        ):
            return True
        if re.search(r"\d{4}-\d{2}-\d{2}", token):
            return True
        return bool(re.search(r"[\*\?\[\]]", token))
