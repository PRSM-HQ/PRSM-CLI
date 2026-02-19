"""Rich tool call formatting with per-tool-type rendering.

Provides a registry-based formatter system that parses tool arguments and
produces structured intermediate representations. Renderers then convert
the IR to Rich markup (TUI) or can be mirrored in JS for the VSCode webview.

Adding a new tool format requires only a single decorated function:

    @tool_formatter("MyTool")
    def _format_my_tool(name, args, result, success):
        return FormattedToolCall(icon="🔧", label=name, summary=..., sections=[...])
"""

from __future__ import annotations

import ast
import json
import os
import shlex
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse


# ── Intermediate Representation ──


@dataclass
class Section:
    """A typed content section in the expanded tool call view.

    Supported kinds:
        "diff"         → content: {"old_lines": list[str], "new_lines": list[str]}
        "code"         → content: {"language": str, "text": str}
        "terminal"     → content: {"command": str, "output": str}
        "path"         → content: str (file path)
        "checklist"    → content: list[{"text": str, "done": bool}]
        "kv"           → content: dict[str, str]
        "plain"        → content: str
        "progress"     → content: {"percent": int, "status": str}
        "agent_prompt" → content: {"number": int|None, "prompt": str, "model": str, "complexity": str}
        "transcript"   → content: list[{"role": str, "text": str, "tool": str|None}]
        "result_block" → content: {"text": str, "status": str} — prominent bordered result
    """

    kind: str
    title: str = ""
    content: Any = None


@dataclass
class FormattedToolCall:
    """Structured representation of a formatted tool call."""

    icon: str = ""
    label: str = ""
    summary: str = ""
    file_path: str = ""  # Primary file path (for click-to-open)
    sections: list[Section] = field(default_factory=list)


# ── Argument Parsing ──


def parse_args(arguments: str) -> dict:
    """Parse a tool arguments string to a dict, falling back gracefully.

    The engine serialises arguments via ``json.dumps(block.input)`` for dicts,
    so ``json.loads`` succeeds for the common case.  Falls back to
    ``ast.literal_eval`` for Python repr format, then to ``{"_raw": arguments}``.
    """
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": arguments}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(arguments)
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": arguments}
    except (ValueError, SyntaxError):
        pass
    return {"_raw": arguments}


# ── Formatter Registry ──

_FORMATTERS: dict[str, Callable[..., FormattedToolCall]] = {}
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
    "replace_string": "Edit",
    "replacestring": "Edit",
    "bash": "Bash",
    "run_bash": "Bash",
    "run_shell_command": "Bash",
    "glob": "Glob",
    "list_directory": "Glob",
    "grep": "Grep",
    "search_files": "Grep",
    "askuserquestion": "ask_user",
    "request_user_input": "ask_user",
}


def tool_formatter(name: str):
    """Decorator to register a formatter for a given tool name."""

    def decorator(fn: Callable[..., FormattedToolCall]):
        _FORMATTERS[name] = fn
        return fn

    return decorator


def _normalize_tool_name(name: str) -> str:
    """Strip MCP server prefix and map provider aliases to canonical names.

    E.g. ``mcp__orchestrator__task_complete`` → ``task_complete`` and
    ``run_bash`` → ``Bash``.
    """
    if name.startswith("mcp__") and name.count("__") >= 2:
        name = name.split("__", 2)[2]
    return _TOOL_NAME_ALIASES.get(name.lower(), name)


def format_tool_call(
    name: str,
    arguments: str,
    result: str | None = None,
    success: bool = True,
) -> FormattedToolCall:
    """Main entry point — dispatch to a registered formatter or the default."""
    args = parse_args(arguments)
    # Try exact match first, then try with MCP prefix stripped
    formatter = _FORMATTERS.get(name)
    if formatter is None:
        bare = _normalize_tool_name(name)
        formatter = _FORMATTERS.get(bare, _format_default)
    return formatter(name, args, result, success)


# ── Helpers ──


def _basename(path: str) -> str:
    """Extract a short display path (last 2 components)."""
    if not path:
        return ""
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _trunc(text: str, length: int = 60) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def _result_section(result: str | None, title: str = "Output") -> list[Section]:
    """Return a plain-text result section if result is available."""
    if result:
        return [Section(kind="plain", title=title, content=result)]
    return []


def _extract_text_result(result: str | None) -> str:
    """Extract human-readable text from structured tool results when possible."""
    if not result:
        return ""
    text = str(result)

    def _extract_from_payload(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str):
                            chunks.append(t)
                if chunks:
                    return "\n".join(chunks)
            for key in ("text", "stdout", "output", "result", "content"):
                val = payload.get(key)
                if isinstance(val, str):
                    return val
                if isinstance(val, (dict, list)):
                    nested = _extract_from_payload(val)
                    if nested:
                        return nested
        if isinstance(payload, list):
            chunks: list[str] = []
            for item in payload:
                nested = _extract_from_payload(item)
                if nested:
                    chunks.append(nested)
            if chunks:
                return "\n".join(chunks)
        return ""

    candidate = text.strip()
    for _ in range(3):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            break
        if isinstance(parsed, str):
            candidate = parsed.strip()
            if '\\"' in candidate:
                candidate = candidate.replace('\\"', '"')
            continue
        extracted = _extract_from_payload(parsed)
        if extracted:
            return extracted
        break

    for _ in range(2):
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            break
        if isinstance(parsed, str):
            candidate = parsed.strip()
            continue
        extracted = _extract_from_payload(parsed)
        if extracted:
            return extracted
        break

    # Fallback for truncated/invalid wrappers: extract embedded text fields.
    matches = re.findall(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if not matches:
        matches = re.findall(r"'text'\s*:\s*'((?:\\.|[^'\\])*)'", text)
    if not matches:
        matches = re.findall(r'\\"text\\"\s*:\s*\\"((?:\\\\.|[^"\\\\])*)\\"', text)
    if matches:
        chunks: list[str] = []
        for raw in matches:
            try:
                chunks.append(json.loads(f'"{raw}"'))
            except (json.JSONDecodeError, TypeError):
                chunks.append(
                    raw.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "\r")
                    .replace("\\'", "'")
                )
        return "\n".join(chunks)

    return text


def _extract_last_non_flag(parts: list[str]) -> str:
    for token in reversed(parts):
        if not token.startswith("-"):
            return token
    return ""


def _parse_sed_substitution(expr: str) -> tuple[str, str] | None:
    """Parse sed substitution expression like s/old/new/g."""
    expr = expr.strip().strip("'\"")
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


def _split_shell_segments(command: str) -> list[str]:
    """Split a shell command on pipes/chain operators while respecting quotes."""
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


def _extract_redirect_target(command: str) -> str:
    """Extract the target file from simple shell redirection operators."""
    if not command:
        return ""
    m = re.search(r"(?:^|[\s])>>?\s*([^\s|;&]+)", command)
    if not m:
        return ""
    return m.group(1).strip("'\"")


def _extract_url(parts: list[str]) -> str:
    for token in parts:
        if token.startswith("http://") or token.startswith("https://"):
            return token
    return ""


def _describe_command_map(mapped_name: str, mapped_args: dict) -> str:
    if mapped_name == "Read":
        file_path = mapped_args.get("file_path", "")
        return f"Read {_basename(file_path)}" if file_path else "Read file"
    if mapped_name == "Write":
        file_path = mapped_args.get("file_path", "")
        return f"Write {_basename(file_path)}" if file_path else "Write file"
    if mapped_name == "Edit":
        file_path = mapped_args.get("file_path", "")
        return f"Edit {_basename(file_path)}" if file_path else "Edit file"
    if mapped_name == "Grep":
        pattern = mapped_args.get("pattern", "")
        return f'Search "{_trunc(pattern, 24)}"'
    if mapped_name == "Glob":
        pattern = mapped_args.get("pattern", "")
        return f"List {pattern or 'files'}"
    if mapped_name == "GitInspect":
        sub = mapped_args.get("subcommand", "inspect")
        return f"Git {sub}"
    if mapped_name == "TestRun":
        return f"Run {_trunc(mapped_args.get('command', 'tests'), 36)}"
    if mapped_name == "BuildRun":
        return f"Build {_trunc(mapped_args.get('command', ''), 36)}".strip()
    if mapped_name == "ProcessControl":
        action = mapped_args.get("action", "process")
        return f"Process {action}"
    if mapped_name == "HttpCall":
        method = mapped_args.get("method", "GET")
        url = mapped_args.get("url", "")
        return f"{method} {_trunc(url, 30)}".strip()
    if mapped_name == "JsonExtract":
        return f"JSON {_trunc(mapped_args.get('query', ''), 32)}".strip()
    if mapped_name == "Stats":
        return f"Stats {_trunc(mapped_args.get('command', ''), 28)}".strip()
    return mapped_name


def _interpret_single_shell_command(command: str) -> tuple[str, dict] | None:
    """Interpret one shell segment as a structured pseudo-tool call."""
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None

    prog = parts[0]

    # list/search/read
    if prog in ("rg", "ripgrep"):
        if "--files" in parts:
            root = _extract_last_non_flag(parts[1:]) or "."
            return ("Glob", {"pattern": "**/*", "path": root})
        pattern = ""
        path = ""
        for token in parts[1:]:
            if token.startswith("-"):
                continue
            if not pattern:
                pattern = token
            else:
                path = token
                break
        return ("Grep", {"pattern": pattern, "path": path})
    if prog == "grep":
        pattern = ""
        path = ""
        for token in parts[1:]:
            if token.startswith("-"):
                continue
            if not pattern:
                pattern = token
            else:
                path = token
                break
        return ("Grep", {"pattern": pattern, "path": path})
    if prog == "ls":
        path = _extract_last_non_flag(parts[1:]) or "."
        return ("Glob", {"pattern": "*", "path": path})
    if prog == "find":
        path = "."
        pattern = "**/*"
        rest = parts[1:]
        if rest and not rest[0].startswith("-"):
            path = rest[0]
            rest = rest[1:]
        for i, token in enumerate(rest):
            if token in ("-name", "-iname") and i + 1 < len(rest):
                candidate = rest[i + 1].strip("'\"")
                if candidate:
                    pattern = candidate
                break
        return ("Glob", {"pattern": pattern, "path": path})
    if prog == "cat":
        file_path = _extract_last_non_flag(parts[1:])
        if file_path:
            return ("Read", {"file_path": file_path})
    if prog in ("head", "tail"):
        limit = 10
        file_path = ""
        for i, token in enumerate(parts[1:], start=1):
            if token == "-n" and i + 1 < len(parts):
                try:
                    limit = max(1, int(parts[i + 1]))
                except ValueError:
                    limit = 10
                continue
            if token.startswith("-n") and len(token) > 2:
                try:
                    limit = max(1, int(token[2:]))
                except ValueError:
                    limit = 10
                continue
            if token.startswith("-"):
                continue
            file_path = token
        if file_path:
            return ("Read", {"file_path": file_path, "limit": limit})

    # sed read/edit
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
                return (
                    "Edit",
                    {
                        "file_path": file_path,
                        "old_string": old_string,
                        "new_string": new_string,
                    },
                )
        if "-n" in parts:
            expr_token = ""
            file_path = ""
            for token in parts[1:]:
                if token.startswith("-"):
                    continue
                if not expr_token:
                    expr_token = token
                else:
                    file_path = token
                    break
            m = re.match(r"^(\d+)(?:,(\d+))?p$", expr_token.strip("'\""))
            if m and file_path:
                start = int(m.group(1))
                end = int(m.group(2) or m.group(1))
                return (
                    "Read",
                    {
                        "file_path": file_path,
                        "offset": start,
                        "limit": max(1, end - start + 1),
                    },
                )

    # write-like shell commands
    if prog == "tee":
        file_path = _extract_last_non_flag(parts[1:])
        if file_path:
            return ("Write", {"file_path": file_path})
    redirect_target = _extract_redirect_target(command)
    if redirect_target and prog in {"cat", "echo", "printf"}:
        return ("Write", {"file_path": redirect_target})

    # git inspect
    if prog == "git" and len(parts) > 1:
        sub = parts[1]
        if sub in {"status", "diff", "log", "show", "blame", "rev-parse"}:
            target = _extract_last_non_flag(parts[2:])
            return ("GitInspect", {"subcommand": sub, "target": target, "command": command})

    # test/build runners
    if prog in {"pytest", "tox", "nosetests", "go", "cargo", "npm", "pnpm", "yarn", "bun", "make"}:
        lower = " ".join(parts).lower()
        if " test" in f" {lower} " or prog in {"pytest", "tox", "nosetests"}:
            return ("TestRun", {"command": command})
        if any(k in lower for k in (" build", " typecheck", " compile", " lint")):
            return ("BuildRun", {"command": command})

    # process control
    if prog in {"pgrep", "pkill", "kill", "killall", "ps", "lsof"}:
        return ("ProcessControl", {"action": prog, "command": command})

    # HTTP requests
    if prog in {"curl", "wget", "http", "xh"}:
        method = "GET"
        url = _extract_url(parts[1:])
        for i, token in enumerate(parts):
            if token in {"-X", "--request"} and i + 1 < len(parts):
                method = parts[i + 1].upper()
                break
        if not url:
            # Handle curl URL passed as final positional token.
            candidate = _extract_last_non_flag(parts[1:])
            if candidate.startswith("http://") or candidate.startswith("https://"):
                url = candidate
        return ("HttpCall", {"method": method, "url": url, "command": command})

    # JSON extraction
    if prog == "jq":
        query = ""
        file_path = ""
        for token in parts[1:]:
            if token.startswith("-"):
                continue
            if not query:
                query = token
            else:
                file_path = token
                break
        return ("JsonExtract", {"query": query, "file_path": file_path, "command": command})

    # stats/metrics
    if prog in {"wc", "du", "sort", "uniq"}:
        return ("Stats", {"command": command})

    return None


def _interpret_bash_as_tool(command: str) -> tuple[str, dict] | None:
    """Map common shell read/search/list commands to analogous tool calls."""
    segments = _split_shell_segments(command)
    if len(segments) > 1:
        steps = []
        for seg in segments:
            mapped = _interpret_single_shell_command(seg)
            if mapped:
                mapped_name, mapped_args = mapped
                steps.append({
                    "command": seg,
                    "tool": mapped_name,
                    "summary": _describe_command_map(mapped_name, mapped_args),
                })
            else:
                steps.append({"command": seg, "tool": "Bash", "summary": _trunc(seg, 48)})
        return ("ShellWorkflow", {"command": command, "steps": steps})
    if not segments:
        return None
    return _interpret_single_shell_command(segments[0])


def _parse_result_kv(result: str) -> dict[str, str] | None:
    """Try to parse a result string as key-value pairs (e.g. 'key: value' lines).

    Returns a dict if the result looks like structured key-value output,
    None otherwise.
    """
    if not result:
        return None
    lines = result.strip().splitlines()
    kv: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if ": " in line:
            key, _, val = line.partition(": ")
            key = key.strip().lstrip("- ")
            if key and not key.startswith("(") and len(key) < 40:
                kv[key] = val.strip()
    # Only return if we found a reasonable number of kv pairs
    if len(kv) >= 2:
        return kv
    return None


def _parse_result_json(result: str) -> dict | None:
    """Try to parse a result string as JSON."""
    if not result:
        return None
    try:
        parsed = json.loads(result.strip())
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _parse_spawn_result(result: str) -> list[Section]:
    """Parse spawn_child result into structured sections."""
    if not result:
        return []
    sections: list[Section] = []

    # Split on "--- Result ---" and "--- Error ---" markers
    main_text = result
    result_body = ""
    error_body = ""

    if "--- Result ---" in result:
        parts = result.split("--- Result ---", 1)
        main_text = parts[0]
        rest = parts[1]
        if "--- Error ---" in rest:
            result_parts = rest.split("--- Error ---", 1)
            result_body = result_parts[0].strip()
            error_body = result_parts[1].strip()
        else:
            result_body = rest.strip()
    elif "--- Error ---" in result:
        parts = result.split("--- Error ---", 1)
        main_text = parts[0]
        error_body = parts[1].strip()

    # Parse the header kv pairs from main_text
    kv = _parse_result_kv(main_text)
    if kv:
        sections.append(Section(kind="kv", title="Details", content=kv))
    elif main_text.strip():
        # First line is usually the status like "Child completed (success=True)."
        first_line = main_text.strip().splitlines()[0]
        if first_line:
            sections.append(Section(kind="plain", title="Status", content=first_line))

    if result_body:
        sections.append(Section(kind="plain", title="Result", content=result_body))
    if error_body:
        sections.append(Section(kind="plain", title="Error", content=error_body))

    return sections


def _parse_wait_message_result(result: str) -> list[Section]:
    """Parse wait_for_message result into structured sections."""
    if not result:
        return []

    # Result format: "Message received:\n  type: ...\n  from: ...\n  correlation_id: ...\n  payload: ..."
    if result.startswith("Message received:"):
        lines = result.splitlines()
        msg_kv: dict[str, str] = {}
        payload_raw = ""
        in_payload = False
        for line in lines[1:]:  # skip "Message received:"
            line = line.strip()
            if in_payload:
                payload_raw += line + "\n"
                continue
            if line.startswith("payload: "):
                payload_raw = line[len("payload: "):]
                in_payload = True
            elif ": " in line:
                key, _, val = line.partition(": ")
                key = key.strip()
                val = val.strip()
                # Shorten agent IDs for readability
                if key in ("from", "correlation_id") and len(val) > 16:
                    val = val[:12] + "..."
                msg_kv[key.strip()] = val

        sections: list[Section] = []

        # Determine message type for smarter display
        msg_type = msg_kv.get("type", "").lower()

        # Show the message metadata as a compact header kv
        if msg_kv:
            sections.append(Section(kind="kv", title="Message", content=msg_kv))

        # Try to parse payload as JSON for structured display
        payload_raw = payload_raw.strip()
        if payload_raw:
            try:
                parsed = json.loads(payload_raw)
                if isinstance(parsed, dict):
                    # TASK_RESULT payloads — show summary prominently in a result_block
                    if "summary" in parsed:
                        sections.append(Section(
                            kind="result_block",
                            title="Child Result",
                            content={
                                "text": str(parsed["summary"]),
                                "status": "success",
                            },
                        ))
                        history = parsed.get("history")
                        if isinstance(history, list) and history:
                            transcript_entries = []
                            for entry in history:
                                if not isinstance(entry, dict):
                                    continue
                                entry_type = str(entry.get("type", "")).lower()
                                tool = entry.get("tool_name", entry.get("tool"))
                                text = str(entry.get("content", ""))
                                if entry_type == "user_message":
                                    role = "user"
                                elif entry_type in ("text", "thinking"):
                                    role = "assistant"
                                elif entry_type in ("tool_call", "tool_result"):
                                    role = "tool"
                                    if entry_type == "tool_call":
                                        text = str(entry.get("tool_args", text))
                                else:
                                    role = entry_type or "unknown"
                                transcript_entries.append({
                                    "role": role,
                                    "text": text,
                                    "tool": tool,
                                })
                            if transcript_entries:
                                sections.append(Section(
                                    kind="kv",
                                    title="History",
                                    content={
                                        "entries": str(len(transcript_entries)),
                                        "detail": str(parsed.get("history_detail", "full")),
                                    },
                                ))
                                sections.append(Section(
                                    kind="transcript",
                                    title="Child Conversation",
                                    content=transcript_entries,
                                ))
                        artifacts = parsed.get("artifacts")
                        if artifacts and isinstance(artifacts, dict):
                            artifact_kv = {
                                k: _trunc(str(v), 200) for k, v in artifacts.items()
                            }
                            sections.append(Section(
                                kind="kv", title="Artifacts", content=artifact_kv,
                            ))
                        # Show any remaining keys (excluding summary/artifacts)
                        extra_kv: dict[str, str] = {}
                        for pk, pv in parsed.items():
                            if pk not in ("summary", "artifacts", "history", "history_detail"):
                                extra_kv[pk] = _trunc(str(pv), 200)
                        if extra_kv:
                            sections.append(Section(
                                kind="kv", title="Details", content=extra_kv,
                            ))
                    # progress_update payloads — visual progress bar
                    elif msg_type == "progress_update" and "status" in parsed:
                        pct = parsed.get("percent_complete", 0)
                        status_text = parsed.get("status", "")
                        sections.append(Section(
                            kind="progress",
                            content={
                                "percent": int(pct) if pct else 0,
                                "status": status_text,
                            },
                        ))
                    # QUESTION payloads — show question in a result_block with "question" status
                    elif "question" in parsed:
                        sections.append(Section(
                            kind="result_block",
                            title="Child Question",
                            content={
                                "text": str(parsed["question"]),
                                "status": "question",
                            },
                        ))
                        # Show correlation context if present
                        q_meta: dict[str, str] = {}
                        for pk, pv in parsed.items():
                            if pk != "question":
                                q_meta[pk] = _trunc(str(pv), 100)
                        if q_meta:
                            sections.append(Section(kind="kv", title="Context", content=q_meta))
                    else:
                        # Generic dict payload — show as kv
                        payload_kv = {
                            k: _trunc(str(v), 200) for k, v in parsed.items()
                        }
                        sections.append(Section(
                            kind="kv", title="Payload", content=payload_kv,
                        ))
                else:
                    sections.append(Section(
                        kind="plain", title="Payload", content=str(parsed),
                    ))
            except (json.JSONDecodeError, TypeError):
                sections.append(Section(
                    kind="plain", title="Payload", content=payload_raw,
                ))
        return sections

    elif result.startswith(("No messages within", "No messages received within")):
        return [Section(kind="plain", title="Result", content=result)]

    return [Section(kind="plain", title="Result", content=result)]


def _parse_children_status_result(result: str) -> list[Section]:
    """Parse get_children_status result into a checklist."""
    if not result:
        return []
    # Format: "Children status: 2 completed, 0 failed, 1 running (total: 3)\n\n- child_id: state=..."
    lines = result.strip().splitlines()
    sections: list[Section] = []

    if lines:
        # First line is the summary
        sections.append(Section(kind="plain", title="Summary", content=lines[0]))

    items = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("- "):
            # "- agent_id: state=completed"
            entry = line[2:]
            done = "completed" in entry.lower()
            items.append({"text": entry, "done": done})
    if items:
        sections.append(Section(kind="checklist", title="Children", content=items))

    return sections


def _parse_check_status_result(result: str) -> list[Section]:
    """Parse check_child_status JSON result into kv sections."""
    if not result:
        return []
    parsed = _parse_result_json(result)
    if parsed:
        # Show important fields first, skip verbose ones
        kv: dict[str, str] = {}
        for key in ["agent_id", "state", "role", "model", "depth",
                     "children_count", "created_at", "completed_at", "error"]:
            if key in parsed and parsed[key] is not None:
                val = str(parsed[key])
                if key == "agent_id":
                    val = val[:16] + "..." if len(val) > 16 else val
                kv[key] = val
        sections: list[Section] = [Section(kind="kv", title="Agent Status", content=kv)]
        if parsed.get("prompt_preview"):
            sections.append(Section(kind="plain", title="Prompt", content=parsed["prompt_preview"]))
        if parsed.get("children_ids"):
            items = [{"text": cid, "done": False} for cid in parsed["children_ids"]]
            sections.append(Section(kind="checklist", title="Children", content=items))
        return sections
    return [Section(kind="plain", title="Status", content=result)]


def _parse_expert_result(result: str) -> list[Section]:
    """Parse consult_expert result into structured sections."""
    if not result:
        return []
    # Format: "Expert 'Name' responded (success=True, duration=3.2s):\n\nResponse text..."
    sections: list[Section] = []
    if result.startswith("Expert "):
        lines = result.split("\n", 2)
        # First line is the header
        sections.append(Section(kind="plain", title="Status", content=lines[0]))
        if len(lines) > 1:
            body = "\n".join(lines[1:]).strip()
            if body:
                sections.append(Section(kind="plain", title="Response", content=body))
        return sections
    return [Section(kind="plain", title="Expert Response", content=result)]


def _parse_peer_result(result: str) -> list[Section]:
    """Parse consult_peer result into structured sections."""
    if not result:
        return []
    sections: list[Section] = []

    # Format: "Peer response (provider=..., model=...):\n\nResponse text\n\nthread_id: ...\n\nOther available peers: ..."
    if result.startswith("Peer response"):
        lines = result.split("\n", 2)
        header = lines[0]
        sections.append(Section(kind="plain", title="Status", content=header))
        if len(lines) > 1:
            body = "\n".join(lines[1:]).strip()
            # Extract thread_id if present
            thread_part = ""
            peers_part = ""
            main_body = body
            if "\nthread_id: " in body:
                parts = body.split("\nthread_id: ", 1)
                main_body = parts[0].strip()
                rest = parts[1]
                if "\nOther available peers: " in rest:
                    tid, _, peers_part = rest.partition("\nOther available peers: ")
                    thread_part = tid.strip()
                    peers_part = peers_part.strip()
                else:
                    thread_part = rest.split("\n")[0].strip()

            if main_body:
                sections.append(Section(kind="plain", title="Response", content=main_body))
            kv: dict[str, str] = {}
            if thread_part:
                kv["thread_id"] = thread_part
            if peers_part:
                kv["other_peers"] = peers_part
            if kv:
                sections.append(Section(kind="kv", title="Info", content=kv))
        return sections

    return [Section(kind="plain", title="Peer Response", content=result)]


def _parse_recommend_result(result: str) -> list[Section]:
    """Parse recommend_model result into structured sections."""
    if not result:
        return []
    # Try kv parsing first (the result is mostly key: value lines)
    kv = _parse_result_kv(result)
    if kv:
        sections: list[Section] = [Section(kind="kv", title="Recommendation", content=kv)]
        # Check for fallback options section
        if "Fallback options" in result:
            fallback_start = result.index("Fallback options")
            fallback_text = result[fallback_start:]
            lines = fallback_text.splitlines()
            items = []
            for line in lines[1:]:
                line = line.strip()
                if line and line[0].isdigit():
                    items.append({"text": line.lstrip("0123456789. "), "done": False})
            if items:
                sections.append(Section(kind="checklist", title="Fallback Options", content=items))
        return sections
    return [Section(kind="plain", title="Recommendation", content=result)]


# ── Per-Tool Formatters ──


@tool_formatter("Edit")
def _format_edit(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    file_path = (
        args.get("file_path")
        or args.get("path")
        or args.get("filePath")
        or ""
    )
    old_string = (
        args.get("old_string")
        or args.get("old")
        or args.get("search")
        or args.get("find")
        or args.get("oldString")
        or ""
    )
    new_string = (
        args.get("new_string")
        or args.get("new")
        or args.get("replace")
        or args.get("replacement")
        or args.get("newString")
        or ""
    )

    sections: list[Section] = []
    if file_path:
        sections.append(Section(kind="path", content=file_path))
    if old_string or new_string:
        sections.append(
            Section(
                kind="diff",
                content={
                    "old_lines": old_string.splitlines() if old_string else [],
                    "new_lines": new_string.splitlines() if new_string else [],
                },
            )
        )
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\u270f",
        label="Edit",
        summary=_basename(file_path),
        file_path=file_path,
        sections=sections,
    )


@tool_formatter("Bash")
def _format_bash(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    command = args.get("command", args.get("_raw", ""))
    description = args.get("description", "")

    sections: list[Section] = []
    sections.append(
        Section(
            kind="terminal",
            title="Terminal",
            content={
                "command": command,
                "output": _extract_text_result(result) if result is not None else "",
            },
        )
    )
    if description:
        sections.append(Section(kind="plain", title="Description", content=description))
    mapped = _interpret_bash_as_tool(command)
    if mapped:
        mapped_name, mapped_args = mapped
        mapped_summary = _describe_command_map(mapped_name, mapped_args)
        sections.append(
            Section(kind="plain", title="Interpreted As", content=mapped_summary)
        )
    return FormattedToolCall(
        icon="$",
        label="Bash",
        summary=_trunc(command),
        sections=sections,
    )


@tool_formatter("Read")
def _format_read(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    file_path = args.get("file_path", "")
    offset = args.get("offset")
    limit = args.get("limit")
    line_info = ""
    if offset or limit:
        parts = []
        if offset:
            parts.append(f"L{offset}")
        if limit:
            parts.append(f"+{limit}")
        line_info = f" ({':'.join(parts)})"

    sections: list[Section] = []
    if file_path:
        sections.append(Section(kind="path", content=file_path))
    display_text = _extract_text_result(result)
    if display_text:
        sections.append(
            Section(
                kind="code",
                title="Content",
                content={"language": "text", "text": display_text},
            )
        )

    return FormattedToolCall(
        icon="\U0001f4c4",
        label="Read",
        summary=_basename(file_path) + line_info,
        file_path=file_path,
        sections=sections,
    )


@tool_formatter("Write")
def _format_write(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    file_path = args.get("file_path", "")

    sections: list[Section] = []
    if file_path:
        sections.append(Section(kind="path", content=file_path))
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\U0001f4dd",
        label="Write",
        summary=_basename(file_path),
        file_path=file_path,
        sections=sections,
    )


@tool_formatter("GitInspect")
def _format_git_inspect(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    sub = args.get("subcommand", "inspect")
    target = args.get("target", "")
    command = args.get("command", "")
    summary = f"{sub} {_basename(target)}".strip()
    sections: list[Section] = []
    if target:
        sections.append(Section(kind="path", content=target))
    if command:
        sections.append(Section(kind="code", title="Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Git Output"))
    return FormattedToolCall(icon="\U0001f330", label="Git", summary=_trunc(summary, 60), file_path=target, sections=sections)


@tool_formatter("TestRun")
def _format_test_run(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    command = args.get("command", "")
    sections: list[Section] = []
    if command:
        sections.append(Section(kind="code", title="Test Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Test Output"))
    return FormattedToolCall(icon="\u2697", label="Test", summary=_trunc(command, 60), sections=sections)


@tool_formatter("BuildRun")
def _format_build_run(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    command = args.get("command", "")
    sections: list[Section] = []
    if command:
        sections.append(Section(kind="code", title="Build Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Build Output"))
    return FormattedToolCall(icon="\U0001f6e0", label="Build", summary=_trunc(command, 60), sections=sections)


@tool_formatter("ProcessControl")
def _format_process_control(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    action = args.get("action", "process")
    command = args.get("command", "")
    sections: list[Section] = []
    if command:
        sections.append(Section(kind="code", title="Process Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Process Output"))
    return FormattedToolCall(icon="\u2699", label="Process", summary=f"{action}: {_trunc(command, 44)}", sections=sections)


@tool_formatter("HttpCall")
def _format_http_call(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    method = args.get("method", "GET")
    url = args.get("url", "")
    command = args.get("command", "")
    sections: list[Section] = []
    kv = {"method": method}
    if url:
        kv["url"] = url
    sections.append(Section(kind="kv", title="Request", content=kv))
    if command:
        sections.append(Section(kind="code", title="Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Response"))
    return FormattedToolCall(icon="\U0001f310", label="HTTP", summary=f"{method} {_trunc(url, 44)}".strip(), sections=sections)


@tool_formatter("JsonExtract")
def _format_json_extract(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    query = args.get("query", "")
    file_path = args.get("file_path", "")
    command = args.get("command", "")
    sections: list[Section] = []
    kv = {}
    if query:
        kv["query"] = query
    if file_path:
        kv["file"] = file_path
    if kv:
        sections.append(Section(kind="kv", title="JSON Query", content=kv))
    if command:
        sections.append(Section(kind="code", title="Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="JSON Output"))
    return FormattedToolCall(icon="\U0001f9fe", label="JSON", summary=_trunc(f"{query} {file_path}".strip(), 60), file_path=file_path, sections=sections)


@tool_formatter("Stats")
def _format_stats(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    command = args.get("command", "")
    sections: list[Section] = []
    if command:
        sections.append(Section(kind="code", title="Stats Command", content={"language": "bash", "text": command}))
    sections.extend(_result_section(result, title="Stats Output"))
    return FormattedToolCall(icon="\U0001f4ca", label="Stats", summary=_trunc(command, 60), sections=sections)


@tool_formatter("ShellWorkflow")
def _format_shell_workflow(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    command = args.get("command", "")
    steps = args.get("steps", [])
    sections: list[Section] = []
    if command:
        sections.append(Section(kind="code", title="Pipeline", content={"language": "bash", "text": command}))
    if isinstance(steps, list) and steps:
        items = []
        for step in steps[:12]:
            if isinstance(step, dict):
                label = step.get("summary") or step.get("tool") or step.get("command") or ""
                if label:
                    items.append({"text": label, "done": False})
        if items:
            sections.append(Section(kind="checklist", title="Steps", content=items))
    sections.extend(_result_section(result, title="Output"))
    return FormattedToolCall(icon="\U0001f9f0", label="Shell Workflow", summary=f"{len(steps) if isinstance(steps, list) else 0} steps", sections=sections)


@tool_formatter("Glob")
def _format_glob(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    pattern = args.get("pattern", "")
    path = args.get("path", "")

    sections: list[Section] = []
    if path:
        sections.append(Section(kind="path", content=path))
    sections.extend(_result_section(result, title="Matches"))

    summary = pattern
    if path:
        summary = f"{pattern} in {_basename(path)}"

    return FormattedToolCall(
        icon="\U0001f50d",
        label="Glob",
        summary=summary,
        file_path=path,
        sections=sections,
    )


@tool_formatter("Grep")
def _format_grep(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    pattern = args.get("pattern", "")
    glob_filter = args.get("glob", "")
    path = args.get("path", "")

    scope = glob_filter or _basename(path) or ""
    summary = f'"{_trunc(pattern, 30)}"'
    if scope:
        summary += f" in {scope}"

    sections: list[Section] = []
    sections.extend(_result_section(result, title="Results"))

    return FormattedToolCall(
        icon="\U0001f50e",
        label="Grep",
        summary=summary,
        file_path=path,
        sections=sections,
    )


@tool_formatter("Task")
def _format_task(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    description = args.get("description", "")
    subagent_type = args.get("subagent_type", "")
    prompt = args.get("prompt", "")

    kv: dict[str, str] = {}
    if subagent_type:
        kv["type"] = subagent_type
    if description:
        kv["description"] = description
    if prompt:
        kv["prompt"] = _trunc(prompt, 100)

    sections: list[Section] = []
    if kv:
        sections.append(Section(kind="kv", content=kv))
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\U0001f500",
        label="Task",
        summary=_trunc(description or prompt, 50),
        sections=sections,
    )


@tool_formatter("TodoWrite")
def _format_todo_write(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    todos = args.get("todos", [])
    items = []
    if isinstance(todos, list):
        for t in todos:
            if isinstance(t, dict):
                text = t.get("content", t.get("text", str(t)))
                done = t.get("status") == "completed"
                items.append({"text": text, "done": done})

    sections: list[Section] = []
    if items:
        sections.append(Section(kind="checklist", content=items))
    sections.extend(_result_section(result))

    count = len(items)
    done_count = sum(1 for i in items if i["done"])
    summary = f"{done_count}/{count} done" if count else "empty"

    return FormattedToolCall(
        icon="\u2611",
        label="TodoWrite",
        summary=summary,
        sections=sections,
    )


@tool_formatter("WebFetch")
def _format_web_fetch(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    url = args.get("url", "")
    prompt = args.get("prompt", "")

    domain = ""
    if url:
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = _trunc(url, 40)

    kv: dict[str, str] = {}
    if url:
        kv["url"] = url
    if prompt:
        kv["prompt"] = _trunc(prompt, 80)

    sections: list[Section] = []
    if kv:
        sections.append(Section(kind="kv", content=kv))
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\U0001f310",
        label="WebFetch",
        summary=domain,
        sections=sections,
    )


@tool_formatter("WebSearch")
def _format_web_search(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    query = args.get("query", "")

    sections: list[Section] = []
    sections.extend(_result_section(result, title="Results"))

    return FormattedToolCall(
        icon="\U0001f50d",
        label="WebSearch",
        summary=_trunc(query, 50),
        sections=sections,
    )


@tool_formatter("NotebookEdit")
def _format_notebook_edit(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    notebook_path = args.get("notebook_path", "")
    edit_mode = args.get("edit_mode", "replace")

    sections: list[Section] = []
    if notebook_path:
        sections.append(Section(kind="path", content=notebook_path))
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\U0001f4d3",
        label="NotebookEdit",
        summary=f"{edit_mode} in {_basename(notebook_path)}",
        file_path=notebook_path,
        sections=sections,
    )


@tool_formatter("Skill")
def _format_skill(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    skill = args.get("skill", "")
    skill_args = args.get("args", "")

    summary = skill
    if skill_args:
        summary += f" {_trunc(skill_args, 40)}"

    sections: list[Section] = []
    kv: dict[str, str] = {"skill": skill}
    if skill_args:
        kv["args"] = skill_args
    sections.append(Section(kind="kv", content=kv))
    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\u26a1",
        label="Skill",
        summary=summary,
        sections=sections,
    )


# ── Orchestrator Tool Formatters ──
# These handle mcp__orchestrator__* tools, registered by their bare names.
# The dispatcher strips the mcp__orchestrator__ prefix before lookup.


@tool_formatter("task_complete")
def _format_task_complete(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    summary_text = args.get("summary", "")
    artifacts = args.get("artifacts", {})

    sections: list[Section] = []

    # Show the summary as a prominent result block — this is the agent's final output
    if summary_text:
        sections.append(Section(
            kind="result_block",
            title="Agent Summary",
            content={
                "text": summary_text,
                "status": "success" if success else "error",
                "markdown": True,
            },
        ))

    # Show artifacts as kv with file paths clickable
    if artifacts and isinstance(artifacts, dict):
        artifact_kv = {k: _trunc(str(v), 200) for k, v in artifacts.items()}
        sections.append(Section(kind="kv", title="Artifacts", content=artifact_kv))

    # Result is typically just "Task marked complete. Session will end." — skip that
    if result and result != "Task marked complete. Session will end.":
        sections.extend(_result_section(result, title="Response"))

    return FormattedToolCall(
        icon="✅",
        label="Task Complete",
        summary=_trunc(summary_text, 60),
        sections=sections,
    )


@tool_formatter("spawn_child")
def _format_spawn_child(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    prompt = args.get("prompt", "")
    model = args.get("model", "")
    complexity = args.get("complexity", "")
    wait = args.get("wait", False)
    cwd = args.get("cwd", "")
    tools = args.get("tools", [])
    mcp_servers = args.get("mcp_servers")
    exclude_plugins = args.get("exclude_plugins")

    sections: list[Section] = []

    # Show the prompt as a prominent agent_prompt section
    if prompt:
        sections.append(Section(
            kind="agent_prompt",
            content={
                "number": None,
                "prompt": prompt,
                "model": model,
                "complexity": complexity,
            },
        ))

    # Configuration details (only non-obvious ones)
    kv: dict[str, str] = {}
    if wait:
        kv["mode"] = "blocking (wait=true)"
    if cwd:
        kv["cwd"] = cwd
    if tools and isinstance(tools, list):
        kv["tools"] = ", ".join(str(t) for t in tools[:6])
        if len(tools) > 6:
            kv["tools"] += f" (+{len(tools) - 6} more)"
    if mcp_servers and isinstance(mcp_servers, dict):
        kv["mcp_servers"] = ", ".join(mcp_servers.keys())
    if exclude_plugins and isinstance(exclude_plugins, list):
        kv["excluded"] = ", ".join(str(p) for p in exclude_plugins)
    if kv:
        sections.append(Section(kind="kv", title="Config", content=kv))

    # Parse structured result (child_id, duration, result body)
    sections.extend(_parse_spawn_result(result))

    # Build collapsed summary: show model/complexity badge + short prompt
    badge = ""
    if model:
        badge = f"[{model}] "
    elif complexity:
        badge = f"[{complexity}] "

    # Add status indicator from result
    status_prefix = ""
    if result:
        if "Child completed" in result:
            status_prefix = "✅ " if "success=True" in result else "❌ "
        elif "Child agent spawned" in result:
            status_prefix = ""
    summary = f"{status_prefix}{badge}{_trunc(prompt, 50)}"

    return FormattedToolCall(
        icon="🚀",
        label="Spawn Agent",
        summary=summary,
        sections=sections,
    )


@tool_formatter("spawn_children_parallel")
def _format_spawn_children_parallel(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    children = args.get("children", [])
    count = len(children) if isinstance(children, list) else 0

    sections: list[Section] = []

    # Show each child as a numbered agent_prompt section
    if isinstance(children, list):
        for i, child in enumerate(children[:10]):
            if isinstance(child, dict):
                prompt = child.get("prompt", "")
                model = child.get("model", "")
                complexity = child.get("complexity", "")
                sections.append(Section(
                    kind="agent_prompt",
                    content={
                        "number": i + 1,
                        "prompt": prompt,
                        "model": model,
                        "complexity": complexity,
                    },
                ))
        if count > 10:
            sections.append(Section(
                kind="plain",
                content=f"... and {count - 10} more agents",
            ))

    # Parse result: extract spawned child IDs
    result_child_count = 0
    if result:
        result_lines = result.strip().splitlines()
        child_ids: list[str] = []
        for line in result_lines:
            line = line.strip()
            if line.startswith("- child_id: "):
                cid = line[len("- child_id: "):]
                child_ids.append(cid[:16] + "..." if len(cid) > 16 else cid)
        result_child_count = len(child_ids)
        if result_child_count == 0:
            for line in result_lines:
                line = line.strip()
                if line.startswith("Spawned ") and " children" in line:
                    try:
                        result_child_count = int(line.split(" ", 2)[1])
                    except (ValueError, IndexError):
                        result_child_count = 0
                    break
        if child_ids:
            id_kv = {f"child {i+1}": cid for i, cid in enumerate(child_ids)}
            sections.append(Section(kind="kv", title="Spawned IDs", content=id_kv))
        # Show any spawn errors
        if "Spawn errors" in result:
            err_start = result.index("Spawn errors")
            sections.append(Section(kind="plain", title="Errors", content=result[err_start:]))

    count = max(count, result_child_count)

    return FormattedToolCall(
        icon="🚀",
        label="Spawn Parallel",
        summary=f"{count} agents",
        sections=sections,
    )


@tool_formatter("restart_child")
def _format_restart_child(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    child_id = args.get("child_agent_id", "")
    prompt = args.get("prompt", "")
    wait = args.get("wait", False)

    short_id = (child_id[:12] + "...") if len(child_id) > 12 else child_id

    sections: list[Section] = []

    # Show agent being restarted
    kv: dict[str, str] = {}
    if short_id:
        kv["agent"] = short_id
    if wait:
        kv["mode"] = "blocking (wait=true)"
    if kv:
        sections.append(Section(kind="kv", title="Target", content=kv))

    # Show the new prompt prominently
    if prompt:
        sections.append(Section(
            kind="agent_prompt",
            content={
                "number": None,
                "prompt": prompt,
                "model": "",
                "complexity": "",
            },
        ))

    # Parse structured result (same format as spawn_child)
    sections.extend(_parse_spawn_result(result))

    summary = f"→ {short_id}" if short_id else _trunc(prompt, 50)

    return FormattedToolCall(
        icon="🔄",
        label="Restart Agent",
        summary=summary,
        sections=sections,
    )


@tool_formatter("ask_parent")
def _format_ask_parent(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    question = args.get("question", "")

    sections: list[Section] = []
    if question:
        sections.append(Section(kind="plain", title="❓ Question", content=question))
    # Result is the parent's answer text
    if result:
        sections.append(Section(kind="plain", title="💬 Answer", content=result))

    # Show whether answered in collapsed summary
    summary = _trunc(question, 50)
    icon = "❓"
    if result:
        icon = "💬"
        summary = f"answered — {summary}"

    return FormattedToolCall(
        icon=icon,
        label="Ask Parent",
        summary=summary,
        sections=sections,
    )


@tool_formatter("ask_user")
def _format_ask_user(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    question = str(args.get("question", "") or "").strip()
    options = args.get("options", [])

    # Support SDK AskUserQuestion shape:
    # {questions: [{question, options: [{label, description}]}]}
    if not question:
        raw_questions = args.get("questions")
        if isinstance(raw_questions, list) and raw_questions:
            first = raw_questions[0] if isinstance(raw_questions[0], dict) else {}
            question = str(first.get("question", "") or "").strip()
            if not isinstance(options, list) or not options:
                options = first.get("options", [])

    sections: list[Section] = []
    if question:
        sections.append(Section(kind="plain", title="Question", content=question))
    if options and isinstance(options, list):
        items = []
        for opt in options[:6]:
            if isinstance(opt, dict):
                label = opt.get("label", "")
                desc = opt.get("description", "")
                text = f"{label} — {desc}" if desc else label
                items.append({"text": text, "done": False})
        if items:
            sections.append(Section(kind="checklist", title="Options", content=items))
    # Result is "User responded: <answer>"
    if result:
        answer = result
        if answer.startswith("User responded: "):
            answer = answer[len("User responded: "):]
        sections.append(Section(kind="plain", title="User's Answer", content=answer))

    return FormattedToolCall(
        icon="💬",
        label="Ask User",
        summary=_trunc(question, 50),
        sections=sections,
    )


@tool_formatter("wait_for_message")
def _format_wait_for_message(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    timeout = args.get("timeout_seconds", "")

    sections: list[Section] = []

    # Parse structured message result (type, from, correlation_id, payload)
    parsed_sections = _parse_wait_message_result(result)
    if parsed_sections:
        sections.extend(parsed_sections)
    else:
        sections.extend(_result_section(result))

    # Build a smart summary from the result — include message type AND source agent
    summary = "waiting..."
    icon = "⏳"
    if result:
        if result.startswith("Message received:"):
            msg_type = ""
            from_id = ""
            # Also try to extract a short preview of the payload summary
            payload_preview = ""
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("type: "):
                    msg_type = line[len("type: "):].strip()
                elif line.startswith("from: "):
                    from_id = line[len("from: "):].strip()
                    if len(from_id) > 12:
                        from_id = from_id[:8] + "…"

            # Try to extract a short payload preview for task_result summaries
            if msg_type.lower() == "task_result":
                try:
                    payload_start = result.find("payload: ")
                    if payload_start >= 0:
                        payload_str = result[payload_start + 9:]
                        payload_json = json.loads(payload_str)
                        if isinstance(payload_json, dict) and "summary" in payload_json:
                            payload_preview = _trunc(str(payload_json["summary"]), 50)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            # Use different icons for different message types
            type_lower = msg_type.lower()
            if type_lower == "task_result":
                icon = "✅"
                if payload_preview:
                    summary = f"{from_id} — {payload_preview}" if from_id else payload_preview
                else:
                    summary = f"result from {from_id}" if from_id else "task_result"
            elif type_lower == "question":
                icon = "❓"
                summary = f"question from {from_id}" if from_id else "question"
            elif type_lower == "progress_update":
                icon = "📊"
                summary = f"progress from {from_id}" if from_id else "progress_update"
            else:
                summary = f"{type_lower}"
                if from_id:
                    summary += f" from {from_id}"
        elif result.startswith("No messages"):
            icon = "⌛"
            summary = "timed out"

    return FormattedToolCall(
        icon=icon,
        label="Wait for Message",
        summary=summary,
        sections=sections,
    )


@tool_formatter("respond_to_child")
def _format_respond_to_child(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    child_id = args.get("child_agent_id", "")
    correlation_id = args.get("correlation_id", "")
    response = args.get("response", "")

    short_id = (child_id[:12] + "...") if len(child_id) > 12 else child_id

    sections: list[Section] = []

    # Show target agent
    kv: dict[str, str] = {}
    if short_id:
        kv["to"] = short_id
    if correlation_id:
        corr_short = (correlation_id[:12] + "...") if len(correlation_id) > 12 else correlation_id
        kv["correlation"] = corr_short
    if kv:
        sections.append(Section(kind="kv", title="Target", content=kv))

    # Show the response content prominently
    if response:
        sections.append(Section(kind="plain", title="Response", content=response))

    # Result is typically "Response delivered to child abc123..."
    if result and not result.startswith("Response delivered"):
        sections.extend(_result_section(result, title="Status"))

    return FormattedToolCall(
        icon="↩️",
        label="Reply to Agent",
        summary=f"→ {short_id}" if short_id else _trunc(response, 40),
        sections=sections,
    )


@tool_formatter("consult_expert")
def _format_consult_expert(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    expert_id = args.get("expert_id", "")
    question = args.get("question", "")

    sections: list[Section] = []
    if question:
        sections.append(Section(kind="plain", title="Question", content=question))
    # Parse structured expert response (status header + response body)
    sections.extend(_parse_expert_result(result))

    return FormattedToolCall(
        icon="🎓",
        label="Consult Expert",
        summary=f"{expert_id}: {_trunc(question, 40)}" if expert_id else _trunc(question, 50),
        sections=sections,
    )


@tool_formatter("consult_peer")
def _format_consult_peer(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    peer = args.get("peer", "")
    question = args.get("question", "")

    sections: list[Section] = []
    if question:
        sections.append(Section(kind="plain", title="Question", content=question))
    # Parse structured peer response (header + body + thread_id)
    sections.extend(_parse_peer_result(result))

    return FormattedToolCall(
        icon="🤝",
        label="Consult Peer",
        summary=f"{peer}: {_trunc(question, 40)}" if peer else _trunc(question, 50),
        sections=sections,
    )


@tool_formatter("report_progress")
def _format_report_progress(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    status = args.get("status", "")
    percent = args.get("percent_complete", 0)

    sections: list[Section] = []

    # Show a visual progress bar section
    if percent or status:
        sections.append(Section(
            kind="progress",
            content={"percent": int(percent) if percent else 0, "status": status},
        ))

    # Result is typically "Progress reported: X% - status" — skip if redundant
    if result and not result.startswith("Progress reported:"):
        sections.extend(_result_section(result))

    # Collapsed summary shows percent + status
    summary = status
    if percent:
        summary = f"{percent}% — {_trunc(status, 40)}"

    return FormattedToolCall(
        icon="📊",
        label="Progress",
        summary=_trunc(summary, 50),
        sections=sections,
    )


@tool_formatter("get_child_history")
def _format_get_child_history(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    child_id = args.get("child_agent_id", "")
    detail = args.get("detail_level", "full")
    short_id = (child_id[:12] + "...") if len(child_id) > 12 else child_id

    kv: dict[str, str] = {}
    if short_id:
        kv["agent"] = short_id
    kv["detail"] = detail

    sections: list[Section] = []
    sections.append(Section(kind="kv", content=kv))

    summary_suffix = ""

    # Result is a JSON transcript — try to parse into a structured transcript view
    if result:
        parsed = _parse_result_json(result)
        if parsed and isinstance(parsed, dict):
            # Show a summary of the history
            turns = parsed.get("turns", [])
            n_turns = len(turns) if isinstance(turns, list) else 0
            history_kv: dict[str, str] = {}
            if "agent_id" in parsed:
                history_kv["agent"] = str(parsed["agent_id"])[:16]
            history_kv["turns"] = str(n_turns)
            if history_kv:
                sections.append(Section(kind="kv", title="History", content=history_kv))

            summary_suffix = f" — {n_turns} turns"

            # Build transcript entries for richer display
            if isinstance(turns, list) and turns:
                transcript_entries = []
                for turn in turns[:20]:
                    if isinstance(turn, dict):
                        role = turn.get("role", "unknown")
                        text = str(turn.get("text", turn.get("content", "")))
                        tool = turn.get("tool", turn.get("tool_name", None))
                        transcript_entries.append({
                            "role": role,
                            "text": _trunc(text, 120),
                            "tool": tool,
                        })
                if transcript_entries:
                    sections.append(Section(
                        kind="transcript",
                        title="Conversation",
                        content=transcript_entries,
                    ))
                if n_turns > 20:
                    sections.append(Section(
                        kind="plain",
                        content=f"... and {n_turns - 20} more turns",
                    ))
        else:
            # Try parsing as an array (alternative format)
            try:
                arr = json.loads(result.strip())
                if isinstance(arr, list):
                    summary_suffix = f" — {len(arr)} entries"
                    transcript_entries = []
                    for entry in arr[:20]:
                        if isinstance(entry, dict):
                            role = entry.get("role", "unknown")
                            text = str(entry.get("content", entry.get("text", "")))
                            tool = entry.get("tool", entry.get("tool_name", None))
                            transcript_entries.append({
                                "role": role,
                                "text": _trunc(text, 120),
                                "tool": tool,
                            })
                    if transcript_entries:
                        sections.append(Section(
                            kind="transcript",
                            title="Transcript",
                            content=transcript_entries,
                        ))
                    if len(arr) > 20:
                        sections.append(Section(
                            kind="plain",
                            content=f"... and {len(arr) - 20} more entries",
                        ))
                else:
                    sections.append(Section(kind="plain", title="History", content=_trunc(result, 500)))
            except (json.JSONDecodeError, TypeError):
                sections.append(Section(kind="plain", title="History", content=_trunc(result, 500)))

    return FormattedToolCall(
        icon="📜",
        label="Agent History",
        summary=short_id + summary_suffix,
        sections=sections,
    )


@tool_formatter("check_child_status")
def _format_check_child_status(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    child_id = args.get("child_agent_id", "")
    short_id = (child_id[:12] + "...") if len(child_id) > 12 else child_id

    sections: list[Section] = []
    # Parse JSON status result into kv display
    sections.extend(_parse_check_status_result(result))

    return FormattedToolCall(
        icon="🔍",
        label="Check Agent",
        summary=short_id,
        sections=sections,
    )


@tool_formatter("send_child_prompt")
def _format_send_child_prompt(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    child_id = args.get("child_agent_id", "")
    prompt = args.get("prompt", "")
    short_id = (child_id[:12] + "...") if len(child_id) > 12 else child_id

    sections: list[Section] = []

    # Show target agent
    if short_id:
        sections.append(Section(kind="kv", title="Target", content={"agent": short_id}))

    # Show the prompt prominently
    if prompt:
        sections.append(Section(
            kind="agent_prompt",
            content={
                "number": None,
                "prompt": prompt,
                "model": "",
                "complexity": "",
            },
        ))

    # "Prompt delivered..." is redundant — only show unexpected results
    if result and not result.startswith("Prompt delivered"):
        sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="📨",
        label="Send to Agent",
        summary=f"→ {short_id}" if short_id else _trunc(prompt, 40),
        sections=sections,
    )


@tool_formatter("get_children_status")
def _format_get_children_status(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    sections: list[Section] = []

    # Parse "X completed, Y failed, Z running" + child list
    parsed_sections = _parse_children_status_result(result)
    if parsed_sections:
        sections.extend(parsed_sections)
    else:
        sections.extend(_result_section(result, title="Children Status"))

    # Extract summary: e.g. "2 completed, 0 failed, 1 running"
    summary = "checking..."
    if result:
        first_line = result.strip().splitlines()[0]
        # Format: "Children status: 2 completed, 0 failed, 1 running (total: 3)"
        if ":" in first_line:
            status_part = first_line.split(":", 1)[1].strip()
            # Remove the "(total: N)" suffix for a cleaner summary
            if "(" in status_part:
                status_part = status_part[:status_part.index("(")].strip().rstrip(",")
            summary = _trunc(status_part, 50)

    return FormattedToolCall(
        icon="👥",
        label="All Agents Status",
        summary=summary,
        sections=sections,
    )


@tool_formatter("recommend_model")
def _format_recommend_model(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    task_desc = args.get("task_description", "")
    complexity = args.get("complexity", "medium")

    kv: dict[str, str] = {}
    if complexity:
        kv["complexity"] = complexity
    if task_desc:
        kv["task"] = _trunc(task_desc, 80)

    sections: list[Section] = []
    if kv:
        sections.append(Section(kind="kv", content=kv))

    # Parse structured recommendation result
    parsed_sections = _parse_recommend_result(result)
    if parsed_sections:
        sections.extend(parsed_sections)
    else:
        sections.extend(_result_section(result, title="Recommendation"))

    # Extract model name for summary if available
    summary = f"[{complexity}] {_trunc(task_desc, 40)}"
    if result:
        for line in result.splitlines():
            line = line.strip()
            if line.startswith("Recommended model:"):
                model_name = line[len("Recommended model:"):].strip()
                summary = model_name
                break

    return FormattedToolCall(
        icon="🧠",
        label="Recommend Model",
        summary=summary,
        sections=sections,
    )


def _format_default(name: str, args: dict, result: str | None, success: bool) -> FormattedToolCall:
    """Fallback formatter for unrecognised tool names."""
    raw = args.get("_raw", "")
    if raw:
        summary = _trunc(raw, 50)
        sections = [Section(kind="plain", content=raw)]
    else:
        # Filter out internal keys, build key-value display
        display_args = {k: _trunc(str(v), 80) for k, v in args.items() if not k.startswith("_")}
        summary = _trunc(", ".join(f"{k}={v}" for k, v in display_args.items()), 50)
        sections = [Section(kind="kv", content=display_args)] if display_args else []

    sections.extend(_result_section(result))

    return FormattedToolCall(
        icon="\U0001f527",
        label=name,
        summary=summary,
        sections=sections,
    )


# ── Rich Markup Renderer (for TUI) ──


def _esc(text: str) -> str:
    """Escape Rich markup characters."""
    return text.replace("[", "\\[")


def render_collapsed_rich(fmt: FormattedToolCall, status: str) -> str:
    """Render a collapsed one-liner as a Rich markup string.

    Args:
        fmt: The formatted tool call IR.
        status: One of "pending", "done", "error".
    """
    status_markup = {
        "pending": "[yellow]\\[pending][/yellow]",
        "done": "[green]done[/green]",
        "error": "[red]error[/red]",
    }.get(status, f"[dim]{_esc(status)}[/dim]")

    icon = fmt.icon
    label = _esc(fmt.label)
    summary = _esc(fmt.summary) if fmt.summary else ""

    parts = [f"[dim]\u25b6[/dim]"]
    if icon:
        parts.append(icon)
    parts.append(f"[cyan]{label}[/cyan]")
    if summary:
        parts.append(f"[dim]{summary}[/dim]")
    parts.append(status_markup)

    return "  ".join(parts)


def render_expanded_rich(
    fmt: FormattedToolCall, status: str, timestamp: str = "",
) -> str:
    """Render the full expanded view as a Rich markup string.

    Args:
        fmt: The formatted tool call IR.
        status: One of "pending", "done", "error".
        timestamp: Optional HH:MM:SS timestamp.
    """
    lines: list[str] = []

    # Header
    status_markup = {
        "pending": "[yellow]\\[pending][/yellow]",
        "done": "[green]done[/green]",
        "error": "[red]error[/red]",
    }.get(status, "")

    icon = fmt.icon
    label = _esc(fmt.label)

    header_parts = ["[dim]\u25bc[/dim]"]
    if timestamp:
        header_parts.append(f"[dim]{timestamp}[/dim]")
    if icon:
        header_parts.append(icon)
    header_parts.append(f"[bold cyan]{label}[/bold cyan]")
    header_parts.append(status_markup)
    lines.append("  ".join(header_parts))

    # Sections
    for section in fmt.sections:
        if section.title:
            lines.append(f"  [bold dim]{_esc(section.title)}[/bold dim]")
        lines.extend(_render_section_rich(section, status))

    return "\n".join(lines)


def _render_section_rich(section: Section, status: str = "") -> list[str]:
    """Render a single section to Rich markup lines."""
    lines: list[str] = []

    if section.kind == "diff":
        content = section.content or {}
        old_lines = content.get("old_lines", [])
        new_lines = content.get("new_lines", [])
        for line in old_lines:
            lines.append(f"  [red]- {_esc(line)}[/red]")
        for line in new_lines:
            lines.append(f"  [green]+ {_esc(line)}[/green]")

    elif section.kind == "code":
        content = section.content or {}
        text = content.get("text", "")
        language = str(content.get("language", "")).lower()
        for code_line in text.splitlines():
            if language == "bash":
                lines.append(f"  [dim]$[/dim] {_esc(code_line)}")
            else:
                lines.append(f"  {_esc(code_line)}")

    elif section.kind == "terminal":
        content = section.content or {}
        command = str(content.get("command", "") or "")
        output = str(content.get("output", "") or "")
        for command_line in command.splitlines() or [""]:
            lines.append(f"  [bold green]$[/bold green] {_esc(command_line)}")
        lines.append("  [dim]────────────────────────────[/dim]")
        if output:
            for output_line in output.splitlines():
                lines.append(f"  {_esc(output_line)}")
        elif status == "pending":
            lines.append("  [yellow]… waiting for output[/yellow]")
        if status == "pending":
            lines.append("  [yellow]▌[/yellow]")

    elif section.kind == "path":
        path = section.content or ""
        lines.append(f"  [underline]{_esc(path)}[/underline]")

    elif section.kind == "checklist":
        items = section.content or []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text", "")
                done = item.get("done", False)
                marker = "[green]\u2713[/green]" if done else "[dim]\u25cb[/dim]"
                lines.append(f"  {marker} {_esc(text)}")

    elif section.kind == "kv":
        kv = section.content or {}
        if isinstance(kv, dict):
            for key, value in kv.items():
                lines.append(f"  [bold]{_esc(key)}:[/bold] {_esc(str(value))}")

    elif section.kind == "progress":
        content = section.content or {}
        percent = content.get("percent", 0)
        status = content.get("status", "")
        # Build a visual progress bar: [████████░░░░] 65%
        bar_width = 20
        filled = round(bar_width * percent / 100)
        empty = bar_width - filled
        bar = "█" * filled + "░" * empty
        lines.append(f"  [green]{bar}[/green] [bold]{percent}%[/bold]")
        if status:
            lines.append(f"  {_esc(status)}")

    elif section.kind == "agent_prompt":
        content = section.content or {}
        number = content.get("number")
        prompt = content.get("prompt", "")
        model = content.get("model", "")
        complexity = content.get("complexity", "")
        # Header with optional number and model badge
        header_parts: list[str] = []
        if number is not None:
            header_parts.append(f"[bold cyan]Agent {number}[/bold cyan]")
        if model:
            header_parts.append(f"[yellow]\\[{_esc(model)}][/yellow]")
        elif complexity:
            header_parts.append(f"[yellow]\\[{_esc(complexity)}][/yellow]")
        if header_parts:
            lines.append(f"  {' '.join(header_parts)}")
        # Show the prompt text indented (with reasonable truncation)
        if prompt:
            prompt_lines = prompt.splitlines()
            for pl in prompt_lines[:15]:
                lines.append(f"    [dim]│[/dim] {_esc(pl)}")
            remaining = len(prompt_lines) - 15
            if remaining > 0:
                lines.append(f"    [dim]│ ... {remaining} more lines[/dim]")

    elif section.kind == "transcript":
        # Render conversation history with role-based icons and styling
        entries = section.content or []
        role_icons = {
            "assistant": "🤖",
            "user": "👤",
            "tool": "🔧",
            "system": "⚙️",
        }
        for entry in entries:
            if isinstance(entry, dict):
                role = entry.get("role", "unknown")
                text = entry.get("text", "")
                tool = entry.get("tool")
                icon = role_icons.get(role, "·")
                # Style differently based on role
                if role == "assistant":
                    role_style = "cyan"
                elif role == "user":
                    role_style = "green"
                elif role == "tool":
                    role_style = "yellow"
                else:
                    role_style = "dim"
                # Build the line
                prefix = f"  {icon} [{role_style}]{_esc(role)}[/{role_style}]"
                if tool:
                    prefix += f" [dim]({_esc(tool)})[/dim]"
                lines.append(prefix)
                # Show text content indented under the role
                if text:
                    for tl in str(text).splitlines():
                        lines.append(f"    [dim]│[/dim] {_esc(tl)}")

    elif section.kind == "result_block":
        # Prominent bordered result block — for agent summaries and final outputs
        content = section.content or {}
        text = content.get("text", "")
        status = content.get("status", "success")
        # When markdown=True, the content will be rendered as proper Markdown
        # at the widget level (e.g. ToolCallWidget uses RichMarkdown).
        # Skip the plain-text rendering here to avoid duplicate display.
        if content.get("markdown"):
            pass
        else:
            # Choose border style based on status
            if status == "success":
                border_color = "green"
                border_char = "┃"
            elif status == "error":
                border_color = "red"
                border_char = "┃"
            elif status == "question":
                border_color = "yellow"
                border_char = "┃"
            else:
                border_color = "dim"
                border_char = "│"
            # Render text lines with a colored left border
            if text:
                for tl in str(text).splitlines()[:30]:
                    lines.append(f"  [{border_color}]{border_char}[/{border_color}] {_esc(tl)}")
                remaining = len(str(text).splitlines()) - 30
                if remaining > 0:
                    lines.append(f"  [{border_color}]{border_char}[/{border_color}] [dim]... {remaining} more lines[/dim]")

    elif section.kind == "plain":
        text = section.content or ""
        for text_line in str(text).splitlines()[:20]:
            lines.append(f"  {_esc(text_line)}")
        remaining = len(str(text).splitlines()) - 20
        if remaining > 0:
            lines.append(f"  [dim]... {remaining} more lines[/dim]")

    return lines
