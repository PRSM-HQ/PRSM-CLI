"""Best-effort cleanup for stale runtime subprocesses.

This targets orphaned helper/provider processes that were originally
spawned by PRSM but outlived their parent process.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    args: str


def _list_processes() -> dict[int, ProcessInfo]:
    """Return process table keyed by PID using `ps` output."""
    out = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,args="],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    table: dict[int, ProcessInfo] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = ProcessInfo(pid=pid, ppid=ppid, args=parts[2])
    return table


def _has_prsm_ancestor(
    proc: ProcessInfo,
    table: dict[int, ProcessInfo],
    current_pid: int,
) -> bool:
    """True when current process tree includes a PRSM server ancestor."""
    cur = proc
    hops = 0
    while hops < 32:
        if cur.pid == current_pid:
            return True
        if "prsm --server" in cur.args:
            return True
        if "prsm.engine.mcp_server.stdio_server" in cur.args:
            return True
        parent = table.get(cur.ppid)
        if parent is None:
            return False
        cur = parent
        hops += 1
    return False


def _is_managed_candidate(args: str, include_claude: bool) -> bool:
    """Match processes that PRSM may spawn and can be safely reaped if orphaned."""
    patterns = [
        r"prsm\.engine\.mcp_server\.orch_proxy",
        r"\bcodex\b.*\bmcp-server\b",
        r"\bcodex\b.*\bexec\b",
        r"\bgemini\b.*\b--output-format\b.*\bstream-json\b",
    ]
    if include_claude:
        # Optional: only when explicitly enabled.
        patterns.append(r"\bclaude\b")
    return any(re.search(pat, args) for pat in patterns)


def cleanup_stale_runtime_processes(
    *,
    current_pid: int | None = None,
    include_claude: bool = False,
    log: Callable[[str], None] | None = None,
) -> int:
    """Kill orphaned PRSM-managed subprocesses.

    Process is considered stale only when:
    - it matches a managed helper/provider signature, and
    - it has no PRSM server ancestry, and
    - it is orphaned (parent is PID 1 or parent is missing).
    """
    pid = current_pid or os.getpid()
    logger = log or (lambda _: None)
    table = _list_processes()
    killed = 0

    for proc in table.values():
        if proc.pid == pid:
            continue
        if not _is_managed_candidate(proc.args, include_claude):
            continue

        parent_exists = proc.ppid in table
        is_orphan = (proc.ppid == 1) or (not parent_exists)
        if not is_orphan:
            continue

        if _has_prsm_ancestor(proc, table, pid):
            continue

        try:
            os.kill(proc.pid, signal.SIGTERM)
            killed += 1
            logger(
                f"Reaped stale runtime process pid={proc.pid} "
                f"ppid={proc.ppid} cmd={proc.args[:180]}"
            )
        except ProcessLookupError:
            continue
        except Exception as exc:
            logger(
                f"Failed to reap stale process pid={proc.pid}: "
                f"{type(exc).__name__}: {exc}"
            )

    return killed

