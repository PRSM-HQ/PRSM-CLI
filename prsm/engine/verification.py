"""Verification pipeline for structured evidence checks."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from .models import VerificationResult


class VerificationPipeline:
    """Runs verification hooks and captures structured results."""

    def __init__(self, artifact_store: Any | None = None) -> None:
        self._artifact_store = artifact_store

    async def run_checks(
        self,
        hooks: list[dict[str, Any]],
        cwd: str = ".",
    ) -> list[VerificationResult]:
        """Execute verification hooks and collect outcomes."""
        results: list[VerificationResult] = []
        for hook in hooks:
            check_type = str(hook.get("type", "custom"))
            command = str(hook.get("command", "")).strip()
            results.append(
                await self._run_single_check(
                    check_type=check_type,
                    command=command,
                    cwd=cwd,
                )
            )
        return results

    async def _run_single_check(
        self,
        check_type: str,
        command: str,
        cwd: str,
    ) -> VerificationResult:
        """Run one check command and persist output if artifact store exists."""
        if not command:
            return VerificationResult(
                check_type=check_type,
                command=command,
                passed=False,
                output_summary="Missing verification command",
            )

        started = time.monotonic()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        duration = time.monotonic() - started
        output = ((stdout or b"") + (stderr or b"")).decode(
            "utf-8", errors="replace"
        )
        summary = output.strip()[:500]
        if not summary:
            summary = "No command output"

        artifact_id: str | None = None
        if self._artifact_store is not None:
            artifact_id = await self._store_output_artifact(
                check_type=check_type,
                command=command,
                output=output,
            )

        return VerificationResult(
            check_type=check_type,
            command=command,
            passed=process.returncode == 0,
            output_summary=summary,
            artifact_id=artifact_id,
            duration_seconds=duration,
        )

    async def _store_output_artifact(
        self,
        check_type: str,
        command: str,
        output: str,
    ) -> str | None:
        """Store full check output via best-effort ArtifactStore adapters."""
        store = self._artifact_store
        metadata = {"check_type": check_type, "command": command}

        if hasattr(store, "store_text"):
            result = store.store_text(output, metadata=metadata)
        elif hasattr(store, "put_text"):
            result = store.put_text(output, metadata=metadata)
        elif hasattr(store, "store"):
            result = store.store(output, metadata=metadata)
        else:
            return None

        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            return None
        return str(result)
