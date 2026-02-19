"""PRSM CLI — main application entry point."""

from __future__ import annotations

import logging
import re
import subprocess
import asyncio
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def _log_runtime_compatibility() -> None:
    """Log SDK/CLI runtime versions and known risk combinations."""
    logger = logging.getLogger(__name__)
    sdk_version = "unknown"
    try:
        from importlib.metadata import version

        sdk_version = version("claude-agent-sdk")
    except Exception:
        logger.debug("Could not resolve claude-agent-sdk version", exc_info=True)

    cli_version = "unknown"
    try:
        out = subprocess.check_output(
            ["claude", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        match = re.search(r"(\d+\.\d+\.\d+)", out)
        if match:
            cli_version = match.group(1)
        else:
            cli_version = out
    except Exception:
        logger.debug("Could not resolve claude CLI version", exc_info=True)

    logger.info(
        "Runtime versions: claude-agent-sdk=%s claude-cli=%s",
        sdk_version,
        cli_version,
    )
    if sdk_version == "0.1.31":
        logger.warning(
            "claude-agent-sdk %s has known intermittent subprocess transport instability. "
            "Retry mitigation is enabled; consider upgrading SDK when available.",
            sdk_version,
        )


async def _run_sdk_preflight_async(timeout_seconds: float = 15.0) -> tuple[bool, str]:
    """Run a minimal SDK query preflight to validate transport/auth health."""
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions
    except Exception as exc:
        return False, f"sdk import failed: {type(exc).__name__}: {exc}"

    options_kwargs = dict(
        system_prompt="Reply with exactly OK.",
        allowed_tools=[],
        permission_mode="bypassPermissions",
        cwd=str(Path.cwd()),
        model=os.getenv("PRSM_MODEL", "claude-opus-4-6"),
    )

    cli_path = os.getenv("PRSM_CLAUDE_CLI_PATH", "claude").strip()
    if cli_path:
        options_kwargs["cli_path"] = cli_path

    options = ClaudeAgentOptions(**options_kwargs)
    result_text = ""

    async def _consume() -> None:
        nonlocal result_text
        async for message in query(prompt="Reply with exactly OK.", options=options):
            if hasattr(message, "result"):
                result_text = (message.result or "").strip()

    try:
        await asyncio.wait_for(_consume(), timeout=timeout_seconds)
    except TimeoutError:
        return False, f"timeout after {timeout_seconds:.1f}s"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if not result_text:
        return False, "empty response"
    return True, result_text


def _run_sdk_preflight() -> tuple[bool, str]:
    timeout = max(5.0, float(os.getenv("PRSM_SDK_PREFLIGHT_TIMEOUT_SECONDS", "15")))
    return asyncio.run(_run_sdk_preflight_async(timeout))


def _prompt_import_depth_cli() -> int | None:
    """Prompt for transcript import depth in CLI mode.

    Returns:
        None for full import, or integer max_turns for recent import.
    """
    if not sys.stdin.isatty():
        print(
            "No interactive terminal detected; defaulting to recent 200 turns. "
            "Pass --import-max-turns to set an explicit value."
        )
        return 200

    print("Choose import depth:")
    print("  1) Recent 200 turns (Recommended)")
    print("  2) Recent 500 turns")
    print("  3) Full transcript")
    while True:
        choice = input("Selection [1]: ").strip() or "1"
        if choice == "1":
            return 200
        if choice == "2":
            return 500
        if choice == "3":
            return None
        print("Please enter 1, 2, or 3.")


def _handle_transcript_import_args(args) -> str | None:
    """Handle transcript import CLI actions.

    Returns:
        Session ID to auto-open in TUI when --import-open is used.
        Returns None when no auto-open is requested.

    Exits the process for handled non-open actions.
    """
    import_flags = [
        bool(getattr(args, "import_list", False)),
        bool(getattr(args, "import_preview", False)),
        bool(getattr(args, "import_run", False)),
    ]
    if not any(import_flags):
        return None
    if sum(import_flags) > 1:
        print(
            "Error: choose exactly one of --import-list, "
            "--import-preview, or --import-run."
        )
        sys.exit(2)

    provider = (args.import_provider or "all").lower().strip()
    if provider not in {"all", "codex", "claude", "gemini"}:
        print(f"Error: unknown --import-provider '{provider}'.")
        sys.exit(2)
    if args.import_max_turns is not None and args.import_max_turns <= 0:
        print("Error: --import-max-turns must be greater than 0.")
        sys.exit(2)

    from datetime import datetime

    from prsm.shared.services.persistence import SessionPersistence
    from prsm.shared.services.transcript_import.service import (
        TranscriptImportService,
    )

    service = TranscriptImportService()

    if args.import_list:
        sessions = service.list_sessions(provider=provider, limit=args.import_limit)
        if not sessions:
            print("No importable sessions found.")
        else:
            print(f"Importable sessions ({len(sessions)}):")
            for item in sessions:
                updated = (
                    item.updated_at.isoformat(timespec="seconds")
                    if item.updated_at
                    else "unknown"
                )
                title = item.title or "(untitled)"
                print(
                    f"  {item.provider}:{item.source_session_id} "
                    f"-- {title} ({item.turn_count} turns, updated {updated})"
                )
        sys.exit(0)

    if args.import_preview:
        if provider == "all":
            print("Error: --import-preview requires --import-provider codex|claude|gemini.")
            sys.exit(2)
        source_id = (args.import_source_id or "").strip()
        if not source_id:
            print("Error: --import-preview requires --import-source-id.")
            sys.exit(2)
        try:
            transcript = service.load_transcript(provider, source_id)
        except FileNotFoundError:
            print(f"Error: import source not found: {provider}:{source_id}")
            sys.exit(1)

        summary = transcript.summary
        print(f"Preview: {summary.provider}:{summary.source_session_id}")
        if summary.title:
            print(f"  title: {summary.title}")
        print(f"  turns: {len(transcript.turns)}")
        if summary.started_at:
            print(f"  started: {summary.started_at.isoformat(timespec='seconds')}")
        if summary.updated_at:
            print(f"  updated: {summary.updated_at.isoformat(timespec='seconds')}")

        preview_turns = transcript.turns[:12]
        for idx, turn in enumerate(preview_turns, start=1):
            text = " ".join(turn.content.split())
            if len(text) > 140:
                text = f"{text[:137]}..."
            tools = f" +{len(turn.tool_calls)} tools" if turn.tool_calls else ""
            print(f"  {idx:02d}. [{turn.role}] {text}{tools}")
        if len(transcript.turns) > len(preview_turns):
            remaining = len(transcript.turns) - len(preview_turns)
            print(f"  ... {remaining} more turns")
        if transcript.warnings:
            print(f"Warnings: {len(transcript.warnings)} parse issue(s).")
        sys.exit(0)

    # --import-run
    if provider == "all":
        print("Error: --import-run requires --import-provider codex|claude|gemini.")
        sys.exit(2)
    source_id = (args.import_source_id or "").strip()
    if not source_id:
        print("Error: --import-run requires --import-source-id.")
        sys.exit(2)

    try:
        max_turns = args.import_max_turns
        if max_turns is None:
            max_turns = _prompt_import_depth_cli()
        result = service.import_to_session(
            provider,
            source_id,
            session_name=args.import_name,
            max_turns=max_turns,
        )
    except FileNotFoundError:
        print(f"Error: import source not found: {provider}:{source_id}")
        sys.exit(1)

    imported_session = result.session
    imported_session.imported_from = (
        TranscriptImportService.session_import_metadata(result)
    )

    persistence = SessionPersistence(cwd=Path.cwd())
    session_name = (
        args.import_name
        or imported_session.name
        or f"imported_{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    imported_session.name = session_name
    save_path = persistence.save(
        imported_session,
        session_name,
        session_id=imported_session.session_id,
    )
    print(
        f"Imported {provider}:{source_id} -> {save_path} "
        f"({result.imported_turns} turns"
        + (f", dropped {result.dropped_turns}" if result.dropped_turns else "")
        + ")"
    )
    warnings = result.metadata.get("warnings", [])
    if warnings:
        print(f"Warnings: {len(warnings)} parse issue(s).")

    if args.import_open:
        return imported_session.session_id
    sys.exit(0)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(
        prog="prsm",
        description="PRSM — Terminal UI for multi-agent orchestration",
    )
    parser.add_argument(
        "--new", action="store_true",
        help="Start a fresh session (skip auto-resume)",
    )
    parser.add_argument(
        "--resume", metavar="NAME",
        help="Resume a specific saved session by name",
    )
    parser.add_argument(
        "--fork", metavar="NAME",
        help="Fork an existing session into a new one",
    )
    parser.add_argument(
        "--fork-snapshot", metavar="SNAPSHOT_ID",
        help="Fork a session from a snapshot without restoring files",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List saved sessions and exit (no TUI)",
    )
    parser.add_argument(
        "--import-list", action="store_true",
        help="List importable provider transcripts and exit",
    )
    parser.add_argument(
        "--import-preview", action="store_true",
        help="Preview a provider transcript by source ID and exit",
    )
    parser.add_argument(
        "--import-run", action="store_true",
        help="Import a provider transcript into a PRSM session",
    )
    parser.add_argument(
        "--import-provider", metavar="PROVIDER", default="all",
        help="Import provider: codex, claude, gemini, or all (list only)",
    )
    parser.add_argument(
        "--import-source-id", metavar="SOURCE_ID",
        help="Provider-native source session ID for preview/run",
    )
    parser.add_argument(
        "--import-name", metavar="NAME",
        help="Optional saved PRSM session name for imported transcript",
    )
    parser.add_argument(
        "--import-max-turns", metavar="N", type=int,
        help="Optionally keep only the most recent N turns during import",
    )
    parser.add_argument(
        "--import-limit", metavar="N", type=int, default=50,
        help="Limit number of sessions shown by --import-list (default: 50)",
    )
    parser.add_argument(
        "--import-open", action="store_true",
        help="After --import-run, open the imported session in TUI",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="Start HTTP+SSE server mode (for VSCode extension)",
    )
    parser.add_argument(
        "--port", type=int, default=0,
        help="Server port (0=random available port)",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="YAML config file for engine, providers, models, experts",
    )
    args = parser.parse_args()

    imported_session_id = _handle_transcript_import_args(args)
    if imported_session_id:
        args.resume = imported_session_id
        args.new = False

    if args.list:
        from prsm.shared.services.persistence import SessionPersistence

        persistence = SessionPersistence(cwd=Path.cwd())
        sessions = persistence.list_sessions()
        if not sessions:
            print("No saved sessions.")
        else:
            for name in sessions:
                print(f"  {name}")
        sys.exit(0)

    if args.server:
        from prsm.vscode.server import PrsmServer
        from prsm.shared.services.process_cleanup import (
            cleanup_stale_runtime_processes,
        )

        # Configure persistent server logging for VSCode startup diagnostics.
        log_level = os.getenv("PRSM_LOG_LEVEL", "INFO").upper()
        log_dir = Path.home() / ".prsm" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "prsm-server.log"

        root = logging.getLogger()
        root.setLevel(getattr(logging, log_level, logging.INFO))
        root.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [pid=%(process)d] %(message)s"
        )
        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
        logging.getLogger(__name__).info(
            "Starting PRSM server mode cwd=%s port=%s config=%s log=%s",
            Path.cwd(),
            args.port,
            args.config or "<none>",
            log_file,
        )
        cleanup_include_claude = os.getenv(
            "PRSM_CLEANUP_STALE_CLAUDE", "0"
        ).lower() in {"1", "true", "yes"}
        try:
            reaped = cleanup_stale_runtime_processes(
                include_claude=cleanup_include_claude,
                log=logging.getLogger(__name__).info,
            )
            if reaped:
                logging.getLogger(__name__).warning(
                    "Reaped %d stale runtime process(es) at startup",
                    reaped,
                )
        except Exception:
            logging.getLogger(__name__).exception(
                "Startup stale-process cleanup failed"
            )
        _log_runtime_compatibility()
        # Disabled by default so startup does not launch Claude before
        # a user explicitly prompts an agent.
        run_preflight = os.getenv("PRSM_SDK_PREFLIGHT", "0").lower() not in {
            "0", "false", "no",
        }
        strict_preflight = os.getenv("PRSM_SDK_PREFLIGHT_STRICT", "0").lower() in {
            "1", "true", "yes",
        }
        preflight_ok: bool | None = None
        preflight_detail: str | None = None
        preflight_checked_at: float | None = None
        if run_preflight:
            preflight_checked_at = time.time()
            ok, detail = _run_sdk_preflight()
            preflight_ok = ok
            preflight_detail = detail
            if ok:
                logging.getLogger(__name__).info(
                    "Claude SDK preflight passed: %s",
                    detail[:120],
                )
            else:
                logging.getLogger(__name__).warning(
                    "Claude SDK preflight failed: %s",
                    detail,
                )
                if strict_preflight:
                    logging.getLogger(__name__).error(
                        "Strict preflight enabled; refusing to start server."
                    )
                    sys.exit(1)

        # Auto-discover .prism/prsm.yaml (preferred) or prsm.yaml (legacy)
        config_path = args.config
        if config_path:
            explicit = Path(config_path)
            logging.getLogger(__name__).info(
                "Using explicit config path: %s (exists=%s)",
                explicit,
                explicit.exists(),
            )
        if not config_path:
            prism_yaml = Path.cwd() / ".prism" / "prsm.yaml"
            legacy_yaml = Path.cwd() / "prsm.yaml"
            prism_exists = prism_yaml.exists()
            legacy_exists = legacy_yaml.exists()
            logging.getLogger(__name__).info(
                "Config auto-discovery candidates: %s (exists=%s), %s (exists=%s)",
                prism_yaml,
                prism_exists,
                legacy_yaml,
                legacy_exists,
            )
            auto_yaml = prism_yaml if prism_exists else legacy_yaml
            if auto_yaml.exists():
                config_path = str(auto_yaml)
                logging.getLogger(__name__).info(
                    "Auto-discovered config: %s", config_path,
                )
            else:
                logging.getLogger(__name__).info(
                    "No config file found (tried %s, %s); using defaults",
                    prism_yaml, legacy_yaml,
                )

        server = PrsmServer(
            port=args.port,
            cwd=str(Path.cwd()),
            config_path=config_path,
            claude_preflight_enabled=run_preflight,
            claude_preflight_ok=preflight_ok,
            claude_preflight_detail=preflight_detail,
            claude_preflight_checked_at=preflight_checked_at,
        )
        asyncio.run(server.start())
        sys.exit(0)

    # TUI mode
    from prsm.tui.app import PrsmApp

    app = PrsmApp()
    app.cli_args = args
    app.run()


if __name__ == "__main__":
    main()
