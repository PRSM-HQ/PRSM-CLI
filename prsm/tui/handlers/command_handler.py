"""Slash-command handler extracted from MainScreen.

Handles all /command dispatching and execution, keeping MainScreen focused
on layout, event consumption, and orchestrator lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from prsm.shared.commands import COMMAND_HELP

if TYPE_CHECKING:
    from prsm.tui.screens.main import MainScreen
    from prsm.tui.widgets.tool_log import ToolLog

logger = logging.getLogger(__name__)


class CommandHandler:
    """Processes slash commands on behalf of MainScreen.

    Keeps a reference to the screen so it can access widgets, session,
    persistence, and plugin state without duplicating any of them.
    """

    def __init__(self, screen: MainScreen) -> None:
        self._screen = screen
        self._import_service = None

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def _tl(self) -> ToolLog:
        from prsm.tui.widgets.tool_log import ToolLog

        return self._screen.query_one("#tool-log", ToolLog)

    # ── public entry point ──────────────────────────────────────────

    def handle_command(self, name: str, args: list[str]) -> bool:
        """Dispatch a slash command.  Returns True if handled."""
        tl = self._tl
        name = name.lower()

        dispatch = {
            "help": lambda: self._cmd_help(tl),
            "new": lambda: self._cmd_new(tl),
            "session": lambda: self._cmd_session(args, tl),
            "sessions": lambda: self._cmd_sessions(tl),
            "fork": lambda: self._cmd_fork(args, tl),
            "save": lambda: self._cmd_save(args, tl),
            "import": lambda: self._cmd_import(args, tl),
            "plugin": lambda: self._cmd_plugin(args, tl),
            "policy": lambda: self._cmd_policy(args, tl),
            "settings": lambda: self._cmd_settings(tl),
            "worktree": lambda: self._cmd_worktree(args, tl),
            "model": lambda: self._cmd_model(args, tl),
        }

        handler = dispatch.get(name)
        if handler:
            handler()
            return True

        tl.write(
            f"[red]Unknown command:[/red] /{name}. "
            "Type /help for available commands."
        )
        return False

    # ── individual commands ─────────────────────────────────────────

    def _cmd_help(self, tl: ToolLog) -> None:
        tl.write("[bold]Available commands:[/bold]")
        for cmd, desc in COMMAND_HELP.items():
            tl.write(f"  [cyan]/{cmd}[/cyan] -- {desc}")

    def _cmd_new(self, tl: ToolLog) -> None:
        s = self._screen
        if s.session.message_count > 0:
            s.save_session()
            tl.write("[dim]Previous session saved[/dim]")

        from prsm.shared.models.session import Session
        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView

        s.session = Session()
        conv = s.query_one("#conversation", ConversationView)
        conv.session = s.session
        container = conv.query_one("#message-container")
        container.remove_children()
        tree = s.query_one("#agent-tree", AgentTree)
        tree.clear_agents()

        s._start_fresh_session(tl)
        tl.write("[green]New session started[/green]")

    def _cmd_session(self, args: list[str], tl: ToolLog) -> None:
        s = self._screen
        if not args:
            tl.write("[red]Usage:[/red] /session SESSION_UUID")
            return
        session_id = args[0]

        if s.session.message_count > 0:
            s.save_session()
            tl.write("[dim]Current session auto-saved[/dim]")

        try:
            from prsm.tui.widgets.agent_tree import AgentTree
            from prsm.tui.widgets.conversation import ConversationView

            loaded = s._persistence.load(session_id)
            s.session = loaded
            conv = s.query_one("#conversation", ConversationView)
            conv.session = s.session
            conv.clear()
            tree = s.query_one("#agent-tree", AgentTree)
            tree.clear_agents()
            s._restore_session(loaded)
            tl.write(f"[green]Loaded[/green] session '{session_id}'")
        except FileNotFoundError:
            tl.write(f"[red]Session '{session_id}' not found[/red]")

    def _cmd_sessions(self, tl: ToolLog) -> None:
        sessions = self._screen._persistence.list_sessions_detailed(
            all_worktrees=True
        )
        if not sessions:
            tl.write("[dim]No saved sessions[/dim]")
            return
        tl.write(f"[bold]Saved sessions ({len(sessions)} across worktrees):[/bold]")
        for info in sessions:
            forked = (
                f" (forked from {info['forked_from']})"
                if info.get("forked_from")
                else ""
            )
            tl.write(
                f"  [cyan]{info['name']}[/cyan] -- "
                f"id={info.get('session_id', '?')}, "
                f"{info.get('agent_count', '?')} agents, "
                f"{info.get('message_count', '?')} msgs{forked}"
            )

    def _cmd_fork(self, args: list[str], tl: ToolLog) -> None:
        s = self._screen
        if s.session.message_count == 0:
            tl.write("[yellow]Nothing to fork -- session is empty[/yellow]")
            return

        fork_name = args[0] if args else None

        current_save = s.save_session()
        if current_save:
            tl.write("[dim]Current session saved before fork[/dim]")

        from prsm.tui.widgets.conversation import ConversationView

        forked = s.session.fork(new_name=fork_name)
        s.session = forked
        conv = s.query_one("#conversation", ConversationView)
        conv.session = s.session

        tl.write(
            "[green]Forked[/green] session"
            + (f" as '{fork_name}'" if fork_name else "")
        )

    def _cmd_save(self, args: list[str], tl: ToolLog) -> None:
        s = self._screen
        if s.session.message_count == 0:
            tl.write("[yellow]Nothing to save -- session is empty[/yellow]")
            return

        if args:
            name = args[0]
            s.session.name = name
            try:
                path = s._persistence.save(s.session, name, session_id=s.session.session_id)
                tl.write(f"[green]Session saved:[/green] {path}")
            except Exception as exc:
                tl.write(f"[red]Save error:[/red] {exc}")
        else:
            path = s.save_session()
            if path:
                tl.write(f"[green]Session saved:[/green] {path}")
            else:
                tl.write("[yellow]Save failed or nothing to save[/yellow]")

    # ── /import ───────────────────────────────────────────────────

    def _cmd_import(self, args: list[str], tl: ToolLog) -> None:
        """Handle /import transcript portability commands."""
        action = args[0].lower() if args else "list"
        if action == "list":
            provider = args[1].lower() if len(args) > 1 else "all"
            self._import_list(provider, tl)
            return
        if action == "preview":
            if len(args) < 3:
                tl.write("[red]Usage:[/red] /import preview PROVIDER SOURCE_ID")
                return
            self._import_preview(args[1], args[2], tl)
            return
        if action == "run":
            self._import_run(args[1:], tl)
            return
        if action == "all":
            provider = args[1].lower() if len(args) > 1 else ""
            if provider not in {"codex", "claude", "prsm"}:
                tl.write("[red]Usage:[/red] /import all PROVIDER")
                return
            self._import_all(provider, tl)
            return

        tl.write(
            f"[red]Unknown import action:[/red] {action}. "
            "Use list|preview|run|all"
        )

    def _get_import_service(self):
        if self._import_service is None:
            from prsm.shared.services.transcript_import.service import (
                TranscriptImportService,
            )

            self._import_service = TranscriptImportService()
        return self._import_service

    def _import_list(self, provider: str, tl: ToolLog) -> None:
        provider = provider.lower()
        if provider not in {"all", "codex", "claude", "prsm"}:
            tl.write(f"[red]Unknown provider:[/red] {provider}")
            return
        service = self._get_import_service()
        sessions = service.list_sessions(provider=provider, limit=25)
        if not sessions:
            tl.write("[dim]No importable sessions found[/dim]")
            return
        rows: list[dict] = []
        for item in sessions:
            rows.append(
                {
                    "provider": item.provider,
                    "source_id": item.source_session_id,
                    "title": item.title or "(untitled)",
                    "turn_count": item.turn_count,
                    "updated": (
                        item.updated_at.isoformat(timespec="seconds")
                        if item.updated_at
                        else "unknown"
                    ),
                }
            )

        from prsm.tui.screens.import_picker import ImportSessionPickerScreen

        def _on_pick(result: dict | None) -> None:
            if not result:
                return
            selected_provider = str(result.get("provider") or "").strip()
            source_id = str(result.get("source_id") or "").strip()
            if not selected_provider or not source_id:
                return
            action = str(result.get("action") or "preview").strip().lower()
            if action == "run":
                title = str(result.get("title") or "").strip()
                run_args = [selected_provider, source_id]
                if title and title != "(untitled)":
                    run_args.append(title)
                self._import_run(run_args, tl)
                return
            self._import_preview(selected_provider, source_id, tl)

        self._screen.app.push_screen(
            ImportSessionPickerScreen(rows, provider_filter=provider),
            callback=_on_pick,
        )

    def _import_preview(self, provider: str, source_id: str, tl: ToolLog) -> None:
        provider = provider.lower()
        if provider not in {"codex", "claude", "prsm"}:
            tl.write("[red]Provider must be codex or claude[/red]")
            return
        service = self._get_import_service()
        try:
            transcript = service.load_transcript(provider, source_id)
        except FileNotFoundError:
            tl.write(
                f"[red]Import source not found:[/red] {provider}:{source_id}"
            )
            return
        except Exception as exc:
            safe = str(exc).replace("[", "\\[")
            tl.write(f"[red]Import preview failed:[/red] {safe}")
            return

        summary = transcript.summary
        tl.write(
            f"[bold]Preview[/bold] [cyan]{summary.provider}:{summary.source_session_id}[/cyan]"
        )
        if summary.title:
            tl.write(f"  title: {summary.title.replace('[', '\\[')}")
        tl.write(f"  turns: {len(transcript.turns)}")
        if summary.updated_at:
            tl.write(
                f"  updated: {summary.updated_at.isoformat(timespec='seconds')}"
            )

        preview_turns = transcript.turns[:10]
        for idx, turn in enumerate(preview_turns, start=1):
            text = " ".join(turn.content.split())
            if len(text) > 120:
                text = f"{text[:117]}..."
            safe = text.replace("[", "\\[")
            tool_count = f" +{len(turn.tool_calls)} tools" if turn.tool_calls else ""
            tl.write(f"  {idx:02d}. ({turn.role}) {safe}{tool_count}")
        if len(transcript.turns) > len(preview_turns):
            tl.write(f"  [dim]... {len(transcript.turns) - len(preview_turns)} more turns[/dim]")
        if transcript.warnings:
            tl.write(
                f"[yellow]Warnings:[/yellow] {len(transcript.warnings)} parse issue(s)"
            )

    def _import_run(self, args: list[str], tl: ToolLog) -> None:
        cleaned_args, max_turns, parse_error = self._parse_import_run_args(args)
        if parse_error:
            tl.write(f"[red]{parse_error}[/red]")
            tl.write(
                "[red]Usage:[/red] /import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]"
            )
            return
        if len(cleaned_args) < 2:
            tl.write(
                "[red]Usage:[/red] /import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]"
            )
            return

        provider = cleaned_args[0].lower()
        source_id = cleaned_args[1]
        if provider not in {"codex", "claude", "prsm"}:
            tl.write("[red]Provider must be codex or claude[/red]")
            return
        session_name = " ".join(cleaned_args[2:]).strip() or None

        if max_turns is None:
            from prsm.tui.screens.import_depth import ImportDepthScreen

            def _on_depth_selected(choice: str | None) -> None:
                if choice is None:
                    tl.write("[dim]Import canceled[/dim]")
                    return
                selected_max_turns: int | None = None
                if choice != "full":
                    selected_max_turns = int(choice)
                self._import_run_execute(
                    provider=provider,
                    source_id=source_id,
                    session_name=session_name,
                    max_turns=selected_max_turns,
                    tl=tl,
                )

            self._screen.app.push_screen(
                ImportDepthScreen(),
                callback=_on_depth_selected,
            )
            return

        self._import_run_execute(
            provider=provider,
            source_id=source_id,
            session_name=session_name,
            max_turns=max_turns,
            tl=tl,
        )

    def _import_run_execute(
        self,
        *,
        provider: str,
        source_id: str,
        session_name: str | None,
        max_turns: int | None,
        tl: ToolLog,
    ) -> None:
        service = self._get_import_service()
        try:
            result = service.import_to_session(
                provider,
                source_id,
                session_name=session_name,
                max_turns=max_turns,
            )
        except FileNotFoundError:
            tl.write(
                f"[red]Import source not found:[/red] {provider}:{source_id}"
            )
            return
        except Exception as exc:
            safe = str(exc).replace("[", "\\[")
            tl.write(f"[red]Import failed:[/red] {safe}")
            return

        from prsm.shared.services.transcript_import.service import (
            TranscriptImportService,
        )

        imported_session = result.session
        imported_session.imported_from = (
            TranscriptImportService.session_import_metadata(result)
        )

        s = self._screen
        if s.session.message_count > 0:
            s.save_session()
            tl.write("[dim]Current session auto-saved before import[/dim]")

        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView

        s.session = imported_session
        conv = s.query_one("#conversation", ConversationView)
        conv.session = s.session
        conv.clear()
        tree = s.query_one("#agent-tree", AgentTree)
        tree.clear_agents()
        s._restore_session(imported_session)

        saved_path = s.save_session()
        if saved_path:
            tl.write(f"[green]Imported session saved:[/green] {saved_path}")
        tl.write(
            f"[green]Imported[/green] {provider}:{source_id} "
            f"({result.imported_turns} turns"
            + (
                f", dropped {result.dropped_turns}"
                if result.dropped_turns
                else ""
            )
            + ")"
        )
        warnings = result.metadata.get("warnings", [])
        if warnings:
            tl.write(
                f"[yellow]Import warnings:[/yellow] {len(warnings)} parse issue(s)"
            )

    def _import_all(self, provider: str, tl: ToolLog) -> None:
        """Import all sessions from a provider at once."""
        service = self._get_import_service()
        tl.write(f"[dim]Importing all {provider} sessions...[/dim]")
        try:
            results = service.import_all_sessions(provider)
        except Exception as exc:
            safe = str(exc).replace("[", "\\[")
            tl.write(f"[red]Import all failed:[/red] {safe}")
            return

        if not results:
            tl.write(f"[dim]No importable {provider} sessions found[/dim]")
            return

        from prsm.shared.services.transcript_import.service import (
            TranscriptImportService,
        )

        s = self._screen
        if s.session.message_count > 0:
            s.save_session()
            tl.write("[dim]Current session auto-saved before import[/dim]")

        for result in results:
            imported_session = result.session
            imported_session.imported_from = (
                TranscriptImportService.session_import_metadata(result)
            )
            session_name = imported_session.name or result.source.source_session_id
            try:
                from prsm.shared.services.persistence import SessionPersistence
                persistence = SessionPersistence(cwd=s.app.cwd if hasattr(s.app, 'cwd') else None)
                persistence.save(
                    imported_session,
                    session_name,
                )
            except Exception as exc:
                safe = str(exc).replace("[", "\\[")
                tl.write(
                    f"[yellow]Warning:[/yellow] Failed to save {session_name}: {safe}"
                )
                continue

        tl.write(
            f"[green]Imported {len(results)} {provider} session(s)[/green]"
        )
        tl.write(
            "[dim]Use /sessions to see imported sessions[/dim]"
        )

    def _parse_import_run_args(
        self, args: list[str]
    ) -> tuple[list[str], int | None, str | None]:
        """Extract optional --max-turns flag from /import run args."""
        cleaned: list[str] = []
        max_turns: int | None = None
        i = 0
        while i < len(args):
            token = args[i]
            if token == "--max-turns":
                if i + 1 >= len(args):
                    return [], None, "Missing value for --max-turns"
                try:
                    max_turns = int(args[i + 1])
                except ValueError:
                    return [], None, "--max-turns must be an integer"
                if max_turns <= 0:
                    return [], None, "--max-turns must be greater than 0"
                i += 2
                continue
            if token.startswith("--max-turns="):
                value = token.split("=", 1)[1]
                try:
                    max_turns = int(value)
                except ValueError:
                    return [], None, "--max-turns must be an integer"
                if max_turns <= 0:
                    return [], None, "--max-turns must be greater than 0"
                i += 1
                continue
            cleaned.append(token)
            i += 1
        return cleaned, max_turns, None

    # ── /model ────────────────────────────────────────────────────

    def _cmd_model(self, args: list[str], tl: ToolLog) -> None:
        """Handle /model — switch the orchestrator model.

        Subcommands:
            (no args)   Open the interactive model picker modal
            list        Show available models in the tool log
            NAME        Switch directly to the named model
        """
        s = self._screen

        if not s._live_mode:
            tl.write("[yellow]Model switching is only available in live mode[/yellow]")
            return

        action = args[0].lower() if args else ""

        if action == "list":
            self._model_list(tl)
        elif action == "":
            # Open interactive model picker
            s.show_model_selector()
        else:
            # Direct model switch: /model opus
            s.switch_model(action)

    def _model_list(self, tl: ToolLog) -> None:
        """Show available models in the tool log."""
        s = self._screen
        models = s.bridge.get_available_models()
        if not models:
            tl.write("[dim]No models available[/dim]")
            return

        current = s.bridge.current_model
        tl.write(f"[bold]Available models[/bold] (current: [cyan]{current}[/cyan]):")

        tier_icons = {
            "frontier": "[bold magenta]⬥ FRONTIER[/bold magenta]",
            "strong": "[bold cyan]◆ STRONG[/bold cyan]",
            "fast": "[bold green]▸ FAST[/bold green]",
            "economy": "[bold yellow]○ ECONOMY[/bold yellow]",
        }

        current_provider = ""
        for m in models:
            if m["provider"] != current_provider:
                current_provider = m["provider"]
                tl.write(f"  [bold]{current_provider.title()}[/bold]")

            tier_label = tier_icons.get(m["tier"], m["tier"])
            avail = "" if m["available"] else " [dim](unavailable)[/dim]"
            marker = " [green]✓[/green]" if m["is_current"] else ""
            safe_id = m["model_id"].replace("[", "\\[")
            tl.write(f"    {tier_label}  {safe_id}{marker}{avail}")

    # ── /settings ───────────────────────────────────────────────────

    def _cmd_settings(self, tl: ToolLog) -> None:
        """Handle /settings — open the settings menu."""
        from prsm.tui.screens.settings import SettingsScreen

        self._screen.app.push_screen(
            SettingsScreen(cwd=self._screen._cwd),
        )

    # ── /policy ─────────────────────────────────────────────────────

    def _cmd_policy(self, args: list[str], tl: ToolLog) -> None:
        """Handle /policy — open the bash command policy manager.

        Subcommands:
            (no args)   Open the interactive policy manager modal
            list        Show current whitelist and blacklist in the tool log
            add-allow   /policy add-allow PATTERN — add to whitelist
            add-block   /policy add-block PATTERN — add to blacklist
            remove      /policy remove PATTERN — remove from both lists
        """
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        store = CommandPolicyStore(self._screen._cwd)
        store.ensure_files()

        action = args[0].lower() if args else ""

        if action == "" or action == "manage":
            self._policy_show_modal()
        elif action == "list":
            self._policy_list(store, tl)
        elif action == "add-allow":
            self._policy_add(store, args, tl, allow=True)
        elif action == "add-block":
            self._policy_add(store, args, tl, allow=False)
        elif action == "remove":
            self._policy_remove(store, args, tl)
        else:
            tl.write(
                f"[red]Unknown policy action:[/red] {action}. "
                "Use list|add-allow|add-block|remove or no args for interactive."
            )

    def _policy_show_modal(self) -> None:
        """Open the interactive command policy manager modal."""
        from prsm.tui.screens.command_policy import CommandPolicyScreen

        self._screen.app.push_screen(
            CommandPolicyScreen(cwd=self._screen._cwd),
        )

    def _policy_list(self, store, tl: ToolLog) -> None:
        """Show current whitelist and blacklist in the tool log."""
        whitelist = store.read_whitelist()
        blacklist = store.read_blacklist()

        tl.write("[bold]Bash Command Policy[/bold]")
        tl.write(f"  [bold green]Whitelist[/bold green] ({len(whitelist)} patterns):")
        if whitelist:
            for pat in whitelist:
                safe = pat.replace("[", "\\[")
                tl.write(f"    [cyan]{safe}[/cyan]")
        else:
            tl.write("    [dim](empty)[/dim]")

        tl.write(f"  [bold red]Blacklist[/bold red] ({len(blacklist)} patterns):")
        if blacklist:
            for pat in blacklist:
                safe = pat.replace("[", "\\[")
                tl.write(f"    [cyan]{safe}[/cyan]")
        else:
            tl.write("    [dim](empty)[/dim]")

    def _policy_add(
        self, store, args: list[str], tl: ToolLog, *, allow: bool
    ) -> None:
        """Add a pattern to whitelist or blacklist."""
        if len(args) < 2:
            kind = "add-allow" if allow else "add-block"
            tl.write(f"[red]Usage:[/red] /policy {kind} PATTERN")
            return
        pattern = " ".join(args[1:])
        if allow:
            store.add_whitelist_pattern(pattern)
            tl.write(
                f"[green]Added to whitelist:[/green] [cyan]{pattern}[/cyan]"
            )
        else:
            store.add_blacklist_pattern(pattern)
            tl.write(
                f"[green]Added to blacklist:[/green] [cyan]{pattern}[/cyan]"
            )

    def _policy_remove(self, store, args: list[str], tl: ToolLog) -> None:
        """Remove a pattern from both whitelist and blacklist."""
        if len(args) < 2:
            tl.write("[red]Usage:[/red] /policy remove PATTERN")
            return
        pattern = " ".join(args[1:])
        removed_wl = store.remove_whitelist_pattern(pattern)
        removed_bl = store.remove_blacklist_pattern(pattern)
        if removed_wl or removed_bl:
            safe = pattern.replace("[", "\\[")
            sources = []
            if removed_wl:
                sources.append("whitelist")
            if removed_bl:
                sources.append("blacklist")
            tl.write(
                f"[green]Removed[/green] [cyan]{safe}[/cyan] "
                f"from {', '.join(sources)}"
            )
        else:
            safe = pattern.replace("[", "\\[")
            tl.write(
                f"[yellow]Pattern not found:[/yellow] [cyan]{safe}[/cyan]"
            )

    # ── /plugin ─────────────────────────────────────────────────────

    def _cmd_plugin(self, args: list[str], tl: ToolLog) -> None:
        pm = self._screen._plugin_manager
        if not pm:
            tl.write("[red]Plugin manager not initialized[/red]")
            return
        if not args:
            tl.write("[red]Usage:[/red] /plugin add|add-json|list|remove|tag")
            return

        action = args[0].lower()

        if action == "list":
            self._plugin_list(pm, tl)
        elif action == "add":
            self._plugin_add(pm, args, tl)
        elif action == "add-json":
            self._plugin_add_json(pm, args, tl)
        elif action == "tag":
            self._plugin_tag(pm, args, tl)
        elif action == "remove":
            self._plugin_remove(pm, args, tl)
        else:
            tl.write(
                f"[red]Unknown plugin action:[/red] {action}. "
                "Use add|add-json|list|remove|tag"
            )

    def _plugin_list(self, pm, tl: ToolLog) -> None:
        plugins = pm.list_plugins()
        if not plugins:
            tl.write("[dim]No plugins loaded[/dim]")
        else:
            tl.write(f"[bold]Loaded plugins ({len(plugins)}):[/bold]")
            for p in plugins:
                if p.type == "stdio":
                    args_str = " ".join(p.args) if p.args else ""
                    detail = f"{p.command} {args_str}".strip()
                else:
                    detail = f"{p.type} {p.url}"
                tags_str = f" [dim]tags={p.tags}[/dim]" if p.tags else ""
                tl.write(
                    f"  [cyan]{p.name}[/cyan] ({p.type}) -- {detail}{tags_str}"
                )

    def _plugin_add(self, pm, args: list[str], tl: ToolLog) -> None:
        if len(args) < 3:
            tl.write("[red]Usage:[/red] /plugin add NAME COMMAND [ARGS...]")
            tl.write(
                "[dim]  or:  /plugin add NAME --type http "
                "--url URL [--header K:V][/dim]"
            )
            return
        name = args[1]

        # Check for --type flag (remote transport)
        if "--type" in args:
            type_idx = args.index("--type")
            transport = args[type_idx + 1] if type_idx + 1 < len(args) else None
            if transport not in ("http", "sse"):
                tl.write(
                    f"[red]Invalid type '{transport}'. Use http or sse.[/red]"
                )
                return

            url = None
            if "--url" in args:
                url_idx = args.index("--url")
                url = args[url_idx + 1] if url_idx + 1 < len(args) else None
            if not url:
                tl.write("[red]Remote plugins require --url URL[/red]")
                return

            headers = {}
            for i, arg in enumerate(args):
                if arg == "--header" and i + 1 < len(args):
                    k, _, v = args[i + 1].partition(":")
                    if k:
                        headers[k.strip()] = v.strip()

            pm.add_remote(name, transport, url, headers or None)
            tl.write(
                f"[green]Plugin added:[/green] {name} ({transport} {url})"
            )
        else:
            # Original stdio path
            command = args[2]
            plugin_args = args[3:]
            pm.add(name, command, plugin_args)
            tl.write(
                f"[green]Plugin added:[/green] {name} "
                f"({command} {' '.join(plugin_args)})"
            )

        tl.write(
            "[dim]Plugin will be available for the next orchestration run[/dim]"
        )

    def _plugin_add_json(self, pm, args: list[str], tl: ToolLog) -> None:
        if len(args) < 3:
            tl.write(
                "[red]Usage:[/red] /plugin add-json NAME "
                "'{\"type\":\"http\",\"url\":\"...\"}'"
            )
            return
        name = args[1]
        json_str = " ".join(args[2:])
        try:
            import json

            config = json.loads(json_str)
            pm.add_json(name, config)
            tl.write(f"[green]Plugin added from JSON:[/green] {name}")
            tl.write(
                "[dim]Plugin will be available for the next "
                "orchestration run[/dim]"
            )
        except json.JSONDecodeError as e:
            tl.write(f"[red]Invalid JSON:[/red] {e}")
        except ValueError as e:
            tl.write(f"[red]Invalid plugin config:[/red] {e}")

    def _plugin_tag(self, pm, args: list[str], tl: ToolLog) -> None:
        if len(args) < 3:
            tl.write("[red]Usage:[/red] /plugin tag NAME tag1 [tag2 ...]")
            return
        name = args[1]
        tags = args[2:]
        plugin = pm._plugins.get(name)
        if not plugin:
            tl.write(f"[red]Plugin '{name}' not found[/red]")
            return
        plugin.tags = tags
        pm._save_project_plugins()
        tl.write(f"[green]Tags set:[/green] {name} -> {tags}")

    def _plugin_remove(self, pm, args: list[str], tl: ToolLog) -> None:
        if len(args) < 2:
            tl.write("[red]Usage:[/red] /plugin remove NAME")
            return
        name = args[1]
        if pm.remove(name):
            tl.write(f"[green]Plugin removed:[/green] {name}")
        else:
            tl.write(f"[red]Plugin '{name}' not found[/red]")

    # ── /worktree ───────────────────────────────────────────────────

    def _cmd_worktree(self, args: list[str], tl: ToolLog) -> None:
        """Handle /worktree subcommands."""
        from prsm.shared.services.project import ProjectManager

        if not ProjectManager.is_git_repo(self._screen._cwd):
            tl.write("[red]Not a git repository[/red]")
            return

        action = args[0].lower() if args else "list"

        if action == "list":
            self._cmd_worktree_list(tl)
        elif action == "create":
            self._cmd_worktree_create_interactive(args[1:], tl)
        elif action == "remove":
            self._cmd_worktree_remove_interactive(args[1:], tl)
        elif action == "switch":
            self._cmd_worktree_switch_interactive(args[1:], tl)
        else:
            tl.write(
                f"[red]Unknown worktree action:[/red] {action}. "
                "Use list|create|remove|switch"
            )

    def _cmd_worktree_list(self, tl: ToolLog) -> None:
        """Show the interactive worktree list modal."""
        from prsm.tui.screens.worktree import WorktreeListScreen

        def _on_list_result(result: str | None) -> None:
            if result is None:
                return
            if result == "create":
                self._cmd_worktree_create_interactive([], tl)
            elif result.startswith("switch:"):
                path = result.removeprefix("switch:")
                self._cmd_worktree_switch_interactive([path], tl)
            elif result.startswith("remove:"):
                path = result.removeprefix("remove:")
                self._cmd_worktree_remove_interactive([path], tl)

        self._screen.app.push_screen(
            WorktreeListScreen(cwd=self._screen._cwd),
            callback=_on_list_result,
        )

    def _cmd_worktree_create_interactive(
        self, args: list[str], tl: ToolLog
    ) -> None:
        """Show the create worktree dialog, or create inline from args."""
        if len(args) >= 1:
            # Inline: /worktree create PATH [BRANCH]
            path = args[0]
            branch = args[1] if len(args) > 1 else None
            self._do_create_worktree(path, branch, tl)
            return

        from prsm.tui.screens.worktree import WorktreeCreateScreen

        def _on_create_result(result: dict | None) -> None:
            if result is None:
                return
            self._do_create_worktree(
                result["path"],
                result.get("branch"),
                tl,
                new_branch=result.get("new_branch"),
            )

        self._screen.app.push_screen(
            WorktreeCreateScreen(cwd=self._screen._cwd),
            callback=_on_create_result,
        )

    def _do_create_worktree(
        self,
        path: str,
        branch: str | None,
        tl: ToolLog,
        new_branch: str | None = None,
    ) -> None:
        """Execute worktree creation and report the result."""
        from prsm.shared.services.project import ProjectManager

        tl.write(
            f"[yellow]Creating[/yellow] worktree at [cyan]{path}[/cyan]..."
        )
        ok, msg = ProjectManager.create_worktree(
            path=path,
            branch=branch,
            new_branch=new_branch,
            cwd=self._screen._cwd,
        )
        safe_msg = msg.replace("[", "\\[")
        if ok:
            tl.write(f"[green]Worktree created:[/green] {safe_msg}")
        else:
            tl.write(f"[red]Failed to create worktree:[/red] {safe_msg}")

    def _cmd_worktree_remove_interactive(
        self, args: list[str], tl: ToolLog
    ) -> None:
        """Show the remove worktree confirmation, or remove inline."""
        if not args:
            tl.write("[red]Usage:[/red] /worktree remove PATH")
            return

        path = args[0]
        force = "--force" in args

        if force:
            # Direct removal without confirmation
            self._do_remove_worktree(path, force=True, tl=tl)
            return

        from prsm.tui.screens.worktree import WorktreeRemoveScreen

        def _on_remove_result(result: str | None) -> None:
            if result is None:
                return
            self._do_remove_worktree(
                path, force=(result == "force"), tl=tl
            )

        self._screen.app.push_screen(
            WorktreeRemoveScreen(worktree_path=path),
            callback=_on_remove_result,
        )

    def _do_remove_worktree(
        self, path: str, force: bool, tl: ToolLog
    ) -> None:
        """Execute worktree removal and report the result."""
        from prsm.shared.services.project import ProjectManager

        force_label = " (force)" if force else ""
        tl.write(
            f"[yellow]Removing[/yellow] worktree "
            f"[cyan]{path}[/cyan]{force_label}..."
        )
        ok, msg = ProjectManager.remove_worktree(
            path=path, force=force, cwd=self._screen._cwd
        )
        safe_msg = msg.replace("[", "\\[")
        if ok:
            tl.write(f"[green]Worktree removed:[/green] {safe_msg}")
        else:
            tl.write(f"[red]Failed to remove worktree:[/red] {safe_msg}")

    def _cmd_worktree_switch_interactive(
        self, args: list[str], tl: ToolLog
    ) -> None:
        """Show switch confirmation and restart PRSM in the new worktree."""
        if not args:
            tl.write("[red]Usage:[/red] /worktree switch PATH")
            return

        path = args[0]
        target = Path(path)

        # Resolve relative paths from cwd
        if not target.is_absolute():
            target = (self._screen._cwd / target).resolve()

        if not target.is_dir():
            tl.write(f"[red]Worktree path does not exist:[/red] {target}")
            return

        from prsm.tui.screens.worktree import WorktreeSwitchScreen

        def _on_switch_result(result: str | None) -> None:
            if result != "switch":
                return
            self._do_switch_worktree(str(target), tl)

        self._screen.app.push_screen(
            WorktreeSwitchScreen(worktree_path=str(target)),
            callback=_on_switch_result,
        )

    def _do_switch_worktree(self, path: str, tl: ToolLog) -> None:
        """Save session and restart PRSM in the target worktree directory."""
        import os
        import sys

        s = self._screen

        # Save current session
        saved = s.save_session()
        if saved:
            tl.write("[dim]Session saved before switch[/dim]")

        tl.write(
            f"[green]Switching[/green] to worktree [cyan]{path}[/cyan]..."
        )

        # Restart the process in the new directory
        os.chdir(path)
        python = sys.executable
        os.execvp(python, [python, "-m", "prsm.app"] + sys.argv[1:])
