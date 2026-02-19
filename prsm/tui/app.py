"""PRSM TUI — Textual application class."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from prsm.tui.screens.main import MainScreen


class PrsmApp(App):
    """Terminal UI for multi-agent orchestration."""

    TITLE = "PRSM"
    SUB_TITLE = "Agent Orchestrator"
    CSS_PATH = Path("styles/app.tcss")
    cli_args = None  # Set by main() before run()

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "cancel_orchestration", "Cancel"),
        ("ctrl+h", "open_help", "Help"),
        ("meta+h", "open_help", "Help"),
        ("ctrl+n", "new_session", "New Session"),
        ("ctrl+t", "focus_tree", "Tree"),
        ("ctrl+e", "focus_editor", "Input"),
        ("ctrl+s", "save_session", "Save"),
        ("f1", "toggle_tool_log", "Tool Log"),
        ("f2", "open_settings", "Settings"),
        ("escape", "cancel_or_blur", "Cancel"),
    ]

    def on_mount(self) -> None:
        self.push_screen(MainScreen())

    def action_focus_tree(self) -> None:
        try:
            self.screen.query_one("#agent-tree").focus()
        except Exception:
            pass

    def action_focus_editor(self) -> None:
        from prsm.tui.widgets.input_bar import InputBar
        try:
            self.screen.query_one(InputBar).focus_input()
        except Exception:
            pass

    def action_toggle_tool_log(self) -> None:
        from prsm.tui.widgets.tool_log import ToolLog
        try:
            self.screen.query_one(ToolLog).toggle()
        except Exception:
            pass

    def action_new_session(self) -> None:
        """Save current session and start a fresh new one."""
        from prsm.tui.screens.main import MainScreen
        screen = self.screen
        if isinstance(screen, MainScreen):
            screen.command_handler.handle_command("new", [])

    def action_cancel_orchestration(self) -> None:
        from prsm.tui.screens.main import MainScreen
        screen = self.screen
        if isinstance(screen, MainScreen):
            screen.request_stop_current_run()

    def action_save_session(self) -> None:
        from prsm.tui.screens.main import MainScreen
        screen = self.screen
        if isinstance(screen, MainScreen):
            path = screen.save_session()
            if path:
                from prsm.tui.widgets.tool_log import ToolLog
                try:
                    screen.query_one(ToolLog).write(
                        f"[green]Session saved:[/green] {path}"
                    )
                except Exception:
                    pass

    async def action_quit(self) -> None:
        """Auto-save session before quitting."""
        from prsm.tui.screens.main import MainScreen
        screen = self.screen
        if isinstance(screen, MainScreen):
            screen.save_session()
        await super().action_quit()

    def action_open_settings(self) -> None:
        """Open the settings menu."""
        from prsm.tui.screens.main import MainScreen
        from prsm.tui.screens.settings import SettingsScreen

        screen = self.screen
        cwd = screen._cwd if isinstance(screen, MainScreen) else None
        self.push_screen(SettingsScreen(cwd=cwd))

    def action_open_help(self) -> None:
        """Open the keyboard/mouse help modal."""
        from prsm.tui.screens.help import HelpScreen

        self.push_screen(HelpScreen())

    def action_cancel_or_blur(self) -> None:
        self.screen.set_focus(None)
