"""Session search modal for quickly loading saved sessions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class SessionSearchScreen(ModalScreen[str | None]):
    """Search and select a saved session by name or session ID."""

    CSS_PATH = "../styles/modal.tcss"
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, sessions: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._sessions = sessions
        self._filtered: list[dict] = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="session-search-dialog"):
            yield Static(
                "[bold $primary]Search Sessions[/bold $primary]",
                id="session-search-title",
                markup=True,
            )
            yield Input(
                placeholder="Type to search by name, id, branch, or fork source...",
                id="session-search-input",
            )
            yield Label(
                "Click a session to load it. This list includes sessions across worktrees.",
                id="session-search-hint",
            )
            yield VerticalScroll(id="session-search-list")
            with Horizontal(id="session-search-actions"):
                yield Button("Cancel", id="session-search-cancel")

    def on_mount(self) -> None:
        self.query_one("#session-search-input", Input).focus()
        self._render_results()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "session-search-input":
            return
        query = event.value.strip().lower()
        if not query:
            self._filtered = self._sessions
        else:
            def _matches(row: dict) -> bool:
                haystack = " ".join([
                    str(row.get("name", "")),
                    str(row.get("session_id", "")),
                    str(row.get("branch", "")),
                    str(row.get("forked_from", "")),
                ]).lower()
                return query in haystack

            self._filtered = [row for row in self._sessions if _matches(row)]
        self._render_results()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "session-search-input":
            return
        if self._filtered:
            self.dismiss(str(self._filtered[0].get("session_id", "")) or None)

    def _render_results(self) -> None:
        container = self.query_one("#session-search-list", VerticalScroll)
        container.remove_children()

        if not self._filtered:
            container.mount(
                Static("[dim]No matching sessions[/dim]", markup=True)
            )
            return

        for idx, row in enumerate(self._filtered):
            name = str(row.get("name") or row.get("session_id") or "<unnamed>")
            session_id = str(row.get("session_id") or "")
            branch = str(row.get("branch") or "")
            counts = (
                f"{row.get('agent_count', '?')} agents · "
                f"{row.get('message_count', '?')} msgs"
            )
            suffix = f" · {branch}" if branch else ""
            label = (
                f"{name}\n"
                f"[dim]{session_id} · {counts}{suffix}[/dim]"
            )
            select_button = Button(
                label,
                id=f"session-search-select-{idx}",
                classes="session-search-item",
            )
            select_button.tooltip = (
                f"Session UUID: {session_id} 📋\n"
                "Click the copy icon button to copy this UUID."
            )
            copy_button = Button(
                "📋",
                id=f"session-search-copy-{idx}",
                classes="session-search-copy",
            )
            copy_button.tooltip = (
                f"Session UUID: {session_id} 📋\n"
                "Click to copy."
            )
            row_container = Horizontal(
                select_button,
                copy_button,
                classes="session-search-row",
            )
            container.mount(row_container)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "session-search-cancel":
            self.dismiss(None)
            return
        if btn_id.startswith("session-search-copy-"):
            try:
                idx = int(btn_id.rsplit("-", 1)[-1])
            except ValueError:
                return
            if 0 <= idx < len(self._filtered):
                session_id = str(self._filtered[idx].get("session_id") or "")
                if session_id:
                    self.app.copy_to_clipboard(session_id)
                    self.notify("Session UUID copied to clipboard", title="Copied")
            return
        if btn_id.startswith("session-search-select-"):
            try:
                idx = int(btn_id.rsplit("-", 1)[-1])
            except ValueError:
                self.dismiss(None)
                return
            if 0 <= idx < len(self._filtered):
                self.dismiss(str(self._filtered[idx].get("session_id", "")) or None)
                return
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
