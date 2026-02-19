"""Transcript import picker modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class ImportSessionPickerScreen(ModalScreen[dict | None]):
    """Search and choose importable provider sessions."""

    CSS_PATH = "../styles/modal.tcss"
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, sessions: list[dict], provider_filter: str = "all", **kwargs) -> None:
        super().__init__(**kwargs)
        self._sessions = sessions
        self._provider_filter = provider_filter
        self._filtered: list[dict] = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="import-picker-dialog"):
            yield Static(
                "[bold $primary]Import Transcript[/bold $primary]",
                id="import-picker-title",
                markup=True,
            )
            yield Input(
                placeholder="Search by provider, source ID, or title...",
                id="import-picker-input",
            )
            filter_hint = (
                f"Provider filter: {self._provider_filter}. "
                if self._provider_filter != "all"
                else ""
            )
            yield Static(
                f"{filter_hint}Use row buttons to preview or import.",
                id="import-picker-hint",
            )
            yield VerticalScroll(id="import-picker-list")
            with Horizontal(id="import-picker-actions"):
                yield Button("Cancel", id="import-picker-cancel")

    def on_mount(self) -> None:
        self.query_one("#import-picker-input", Input).focus()
        self._render_results()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "import-picker-input":
            return
        query = event.value.strip().lower()
        if not query:
            self._filtered = self._sessions
        else:
            def _matches(row: dict) -> bool:
                haystack = " ".join([
                    str(row.get("provider", "")),
                    str(row.get("source_id", "")),
                    str(row.get("title", "")),
                    str(row.get("updated", "")),
                ]).lower()
                return query in haystack

            self._filtered = [row for row in self._sessions if _matches(row)]
        self._render_results()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "import-picker-input":
            return
        if self._filtered:
            row = self._filtered[0]
            self.dismiss(
                {
                    "action": "preview",
                    "provider": row.get("provider"),
                    "source_id": row.get("source_id"),
                    "title": row.get("title"),
                }
            )

    def _render_results(self) -> None:
        container = self.query_one("#import-picker-list", VerticalScroll)
        container.remove_children()

        if not self._filtered:
            container.mount(Static("[dim]No matching import sessions[/dim]", markup=True))
            return

        for idx, row in enumerate(self._filtered):
            provider = str(row.get("provider") or "unknown")
            source_id = str(row.get("source_id") or "")
            title = str(row.get("title") or "(untitled)")
            updated = str(row.get("updated") or "unknown")
            turn_count = row.get("turn_count", "?")
            safe_title = title.replace("[", "\\[")
            label = Static(
                f"[bold]{safe_title}[/bold]\n"
                f"[dim]{provider}:{source_id} · {turn_count} turns · {updated}[/dim]",
                markup=True,
                classes="import-picker-label",
            )

            row_container = Horizontal(
                label,
                Button("Preview", id=f"import-picker-preview-{idx}", classes="import-picker-preview"),
                Button("Import", id=f"import-picker-run-{idx}", classes="import-picker-run"),
                classes="import-picker-row",
            )
            container.mount(row_container)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "import-picker-cancel":
            self.dismiss(None)
            return
        if btn_id.startswith("import-picker-preview-"):
            idx = self._index_from_button_id(btn_id)
            if idx is None:
                return
            row = self._filtered[idx]
            self.dismiss(
                {
                    "action": "preview",
                    "provider": row.get("provider"),
                    "source_id": row.get("source_id"),
                    "title": row.get("title"),
                }
            )
            return
        if btn_id.startswith("import-picker-run-"):
            idx = self._index_from_button_id(btn_id)
            if idx is None:
                return
            row = self._filtered[idx]
            self.dismiss(
                {
                    "action": "run",
                    "provider": row.get("provider"),
                    "source_id": row.get("source_id"),
                    "title": row.get("title"),
                }
            )

    def _index_from_button_id(self, btn_id: str) -> int | None:
        try:
            idx = int(btn_id.rsplit("-", 1)[-1])
        except ValueError:
            return None
        if idx < 0 or idx >= len(self._filtered):
            return None
        return idx

    def action_cancel(self) -> None:
        self.dismiss(None)

