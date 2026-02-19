"""Command policy management modal — add/remove bash whitelist & blacklist patterns.

Displays the current whitelist and blacklist stored in
<workspace>/.prism/command_whitelist.txt and command_blacklist.txt.
Allows the user to add new regex patterns or remove existing ones.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from prsm.shared.services.command_policy_store import CommandPolicyStore


class CommandPolicyScreen(ModalScreen[str | None]):
    """Modal dialog for managing bash command whitelist & blacklist.

    Returns None when dismissed (all mutations happen in-place on the
    policy files; the caller does not need to process a return value).
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, cwd: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd or Path.cwd()
        self._store = CommandPolicyStore(self._cwd)
        self._store.ensure_files()

    def compose(self) -> ComposeResult:
        whitelist = self._store.read_whitelist()
        blacklist = self._store.read_blacklist()

        with Vertical(id="cmd-policy-dialog"):
            yield Static(
                "[bold $primary]Bash Command Policy[/bold $primary]",
                id="cmd-policy-title",
            )
            yield Static(
                "[dim]Patterns are regex. Whitelist auto-allows matching commands; "
                "blacklist forces a permission prompt.[/dim]",
                id="cmd-policy-hint",
            )

            # ── Whitelist section ──
            yield Static(
                "[bold green]Whitelist[/bold green] (auto-allow)",
                id="cmd-policy-wl-header",
            )
            with VerticalScroll(id="cmd-policy-wl-list"):
                if whitelist:
                    for i, pat in enumerate(whitelist):
                        yield _PatternEntry(
                            pattern=pat,
                            list_kind="wl",
                            index=i,
                        )
                else:
                    yield Static(
                        "[dim]No whitelist patterns[/dim]",
                        id="cmd-policy-wl-empty",
                    )
            with Horizontal(classes="cmd-policy-add-row"):
                yield Input(
                    placeholder="e.g. npm\\s+test",
                    id="cmd-policy-wl-input",
                )
                yield Button(
                    "Add",
                    variant="success",
                    id="btn-cmd-wl-add",
                )

            # ── Blacklist section ──
            yield Static(
                "[bold red]Blacklist[/bold red] (always prompt)",
                id="cmd-policy-bl-header",
            )
            with VerticalScroll(id="cmd-policy-bl-list"):
                if blacklist:
                    for i, pat in enumerate(blacklist):
                        yield _PatternEntry(
                            pattern=pat,
                            list_kind="bl",
                            index=i,
                        )
                else:
                    yield Static(
                        "[dim]No blacklist patterns[/dim]",
                        id="cmd-policy-bl-empty",
                    )
            with Horizontal(classes="cmd-policy-add-row"):
                yield Input(
                    placeholder="e.g. docker\\s+volume\\s+rm",
                    id="cmd-policy-bl-input",
                )
                yield Button(
                    "Add",
                    variant="error",
                    id="btn-cmd-bl-add",
                )

            # ── Close ──
            with Horizontal(id="cmd-policy-actions"):
                yield Button("Close", variant="default", id="btn-cmd-close")

    # ── Event handling ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id == "btn-cmd-close":
            self.dismiss(None)
            return

        if btn_id == "btn-cmd-wl-add":
            self._add_to_whitelist()
            return

        if btn_id == "btn-cmd-bl-add":
            self._add_to_blacklist()
            return

        if btn_id.startswith("btn-cmd-rm-wl-"):
            idx = int(btn_id.removeprefix("btn-cmd-rm-wl-"))
            self._remove_from_whitelist(idx)
            return

        if btn_id.startswith("btn-cmd-rm-bl-"):
            idx = int(btn_id.removeprefix("btn-cmd-rm-bl-"))
            self._remove_from_blacklist(idx)
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Allow Enter in an input field to submit."""
        if event.input.id == "cmd-policy-wl-input":
            self._add_to_whitelist()
        elif event.input.id == "cmd-policy-bl-input":
            self._add_to_blacklist()

    def key_escape(self) -> None:
        self.dismiss(None)

    # ── Mutations ──

    def _add_to_whitelist(self) -> None:
        inp = self.query_one("#cmd-policy-wl-input", Input)
        pattern = inp.value.strip()
        if not pattern:
            inp.add_class("error")
            return
        self._store.add_whitelist_pattern(pattern)
        inp.value = ""
        inp.remove_class("error")
        self._rebuild()

    def _add_to_blacklist(self) -> None:
        inp = self.query_one("#cmd-policy-bl-input", Input)
        pattern = inp.value.strip()
        if not pattern:
            inp.add_class("error")
            return
        self._store.add_blacklist_pattern(pattern)
        inp.value = ""
        inp.remove_class("error")
        self._rebuild()

    def _remove_from_whitelist(self, index: int) -> None:
        patterns = self._store.read_whitelist()
        if 0 <= index < len(patterns):
            self._store.remove_whitelist_pattern(patterns[index])
            self._rebuild()

    def _remove_from_blacklist(self, index: int) -> None:
        patterns = self._store.read_blacklist()
        if 0 <= index < len(patterns):
            self._store.remove_blacklist_pattern(patterns[index])
            self._rebuild()

    def _rebuild(self) -> None:
        """Re-render the entire modal content after a mutation."""
        # Remove all children and re-compose
        dialog = self.query_one("#cmd-policy-dialog", Vertical)
        dialog.remove_children()
        # Re-compose content into the existing dialog
        whitelist = self._store.read_whitelist()
        blacklist = self._store.read_blacklist()

        dialog.mount(Static(
            "[bold $primary]Bash Command Policy[/bold $primary]",
            id="cmd-policy-title",
        ))
        dialog.mount(Static(
            "[dim]Patterns are regex. Whitelist auto-allows matching commands; "
            "blacklist forces a permission prompt.[/dim]",
            id="cmd-policy-hint",
        ))

        # Whitelist
        dialog.mount(Static(
            "[bold green]Whitelist[/bold green] (auto-allow)",
            id="cmd-policy-wl-header",
        ))
        wl_scroll = VerticalScroll(id="cmd-policy-wl-list")
        dialog.mount(wl_scroll)
        if whitelist:
            for i, pat in enumerate(whitelist):
                wl_scroll.mount(_PatternEntry(
                    pattern=pat, list_kind="wl", index=i,
                ))
        else:
            wl_scroll.mount(Static(
                "[dim]No whitelist patterns[/dim]",
                id="cmd-policy-wl-empty",
            ))
        wl_add_row = Horizontal(classes="cmd-policy-add-row")
        dialog.mount(wl_add_row)
        wl_add_row.mount(Input(
            placeholder="e.g. npm\\s+test",
            id="cmd-policy-wl-input",
        ))
        wl_add_row.mount(Button(
            "Add", variant="success", id="btn-cmd-wl-add",
        ))

        # Blacklist
        dialog.mount(Static(
            "[bold red]Blacklist[/bold red] (always prompt)",
            id="cmd-policy-bl-header",
        ))
        bl_scroll = VerticalScroll(id="cmd-policy-bl-list")
        dialog.mount(bl_scroll)
        if blacklist:
            for i, pat in enumerate(blacklist):
                bl_scroll.mount(_PatternEntry(
                    pattern=pat, list_kind="bl", index=i,
                ))
        else:
            bl_scroll.mount(Static(
                "[dim]No blacklist patterns[/dim]",
                id="cmd-policy-bl-empty",
            ))
        bl_add_row = Horizontal(classes="cmd-policy-add-row")
        dialog.mount(bl_add_row)
        bl_add_row.mount(Input(
            placeholder="e.g. docker\\s+volume\\s+rm",
            id="cmd-policy-bl-input",
        ))
        bl_add_row.mount(Button(
            "Add", variant="error", id="btn-cmd-bl-add",
        ))

        # Close
        close_row = Horizontal(id="cmd-policy-actions")
        dialog.mount(close_row)
        close_row.mount(Button("Close", variant="default", id="btn-cmd-close"))


class _PatternEntry(Static):
    """A single pattern row with a remove button."""

    def __init__(
        self,
        pattern: str,
        list_kind: str,
        index: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._pattern = pattern
        self._list_kind = list_kind
        self._index = index

    def compose(self) -> ComposeResult:
        safe_pat = self._pattern.replace("[", "\\[")
        with Horizontal(classes="cmd-policy-entry"):
            yield Static(f"  [cyan]{safe_pat}[/cyan]", classes="cmd-policy-pattern")
            yield Button(
                "Remove",
                variant="warning",
                id=f"btn-cmd-rm-{self._list_kind}-{self._index}",
                classes="cmd-policy-rm-btn",
            )
