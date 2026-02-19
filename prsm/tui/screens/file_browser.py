"""File browser modal — navigate filesystem and select archive files."""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


_ARCHIVE_EXTENSIONS = {".tar.gz", ".tgz", ".zip", ".tar", ".tar.bz2"}


def _is_archive(path: Path) -> bool:
    """Check if a path has a supported archive extension."""
    name = path.name.lower()
    return any(name.endswith(ext) for ext in _ARCHIVE_EXTENSIONS)


class FileBrowserScreen(ModalScreen[Path | None]):
    """Modal file browser for selecting archive files.

    Allows navigating directories and selecting .tar.gz, .zip, .tar,
    .tar.bz2, or .tgz files. Returns the selected Path or None.
    """

    CSS_PATH = "../styles/modal.tcss"
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        start_dir: Path | None = None,
        title: str = "Select Archive File",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._current_dir = (start_dir or Path.home()).resolve()
        self._title = title
        self._selected: Path | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="file-browser-dialog"):
            yield Static(
                f"[bold $primary]{self._title}[/bold $primary]",
                id="file-browser-title",
                markup=True,
            )
            yield Static(
                f"[dim]Navigate to select a .tar.gz, .zip, or .tar archive file.[/dim]",
                classes="settings-desc",
            )
            with Horizontal(id="file-browser-path-row"):
                yield Input(
                    value=str(self._current_dir),
                    placeholder="Enter path...",
                    id="file-browser-path-input",
                )
                yield Button("Go", variant="primary", id="btn-file-browser-go")
            yield Static(
                "",
                id="file-browser-location",
            )
            yield VerticalScroll(id="file-browser-list")
            yield Static("", id="file-browser-status")
            with Horizontal(id="file-browser-actions"):
                yield Button("Cancel", id="btn-file-browser-cancel")

    def on_mount(self) -> None:
        self._refresh_listing()

    def _refresh_listing(self) -> None:
        """Refresh the file listing for the current directory."""
        listing = self.query_one("#file-browser-list", VerticalScroll)
        listing.remove_children()

        location = self.query_one("#file-browser-location", Static)
        location.update(f"[bold]Location:[/bold] {self._current_dir}")

        path_input = self.query_one("#file-browser-path-input", Input)
        path_input.value = str(self._current_dir)

        self._set_status("")

        try:
            entries = sorted(self._current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            self._set_status("[red]Permission denied[/red]", error=True)
            return
        except Exception as exc:
            self._set_status(f"[red]Error: {exc}[/red]", error=True)
            return

        # Parent directory entry
        if self._current_dir.parent != self._current_dir:
            listing.mount(
                _FileEntry(
                    path=self._current_dir.parent,
                    display_name="[bold cyan]📁 ..[/bold cyan]",
                    is_dir=True,
                    index=0,
                )
            )

        index = 1
        for entry in entries:
            if entry.name.startswith("."):
                continue  # Skip hidden files

            if entry.is_dir():
                listing.mount(
                    _FileEntry(
                        path=entry,
                        display_name=f"[cyan]📁 {entry.name}/[/cyan]",
                        is_dir=True,
                        index=index,
                    )
                )
                index += 1
            elif _is_archive(entry):
                size = self._format_size(entry.stat().st_size)
                listing.mount(
                    _FileEntry(
                        path=entry,
                        display_name=f"[green]📦 {entry.name}[/green] [dim]({size})[/dim]",
                        is_dir=False,
                        index=index,
                    )
                )
                index += 1

        if index <= 1 and self._current_dir.parent == self._current_dir:
            listing.mount(Static("[dim]No archive files found[/dim]"))

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _set_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#file-browser-status", Static)
        status.update(message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id == "btn-file-browser-cancel":
            self.dismiss(None)
            return

        if btn_id == "btn-file-browser-go":
            path_input = self.query_one("#file-browser-path-input", Input)
            new_path = Path(path_input.value.strip()).expanduser().resolve()
            if new_path.is_dir():
                self._current_dir = new_path
                self._refresh_listing()
            elif new_path.is_file() and _is_archive(new_path):
                self.dismiss(new_path)
            else:
                self._set_status(f"[red]Invalid path: {new_path}[/red]", error=True)
            return

        if btn_id.startswith("btn-file-entry-"):
            idx = int(btn_id.removeprefix("btn-file-entry-"))
            self._handle_entry_click(idx)
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "file-browser-path-input":
            new_path = Path(event.input.value.strip()).expanduser().resolve()
            if new_path.is_dir():
                self._current_dir = new_path
                self._refresh_listing()
            elif new_path.is_file() and _is_archive(new_path):
                self.dismiss(new_path)
            else:
                self._set_status(f"[red]Invalid path: {new_path}[/red]", error=True)

    def _handle_entry_click(self, index: int) -> None:
        listing = self.query_one("#file-browser-list", VerticalScroll)
        for child in listing.children:
            if isinstance(child, _FileEntry) and child._index == index:
                if child._is_dir:
                    self._current_dir = child._path.resolve()
                    self._refresh_listing()
                else:
                    self.dismiss(child._path)
                return

    def action_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class _FileEntry(Static):
    """Single file/directory entry in the browser."""

    def __init__(
        self,
        path: Path,
        display_name: str,
        is_dir: bool,
        index: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._path = path
        self._display_name = display_name
        self._is_dir = is_dir
        self._index = index

    def compose(self) -> ComposeResult:
        with Horizontal(classes="file-browser-entry"):
            yield Static(self._display_name, classes="file-browser-name")
            label = "Open" if self._is_dir else "Select"
            variant = "default" if self._is_dir else "success"
            yield Button(
                label,
                variant=variant,
                id=f"btn-file-entry-{self._index}",
                classes="file-browser-entry-btn",
            )
