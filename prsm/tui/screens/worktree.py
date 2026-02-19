"""Worktree management modals — list, create, remove, and switch worktrees."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from prsm.shared.services.project import ProjectManager, WorktreeInfo


class WorktreeListScreen(ModalScreen[str | None]):
    """Modal dialog listing all worktrees with actions.

    Returns:
        "switch:<path>" to switch to a worktree,
        "remove:<path>" to remove a worktree,
        "create" to open the create dialog,
        or None if dismissed.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, cwd: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd or Path.cwd()
        self._worktrees: list[WorktreeInfo] = []
        self._current_root: Path | None = None

    def compose(self) -> ComposeResult:
        self._worktrees = ProjectManager.list_worktrees(self._cwd)
        self._current_root = ProjectManager.get_worktree_root(self._cwd)

        with Vertical(id="worktree-dialog"):
            yield Static(
                "[bold $primary]Git Worktrees[/bold $primary]",
                id="worktree-title",
            )
            if not self._worktrees:
                yield Static(
                    "[dim]No worktrees found. Is this a git repository?[/dim]",
                    id="worktree-empty",
                )
            else:
                with VerticalScroll(id="worktree-list"):
                    for wt in self._worktrees:
                        is_current = (
                            self._current_root is not None
                            and Path(wt.path).resolve()
                            == self._current_root.resolve()
                        )
                        yield WorktreeEntry(wt, is_current=is_current)
            with Horizontal(id="worktree-actions"):
                yield Button(
                    "New Worktree",
                    variant="success",
                    id="btn-wt-create",
                )
                yield Button("Close", variant="default", id="btn-wt-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "btn-wt-close":
            self.dismiss(None)
        elif btn_id == "btn-wt-create":
            self.dismiss("create")
        elif btn_id.startswith("btn-wt-switch-"):
            idx = int(btn_id.removeprefix("btn-wt-switch-"))
            if 0 <= idx < len(self._worktrees):
                self.dismiss(f"switch:{self._worktrees[idx].path}")
        elif btn_id.startswith("btn-wt-remove-"):
            idx = int(btn_id.removeprefix("btn-wt-remove-"))
            if 0 <= idx < len(self._worktrees):
                self.dismiss(f"remove:{self._worktrees[idx].path}")

    def key_escape(self) -> None:
        self.dismiss(None)


class WorktreeEntry(Static):
    """A single worktree row in the list."""

    def __init__(
        self, wt: WorktreeInfo, is_current: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.wt = wt
        self.is_current = is_current

    def compose(self) -> ComposeResult:
        wt = self.wt
        # Determine branch display
        if wt.branch:
            # refs/heads/main -> main
            branch_display = wt.branch.removeprefix("refs/heads/")
        elif wt.detached:
            branch_display = f"(detached {wt.head[:8]})"
        elif wt.bare:
            branch_display = "(bare)"
        else:
            branch_display = wt.head[:8]

        current_marker = " [bold green]\u25c0 current[/bold green]" if self.is_current else ""
        lock_marker = " [yellow]\U0001f512 locked[/yellow]" if wt.locked else ""
        lock_reason = f" [dim]({wt.lock_reason})[/dim]" if wt.lock_reason else ""

        # Find the index in the parent's worktree list for button IDs
        parent = self.parent
        idx = 0
        if parent is not None:
            for i, child in enumerate(parent.children):
                if child is self:
                    idx = i
                    break

        with Vertical(classes="worktree-entry"):
            yield Static(
                f"[bold]{branch_display}[/bold]{current_marker}{lock_marker}{lock_reason}",
            )
            yield Static(
                f"  [dim]{wt.path}[/dim]",
            )
            if not self.is_current:
                with Horizontal(classes="worktree-entry-buttons"):
                    yield Button(
                        "Switch",
                        variant="primary",
                        id=f"btn-wt-switch-{idx}",
                    )
                    yield Button(
                        "Remove",
                        variant="error",
                        id=f"btn-wt-remove-{idx}",
                    )


class WorktreeCreateScreen(ModalScreen[dict | None]):
    """Modal dialog for creating a new worktree.

    Returns a dict with keys: path, branch, new_branch
    or None if dismissed.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, cwd: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd or Path.cwd()

    def compose(self) -> ComposeResult:
        with Vertical(id="worktree-create-dialog"):
            yield Static(
                "[bold $primary]Create Worktree[/bold $primary]",
                id="worktree-create-title",
            )
            yield Label("Path for new worktree (absolute or relative to repo):")
            yield Input(
                placeholder="../my-feature-branch",
                id="wt-path-input",
            )
            yield Label("Branch name (leave empty to detach at HEAD):")
            yield Input(
                placeholder="feature/my-branch",
                id="wt-branch-input",
            )
            yield Static(
                "[dim]If the branch doesn't exist, it will be created.[/dim]",
                id="wt-create-hint",
            )
            with Horizontal(id="worktree-create-buttons"):
                yield Button(
                    "Create",
                    variant="success",
                    id="btn-wt-do-create",
                )
                yield Button(
                    "Cancel",
                    variant="default",
                    id="btn-wt-cancel-create",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-wt-cancel-create":
            self.dismiss(None)
        elif event.button.id == "btn-wt-do-create":
            self._do_create()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Allow Enter in the branch input to submit the form."""
        self._do_create()

    def _do_create(self) -> None:
        path_input = self.query_one("#wt-path-input", Input)
        branch_input = self.query_one("#wt-branch-input", Input)

        path = path_input.value.strip()
        branch = branch_input.value.strip()

        if not path:
            path_input.add_class("error")
            return

        result: dict = {"path": path}

        if branch:
            # Check if the branch already exists
            existing = self._branch_exists(branch)
            if existing:
                result["branch"] = branch
            else:
                result["new_branch"] = branch
        # If no branch specified, result has only path -> detached HEAD

        self.dismiss(result)

    def _branch_exists(self, branch: str) -> bool:
        """Check if a git branch exists."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def key_escape(self) -> None:
        self.dismiss(None)


class WorktreeRemoveScreen(ModalScreen[str | None]):
    """Confirmation dialog for removing a worktree.

    Returns "remove" for normal removal, "force" for forced removal,
    or None if cancelled.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, worktree_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worktree_path = worktree_path

    def compose(self) -> ComposeResult:
        with Vertical(id="worktree-remove-dialog"):
            yield Static(
                "[bold $warning]Remove Worktree?[/bold $warning]",
                id="worktree-remove-title",
            )
            safe_path = self.worktree_path.replace("[", "\\[")
            yield Static(
                f"Are you sure you want to remove the worktree at:\n"
                f"[cyan]{safe_path}[/cyan]",
            )
            yield Static(
                "[dim]This removes the worktree directory and its git link. "
                "Uncommitted changes will be lost.[/dim]",
            )
            with Horizontal(id="worktree-remove-buttons"):
                yield Button(
                    "Remove",
                    variant="error",
                    id="btn-wt-do-remove",
                )
                yield Button(
                    "Force Remove",
                    variant="warning",
                    id="btn-wt-do-force",
                )
                yield Button(
                    "Cancel",
                    variant="default",
                    id="btn-wt-cancel-remove",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result_map = {
            "btn-wt-do-remove": "remove",
            "btn-wt-do-force": "force",
            "btn-wt-cancel-remove": None,
        }
        self.dismiss(result_map.get(event.button.id))

    def key_escape(self) -> None:
        self.dismiss(None)


class WorktreeSwitchScreen(ModalScreen[str | None]):
    """Confirmation dialog for switching to a different worktree.

    Switching restarts PRSM in the new worktree directory.

    Returns "switch" to confirm, or None if cancelled.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, worktree_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worktree_path = worktree_path

    def compose(self) -> ComposeResult:
        with Vertical(id="worktree-switch-dialog"):
            yield Static(
                "[bold $primary]Switch Worktree[/bold $primary]",
                id="worktree-switch-title",
            )
            safe_path = self.worktree_path.replace("[", "\\[")
            yield Static(
                f"Switch to worktree at:\n[cyan]{safe_path}[/cyan]\n\n"
                "This will save your current session and restart PRSM "
                "in the new working directory.",
            )
            with Horizontal(id="worktree-switch-buttons"):
                yield Button(
                    "Switch",
                    variant="primary",
                    id="btn-wt-do-switch",
                )
                yield Button(
                    "Cancel",
                    variant="default",
                    id="btn-wt-cancel-switch",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result_map = {
            "btn-wt-do-switch": "switch",
            "btn-wt-cancel-switch": None,
        }
        self.dismiss(result_map.get(event.button.id))

    def key_escape(self) -> None:
        self.dismiss(None)
