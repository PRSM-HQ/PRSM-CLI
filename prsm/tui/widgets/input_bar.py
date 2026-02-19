"""Input bar — prompt input with submit handling, history, and @ file completion."""

from __future__ import annotations

from pathlib import Path

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import TextArea, Button, Markdown

from prsm.shared.services.preferences import UserPreferences

class PromptInput(TextArea):
    """TextArea that fires SubmitRequested on Enter (Shift+Enter for newlines).

    Also detects '@' keystrokes for file completion and intercepts
    navigation keys when the completer is active.
    """

    class SubmitRequested(Message):
        """Fired when bare Enter is pressed."""

    class CompletionRequested(Message):
        """Fired when '@' is typed and completion should start."""

        def __init__(self, anchor: tuple[int, int]) -> None:
            self.anchor = anchor  # (row, col) where @ was inserted
            super().__init__()

    class CompletionNavigate(Message):
        """Fired for Up/Down navigation in completer."""

        def __init__(self, delta: int) -> None:
            self.delta = delta
            super().__init__()

    class CompletionConfirm(Message):
        """Fired for Tab/Enter to select completion."""

    class CompletionCancel(Message):
        """Fired for Escape to dismiss completer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._completion_active: bool = False
        self._completion_anchor: tuple[int, int] | None = None

    @property
    def completion_active(self) -> bool:
        return self._completion_active

    def set_completion_active(
        self,
        active: bool,
        anchor: tuple[int, int] | None = None,
    ) -> None:
        self._completion_active = active
        self._completion_anchor = anchor if active else None

    def watch_text(self, value: str) -> None:
        """Watch for changes in the text to apply command-mode styling."""
        if value.startswith("/"):
            self.add_class("command-mode")
        else:
            self.remove_class("command-mode")

    def get_completion_prefix(self) -> str:
        """Get the text typed after @ up to the current cursor position."""
        if self._completion_anchor is None:
            return ""
        anchor_row, anchor_col = self._completion_anchor
        cursor_row, cursor_col = self.cursor_location
        # If cursor moved to a different line, completion is void
        if cursor_row != anchor_row:
            return ""
        # Extract text from anchor+1 (after @) to cursor
        line_text = self.get_text_range(
            (anchor_row, anchor_col + 1),
            (cursor_row, cursor_col),
        )
        return line_text

    async def _on_key(self, event: events.Key) -> None:
        # ── Completion mode intercepts ──
        if self._completion_active:
            if event.key == "up":
                event.stop()
                event.prevent_default()
                self.post_message(self.CompletionNavigate(-1))
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                self.post_message(self.CompletionNavigate(1))
                return
            if event.key in ("tab", "enter"):
                event.stop()
                event.prevent_default()
                self.post_message(self.CompletionConfirm())
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(self.CompletionCancel())
                return
            # Space ends completion mode
            if event.key == "space":
                self._completion_active = False
                self._completion_anchor = None
                # Fall through to normal handling

        # ── Normal key handling ──
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.SubmitRequested())
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self._replace_via_keyboard("\n", *self.selection)
            return

        # Delegate to TextArea, then check if @ was typed
        await super()._on_key(event)

        # After TextArea processes the key, check for @ trigger
        if event.character == "@":
            row, col = self.cursor_location
            # Anchor is the position of the @ character itself (col - 1)
            anchor = (row, col - 1)
            self.post_message(self.CompletionRequested(anchor))


class InputBar(Widget):
    """Prompt input area with send button, Up/Down history, and @file completion."""

    class Submitted(Message):
        """Posted when user submits a prompt."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    class CommandSubmitted(Message):
        """Posted when user submits a slash command."""

        def __init__(self, name: str, args: list[str], raw: str) -> None:
            self.name = name
            self.args = args
            self.raw = raw
            super().__init__()

    class ModelSwitchRequested(Message):
        """Posted when user clicks the model selector button."""

    class NewSessionRequested(Message):
        """Posted when user clicks the new-session button."""

    class SearchSessionsRequested(Message):
        """Posted when user clicks the search button."""

    class SettingsRequested(Message):
        """Posted when user clicks the settings button."""

    DEFAULT_CSS = """
    InputBar .quick-action-btn {
        width: 4;
        min-width: 4;
        max-width: 4;
        background: $surface-lighten-1;
        color: $text;
        border: none;
        margin: 0 1 0 0;
        height: 3;
    }
    InputBar .quick-action-btn:hover {
        background: $accent-darken-2;
        color: $text;
    }
    InputBar .model-btn {
        min-width: 10;
        max-width: 40;
        background: $surface-lighten-1;
        color: $text-muted;
        border: none;
        margin: 0 1 0 0;
        height: 3;
    }
    InputBar .model-btn:hover {
        background: $accent-darken-2;
        color: $text;
    }
    InputBar #prompt-preview {
        height: auto;
        max-height: 12;
        margin: 1 0 0 0;
        padding: 1 2;
        border: tall $surface-lighten-1;
        background: $surface-lighten-1;
    }
    InputBar #prompt-preview.hidden {
        display: none;
    }
    """

    def __init__(self, cwd: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""
        self._cwd: Path = cwd or Path.cwd()
        self._preview_enabled: bool = True

    def compose(self) -> ComposeResult:
        from prsm.tui.widgets.file_completer import FileCompleter

        yield FileCompleter(cwd=self._cwd, id="file-completer")
        with Horizontal():
            yield Button("+", id="new-session-btn", classes="quick-action-btn")
            yield Button("🔍", id="search-session-btn", classes="quick-action-btn")
            yield Button("⚙", id="settings-btn", classes="quick-action-btn")
            yield Button("🔧 Model", id="model-btn", classes="model-btn")
            yield PromptInput(id="prompt-input")
            yield Button("Send", id="send-btn", variant="primary")
        yield Markdown("", id="prompt-preview")

    def on_mount(self) -> None:
        prefs = UserPreferences.load()
        self.set_markdown_preview_enabled(prefs.markdown_preview_enabled)
        self._update_preview()

    # ── Completion event handlers ──

    def on_prompt_input_completion_requested(
        self,
        event: PromptInput.CompletionRequested,
    ) -> None:
        """@ was typed — show the file completer."""
        from prsm.tui.widgets.file_completer import FileCompleter

        editor = self.query_one("#prompt-input", PromptInput)
        completer = self.query_one("#file-completer", FileCompleter)
        editor.set_completion_active(True, event.anchor)
        completer.show("")

    def on_prompt_input_completion_navigate(
        self,
        event: PromptInput.CompletionNavigate,
    ) -> None:
        from prsm.tui.widgets.file_completer import FileCompleter

        completer = self.query_one("#file-completer", FileCompleter)
        completer.move_highlight(event.delta)

    def on_prompt_input_completion_confirm(self) -> None:
        from prsm.tui.widgets.file_completer import FileCompleter

        completer = self.query_one("#file-completer", FileCompleter)
        completer.confirm()

    def on_prompt_input_completion_cancel(self) -> None:
        self._dismiss_completer()

    def on_file_completer_selected(self, event) -> None:
        """A file/dir was selected — insert path into the editor."""
        from prsm.tui.widgets.file_completer import FileCompleter

        editor = self.query_one("#prompt-input", PromptInput)
        anchor = editor._completion_anchor
        if anchor is not None:
            row, col = anchor
            cursor_row, cursor_col = editor.cursor_location
            start = (row, col + 1)  # After the @
            end = (cursor_row, cursor_col)
            editor.replace(event.path, start, end)

        # If directory was selected, re-trigger completion for drilling
        if event.is_dir and anchor is not None:
            # Keep the same @ anchor position
            editor.set_completion_active(True, anchor)
            prefix = event.path
            completer = self.query_one("#file-completer", FileCompleter)
            completer.show(prefix)
        else:
            self._dismiss_completer()

    def on_file_completer_dismissed(self) -> None:
        self._dismiss_completer()

    def _dismiss_completer(self) -> None:
        from prsm.tui.widgets.file_completer import FileCompleter

        editor = self.query_one("#prompt-input", PromptInput)
        completer = self.query_one("#file-completer", FileCompleter)
        editor.set_completion_active(False)
        completer.hide()

    # ── Text change watcher for filtering ──

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """As user types after @, update the completer filter."""
        from prsm.tui.widgets.file_completer import FileCompleter

        editor = self.query_one("#prompt-input", PromptInput)
        if editor.completion_active:
            prefix = editor.get_completion_prefix()
            completer = self.query_one("#file-completer", FileCompleter)
            completer.update_filter(prefix)
        self._update_preview()

    # ── Submit and history ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-btn":
            self._submit()
        elif event.button.id == "model-btn":
            self.post_message(self.ModelSwitchRequested())
        elif event.button.id == "new-session-btn":
            self.post_message(self.NewSessionRequested())
        elif event.button.id == "search-session-btn":
            self.post_message(self.SearchSessionsRequested())
        elif event.button.id == "settings-btn":
            self.post_message(self.SettingsRequested())

    def update_model_label(self, model: str) -> None:
        """Update the model button label to show the current model."""
        try:
            btn = self.query_one("#model-btn", Button)
            btn.label = f"🔧 {model}"
        except Exception:
            pass

    def on_prompt_input_submit_requested(self) -> None:
        self._submit()

    def on_key(self, event) -> None:
        editor = self.query_one("#prompt-input", PromptInput)
        if not editor.has_focus:
            return
        # Don't handle history when completer is active
        if editor.completion_active:
            return

        if event.key == "up":
            if self._history:
                event.prevent_default()
                event.stop()
                self._navigate_history(-1)
        elif event.key == "down":
            if self._history_index >= 0:
                event.prevent_default()
                event.stop()
                self._navigate_history(1)

    def _submit(self) -> None:
        # Dismiss any active completion first
        self._dismiss_completer()

        editor = self.query_one("#prompt-input", PromptInput)
        text = editor.text.strip()
        if not text:
            return

        self._history.append(text)
        self._history_index = -1
        self._draft = ""

        if text.startswith("/"):
            from prsm.shared.commands import parse_command

            cmd = parse_command(text)
            if cmd and cmd.name:
                self.post_message(self.CommandSubmitted(
                    name=cmd.name, args=cmd.args, raw=cmd.raw,
                ))
            else:
                self.post_message(self.Submitted(text))
        else:
            # Resolve @file references before submitting
            from prsm.shared.file_utils import resolve_references

            resolved_text, attachments = resolve_references(text, self._cwd)

            if attachments:
                context_parts = []
                for att in attachments:
                    tag = "directory" if att.is_directory else "file"
                    warning = " (truncated)" if att.truncated else ""
                    context_parts.append(
                        f"<{tag} path=\"{att.path}\"{warning}>\n"
                        f"{att.content}\n"
                        f"</{tag}>"
                    )
                composite = resolved_text + "\n\n" + "\n\n".join(context_parts)
                self.post_message(self.Submitted(composite))
            else:
                self.post_message(self.Submitted(text))

        editor.clear()
        editor.focus()
        self._update_preview()

    def _navigate_history(self, direction: int) -> None:
        """Navigate prompt history. direction: -1=older, 1=newer."""
        editor = self.query_one("#prompt-input", PromptInput)

        if self._history_index == -1 and direction == -1:
            self._draft = editor.text
            self._history_index = len(self._history) - 1
        elif direction == -1 and self._history_index > 0:
            self._history_index -= 1
        elif direction == 1:
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
            else:
                # Back to draft
                self._history_index = -1
                editor.clear()
                editor.insert(self._draft)
                return

        if 0 <= self._history_index < len(self._history):
            editor.clear()
            editor.insert(self._history[self._history_index])

    def set_placeholder(self, text: str) -> None:
        """Update the prompt input placeholder text."""
        editor = self.query_one("#prompt-input", PromptInput)
        editor.placeholder = text

    def focus_input(self) -> None:
        self.query_one("#prompt-input", PromptInput).focus()

    def set_text(self, text: str) -> None:
        """Pre-fill the input with text (e.g. for resend/edit)."""
        editor = self.query_one("#prompt-input", PromptInput)
        editor.clear()
        editor.insert(text)
        editor.focus()
        self._update_preview()

    def set_markdown_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = bool(enabled)
        preview = self.query_one("#prompt-preview", Markdown)
        if self._preview_enabled:
            preview.remove_class("hidden")
        else:
            preview.add_class("hidden")
        self._update_preview()

    def _update_preview(self) -> None:
        preview = self.query_one("#prompt-preview", Markdown)
        if not self._preview_enabled:
            preview.update("")
            return
        editor = self.query_one("#prompt-input", PromptInput)
        preview.update(editor.text or "")
