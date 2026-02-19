"""Conversation view — scrollable message area with per-agent switching and streaming."""

from __future__ import annotations

import re
from typing import AsyncIterator

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.widget import Widget
from textual.widgets import Button, Markdown, Static

from prsm.shared.models.message import Message, MessageRole
from prsm.shared.models.session import Session


def _esc(text: str) -> str:
    """Escape Rich markup characters in dynamic content."""
    return text.replace("[", "\\[")


def _has_closed_fenced_code_block(text: str) -> bool:
    """Return True when text contains at least one closed triple-backtick fence."""
    return re.search(r"```[\s\S]*?```", text) is not None


class QueuedPromptWidget(Widget):
    """A queued prompt pinned to the bottom of the conversation.

    Visually distinct from normal user messages — uses yellow styling
    with a plan index prefix (e.g. "Plan 1", "Plan 2") and a cancel
    button. The widget is not clickable for resend and does not look
    like a regular user message.
    """

    class Cancelled(TextualMessage):
        """Fired when the user cancels a queued prompt."""

        def __init__(self, plan_index: int, prompt: str) -> None:
            self.plan_index = plan_index
            self.prompt = prompt
            super().__init__()

    DEFAULT_CSS = """
    QueuedPromptWidget {
        background: #3d3200;
        margin: 1 0;
        padding: 1 2;
        border-left: thick #e6a817;
        height: auto;
    }

    QueuedPromptWidget .queued-header {
        height: auto;
        margin: 0 0 1 0;
    }

    QueuedPromptWidget .queued-cancel-btn {
        dock: right;
        min-width: 10;
        max-height: 1;
        background: #5c4a00;
        color: #ffd54f;
        border: none;
        margin: 0 0 0 1;
    }

    QueuedPromptWidget .queued-cancel-btn:hover {
        background: #7a1a1a;
        color: #ff6b6b;
    }
    """

    def __init__(self, plan_index: int, prompt: str, **kwargs) -> None:
        self.plan_index = plan_index
        self.prompt = prompt
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="queued-header"):
            yield Static(
                f"[bold #e6a817]📋 Plan {self.plan_index}[/bold #e6a817]  "
                f"[dim #ffd54f]queued[/dim #ffd54f]",
                markup=True,
            )
            yield Button("✕ Cancel", classes="queued-cancel-btn", variant="default")
        yield Static(
            f"[#ffd54f]{_esc(self.prompt)}[/#ffd54f]",
            markup=True,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle cancel button click."""
        event.stop()
        self.post_message(self.Cancelled(
            plan_index=self.plan_index,
            prompt=self.prompt,
        ))


class ModelSwitchIndicator(Widget):
    """Visual separator in the conversation indicating a model switch.

    Shows a horizontal divider with the old and new model names,
    making it clear that subsequent messages use a different model.
    """

    DEFAULT_CSS = """
    ModelSwitchIndicator {
        height: auto;
        margin: 1 0;
        padding: 0 2;
    }

    ModelSwitchIndicator .model-switch-line {
        height: 1;
        width: 100%;
        text-align: center;
    }
    """

    def __init__(
        self,
        old_model: str,
        new_model: str,
        **kwargs,
    ) -> None:
        self.old_model = old_model
        self.new_model = new_model
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(
            f"[dim]───── [/dim]"
            f"[bold #e6a817]🔄 Model switched[/bold #e6a817]"
            f"[dim] ── [/dim]"
            f"[dim italic]{_esc(self.old_model)}[/dim italic]"
            f"[dim] → [/dim]"
            f"[bold cyan]{_esc(self.new_model)}[/bold cyan]"
            f"[dim] ─────[/dim]",
            classes="model-switch-line",
            markup=True,
        )


class RunStoppedIndicator(Widget):
    """Visual separator indicating a run was explicitly stopped."""

    DEFAULT_CSS = """
    RunStoppedIndicator {
        height: auto;
        margin: 1 0;
        padding: 0 2;
    }

    RunStoppedIndicator .run-stopped-line {
        height: 1;
        width: 100%;
        text-align: center;
    }
    """

    def __init__(self, reason: str = "Stopped by user", **kwargs) -> None:
        self.reason = reason
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(
            f"[dim]───── [/dim]"
            f"[bold red]⏹ Run stopped[/bold red]"
            f"[dim] ── [/dim]"
            f"[dim italic]{_esc(self.reason)}[/dim italic]"
            f"[dim] ─────[/dim]",
            classes="run-stopped-line",
            markup=True,
        )


class MessageWidget(Widget):
    """A single rendered message with timestamp and role badge.

    User and assistant messages are rendered using Textual's ``Markdown``
    widget so that fenced code blocks, inline code, and other Markdown
    constructs are sized and displayed correctly.  System messages remain
    plain ``Static`` text.

    User messages are clickable — clicking them emits a ResendRequested
    event that allows reverting the conversation to that point and
    re-submitting the prompt.
    """

    class ResendRequested(TextualMessage):
        """Fired when a user clicks on a previously sent user message."""

        def __init__(
            self,
            message: Message,
            agent_id: str,
            message_index: int,
        ) -> None:
            self.msg = message
            self.agent_id = agent_id
            self.message_index = message_index
            super().__init__()

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
    }
    MessageWidget .msg-header {
        height: auto;
    }
    MessageWidget .msg-body {
        height: auto;
        margin: 0;
        padding: 0;
    }
    """

    def __init__(self, message: Message, message_index: int = -1, **kwargs) -> None:
        self.message = message
        self._message_index = message_index
        css_class = {
            MessageRole.USER: "message-user",
            MessageRole.ASSISTANT: "message-assistant",
            MessageRole.SYSTEM: "message-system",
        }.get(message.role, "message-system")
        # User messages get a "clickable" class for hover styling
        if message.role == MessageRole.USER and message.agent_id != "__pending__":
            css_class += " message-clickable"
        super().__init__(classes=css_class, **kwargs)

    def compose(self) -> ComposeResult:
        msg = self.message
        ts = msg.timestamp.strftime("%H:%M:%S")
        timestamp_str = f"[dim]{ts}[/dim]"

        if msg.role == MessageRole.USER:
            resend_hint = (
                "  [dim italic]click to edit and resend[/dim italic]"
                if msg.agent_id != "__pending__"
                else ""
            )
            yield Static(
                f"[bold $primary]You[/bold $primary] {timestamp_str}{resend_hint}",
                classes="msg-header",
                markup=True,
            )
            yield Markdown(msg.content, classes="msg-body")

        elif msg.role == MessageRole.ASSISTANT:
            yield Static(
                f"[bold cyan]Assistant[/bold cyan] {timestamp_str}",
                classes="msg-header",
                markup=True,
            )
            yield Markdown(msg.content, classes="msg-body")

        elif msg.role == MessageRole.SYSTEM:
            yield Static(
                f"[dim]System[/dim] {timestamp_str} {_esc(msg.content)}",
                markup=True,
            )

        else:
            yield Static(_esc(msg.content), markup=True)

    def on_click(self, event: events.Click) -> None:
        """When a user message is left-clicked, emit ResendRequested."""
        if event.button != 1:
            return
        if self.message.role == MessageRole.USER and self.message.agent_id != "__pending__":
            event.stop()
            self.post_message(self.ResendRequested(
                message=self.message,
                agent_id=self.message.agent_id,
                message_index=self._message_index,
            ))


class ToolCallWidget(Static):
    """A collapsible tool call widget with rich per-tool-type formatting.

    Collapsed by default, click to expand. Uses the formatter registry
    from ``prsm.shared.formatters.tool_call`` to produce structured
    previews (diffs for Edit, command blocks for Bash, etc.).

    File paths are clickable — clicking the file name in the collapsed or
    expanded view opens the file in the user's editor.
    """

    def __init__(self, message: Message, **kwargs) -> None:
        self.message = message
        self._expanded = False
        self._formatted = self._build_formatted()
        content = self._format_collapsed()
        super().__init__(content, classes="tool-call", markup=True, **kwargs)

    def _build_formatted(self):
        """Build the FormattedToolCall IR from the message's first tool call."""
        from prsm.shared.formatters.tool_call import format_tool_call

        tc = self.message.tool_calls[0] if self.message.tool_calls else None
        if not tc:
            return None
        return format_tool_call(tc.name, tc.arguments, tc.result, tc.success)

    def _get_status(self) -> str:
        """Derive the status string from the first tool call."""
        tc = self.message.tool_calls[0] if self.message.tool_calls else None
        if not tc:
            return "pending"
        if bool(getattr(tc, "_prsm_pending", False)):
            return "pending"
        if tc.result is None:
            return "pending"
        return "done" if tc.success else "error"

    def _format_collapsed(self) -> str:
        """One-line summary with smart per-tool preview."""
        from prsm.shared.formatters.tool_call import render_collapsed_rich

        status = self._get_status()
        status_icon = ""
        status_color = "dim"

        if status == "done":
            status_icon = "✔"  # Checkmark
            status_color = "green"
        elif status == "error":
            status_icon = "✘"  # Cross
            status_color = "red"
        elif status == "pending":
            status_icon = "⏳" # Hourglass
            status_color = "yellow"

        if not self._formatted:
            # Prepend icon and color even if no formatted content
            return f"[{status_color}]{status_icon}[/{status_color}] [dim]\\[\u25b6][/dim] Tool call"

        # Prepend icon and color to the existing markup
        markup = render_collapsed_rich(self._formatted, status)
        return f"[{status_color}]{status_icon}[/{status_color}] {self._inject_file_link_collapsed(markup)}"

    def _format_expanded(self):
        """Full detail view with structured sections.

        Returns a Rich markup string for most tools, or a Rich Group
        renderable when the tool has markdown content (e.g. task_complete
        summaries) so that the summary is rendered as proper markdown.
        """
        from prsm.shared.formatters.tool_call import render_expanded_rich

        if not self._formatted:
            return "[dim]\\[\u25bc][/dim] Tool call"
        ts = self.message.timestamp.strftime("%H:%M:%S")

        # Check if any section has markdown content that should be
        # rendered as real markdown instead of plain escaped text.
        markdown_sections = []
        if self._formatted.sections:
            for section in self._formatted.sections:
                if (
                    section.kind == "result_block"
                    and isinstance(section.content, dict)
                    and section.content.get("markdown")
                    and section.content.get("text")
                ):
                    markdown_sections.append(section)

        if markdown_sections:
            # Build expanded view with markdown sections rendered via RichMarkdown.
            # The header and non-markdown sections use the standard Rich markup
            # renderer (which skips markdown-flagged result_blocks), and each
            # markdown section is appended as a RichMarkdown renderable.
            markup = render_expanded_rich(
                self._formatted, self._get_status(), ts,
            )
            markup = self._inject_file_link_expanded(markup)
            parts: list = [Text.from_markup(markup)]
            for section in markdown_sections:
                parts.append(RichMarkdown(section.content["text"]))
            return Group(*parts)

        markup = render_expanded_rich(self._formatted, self._get_status(), ts)
        return self._inject_file_link_expanded(markup)

    def _inject_file_link_collapsed(self, markup: str) -> str:
        """Wrap the file-path summary in a clickable @click action link."""
        if not self._formatted or not self._formatted.file_path:
            return markup
        summary = _esc(self._formatted.summary) if self._formatted.summary else ""
        if not summary:
            return markup
        # Replace the dim summary span with a clickable link
        dim_summary = f"[dim]{summary}[/dim]"
        link_summary = f"[@click=open_file][bold underline]{summary}[/bold underline][/]"
        return markup.replace(dim_summary, link_summary, 1)

    def _inject_file_link_expanded(self, markup: str) -> str:
        """Wrap expanded path sections in clickable @click action links."""
        if not self._formatted or not self._formatted.file_path:
            return markup
        file_path = self._formatted.file_path
        escaped_path = _esc(file_path)
        # The expanded renderer outputs path sections as:
        #   [underline]{path}[/underline]
        old = f"[underline]{escaped_path}[/underline]"
        new = f"[@click=open_file][bold underline]{escaped_path}[/bold underline][/]"
        return markup.replace(old, new)

    def action_open_file(self) -> None:
        """Open the tool call's file in the user's editor."""
        import os
        import subprocess

        if not self._formatted or not self._formatted.file_path:
            return

        file_path = self._formatted.file_path
        if not os.path.exists(file_path):
            self.notify(f"File not found: {file_path}", severity="warning")
            return

        # Try editors in order: $EDITOR, code, xdg-open
        editor = os.environ.get("EDITOR", "")
        if editor:
            cmd = [editor, file_path]
        else:
            # Default: try 'code' (VS Code), then 'xdg-open'
            import shutil
            if shutil.which("code"):
                cmd = ["code", file_path]
            elif shutil.which("xdg-open"):
                cmd = ["xdg-open", file_path]
            else:
                self.notify(f"No editor found. Set $EDITOR env var.", severity="warning")
                return

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            self.notify(f"Failed to open file: {exc}", severity="error")

    def on_click(self) -> None:
        """Toggle between collapsed and expanded views."""
        self._expanded = not self._expanded
        if self._expanded:
            self.add_class("expanded")
            self.update(self._format_expanded())
        else:
            self.remove_class("expanded")
            self.update(self._format_collapsed())

    def refresh_content(self) -> None:
        """Re-render the widget with current message data (e.g. result arrived)."""
        self._formatted = self._build_formatted()
        if self._expanded:
            self.update(self._format_expanded())
        else:
            self.update(self._format_collapsed())


class ConversationView(Widget):
    """Scrollable conversation pane that switches between agent histories."""

    def __init__(self, session: Session | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session or Session()
        self._current_agent_id: str | None = None
        # Buffer for streaming chunks from non-active agents
        self._stream_buffers: dict[str, list[str]] = {}
        # Track pending tool calls: tool_id → (Message, ToolCallWidget)
        self._pending_tool_calls: dict[str, tuple[Message, ToolCallWidget | None]] = {}
        self._stderr_prefix_emitted: set[str] = set()

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="message-container")

    # ── Scroll helpers ──

    def is_near_bottom(self) -> bool:
        """Check if the scroll container is at or near the bottom."""
        container = self._message_container()
        if container is None:
            return False
        if container.max_scroll_y == 0:
            return True
        return container.scroll_y >= container.max_scroll_y - 3

    def _smart_scroll(self) -> None:
        """Scroll to bottom only if already near the bottom."""
        if self.is_near_bottom():
            container = self._message_container()
            if container is None:
                return
            container.scroll_end(animate=False)

    def _message_container(self) -> VerticalScroll | None:
        """Return the message container when mounted."""
        try:
            return self.query_one("#message-container", VerticalScroll)
        except NoMatches:
            return None

    def _first_queued_widget(self) -> QueuedPromptWidget | None:
        """Return the first QueuedPromptWidget in the container, or None."""
        container = self._message_container()
        if container is None:
            return None
        try:
            return container.query(QueuedPromptWidget).first()
        except NoMatches:
            return None

    def mount_before_queued(self, widget: Widget) -> None:
        """Mount a widget in the conversation, inserting it before any
        QueuedPromptWidget instances so queued plans stay pinned to the
        bottom of the chat window."""
        container = self._message_container()
        if container is None:
            return
        first_queued = self._first_queued_widget()
        if first_queued is not None:
            container.mount(widget, before=first_queued)
        else:
            container.mount(widget)

    # ── Agent switching ──

    def show_agent(self, agent_id: str, *, force: bool = False) -> None:
        """Switch to displaying a specific agent's conversation.

        Args:
            agent_id: The agent whose messages to display.
            force: If True, rebuild even if already showing this agent.
                   Needed after session truncation (resend) where the
                   message list changed but the active agent didn't.
        """
        if agent_id == self._current_agent_id and not force:
            return
        self._current_agent_id = agent_id
        self._rebuild()

        # Replay buffered stream chunks if any exist
        if agent_id in self._stream_buffers:
            buffered = self._stream_buffers.pop(agent_id)
            if buffered:
                full_text = "".join(buffered)
                # Add as a static message (stream already completed or in progress)
                self.add_assistant_message(agent_id, full_text)

    def _rebuild(self) -> None:
        """Re-render all messages for the current agent from session store."""
        container = self._message_container()
        if container is None:
            return
        container.remove_children()
        if self._current_agent_id is None:
            return
        for idx, msg in enumerate(self.session.get_messages(self._current_agent_id)):
            if msg.role == MessageRole.TOOL:
                container.mount(ToolCallWidget(msg))
            else:
                container.mount(MessageWidget(msg, message_index=idx))
        container.scroll_end(animate=False)

    def append_message(self, message: Message) -> None:
        """Append a message. If it belongs to the active agent, render it live.

        Inserts before any QueuedPromptWidget instances to keep queued
        plans pinned to the bottom.
        """
        if message.agent_id == self._current_agent_id:
            # Compute the message index (position in the agent's message list)
            msg_index = len(self.session.get_messages(message.agent_id)) - 1
            self.mount_before_queued(MessageWidget(message, message_index=msg_index))
            self._smart_scroll()

    # ── Streaming ──

    async def stream_assistant_message(
        self, agent_id: str, chunks: AsyncIterator[str],
    ) -> Message:
        """Stream an assistant response token-by-token using Markdown.get_stream().

        Mounts a Markdown widget, anchors the scroll container to auto-follow
        (only if already near bottom), and feeds chunks through MarkdownStream.
        When done, stores the final content as a Message in the session.

        Returns the completed Message.
        """
        container = self.query_one("#message-container")

        # Mount a streaming Markdown widget with assistant styling,
        # inserting before any queued prompt widgets
        md = Markdown("", classes="message-assistant")
        self.mount_before_queued(md)

        # Only anchor (auto-follow) if user is already near the bottom
        if self.is_near_bottom():
            container.anchor()

        # Stream chunks into the Markdown widget
        stream = Markdown.get_stream(md)
        accumulated = []
        try:
            async for chunk in chunks:
                accumulated.append(chunk)
                await stream.write(chunk)
        finally:
            await stream.stop()

        # Store the final content in the session
        full_text = "".join(accumulated)
        msg = self.session.add_message(agent_id, MessageRole.ASSISTANT, full_text)
        return msg

    # ── Convenience methods that add to session + render ──

    def add_user_message(
        self,
        agent_id: str,
        text: str,
        *,
        snapshot_id: str | None = None,
    ) -> Message:
        msg = self.session.add_message(
            agent_id,
            MessageRole.USER,
            text,
            snapshot_id=snapshot_id,
        )
        self.append_message(msg)
        return msg

    def add_assistant_message(self, agent_id: str, text: str) -> Message:
        msg = self.session.add_message(agent_id, MessageRole.ASSISTANT, text)
        self.append_message(msg)
        return msg

    def add_system_message(self, agent_id: str, text: str) -> Message:
        msg = self.session.add_message(agent_id, MessageRole.SYSTEM, text)
        self.append_message(msg)
        return msg

    def add_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: str,
        result: str,
        success: bool = True,
        tool_id: str | None = None,
    ) -> Message:
        from prsm.shared.models.message import ToolCall

        call_id = tool_id or f"{tool_name}-{len(self.session.get_messages(agent_id))}"
        tc = ToolCall(
            id=call_id,
            name=tool_name,
            arguments=args,
            result=result if result else None,
            success=success,
        )
        setattr(tc, "_prsm_pending", not bool(result))
        msg = self.session.add_message(
            agent_id, MessageRole.TOOL, "", tool_calls=[tc]
        )
        # Render and track if result is pending
        widget = None
        if msg.agent_id == self._current_agent_id:
            widget = ToolCallWidget(msg)
            self.mount_before_queued(widget)
            self._smart_scroll()

        if not result:
            # Result will arrive later via update_tool_result
            self._pending_tool_calls[call_id] = (msg, widget)

        return msg

    def update_tool_result(
        self,
        tool_id: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """Update a pending tool call with its result."""
        entry = self._pending_tool_calls.pop(tool_id, None)
        if entry is None:
            return
        self._stderr_prefix_emitted.discard(tool_id)

        msg, widget = entry
        # Update the ToolCall in the message
        for tc in msg.tool_calls:
            if tc.id == tool_id:
                existing = str(tc.result or "")
                completed = str(result or "")
                streamed_summary = "Output streamed previously." in completed
                if not (streamed_summary and existing.strip()):
                    tc.result = result
                tc.success = not is_error
                setattr(tc, "_prsm_pending", False)
                break

        # Re-render the widget if it's mounted
        if widget is not None and widget.is_mounted:
            widget.refresh_content()

    def update_bash_live_output(
        self,
        agent_id: str,
        tool_id: str,
        delta: str,
        stream: str = "stdout",
    ) -> None:
        """Append live bash output to an in-flight tool call widget."""
        if not delta:
            return

        entry = None
        resolved_tool_id = tool_id
        if tool_id:
            entry = self._pending_tool_calls.get(tool_id)

        if entry is None:
            for pending_id, pending_entry in reversed(list(self._pending_tool_calls.items())):
                msg, _widget = pending_entry
                tc = msg.tool_calls[0] if msg.tool_calls else None
                if (
                    msg.agent_id == agent_id
                    and tc is not None
                    and str(tc.name).lower() in {"bash", "run_bash"}
                ):
                    entry = pending_entry
                    resolved_tool_id = pending_id
                    break

        if entry is None:
            return

        msg, widget = entry
        for tc in msg.tool_calls:
            if resolved_tool_id and tc.id != resolved_tool_id:
                continue
            if str(tc.name).lower() not in {"bash", "run_bash"}:
                continue
            if tc.result is None:
                tc.result = ""
            if stream == "stderr" and resolved_tool_id:
                if resolved_tool_id not in self._stderr_prefix_emitted:
                    tc.result += "\n\nSTDERR:\n"
                    self._stderr_prefix_emitted.add(resolved_tool_id)
            tc.result += delta
            break

        if widget is not None and widget.is_mounted:
            widget.refresh_content()

    def buffer_stream_chunk(self, agent_id: str, text: str) -> None:
        """Buffer a stream chunk for a non-active agent."""
        if agent_id not in self._stream_buffers:
            self._stream_buffers[agent_id] = []
        self._stream_buffers[agent_id].append(text)

    def flush_stream_buffer(self, agent_id: str) -> None:
        """Flush buffered stream chunks to session as a message."""
        buffered = self._stream_buffers.pop(agent_id, [])
        if buffered:
            full_text = "".join(buffered)
            self.session.add_message(agent_id, MessageRole.ASSISTANT, full_text)

    def clear(self, agent_id: str | None = None) -> None:
        """Clear the message display (UI only — session data is preserved)."""
        if agent_id is None or agent_id == self._current_agent_id:
            container = self.query_one("#message-container")
            container.remove_children()
