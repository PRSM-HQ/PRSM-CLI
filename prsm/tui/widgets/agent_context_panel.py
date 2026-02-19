"""Agent context panel — side panel showing full agent details and conversation."""

from __future__ import annotations

import re

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from prsm.shared.models.agent import AgentNode
from prsm.shared.models.message import Message as ChatMessage, MessageRole
from prsm.shared.models.session import Session
from prsm.adapters.agent_adapter import ROLE_DISPLAY, STATE_DISPLAY, STATE_ICONS


def _esc(text: str) -> str:
    """Escape Rich markup characters in dynamic content."""
    return text.replace("[", "\\[")


def _has_closed_fenced_code_block(text: str) -> bool:
    """Return True when text contains at least one closed triple-backtick fence."""
    return re.search(r"```[\s\S]*?```", text) is not None


class AgentContextPanel(Widget):
    """Side panel showing comprehensive agent context.

    Displays:
    - Agent metadata (name, role, model, state, parent)
    - Agent hierarchy tree (orchestrator -> worker -> expert chain)
    - Full conversation history (messages, tool calls, responses)
    """

    DEFAULT_CSS = """
    AgentContextPanel {
        dock: right;
        width: 60;
        min-width: 40;
        max-width: 80;
        border-left: tall $surface-lighten-1;
        background: $surface;
        display: none;
        layout: vertical;
    }

    AgentContextPanel.visible {
        display: block;
    }

    AgentContextPanel #acp-header {
        height: auto;
        padding: 1 2;
        background: $primary-darken-3;
    }

    AgentContextPanel #acp-header-row {
        height: 1;
    }

    AgentContextPanel #acp-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }

    AgentContextPanel #acp-close-btn {
        min-width: 3;
        width: 3;
        height: 1;
        background: transparent;
        color: $text-muted;
        border: none;
    }

    AgentContextPanel #acp-close-btn:hover {
        color: $error;
    }

    AgentContextPanel #acp-metadata {
        height: auto;
        padding: 1 2;
        border-bottom: solid $surface-lighten-1;
    }

    AgentContextPanel .acp-meta-row {
        height: 1;
    }

    AgentContextPanel .acp-meta-label {
        width: 12;
        color: $text-muted;
    }

    AgentContextPanel .acp-meta-value {
        width: 1fr;
    }

    AgentContextPanel #acp-hierarchy {
        height: auto;
        max-height: 10;
        padding: 0 2;
        border-bottom: solid $surface-lighten-1;
    }

    AgentContextPanel #acp-hierarchy-title {
        text-style: bold;
        color: $accent;
        padding: 1 0 0 0;
    }

    AgentContextPanel #acp-hierarchy-tree {
        height: auto;
        max-height: 8;
        padding: 0 0 1 0;
    }

    AgentContextPanel #acp-conversation {
        height: 1fr;
        padding: 0;
    }

    AgentContextPanel #acp-conv-title {
        text-style: bold;
        color: $accent;
        padding: 1 2 0 2;
    }

    AgentContextPanel #acp-messages {
        height: 1fr;
        padding: 0 1;
    }

    AgentContextPanel .acp-msg-user {
        background: $primary-darken-3;
        margin: 1 0;
        padding: 1 2;
        border-left: thick $primary;
    }

    AgentContextPanel .acp-msg-assistant {
        margin: 1 0;
        padding: 1 2;
    }

    AgentContextPanel .acp-msg-system {
        color: $text-muted;
        margin: 0 0;
        padding: 0 2;
    }

    AgentContextPanel .acp-msg-tool {
        margin: 0 0 0 2;
        padding: 0 2;
        color: $text-muted;
    }

    AgentContextPanel .acp-msg-tool:hover {
        background: $surface-lighten-1;
    }

    AgentContextPanel .acp-msg-tool.expanded {
        background: $surface-darken-1;
        padding: 1 2;
        border: round $accent-darken-1;
        color: $text;
    }

    AgentContextPanel .acp-msg-highlighted {
        background: $warning-darken-2 !important;
        border: thick $warning;
        padding: 1 2;
        margin: 1 0;
    }

    AgentContextPanel .acp-prompt-section {
        height: auto;
        padding: 1 2;
        border-bottom: solid $surface-lighten-1;
    }

    AgentContextPanel .acp-prompt-label {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    AgentContextPanel .acp-prompt-content {
        max-height: 6;
        overflow-y: auto;
        background: $surface-darken-1;
        padding: 1;
        color: $text-muted;
    }

    AgentContextPanel #acp-footer {
        height: 3;
        dock: bottom;
        padding: 0 2;
        border-top: solid $surface-lighten-1;
    }

    AgentContextPanel #acp-footer Button {
        margin: 0 1 0 0;
        min-width: 16;
    }
    """

    class CloseRequested(Message):
        """Posted when the user clicks the close button."""

    class NavigateToAgent(Message):
        """Posted when the user clicks an agent in the hierarchy tree."""

        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_id: str | None = None
        self._session: Session | None = None
        self._highlight_message_index: int | None = None
        self._highlight_tool_call_id: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="acp-header"):
            with Horizontal(id="acp-header-row"):
                yield Static("Agent Context", id="acp-title")
                yield Button("\u2715", id="acp-close-btn")

        yield Vertical(id="acp-metadata")
        yield Vertical(id="acp-hierarchy")
        yield Vertical(id="acp-conversation")
        with Horizontal(id="acp-footer"):
            yield Button("Switch To", variant="primary", id="acp-switch-btn")
            yield Button("Close", variant="default", id="acp-close-footer-btn")

    def show_agent(
        self,
        agent_id: str,
        session: Session,
        highlight_message_index: int | None = None,
        highlight_tool_call_id: str | None = None,
    ) -> None:
        """Populate the panel with data for the given agent.

        Args:
            agent_id: The agent to display
            session: Current session
            highlight_message_index: Optional message index to highlight
            highlight_tool_call_id: Optional tool call ID to highlight
        """
        self._agent_id = agent_id
        self._session = session
        self._highlight_message_index = highlight_message_index
        self._highlight_tool_call_id = highlight_tool_call_id
        agent = session.agents.get(agent_id)
        if not agent:
            return

        self.add_class("visible")
        self._render_metadata(agent)
        self._render_hierarchy(agent, session)
        self._render_conversation(agent_id, session)

    def hide(self) -> None:
        """Hide the panel."""
        self.remove_class("visible")
        self._agent_id = None
        self._highlight_message_index = None
        self._highlight_tool_call_id = None

    @property
    def is_visible(self) -> bool:
        return self.has_class("visible")

    # ── Metadata section ──

    def _render_metadata(self, agent: AgentNode) -> None:
        """Render the agent metadata section."""
        container = self.query_one("#acp-metadata", Vertical)
        container.remove_children()

        lines = []

        # Agent Name (more prominent)
        lines.append(f"[bold $primary]{_esc(agent.name)}[/bold $primary]")
        lines.append("")

        # Role and State (grouped)
        role_str = ROLE_DISPLAY.get(agent.role, "Unknown") if agent.role else "Unknown"
        state_text, state_color = STATE_DISPLAY.get(
            agent.state, ("Unknown", "dim"),
        )
        lines.append(f"[bold dim]Role:[/bold dim]   {role_str}")
        lines.append(f"[bold dim]State:[/bold dim]  [{state_color}]{state_text}[/{state_color}]")
        lines.append(f"[bold dim]Model:[/bold dim]  {_esc(agent.model)}")
        lines.append("") # Separator

        # Identifiers
        lines.append(f"[bold dim]ID:[/bold dim]     [dim]{agent.id}[/dim]") # Show full ID, not truncated
        
        # Parent
        if agent.parent_id and self._session:
            parent = self._session.agents.get(agent.parent_id)
            parent_name = parent.name if parent else agent.parent_id
            lines.append(f"[bold dim]Parent:[/bold dim] {_esc(parent_name)} [dim]({agent.parent_id[:8]})[/dim]")
        
        # Children
        if self._session:
            children = [
                a for a in self._session.agents.values()
                if a.parent_id == agent.id
            ]
            if children:
                child_names = ", ".join(_esc(c.name) for c in children[:3]) # Limit to 3 for brevity
                if len(children) > 3:
                    child_names += f" (+{len(children) - 3} more)"
                lines.append(f"[bold dim]Children:[/bold dim] {child_names}")
        
        lines.append("") # Separator
        
        # Created/Completed
        if agent.created_at:
            lines.append(f"[bold dim]Created:[/bold dim] {agent.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if agent.completed_at:
            lines.append(f"[bold dim]Completed:[/bold dim] {agent.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")


        container.mount(Static(
            "\n".join(lines),
            markup=True,
        ))

    # ── Hierarchy section ──

    def _render_hierarchy(self, agent: AgentNode, session: Session) -> None:
        """Render the agent hierarchy tree."""
        container = self.query_one("#acp-hierarchy", Vertical)
        container.remove_children()

        container.mount(Static(
            "Hierarchy",
            id="acp-hierarchy-title",
            markup=True,
        ))

        # Build the chain from root to this agent
        chain = self._build_ancestor_chain(agent, session)
        # Also include immediate children
        children = [
            a for a in session.agents.values()
            if a.parent_id == agent.id
        ]

        # Render as indented text (simpler and more compact than a Tree widget)
        lines = []
        for depth, node in enumerate(chain):
            indent = "  " * depth
            icon, color = STATE_DISPLAY.get(node.state, ("?", "dim"))
            icon_char = icon.split(" ")[0]  # Just the icon character
            role_tag = f" \\[{node.role.value}]" if node.role else ""

            if node.id == agent.id:
                # Highlight the current agent
                lines.append(
                    f"{indent}[{color}]{icon_char}[/{color}] "
                    f"[bold reverse] {_esc(node.name)} [/bold reverse]"
                    f"[dim]{role_tag}[/dim]"
                )
            else:
                lines.append(
                    f"{indent}[{color}]{icon_char}[/{color}] "
                    f"{_esc(node.name)}"
                    f"[dim]{role_tag}[/dim]"
                )

        # Add children
        current_depth = len(chain)
        for child in children:
            indent = "  " * current_depth
            icon, color = STATE_DISPLAY.get(child.state, ("?", "dim"))
            icon_char = icon.split(" ")[0]
            role_tag = f" \\[{child.role.value}]" if child.role else ""
            lines.append(
                f"{indent}[{color}]{icon_char}[/{color}] "
                f"{_esc(child.name)}"
                f"[dim]{role_tag}[/dim]"
            )
            # Show grandchildren count
            grandchildren = [
                a for a in session.agents.values()
                if a.parent_id == child.id
            ]
            if grandchildren:
                lines.append(
                    f"{indent}  [dim]... {len(grandchildren)} sub-agent(s)[/dim]"
                )

        hierarchy_text = "\n".join(lines) if lines else "[dim]No hierarchy data[/dim]"
        container.mount(Static(
            hierarchy_text,
            id="acp-hierarchy-tree",
            markup=True,
        ))

    def _build_ancestor_chain(
        self, agent: AgentNode, session: Session,
    ) -> list[AgentNode]:
        """Build the chain from root ancestor to the given agent."""
        chain = [agent]
        current = agent
        visited: set[str] = {agent.id}  # Guard against cycles

        while current.parent_id and current.parent_id not in visited:
            parent = session.agents.get(current.parent_id)
            if not parent:
                break
            visited.add(parent.id)
            chain.insert(0, parent)
            current = parent

        return chain

    # ── Conversation section ──

    def _render_conversation(self, agent_id: str, session: Session) -> None:
        """Render the full conversation history for this agent."""
        container = self.query_one("#acp-conversation", Vertical)
        container.remove_children()

        messages = session.get_messages(agent_id)
        agent = session.agents.get(agent_id)

        # Task prompt section (if agent has prompt_preview)
        if agent and agent.prompt_preview:
            container.mount(Static(
                "[bold $accent]Task Prompt[/bold $accent]",
                classes="acp-prompt-label",
                markup=True,
            ))
            container.mount(Static(
                _esc(agent.prompt_preview),
                classes="acp-prompt-content",
                markup=True,
            ))

        container.mount(Static(
            f"Conversation ({len(messages)} messages)",
            id="acp-conv-title",
            markup=True,
        ))

        scroll = VerticalScroll(id="acp-messages")
        container.mount(scroll)

        if not messages:
            scroll.mount(Static(
                "[dim]No messages yet[/dim]",
                classes="acp-msg-system",
                markup=True,
            ))
            return

        highlighted_widget = None
        for idx, msg in enumerate(messages):
            widget = self._render_message(msg, idx)
            scroll.mount(widget)
            # Track if this is the highlighted message
            if self._highlight_message_index is not None and idx == self._highlight_message_index:
                highlighted_widget = widget
            elif self._highlight_tool_call_id is not None and msg.tool_calls:
                # Check if any tool call in this message matches
                for tc in msg.tool_calls:
                    if tc.id == self._highlight_tool_call_id:
                        highlighted_widget = widget
                        break

        # Scroll to highlighted message if present, otherwise to bottom
        if highlighted_widget:
            highlighted_widget.scroll_visible(animate=True)
        else:
            scroll.scroll_end(animate=False)

    def _render_message(self, msg: ChatMessage, idx: int) -> Static:
        """Render a single message as a Static widget."""
        # Convert to local time for display
        local_ts = msg.timestamp.astimezone() if msg.timestamp.tzinfo else msg.timestamp
        ts = local_ts.strftime("%H:%M:%S")

        # Determine if this message should be highlighted
        is_highlighted = False
        if self._highlight_message_index is not None and idx == self._highlight_message_index:
            is_highlighted = True
        elif self._highlight_tool_call_id is not None and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id == self._highlight_tool_call_id:
                    is_highlighted = True
                    break

        if msg.role == MessageRole.USER:
            classes = "acp-msg-user" + (" acp-msg-highlighted" if is_highlighted else "")
            if _has_closed_fenced_code_block(msg.content):
                header = Text.from_markup(
                    f"[dim]{ts}[/dim]  [bold $primary]You[/bold $primary]"
                )
                return Static(
                    Group(header, RichMarkdown(msg.content)),
                    classes=classes,
                    markup=False,
                )
            content = f"[dim]{ts}[/dim]  [bold $primary]You[/bold $primary]\n{_esc(msg.content)}"
            return Static(content, classes=classes, markup=True)

        if msg.role == MessageRole.ASSISTANT:
            text = msg.content
            if len(text) > 2000:
                text = text[:2000] + "\n... (truncated)"
            content = f"[dim]{ts}[/dim]  [bold cyan]Assistant[/bold cyan]\n{_esc(text)}"
            classes = "acp-msg-assistant" + (" acp-msg-highlighted" if is_highlighted else "")
            return Static(content, classes=classes, markup=True)

        if msg.role == MessageRole.SYSTEM:
            content = f"[dim]{ts}  {_esc(msg.content)}[/dim]"
            classes = "acp-msg-system" + (" acp-msg-highlighted" if is_highlighted else "")
            return Static(content, classes=classes, markup=True)

        if msg.role == MessageRole.TOOL:
            return self._render_tool_message(msg, ts, is_highlighted)

        classes = "acp-msg-system" + (" acp-msg-highlighted" if is_highlighted else "")
        return Static(_esc(msg.content), classes=classes, markup=True)

    def _render_tool_message(self, msg: ChatMessage, ts: str, is_highlighted: bool = False) -> Static:
        """Render a tool call message."""
        lines = []
        for tc in msg.tool_calls:
            brief_args = tc.arguments
            if len(brief_args) > 80:
                brief_args = brief_args[:77] + "..."

            if tc.result is None:
                status = "[yellow]pending[/yellow]"
            elif tc.success:
                status = "[green]done[/green]"
            else:
                status = "[red]error[/red]"

            lines.append(
                f"[dim]{ts}[/dim]  [cyan]{_esc(tc.name)}[/cyan]"
                f"({_esc(brief_args)}) {status}"
            )

            if tc.result:
                result_text = tc.result
                if len(result_text) > 300:
                    result_text = result_text[:297] + "..."
                color = "green" if tc.success else "red"
                lines.append(f"  [{color}]\u2192 {_esc(result_text)}[/{color}]")

        if msg.content:
            lines.append(_esc(msg.content))

        classes = "acp-msg-tool" + (" acp-msg-highlighted" if is_highlighted else "")
        return Static(
            "\n".join(lines),
            classes=classes,
            markup=True,
        )

    # ── Event handlers ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id in ("acp-close-btn", "acp-close-footer-btn"):
            self.hide()
            self.post_message(self.CloseRequested())
            event.stop()
        elif event.button.id == "acp-switch-btn":
            if self._agent_id:
                self.post_message(self.NavigateToAgent(self._agent_id))
            event.stop()
