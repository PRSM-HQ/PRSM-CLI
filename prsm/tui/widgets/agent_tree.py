"""Agent tree widget — hierarchical view of orchestrated agents."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.events import Click
from textual.message import Message
from textual.strip import Strip
from textual.widgets import Tree
from textual.widgets._tree import TreeNode

from prsm.engine.models import AgentState
from prsm.shared.models.agent import AgentNode
from prsm.adapters.agent_adapter import STATE_ICONS

# Active states — agents in these states show "active"
_ACTIVE_STATES = frozenset({
    AgentState.RUNNING,
    AgentState.STARTING,
    AgentState.PENDING,
    AgentState.WAITING_FOR_PARENT,
    AgentState.WAITING_FOR_CHILD,
    AgentState.WAITING_FOR_EXPERT,
})

_SPIN_ANIMATION_STATES = frozenset({
    AgentState.RUNNING,
    AgentState.STARTING,
    AgentState.PENDING,
    AgentState.WAITING_FOR_PARENT,
    AgentState.WAITING_FOR_EXPERT,
})

_SPIN_FRAMES = (
    "\u2191",  # ↑
    "\u2197",  # ↗
    "\u2192",  # →
    "\u2198",  # ↘
    "\u2193",  # ↓
    "\u2199",  # ↙
    "\u2190",  # ←
    "\u2196",  # ↖
)

_WAITING_CHILD_FRAMES = (
    "\u25cf",  # ●
    "\u25cb",  # ○
)


def _state_uses_animation(state: AgentState) -> bool:
    """Return True when this state should animate in the tree."""
    return state in _SPIN_ANIMATION_STATES or state == AgentState.WAITING_FOR_CHILD


def _animated_state_icon(
    state: AgentState,
    spin_frame_index: int,
    waiting_child_blink_on: bool,
) -> tuple[str, str]:
    """Resolve icon and color for a state, including animated frames."""
    if state in _SPIN_ANIMATION_STATES:
        icon = _SPIN_FRAMES[spin_frame_index % len(_SPIN_FRAMES)]
        return icon, "green"
    if state == AgentState.WAITING_FOR_CHILD:
        icon = _WAITING_CHILD_FRAMES[0 if waiting_child_blink_on else 1]
        return icon, "yellow"
    return STATE_ICONS.get(state, ("?", "white"))


def _format_relative_time(dt: datetime) -> str:
    """Format a datetime as a compact relative time string."""
    now = datetime.now(timezone.utc)
    # Ensure dt is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def _get_timestamp_info(agent: AgentNode) -> tuple[str, str]:
    """Get the timestamp string and style for an agent.

    Returns:
        Tuple of (time_str, style_name).  time_str is empty when there is
        nothing to display.
    """
    if agent.state in _ACTIVE_STATES:
        return ("active", "green")
    if agent.last_active:
        return (_format_relative_time(agent.last_active), "dim")
    if agent.completed_at:
        return (_format_relative_time(agent.completed_at), "dim")
    if agent.created_at:
        return (_format_relative_time(agent.created_at), "dim")
    return ("", "dim")


class AgentTree(Tree[AgentNode]):
    """Hierarchical view of orchestrated agents and their tasks."""

    class ContextMenuRequested(Message):
        """Posted on right-click to open the context menu."""

        def __init__(self, agent: AgentNode) -> None:
            super().__init__()
            self.agent = agent

    class KillRequested(Message):
        """Posted on Delete key to kill the selected agent directly."""

        def __init__(self, agent: AgentNode) -> None:
            super().__init__()
            self.agent = agent

    class ViewContextRequested(Message):
        """Posted when the user presses 'i' to view agent context."""

        def __init__(self, agent: AgentNode) -> None:
            super().__init__()
            self.agent = agent

    BINDINGS = [
        ("enter", "select_cursor", "Select"),
        ("space", "toggle_node", "Expand/Collapse"),
        ("i", "view_context", "View Context"),
        ("delete", "kill_selected", "Kill Agent"),
    ]

    # Refresh timestamps every 10 seconds so relative times stay accurate
    _TIMESTAMP_REFRESH_INTERVAL = 10.0
    # Icon animation refresh (only repaints while at least one agent animates)
    _ANIMATION_REFRESH_INTERVAL = 0.2

    def __init__(self, **kwargs) -> None:
        super().__init__("Agents", **kwargs)
        self.show_root = True
        self.guide_depth = 3
        self._timestamp_timer = None
        self._animation_timer = None
        self._spin_frame_index = 0
        self._waiting_child_blink_on = True
        self._animating_agent_ids: set[str] = set()

    def on_mount(self) -> None:
        """Start periodic refresh for relative timestamps."""
        self._timestamp_timer = self.set_interval(
            self._TIMESTAMP_REFRESH_INTERVAL, self._refresh_timestamps
        )
        self._animation_timer = self.set_interval(
            self._ANIMATION_REFRESH_INTERVAL, self._refresh_animations
        )

    def on_unmount(self) -> None:
        """Stop timers when the widget is removed."""
        if self._timestamp_timer is not None:
            self._timestamp_timer.stop()
            self._timestamp_timer = None
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def _refresh_timestamps(self) -> None:
        """Invalidate the tree so relative timestamps are recalculated."""
        # Only refresh if we have agents in the tree (avoid needless repaints)
        if self.root.children:
            self._invalidate()

    def _refresh_animations(self) -> None:
        """Advance animation frames and repaint only when needed."""
        if not self._animating_agent_ids:
            return
        self._spin_frame_index = (self._spin_frame_index + 1) % len(_SPIN_FRAMES)
        self._waiting_child_blink_on = not self._waiting_child_blink_on
        self._invalidate()

    def render_label(
        self,
        node: TreeNode[AgentNode],
        base_style,
        style,
    ) -> Text:
        agent = node.data
        if agent is None:
            return Text(str(node.label), style=style)

        icon, color = _animated_state_icon(
            agent.state,
            self._spin_frame_index,
            self._waiting_child_blink_on,
        )
        label = Text()
        label.append(f" {icon} ", style=color)
        label.append(str(node.label), style=style)

        # Append contextual waiting message if applicable
        if agent.state == AgentState.WAITING_FOR_PARENT:
            label.append(" (waiting for parent)", style="dim yellow")
        elif agent.state == AgentState.WAITING_FOR_CHILD:
            label.append(" (waiting for child)", style="dim yellow")
        elif agent.state == AgentState.WAITING_FOR_EXPERT:
            label.append(" (waiting for expert)", style="dim yellow")

        if agent.role:
            label.append(f" [{agent.role.value}]", style="dim")

        return label

    def render_line(self, y: int) -> Strip:
        """Render a line, then splice in a right-justified timestamp."""
        strip = super().render_line(y)
        width = self.size.width
        if width <= 0:
            return strip

        # Find the agent node for this line
        scroll_y = self.scroll_offset.y
        line_index = y + scroll_y
        tree_lines = self._tree_lines
        if line_index >= len(tree_lines):
            return strip

        node = tree_lines[line_index].node
        agent = node.data
        if agent is None:
            return strip

        time_str, time_style = _get_timestamp_info(agent)
        if not time_str:
            return strip

        # Build the timestamp segment with a leading space
        ts_text = f" {time_str} "
        ts_len = len(ts_text)

        if ts_len >= width:
            return strip

        # Parse the style
        ts_style = Style.parse(time_style)

        # Build new segments: original content cropped + timestamp at right edge
        # Crop the strip to make room for the timestamp
        cropped = strip.crop(0, width - ts_len)

        # Get segments from the cropped strip and pad to the timestamp position
        segments = list(cropped._segments)
        cropped_len = cropped.cell_length
        gap = width - ts_len - cropped_len
        if gap > 0:
            segments.append(Segment(" " * gap))

        # Append the timestamp
        segments.append(Segment(ts_text, ts_style))

        return Strip(segments, width)

    def on_click(self, event: Click) -> None:
        """Handle right-click to open context menu."""
        if event.button == 3:
            line = event.y + self.scroll_offset.y
            node = self.get_node_at_line(line)
            if node is not None and node.data and not node.is_root:
                self.post_message(self.ContextMenuRequested(node.data))

    def action_view_context(self) -> None:
        """View context for the currently selected agent (i key)."""
        node = self.cursor_node
        if node is not None and node.data and not node.is_root:
            self.post_message(self.ViewContextRequested(node.data))

    def action_kill_selected(self) -> None:
        """Kill the currently selected agent (Delete key)."""
        node = self.cursor_node
        if node is not None and node.data and not node.is_root:
            self.post_message(self.KillRequested(node.data))

    def add_agent(
        self,
        parent_id: str | None,
        agent: AgentNode,
    ) -> TreeNode[AgentNode]:
        parent = self._find_node(parent_id) if parent_id else self.root
        if parent is None:
            parent = self.root
        if _state_uses_animation(agent.state):
            self._animating_agent_ids.add(agent.id)
        return parent.add(agent.name, data=agent, expand=True)

    def update_agent_state(self, agent_id: str, state: AgentState) -> None:
        node = self._find_node(agent_id)
        if node and node.data:
            node.data.state = state
            if _state_uses_animation(state):
                self._animating_agent_ids.add(agent_id)
            else:
                self._animating_agent_ids.discard(agent_id)
            node.refresh()

    def remove_agent(self, agent_id: str) -> None:
        node = self._find_node(agent_id)
        if node and not node.is_root:
            self._animating_agent_ids.discard(agent_id)
            node.remove()

    def clear_agents(self) -> None:
        """Remove all agents from the tree, resetting to an empty root."""
        self._animating_agent_ids.clear()
        self.root.remove_children()
        self.root.data = None
        self.root.set_label("Agents")

    def sort_by_activity(self) -> None:
        """Re-sort all children at each level by last_active (most recent first).

        Preserves parent-child hierarchy while reordering siblings.
        """
        self._sort_children(self.root)
        self._invalidate()

    def _sort_children(self, parent: TreeNode[AgentNode]) -> None:
        """Recursively sort children of a node by last_active descending."""
        children = list(parent.children)
        if not children:
            return

        # Sort by last_active descending (most recent first), with fallbacks
        def sort_key(node: TreeNode[AgentNode]) -> float:
            agent = node.data
            if agent is None:
                return 0.0
            ts = agent.last_active or agent.completed_at or agent.created_at
            if ts is None:
                return 0.0
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()

        sorted_children = sorted(children, key=sort_key, reverse=True)

        # Reorder via the internal structure
        parent._children = sorted_children
        parent._updates += 1

        # Recurse into each child
        for child in sorted_children:
            self._sort_children(child)

    def _find_node(self, agent_id: str) -> TreeNode[AgentNode] | None:
        for _, node in self._tree_nodes.items():
            if node.data and node.data.id == agent_id:
                return node
        return None
