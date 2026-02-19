"""Status bar \u2014 bottom bar showing agent state and metrics."""

from __future__ import annotations

import time
from typing import Optional

from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from rich.text import Text


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    elif secs < 3600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(secs, 3600)
        m = remainder // 60
        return f"{h}h {m}m"


class StatusBar(Widget):
    """Single-line status bar with agent info and connection state."""

    agent_name: reactive[str] = reactive("No agent")
    model: reactive[str] = reactive("\u2014")
    tokens_used: reactive[int] = reactive(0)
    status: reactive[str] = reactive("disconnected")
    mode: reactive[str] = reactive("live")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine_started_at: Optional[float] = None
        self._elapsed_timer: Timer | None = None

    def watch_status(self, old_value: str, new_value: str) -> None:
        """Track engine elapsed time when status transitions to/from streaming."""
        if new_value == "streaming" and old_value != "streaming":
            self._engine_started_at = time.monotonic()
            # Start a 1s timer to refresh the elapsed display
            if self._elapsed_timer is None:
                self._elapsed_timer = self.set_interval(1.0, self._refresh_elapsed)
        elif old_value == "streaming" and new_value != "streaming":
            self._engine_started_at = None
            if self._elapsed_timer is not None:
                self._elapsed_timer.stop()
                self._elapsed_timer = None

    def _refresh_elapsed(self) -> None:
        """Trigger a re-render to update the elapsed time display."""
        self.refresh()

    def watch_mode(self, value: str) -> None:
        if value == "demo":
            self.add_class("demo-mode")
        else:
            self.remove_class("demo-mode")

    def render(self) -> Text:
        status_colors = {
            "connected": "green",
            "streaming": "yellow",
            "disconnected": "red",
            "error": "red bold",
        }
        color = status_colors.get(self.status, "white")

        bar = Text()

        if self.mode == "demo":
            bar.append(" \u26a0 DEMO ", style="bold black on yellow")
            bar.append(" ", style="dim")

        bar.append(f" {self.agent_name} ", style="bold")
        bar.append(" \u2502 ", style="dim")
        bar.append(self.model, style="cyan")
        bar.append(" \u2502 ", style="dim")
        bar.append(f"{self.tokens_used:,} tokens", style="dim")
        bar.append(" \u2502 ", style="dim")

        status_display = f"\u25cf {self.status}"
        if self._engine_started_at is not None:
            elapsed = _format_elapsed(time.monotonic() - self._engine_started_at)
            status_display += f" ({elapsed})"
        bar.append(status_display, style=color)

        if self.mode == "demo":
            bar.append("  ", style="dim")
            bar.append("Simulated responses \u2014 run `claude auth login` for live mode",
                       style="dim italic yellow")

        return bar
