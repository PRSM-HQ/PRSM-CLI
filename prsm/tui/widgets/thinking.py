"""Thinking indicator — animated widget shown while agent is processing."""

from __future__ import annotations

import random
import time
from pathlib import Path

from prsm.shared.services.preferences import UserPreferences
from textual.timer import Timer
from textual.widgets import Static


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string.

    Under 60s: Xs, 60-3600: Xm Ys, over 3600: Xh Ym.
    """
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


def _load_verbs_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_thinking_verbs() -> list[str]:
    """Load base + optional NSFW + custom thinking verbs."""
    shared_ui_dir = Path(__file__).parent.parent.parent / "shared_ui"
    safe_verbs = _load_verbs_from_file(shared_ui_dir / "thinking_verbs.txt")
    nsfw_verbs = _load_verbs_from_file(shared_ui_dir / "nsfw_thinking_verbs.txt")

    prefs = UserPreferences.load()
    merged = list(safe_verbs)

    if prefs.enable_nsfw_thinking_verbs:
        for verb in nsfw_verbs:
            if verb not in merged:
                merged.append(verb)

    for verb in prefs.custom_thinking_verbs:
        if verb not in merged:
            merged.append(verb)

    if merged:
        return merged
    return ["Thinking", "Processing", "Analyzing"]


THINKING_VERBS: list[str] = _load_thinking_verbs()


class ThinkingIndicator(Static):
    """Animated thinking indicator that cycles through verbs with animated dots."""

    def __init__(
        self,
        agent_name: str | None = None,
        status_text: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__("", classes="thinking-indicator", markup=True, **kwargs)
        self._verbs: list[str] = list(THINKING_VERBS)
        self._verb_index: int = random.randrange(len(self._verbs))
        self._dot_count: int = 1
        self._tick_count: int = 0
        self._timer: Timer | None = None
        self._pissing_lock_ticks: int = 0  # Tracks how many ticks "Pissing" should stay locked
        self._agent_name = agent_name  # Store the agent name
        self._status_text = status_text
        self._started_at: float = 0.0  # Set on mount

    @property
    def agent_name(self) -> str | None:
        return self._agent_name

    @property
    def status_text(self) -> str | None:
        return self._status_text

    def on_mount(self) -> None:
        self._started_at = time.monotonic()
        self._render_text()
        if self._status_text is None:
            self._timer = self.set_interval(0.4, self._rotate)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _rotate(self) -> None:
        """Advance dots; randomly sample a new verb every 4 ticks with conditional logic."""
        self._tick_count += 1
        self._dot_count = (self._dot_count % 3) + 1

        # Handle "Pissing" lock - if locked, decrement and skip verb change
        if self._pissing_lock_ticks > 0:
            self._pissing_lock_ticks -= 1
            self._render_text()
            return

        if self._tick_count % 4 == 0:
            current_verb = self._verbs[self._verb_index]
            next_verb = None

            # Apply transition rules (20% chance each)
            if current_verb == "Shitting" and random.random() < 0.2:
                next_verb = "Wiping"
            elif current_verb == "Sharting" and random.random() < 0.2:
                next_verb = "Wiping"
            elif current_verb == "Sleeping" and random.random() < 0.2:
                next_verb = "Waking up"
            elif current_verb == "Drinking":
                roll = random.random()
                if roll < 0.2:
                    next_verb = "Blacking out"
                elif roll < 0.4:  # Another 20% chance for Pissing
                    next_verb = "Pissing"

            # If a transition was triggered, use it
            if next_verb and next_verb in self._verbs:
                self._verb_index = self._verbs.index(next_verb)
            else:
                # Otherwise, random selection with 20% chance for Pissing lock
                self._verb_index = random.randrange(len(self._verbs))
                if self._verbs[self._verb_index] == "Pissing" and random.random() < 0.2:
                    # Lock "Pissing" for 1 minute = 150 ticks (60s / 0.4s per tick)
                    self._pissing_lock_ticks = 150

        self._render_text()

    def _render_text(self) -> None:
        elapsed = _format_elapsed(time.monotonic() - self._started_at) if self._started_at else ""
        elapsed_suffix = f" [dim]({elapsed})[/]" if elapsed else ""
        if self._status_text is not None:
            display_text = f"[italic $accent]{self._status_text}[/]{elapsed_suffix}"
        else:
            verb = self._verbs[self._verb_index]
            dots = "." * self._dot_count
            padding = " " * (3 - self._dot_count)
            display_text = f"[italic $accent]{verb}{dots}{padding}[/]{elapsed_suffix}"
        if self._agent_name:
            display_text = f"[bold dim]{self._agent_name}: [/]{display_text}"

        self.update(display_text)
