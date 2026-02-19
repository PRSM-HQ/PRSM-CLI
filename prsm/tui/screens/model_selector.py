"""Model selector modal — pick a model for the current/next orchestration run.

Shows available models grouped by provider with tier badges.
Returns the selected model_id string or None if cancelled.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


@dataclass
class ModelOption:
    """A model available for selection."""
    model_id: str
    provider: str
    tier: str  # frontier, strong, fast, economy
    available: bool = True
    is_current: bool = False
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model_id


_TIER_STYLES = {
    "frontier": ("bold magenta", "⬥"),
    "strong": ("bold cyan", "◆"),
    "fast": ("bold green", "▸"),
    "economy": ("bold yellow", "○"),
}

_PROVIDER_ICONS = {
    "claude": "🤖",
    "codex": "⚡",
    "gemini": "✦",
    "minimax": "◈",
}


def _sanitize_id(raw: str) -> str:
    """Sanitize a string for use as a Textual CSS identifier.

    Textual identifiers may only contain letters, numbers, underscores, and
    hyphens, and must not begin with a number.  Model IDs such as
    ``gpt-5.2-codex`` contain dots which are illegal, so we replace any
    disallowed character with an underscore.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


class ModelSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for selecting an AI model."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    _MOUNT_GUARD_SECONDS = 0.3

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }
    ModelSelectorScreen > Vertical {
        width: 72;
        height: auto;
        max-height: 32;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    ModelSelectorScreen .model-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        width: 100%;
    }
    ModelSelectorScreen .model-subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
        width: 100%;
    }
    ModelSelectorScreen .model-list {
        max-height: 20;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    ModelSelectorScreen .provider-header {
        margin: 1 0 0 0;
        text-style: bold;
        color: $text;
    }
    ModelSelectorScreen .model-btn {
        width: 100%;
        margin: 0;
        height: 3;
    }
    ModelSelectorScreen .model-btn:focus {
        border: tall $accent;
    }
    ModelSelectorScreen .model-btn.current-model {
        background: $accent-darken-2;
    }
    ModelSelectorScreen .model-btn.unavailable {
        opacity: 0.4;
    }
    ModelSelectorScreen .modal-actions {
        layout: horizontal;
        height: 3;
        align: center middle;
    }
    ModelSelectorScreen .modal-actions Button {
        margin: 0 1;
        min-width: 14;
    }
    """

    def __init__(
        self,
        models: list[ModelOption],
        current_model: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._models = models
        self._current_model = current_model
        self._current_display_model = current_model
        for model in models:
            if model.model_id == current_model:
                self._current_display_model = model.display_name
                break
        self._mount_time = 0.0
        # Maps sanitized button ID → real model_id for reverse lookup
        self._id_to_model: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🔧 Select Model", classes="model-title")
            if self._current_model:
                yield Static(
                    f"Current: [bold]{self._current_display_model}[/bold]",
                    classes="model-subtitle",
                    markup=True,
                )

            with VerticalScroll(classes="model-list"):
                # Group by provider
                providers: dict[str, list[ModelOption]] = {}
                for m in self._models:
                    providers.setdefault(m.provider, []).append(m)

                for provider, models in providers.items():
                    icon = _PROVIDER_ICONS.get(provider, "●")
                    yield Static(
                        f"{icon} {provider.title()}",
                        classes="provider-header",
                    )
                    for model in models:
                        tier_style, tier_icon = _TIER_STYLES.get(
                            model.tier, ("dim", "·")
                        )
                        current_mark = " ✓" if model.is_current else ""
                        label = (
                            f"{tier_icon} [{tier_style}]{model.tier.upper()}[/{tier_style}] "
                            f"{model.display_name}{current_mark}"
                        )
                        safe_id = f"model-{_sanitize_id(model.model_id)}"
                        self._id_to_model[safe_id] = model.model_id
                        btn = Button(
                            label,
                            id=safe_id,
                            classes="model-btn"
                            + (" current-model" if model.is_current else "")
                            + (" unavailable" if not model.available else ""),
                            disabled=not model.available,
                        )
                        yield btn

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="model-cancel")

    def on_mount(self) -> None:
        self._mount_time = time.monotonic()
        # Focus the current model button if possible
        if self._current_model:
            safe_current = f"model-{_sanitize_id(self._current_model)}"
            try:
                btn = self.query_one(f"#{safe_current}", Button)
                btn.focus()
            except Exception:
                pass

    def _is_guarded(self) -> bool:
        return (time.monotonic() - self._mount_time) < self._MOUNT_GUARD_SECONDS

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._is_guarded():
            return
        btn_id = event.button.id or ""
        if btn_id == "model-cancel":
            self.dismiss(None)
        elif btn_id.startswith("model-"):
            # Look up the real model_id from the sanitized button ID
            model_id = self._id_to_model.get(btn_id, btn_id[len("model-"):])
            self.dismiss(model_id)

    def action_cancel(self) -> None:
        if self._is_guarded():
            return
        self.dismiss(None)
