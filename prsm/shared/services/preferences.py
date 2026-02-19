"""User preferences — persistent settings stored in ~/.prsm/preferences.json.

Manages per-user preferences like file revert behavior when resending prompts.
Settings are global (not per-project) since they reflect user preferences
rather than project configuration.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_PATH = Path.home() / ".prsm" / "preferences.json"


@dataclass
class UserPreferences:
    """User preference settings.

    Attributes:
        file_revert_on_resend: Controls behavior when resending a prompt.
            - "ask": Ask each time whether to revert file changes (default).
            - "always": Always revert file changes without asking.
            - "never": Never revert file changes.
    """

    file_revert_on_resend: str = "ask"  # "ask", "always", "never"
    enable_nsfw_thinking_verbs: bool = True
    custom_thinking_verbs: list[str] = field(default_factory=list)
    markdown_preview_enabled: bool = True

    def validate(self) -> None:
        """Ensure all values are within allowed ranges."""
        if self.file_revert_on_resend not in ("ask", "always", "never"):
            self.file_revert_on_resend = "ask"
        if not isinstance(self.enable_nsfw_thinking_verbs, bool):
            self.enable_nsfw_thinking_verbs = True
        if isinstance(self.custom_thinking_verbs, list):
            cleaned: list[str] = []
            for verb in self.custom_thinking_verbs:
                if not isinstance(verb, str):
                    continue
                stripped = verb.strip()
                if stripped and stripped not in cleaned:
                    cleaned.append(stripped)
            self.custom_thinking_verbs = cleaned
        else:
            self.custom_thinking_verbs = []
        if not isinstance(self.markdown_preview_enabled, bool):
            self.markdown_preview_enabled = True

    def save(self, path: Path | None = None) -> None:
        """Persist preferences to disk."""
        target = path or PREFS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(json.dumps(asdict(self), indent=2))
        except Exception:
            logger.debug("Failed to save preferences to %s", target)

    @classmethod
    def load(cls, path: Path | None = None) -> UserPreferences:
        """Load preferences from disk, returning defaults if missing/corrupt."""
        target = path or PREFS_PATH
        try:
            if target.exists():
                data = json.loads(target.read_text())
                prefs = cls(**{
                    k: v for k, v in data.items()
                    if k in cls.__dataclass_fields__
                })
                prefs.validate()
                logger.debug("Loaded preferences from %s", target)
                return prefs
            else:
                logger.debug("Preferences file not found at %s; using defaults", target)
        except Exception:
            logger.warning("Failed to load preferences from %s; using defaults", target)
        return cls()
