"""Settings menu — hub screen for managing PRSM preferences.

Provides settings for:
- Bash command whitelist/blacklist management
- File-revert-on-resend preference
- Session import/export
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, TextArea

from prsm.shared.services.command_policy_store import CommandPolicyStore
from prsm.shared.services.preferences import UserPreferences
from prsm.shared.services.session_archive_import import SessionArchiveImportService
from prsm.shared.services.session_export import SessionExportService
from prsm.tui.screens.file_browser import FileBrowserScreen
from prsm.tui.widgets.input_bar import InputBar

logger = logging.getLogger(__name__)


class SettingsScreen(ModalScreen[str | None]):
    """Modal dialog for managing PRSM settings.

    Returns None on dismiss.  Side-effects (preference saves, policy
    edits) happen in-place.
    """

    CSS_PATH = "../styles/modal.tcss"

    def __init__(self, cwd: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd or Path.cwd()
        self._prefs = UserPreferences.load()
        self._policy_store = CommandPolicyStore(self._cwd)
        self._policy_store.ensure_files()
        self._prsm_path = self._cwd / ".prism" / "prsm.yaml"
        self._models_path = Path.home() / ".prsm" / "models.yaml"
        self._orchestration = self._load_orchestration_config()

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        logger.debug(
            "SettingsScreen._load_yaml: attempting to load %s (exists=%s)",
            path, path.exists()
        )
        if not path.exists():
            logger.debug("SettingsScreen._load_yaml: file does not exist: %s", path)
            return {}
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            result = data if isinstance(data, dict) else {}
            logger.debug(
                "SettingsScreen._load_yaml: loaded %s successfully (keys=%s)",
                path, list(result.keys()) if result else "empty"
            )
            return result
        except yaml.YAMLError as exc:
            logger.warning(
                "SettingsScreen._load_yaml: YAML parse error in %s: %s",
                path, exc
            )
            return {}
        except Exception as exc:
            logger.warning(
                "SettingsScreen._load_yaml: unexpected error reading %s: %s",
                path, exc
            )
            return {}

    def _load_orchestration_config(self) -> dict:
        logger.info(
            "SettingsScreen: loading orchestration config from %s and %s",
            self._prsm_path, self._models_path
        )
        prsm_raw = self._load_yaml(self._prsm_path)
        models_raw = self._load_yaml(self._models_path)
        defaults = prsm_raw.get("defaults", {}) or {}
        engine = prsm_raw.get("engine", {}) or {}
        models = models_raw.get("models", {}) or prsm_raw.get("models", {}) or {}
        model_registry = (
            models_raw.get("model_registry", {})
            or prsm_raw.get("model_registry", {})
            or {}
        )
        if not model_registry and models:
            model_registry = self._derive_model_registry_from_models(models)
        result = {
            "engine": engine,
            "defaults": defaults,
            "providers": prsm_raw.get("providers", {}) or {},
            "models": models,
            "model_registry": model_registry,
        }
        logger.info(
            "SettingsScreen: loaded config with providers=%d, models=%d, "
            "registry_entries=%d, default_model=%s",
            len(result["providers"]),
            len(result["models"]),
            len(result["model_registry"]),
            defaults.get("model", "<none>")
        )
        return result

    @staticmethod
    def _dump_yaml_fragment(data: dict) -> str:
        if not data:
            return ""
        return yaml.safe_dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).strip()

    @staticmethod
    def _derive_model_registry_from_models(
        models: dict,
    ) -> dict[str, dict]:
        """Build model_registry entries from capability-style models."""
        derived: dict[str, dict] = {}
        for alias, cfg in (models or {}).items():
            if not isinstance(cfg, dict):
                continue
            if not any(
                key in cfg
                for key in ("tier", "cost_factor", "speed_factor", "affinities")
            ):
                continue
            model_id = str(cfg.get("model_id") or alias)
            effort = cfg.get("reasoning_effort")
            registry_id = (
                f"{model_id}::reasoning_effort={effort}"
                if effort
                else model_id
            )
            entry: dict[str, object] = {}
            for key in (
                "tier",
                "provider",
                "cost_factor",
                "speed_factor",
                "available",
                "affinities",
            ):
                if key in cfg:
                    entry[key] = cfg[key]
            if entry:
                derived[registry_id] = entry
        return derived

    def compose(self) -> ComposeResult:
        whitelist = self._policy_store.read_whitelist()
        blacklist = self._policy_store.read_blacklist()
        defaults = self._orchestration.get("defaults", {}) or {}
        engine = self._orchestration.get("engine", {}) or {}
        peers = defaults.get("peer_models")
        if not peers and defaults.get("peer_model"):
            peers = [defaults.get("peer_model")]
        peer_models = ", ".join(str(p) for p in (peers or []))
        user_question_timeout = engine.get("user_question_timeout_seconds", 0)
        if isinstance(user_question_timeout, float) and user_question_timeout.is_integer():
            user_question_timeout = int(user_question_timeout)
        preview_value = "on" if self._prefs.markdown_preview_enabled else "off"

        with Vertical(id="settings-dialog"):
            yield Static(
                "[bold $primary]Settings[/bold $primary]",
                id="settings-title",
            )

            # ── Bash Command Policy ──
            yield Static(
                "[bold]Bash Command Policy[/bold]",
                id="settings-policy-header",
            )
            yield Static(
                "[dim]Manage regex patterns that auto-allow or always-prompt "
                "for bash commands.[/dim]",
                classes="settings-desc",
            )
            yield Static(
                f"[bold green]Whitelist[/bold green] ({len(whitelist)} patterns)",
                id="settings-wl-header",
            )
            with VerticalScroll(id="settings-wl-list"):
                if whitelist:
                    for i, pat in enumerate(whitelist):
                        yield _PolicyPatternEntry(
                            pattern=pat,
                            list_kind="wl",
                            index=i,
                        )
                else:
                    yield Static("[dim]No whitelist patterns[/dim]")
            with Horizontal(classes="settings-policy-add-row"):
                yield Input(placeholder="e.g. npm\\s+test", id="settings-wl-input")
                yield Button("Add", variant="success", id="btn-settings-wl-add")

            yield Static(
                f"[bold red]Blacklist[/bold red] ({len(blacklist)} patterns)",
                id="settings-bl-header",
            )
            with VerticalScroll(id="settings-bl-list"):
                if blacklist:
                    for i, pat in enumerate(blacklist):
                        yield _PolicyPatternEntry(
                            pattern=pat,
                            list_kind="bl",
                            index=i,
                        )
                else:
                    yield Static("[dim]No blacklist patterns[/dim]")
            with Horizontal(classes="settings-policy-add-row"):
                yield Input(
                    placeholder="e.g. docker\\s+volume\\s+rm",
                    id="settings-bl-input",
                )
                yield Button("Add", variant="error", id="btn-settings-bl-add")

            # ── File Revert on Resend ──
            yield Static(
                "[bold]File Revert on Resend[/bold]",
                id="settings-revert-header",
            )
            yield Static(
                "[dim]When resending a previous prompt, should file changes "
                "made after that point be reverted?[/dim]",
                classes="settings-desc",
            )
            yield Select(
                [
                    ("Ask each time", "ask"),
                    ("Always revert", "always"),
                    ("Never revert", "never"),
                ],
                value=self._prefs.file_revert_on_resend,
                id="settings-revert-select",
            )

            # ── Input Markdown Preview ──
            yield Static(
                "[bold]Input Markdown Preview[/bold]",
                id="settings-preview-header",
            )
            yield Static(
                "[dim]Show a live markdown preview below the prompt input.[/dim]",
                classes="settings-desc",
            )
            yield Select(
                [
                    ("On", "on"),
                    ("Off", "off"),
                ],
                value=preview_value,
                id="settings-preview-select",
            )

            # ── Orchestration Config (.prism/*.yaml) ──
            yield Static(
                "[bold]Model & Provider Config[/bold]",
                id="settings-orch-header",
            )
            yield Static(
                "[dim]Auto-loaded from .prism/prsm.yaml and ~/.prsm/models.yaml. "
                "Edit and save to persist.[/dim]",
                classes="settings-desc",
            )
            with Horizontal(classes="settings-policy-add-row"):
                yield Input(
                    value=str(defaults.get("model", "")),
                    placeholder="Default model alias",
                    id="settings-default-model",
                )
                yield Input(
                    value=peer_models,
                    placeholder="Peer models (comma-separated aliases)",
                    id="settings-peer-models",
                )
            with Horizontal(classes="settings-policy-add-row"):
                yield Input(
                    value=str(user_question_timeout),
                    placeholder="User question timeout seconds (0 disables)",
                    id="settings-user-question-timeout",
                )
            yield Static(
                "[dim]Set timeout seconds for ask_user/request_user_input. "
                "Use 0 to disable timeout.[/dim]",
                classes="settings-desc",
            )

            yield Static("[bold]Providers (YAML map)[/bold]", classes="settings-desc")
            yield TextArea(
                self._dump_yaml_fragment(self._orchestration.get("providers", {}) or {}),
                id="settings-providers-yaml",
            )
            yield Static("[bold]Models (YAML map)[/bold]", classes="settings-desc")
            yield TextArea(
                self._dump_yaml_fragment(self._orchestration.get("models", {}) or {}),
                id="settings-models-yaml",
            )
            yield Static("[bold]Model Registry (YAML map)[/bold]", classes="settings-desc")
            yield TextArea(
                self._dump_yaml_fragment(self._orchestration.get("model_registry", {}) or {}),
                id="settings-registry-yaml",
            )
            yield Static("", id="settings-config-status")

            # ── Close ──
            with Horizontal(id="settings-actions"):
                yield Button("Save Config", variant="success", id="btn-settings-save-config")
                yield Button("Close", variant="default", id="btn-settings-close")

    # ── Event handling ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id == "btn-settings-close":
            self.dismiss(None)
            return

        if btn_id == "btn-settings-save-config":
            self._save_orchestration_config()
            return

        if btn_id == "btn-settings-wl-add":
            self._add_to_whitelist()
            return

        if btn_id == "btn-settings-bl-add":
            self._add_to_blacklist()
            return

        if btn_id.startswith("btn-settings-rm-wl-"):
            idx = int(btn_id.removeprefix("btn-settings-rm-wl-"))
            self._remove_from_whitelist(idx)
            return

        if btn_id.startswith("btn-settings-rm-bl-"):
            idx = int(btn_id.removeprefix("btn-settings-rm-bl-"))
            self._remove_from_blacklist(idx)
            return

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "settings-revert-select":
            value = str(event.value)
            if value in ("ask", "always", "never"):
                self._prefs.file_revert_on_resend = value
                self._prefs.save()
        elif event.select.id == "settings-preview-select":
            value = str(event.value)
            self._prefs.markdown_preview_enabled = value == "on"
            self._prefs.save()
            try:
                input_bar = self.app.query_one("#input-bar", InputBar)
                input_bar.set_markdown_preview_enabled(
                    self._prefs.markdown_preview_enabled
                )
            except Exception:
                pass

    def key_escape(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "settings-wl-input":
            self._add_to_whitelist()
        elif event.input.id == "settings-bl-input":
            self._add_to_blacklist()

    @staticmethod
    def _parse_yaml_map(raw: str) -> dict:
        text = raw.strip()
        if not text:
            return {}
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Expected a YAML mapping")
        return data

    def _set_config_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#settings-config-status", Static)
        if error:
            status.update(f"[red]{message}[/red]")
        else:
            status.update(f"[green]{message}[/green]")

    def _save_orchestration_config(self) -> None:
        try:
            default_model = self.query_one("#settings-default-model", Input).value.strip()
            peer_raw = self.query_one("#settings-peer-models", Input).value.strip()
            user_question_timeout_raw = self.query_one(
                "#settings-user-question-timeout",
                Input,
            ).value.strip()
            providers_raw = self.query_one("#settings-providers-yaml", TextArea).text
            models_raw = self.query_one("#settings-models-yaml", TextArea).text
            registry_raw = self.query_one("#settings-registry-yaml", TextArea).text

            providers = self._parse_yaml_map(providers_raw)
            models = self._parse_yaml_map(models_raw)
            model_registry = self._parse_yaml_map(registry_raw)
            peer_models = [p.strip() for p in peer_raw.split(",") if p.strip()]
            user_question_timeout = 0.0
            if user_question_timeout_raw:
                user_question_timeout = float(user_question_timeout_raw)
            if user_question_timeout < 0:
                raise ValueError("User question timeout must be 0 or greater")

            self._prsm_path.parent.mkdir(parents=True, exist_ok=True)
            self._models_path.parent.mkdir(parents=True, exist_ok=True)
            prsm_cfg = self._load_yaml(self._prsm_path)
            defaults = dict(prsm_cfg.get("defaults", {}) or {})
            engine = dict(prsm_cfg.get("engine", {}) or {})

            if default_model:
                defaults["model"] = default_model
            else:
                defaults.pop("model", None)
            if peer_models:
                defaults["peer_models"] = peer_models
                defaults.pop("peer_model", None)
            else:
                defaults.pop("peer_models", None)
                defaults.pop("peer_model", None)
            engine["user_question_timeout_seconds"] = (
                int(user_question_timeout)
                if float(user_question_timeout).is_integer()
                else user_question_timeout
            )
            prsm_cfg["defaults"] = defaults
            prsm_cfg["engine"] = engine
            prsm_cfg["providers"] = providers
            prsm_cfg.pop("models", None)
            prsm_cfg.pop("model_registry", None)

            models_cfg = self._load_yaml(self._models_path)
            models_cfg["models"] = models
            models_cfg["model_registry"] = model_registry

            with open(self._prsm_path, "w") as f:
                yaml.safe_dump(
                    prsm_cfg,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            with open(self._models_path, "w") as f:
                yaml.safe_dump(
                    models_cfg,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            self._set_config_status(
                "Saved .prism/prsm.yaml and ~/.prsm/models.yaml",
                error=False,
            )
        except Exception as exc:
            self._set_config_status(f"Config save failed: {exc}", error=True)

    # ── Policy mutations ──

    def _add_to_whitelist(self) -> None:
        inp = self.query_one("#settings-wl-input", Input)
        pattern = inp.value.strip()
        if not pattern:
            inp.add_class("error")
            return
        self._policy_store.add_whitelist_pattern(pattern)
        inp.value = ""
        inp.remove_class("error")
        self._refresh_policy_lists()

    def _add_to_blacklist(self) -> None:
        inp = self.query_one("#settings-bl-input", Input)
        pattern = inp.value.strip()
        if not pattern:
            inp.add_class("error")
            return
        self._policy_store.add_blacklist_pattern(pattern)
        inp.value = ""
        inp.remove_class("error")
        self._refresh_policy_lists()

    def _remove_from_whitelist(self, index: int) -> None:
        patterns = self._policy_store.read_whitelist()
        if 0 <= index < len(patterns):
            self._policy_store.remove_whitelist_pattern(patterns[index])
            self._refresh_policy_lists()

    def _remove_from_blacklist(self, index: int) -> None:
        patterns = self._policy_store.read_blacklist()
        if 0 <= index < len(patterns):
            self._policy_store.remove_blacklist_pattern(patterns[index])
            self._refresh_policy_lists()

    def _refresh_policy_lists(self) -> None:
        whitelist = self._policy_store.read_whitelist()
        blacklist = self._policy_store.read_blacklist()

        self.query_one("#settings-wl-header", Static).update(
            f"[bold green]Whitelist[/bold green] ({len(whitelist)} patterns)"
        )
        self.query_one("#settings-bl-header", Static).update(
            f"[bold red]Blacklist[/bold red] ({len(blacklist)} patterns)"
        )

        wl_list = self.query_one("#settings-wl-list", VerticalScroll)
        bl_list = self.query_one("#settings-bl-list", VerticalScroll)
        wl_list.remove_children()
        bl_list.remove_children()

        if whitelist:
            for i, pat in enumerate(whitelist):
                wl_list.mount(_PolicyPatternEntry(pattern=pat, list_kind="wl", index=i))
        else:
            wl_list.mount(Static("[dim]No whitelist patterns[/dim]"))

        if blacklist:
            for i, pat in enumerate(blacklist):
                bl_list.mount(_PolicyPatternEntry(pattern=pat, list_kind="bl", index=i))
        else:
            bl_list.mount(Static("[dim]No blacklist patterns[/dim]"))


class _PolicyPatternEntry(Static):
    """Single command policy pattern row with remove action."""

    def __init__(self, pattern: str, list_kind: str, index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pattern = pattern
        self._list_kind = list_kind
        self._index = index

    def compose(self) -> ComposeResult:
        safe_pat = self._pattern.replace("[", "\\[")
        with Horizontal(classes="settings-policy-entry"):
            yield Static(f"  [cyan]{safe_pat}[/cyan]", classes="settings-policy-pattern")
            yield Button(
                "Remove",
                variant="warning",
                id=f"btn-settings-rm-{self._list_kind}-{self._index}",
                classes="settings-policy-rm-btn",
            )
