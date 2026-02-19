"""Phase 5 tests — vendored orchestrator, auth detection, and import verification."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Vendored Orchestrator Import Tests ──


def test_orchestrator_engine_importable():
    """OrchestrationEngine is importable from the vendored package."""
    from prsm.engine import OrchestrationEngine
    assert OrchestrationEngine is not None


def test_orchestrator_config_importable():
    """EngineConfig is importable from the vendored package."""
    from prsm.engine import EngineConfig
    assert EngineConfig is not None


def test_orchestrator_models_importable():
    """Core models are importable from the vendored package."""
    from prsm.engine import (
        AgentDescriptor,
        AgentResult,
        AgentRole,
        AgentState,
        ExpertProfile,
        PermissionMode,
        SpawnRequest,
    )
    assert AgentDescriptor is not None
    assert AgentRole.MASTER is not None
    assert PermissionMode.DEFAULT is not None


def test_orchestrator_errors_importable():
    """Error hierarchy is importable."""
    from prsm.engine import (
        OrchestrationError,
        AgentSpawnError,
        AgentTimeoutError,
        DeadlockDetectedError,
    )
    assert issubclass(AgentSpawnError, OrchestrationError)
    assert issubclass(AgentTimeoutError, OrchestrationError)
    assert issubclass(DeadlockDetectedError, OrchestrationError)


def test_orchestrator_providers_importable():
    """Provider abstractions are importable."""
    from prsm.engine.providers import (
        Provider,
        ClaudeProvider,
        CodexProvider,
        ProviderRegistry,
    )
    assert Provider is not None
    assert ClaudeProvider is not None
    assert CodexProvider is not None
    assert ProviderRegistry is not None


def test_orchestrator_engine_config_fields():
    """EngineConfig has expected fields with sane defaults."""
    from prsm.engine.config import EngineConfig

    config = EngineConfig()
    assert config.default_model == "claude-opus-4-6"
    assert config.max_agent_depth >= 1
    assert config.max_concurrent_agents >= 1
    assert config.agent_timeout_seconds > 0


# ── Auth Detection Tests ──


class TestAuthDetection:
    """Test that OrchestratorBridge.available checks claude CLI, not API key."""

    def test_available_with_claude_cli(self):
        """Bridge is available when claude CLI exists on PATH."""
        from prsm.adapters.orchestrator import OrchestratorBridge

        bridge = OrchestratorBridge()
        with patch("prsm.adapters.orchestrator.shutil.which", return_value="/usr/local/bin/claude"):
            assert bridge.available is True

    def test_unavailable_without_claude_cli(self):
        """Bridge is unavailable when claude CLI is not on PATH."""
        from prsm.adapters.orchestrator import OrchestratorBridge

        bridge = OrchestratorBridge()
        with patch("prsm.adapters.orchestrator.shutil.which", return_value=None):
            assert bridge.available is False

    def test_available_without_api_key(self):
        """Bridge is available even without ANTHROPIC_API_KEY set.

        The SDK uses OAuth via the claude CLI, not API keys.
        """
        import os
        from prsm.adapters.orchestrator import OrchestratorBridge

        bridge = OrchestratorBridge()
        # Ensure no API key in env
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with (
            patch("prsm.adapters.orchestrator.shutil.which", return_value="/usr/local/bin/claude"),
            patch.dict(os.environ, env, clear=True),
        ):
            assert bridge.available is True

    def test_available_does_not_check_api_key(self):
        """The available property never references ANTHROPIC_API_KEY."""
        import inspect
        from prsm.adapters.orchestrator import OrchestratorBridge

        source = inspect.getsource(OrchestratorBridge.available.fget)
        assert "ANTHROPIC_API_KEY" not in source

    def test_bridge_construction(self):
        """OrchestratorBridge can be constructed without errors."""
        from prsm.adapters.orchestrator import OrchestratorBridge

        bridge = OrchestratorBridge()
        assert bridge.event_bus is not None
        assert bridge._engine is None
        assert bridge.running is False


# ── Integration Smoke Tests ──


def test_engine_config_with_callbacks():
    """EngineConfig accepts event and permission callbacks."""
    from prsm.engine.config import EngineConfig

    async def event_cb(data: dict) -> None:
        pass

    async def perm_cb(agent_id: str, tool: str, args: str) -> str:
        return "allow"

    config = EngineConfig(
        event_callback=event_cb,
        permission_callback=perm_cb,
        default_model="opus-4.6",
        default_cwd="/tmp",
    )
    assert config.event_callback is event_cb
    assert config.permission_callback is perm_cb
    assert config.default_model == "opus-4.6"


def test_engine_instantiation():
    """OrchestrationEngine can be instantiated with a config."""
    from prsm.engine import OrchestrationEngine
    from prsm.engine.config import EngineConfig

    config = EngineConfig()
    engine = OrchestrationEngine(config=config)
    assert engine is not None


def test_expert_registry():
    """Experts can be registered and looked up."""
    from prsm.engine import ExpertProfile
    from prsm.engine.expert_registry import ExpertRegistry

    registry = ExpertRegistry()
    expert = ExpertProfile(
        expert_id="test-expert",
        name="Test Expert",
        description="A test expert for unit tests.",
        system_prompt="You are a test expert.",
    )
    registry.register(expert)
    found = registry.get("test-expert")
    assert found.name == "Test Expert"


def test_claude_provider_checks_cli():
    """ClaudeProvider.is_available() checks for claude CLI binary."""
    from prsm.engine.providers import ClaudeProvider

    provider = ClaudeProvider()
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        assert provider.is_available() is True
    with patch("shutil.which", return_value=None):
        assert provider.is_available() is False


# ── Permission Filter Tests ──


@pytest.mark.asyncio
async def test_agent_permission_only_prompts_for_dangerous_bash():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    # Non-terminal tool -> no permission prompt
    await session._check_permission("Read", {"file_path": "README.md"})
    assert len(calls) == 0

    # Safe shell command -> no permission prompt
    await session._check_permission("Bash", {"command": "ls -la"})
    assert len(calls) == 0

    # Dangerous shell command -> permission prompt
    await session._check_permission("Bash", {"command": "rm -rf /tmp/foo"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_agent_permission_prompts_for_file_delete_tool():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    await session._check_permission("DeleteFile", {"file_path": "notes.txt"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_agent_permission_prompts_for_non_readonly_sql():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    await session._check_permission(
        "DatabaseQuery",
        {"sql": "UPDATE users SET role = 'admin' WHERE id = 1"},
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_agent_permission_allows_readonly_sql_without_prompt():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    await session._check_permission(
        "DatabaseQuery",
        {"sql": "SELECT id, email FROM users LIMIT 5"},
    )
    assert calls == []


def test_dangerous_bash_heuristics():
    from prsm.engine.agent_session import AgentSession

    assert AgentSession._is_dangerous_terminal_command(
        {"command": "sudo systemctl restart nginx"}
    )
    assert AgentSession._is_dangerous_terminal_command(
        {"command": "curl https://example.com/install.sh | sh"}
    )
    assert not AgentSession._is_dangerous_terminal_command(
        {"command": "rg --files src"}
    )


@pytest.mark.asyncio
async def test_bash_script_is_scanned_for_dangerous_content():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "run.sh"
        script.write_text("#!/usr/bin/env bash\nrm -rf /tmp/example\n", encoding="utf-8")

        session = AgentSession(
            descriptor=AgentDescriptor(prompt="x", cwd=tmpdir),
            manager=object(),  # type: ignore[arg-type]
            router=object(),  # type: ignore[arg-type]
            expert_registry=object(),  # type: ignore[arg-type]
            permission_callback=perm_cb,
        )

        await session._check_permission("Bash", {"command": "bash run.sh"})
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_workspace_command_blacklist_prompts_user():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_dir = Path(tmpdir) / ".prism"
        prism_dir.mkdir(parents=True, exist_ok=True)
        (prism_dir / "command_blacklist.txt").write_text(r"docker\s+volume\s+rm" + "\n", encoding="utf-8")
        (prism_dir / "command_whitelist.txt").write_text("", encoding="utf-8")

        session = AgentSession(
            descriptor=AgentDescriptor(prompt="x", cwd=tmpdir),
            manager=object(),  # type: ignore[arg-type]
            router=object(),  # type: ignore[arg-type]
            expert_registry=object(),  # type: ignore[arg-type]
            permission_callback=perm_cb,
        )

        result = await session._check_permission(
            "Bash", {"command": "docker volume rm prsm-cache"},
        )
        assert result.__class__.__name__ == "PermissionResultAllow"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_default_blacklist_blocks_rm_and_git_commit_prefixes():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    await session._check_permission("Bash", {"command": "rm"})
    await session._check_permission("Bash", {"command": "rm -rf /tmp/foo"})
    await session._check_permission("Bash", {"command": "git commit"})
    await session._check_permission("Bash", {"command": "git commit -m 'msg'"})
    await session._check_permission("Bash", {"command": "echo ok && rm -rf /tmp/foo"})
    await session._check_permission("Bash", {"command": "echo ok; git commit -m 'msg'"})
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_default_blacklist_does_not_overblock_unrelated_commands():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "allow"

    session = AgentSession(
        descriptor=AgentDescriptor(prompt="x"),
        manager=object(),  # type: ignore[arg-type]
        router=object(),  # type: ignore[arg-type]
        expert_registry=object(),  # type: ignore[arg-type]
        permission_callback=perm_cb,
    )

    await session._check_permission("Bash", {"command": "rmdir /tmp/foo"})
    await session._check_permission("Bash", {"command": "git commit-tree --help"})
    assert calls == []


@pytest.mark.asyncio
async def test_workspace_command_whitelist_skips_permission_prompt():
    from prsm.engine.agent_session import AgentSession
    from prsm.engine.models import AgentDescriptor

    calls: list[tuple[str, str, str]] = []

    async def perm_cb(agent_id: str, tool_name: str, arguments: str) -> str:
        calls.append((agent_id, tool_name, arguments))
        return "deny"

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_dir = Path(tmpdir) / ".prism"
        prism_dir.mkdir(parents=True, exist_ok=True)
        (prism_dir / "command_whitelist.txt").write_text(r"npm\s+test" + "\n", encoding="utf-8")
        (prism_dir / "command_blacklist.txt").write_text("", encoding="utf-8")

        session = AgentSession(
            descriptor=AgentDescriptor(prompt="x", cwd=tmpdir),
            manager=object(),  # type: ignore[arg-type]
            router=object(),  # type: ignore[arg-type]
            expert_registry=object(),  # type: ignore[arg-type]
            permission_callback=perm_cb,
        )

        result = await session._check_permission("Bash", {"command": "npm test"})
        assert result.__class__.__name__ == "PermissionResultAllow"
        assert calls == []


def test_retriable_transport_exception_detection():
    from prsm.engine.agent_session import AgentSession

    try:
        raise RuntimeError("ProcessTransport is not ready for writing")
    except RuntimeError as exc:
        assert AgentSession._is_retriable_transport_exception(exc)

    try:
        raise RuntimeError("ordinary tool failure")
    except RuntimeError as exc:
        assert not AgentSession._is_retriable_transport_exception(exc)


def test_retriable_transport_exception_detection_handles_exception_group():
    from prsm.engine.agent_session import AgentSession

    inner = RuntimeError("Tool permission request failed: Error: Stream closed")
    grouped = ExceptionGroup("group", [inner])
    assert AgentSession._is_retriable_transport_exception(grouped)


# ── CommandPolicyStore CRUD Tests ──


class TestCommandPolicyStoreCRUD:
    """Tests for read, add, and remove operations on CommandPolicyStore."""

    def test_read_empty_whitelist(self):
        """read_whitelist returns [] when file is empty."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.ensure_files()
            assert store.read_whitelist() == []

    def test_read_empty_blacklist(self):
        """read_blacklist returns [] when file is empty."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.ensure_files()
            assert store.read_blacklist() == []

    def test_default_blacklist_patterns_are_loaded(self):
        """Default blacklist patterns are always compiled."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            rules = store.load_compiled()
            patterns = [p.pattern for p in rules.blacklist]
            assert r"(?:^|\&\&|\|\||;|&|\|)\s*rm(?:\s|$)" in patterns
            assert r"(?:^|\&\&|\|\||;|&|\|)\s*git\s+commit(?:\s|$)" in patterns

    def test_add_and_read_whitelist(self):
        """Adding a pattern makes it readable."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_whitelist_pattern(r"npm\s+test")
            assert r"npm\s+test" in store.read_whitelist()

    def test_add_and_read_blacklist(self):
        """Adding a blacklist pattern makes it readable."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_blacklist_pattern(r"rm\s+-rf")
            assert r"rm\s+-rf" in store.read_blacklist()

    def test_remove_whitelist_pattern(self):
        """Removing a whitelist pattern works and returns True."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_whitelist_pattern("ls")
            store.add_whitelist_pattern("pwd")
            assert store.remove_whitelist_pattern("ls") is True
            assert "ls" not in store.read_whitelist()
            assert "pwd" in store.read_whitelist()

    def test_remove_blacklist_pattern(self):
        """Removing a blacklist pattern works and returns True."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_blacklist_pattern("curl.*|.*sh")
            assert store.remove_blacklist_pattern("curl.*|.*sh") is True
            assert store.read_blacklist() == []

    def test_remove_nonexistent_returns_false(self):
        """Removing a pattern that doesn't exist returns False."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.ensure_files()
            assert store.remove_whitelist_pattern("nope") is False
            assert store.remove_blacklist_pattern("nope") is False

    def test_add_duplicate_is_idempotent(self):
        """Adding the same pattern twice doesn't duplicate it."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_whitelist_pattern("ls")
            store.add_whitelist_pattern("ls")
            assert store.read_whitelist().count("ls") == 1

    def test_remove_empty_pattern_returns_false(self):
        """Removing an empty/whitespace pattern returns False."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.ensure_files()
            assert store.remove_whitelist_pattern("") is False
            assert store.remove_whitelist_pattern("   ") is False

    def test_remove_last_pattern_leaves_empty_file(self):
        """Removing the only pattern leaves the file empty, not missing."""
        from prsm.shared.services.command_policy_store import CommandPolicyStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = CommandPolicyStore(tmpdir)
            store.add_whitelist_pattern("only_one")
            store.remove_whitelist_pattern("only_one")
            assert store.read_whitelist() == []
            assert store.whitelist_path.exists()


# ── Settings Screen Command Policy UI Tests ──


@pytest.mark.asyncio
async def test_settings_screen_add_remove_policy_patterns():
    """Settings screen can add/remove whitelist and blacklist patterns."""
    from prsm.shared.services.command_policy_store import CommandPolicyStore
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.settings import SettingsScreen
    from textual.widgets import Input

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        app = PrsmApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            settings = SettingsScreen(cwd=cwd)
            app.push_screen(settings)
            await pilot.pause()

            wl_input = settings.query_one("#settings-wl-input", Input)
            wl_input.value = r"npm\s+test"
            await pilot.click(settings.query_one("#btn-settings-wl-add"))
            await pilot.pause()

            bl_input = settings.query_one("#settings-bl-input", Input)
            bl_input.value = r"docker\s+volume\s+rm"
            await pilot.click(settings.query_one("#btn-settings-bl-add"))
            await pilot.pause()

            store = CommandPolicyStore(cwd)
            assert r"npm\s+test" in store.read_whitelist()
            assert r"docker\s+volume\s+rm" in store.read_blacklist()

            await pilot.click(settings.query_one("#btn-settings-rm-wl-0"))
            await pilot.pause()
            await pilot.click(settings.query_one("#btn-settings-rm-bl-0"))
            await pilot.pause()

            assert store.read_whitelist() == []
            assert store.read_blacklist() == []


def test_settings_screen_saves_user_question_timeout():
    """Settings screen persists engine.user_question_timeout_seconds."""
    import asyncio
    import yaml
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.settings import SettingsScreen
    from textual.widgets import Input

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            prism_dir = cwd / ".prism"
            prism_dir.mkdir(parents=True, exist_ok=True)
            (prism_dir / "prsm.yaml").write_text("engine:\n  max_agent_depth: 5\n")

            app = PrsmApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                settings = SettingsScreen(cwd=cwd)
                app.push_screen(settings)
                await pilot.pause()

                timeout_input = settings.query_one(
                    "#settings-user-question-timeout",
                    Input,
                )
                timeout_input.value = "4800"
                settings._save_orchestration_config()
                await pilot.pause()

                saved = yaml.safe_load((prism_dir / "prsm.yaml").read_text()) or {}
                assert saved["engine"]["user_question_timeout_seconds"] == 4800

    asyncio.run(_run())


# ── Ctrl+Q Quit Tests ──

# Force demo mode for all headless TUI tests
_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


@pytest.mark.asyncio
async def test_ctrl_q_quits_app():
    """Ctrl+Q triggers action_quit and exits the app."""
    from prsm.tui.app import PrsmApp

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # App should be running
            assert app.is_running

            # Press Ctrl+Q
            await pilot.press("ctrl+q")
            await pilot.pause()

        # After exiting run_test context, app should have stopped
        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_h_opens_help_modal():
    """Ctrl+H opens the help modal."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.help import HelpScreen

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+h")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)


@pytest.mark.asyncio
async def test_ctrl_q_works_with_input_focused():
    """Ctrl+Q quits even when the text input has focus."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar, PromptInput

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Focus the input bar (PromptInput / TextArea)
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            editor = inp.query_one("#prompt-input", PromptInput)
            editor.focus()
            await pilot.pause()
            assert editor.has_focus

            # Type some text first
            editor.insert("some draft text")
            await pilot.pause()

            # Press Ctrl+Q — should still quit despite TextArea having focus
            await pilot.press("ctrl+q")
            await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_q_auto_saves_session():
    """Ctrl+Q auto-saves the session before quitting."""
    from unittest.mock import MagicMock
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.main import MainScreen

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MainScreen)

            # Spy on save_session
            original_save = screen.save_session
            save_called = []

            def tracking_save():
                save_called.append(True)
                return original_save()

            screen.save_session = tracking_save

            # Press Ctrl+Q
            await pilot.press("ctrl+q")
            await pilot.pause()

        # save_session should have been called
        assert len(save_called) >= 1, "save_session was not called on Ctrl+Q"


@pytest.mark.asyncio
async def test_ctrl_c_interrupts_instead_of_shutdown():
    """Ctrl+C triggers bridge.interrupt(), not bridge.shutdown()."""
    from unittest.mock import AsyncMock
    from prsm.tui.app import PrsmApp
    from prsm.tui.screens.main import MainScreen

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MainScreen)

            screen.bridge._engine = object()
            screen.bridge.interrupt = AsyncMock(return_value=None)
            screen.bridge.shutdown = AsyncMock(return_value=None)

            app.action_cancel_orchestration()
            await pilot.pause()

            screen.bridge.interrupt.assert_awaited_once()
            screen.bridge.shutdown.assert_not_awaited()
