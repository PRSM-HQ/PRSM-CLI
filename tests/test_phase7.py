"""Phase 7 tests — transport parity, per-agent plugins, intelligent auto-loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.shared.services.plugins import PluginConfig, PluginManager, PluginMatcher
from prsm.engine.models import (
    AgentDescriptor,
    AgentRole,
    ExpertProfile,
    PermissionMode,
    SpawnRequest,
)

# Force demo mode in all headless TUI tests
_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


# ── Transport Parity Tests ──


class TestTransportParity:
    def test_add_http_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_remote("github", "http", "https://mcp.github.com/v1",
                          headers={"Authorization": "Bearer tok"})

            configs = pm.get_mcp_server_configs()
            assert "github" in configs
            assert configs["github"]["type"] == "http"
            assert configs["github"]["url"] == "https://mcp.github.com/v1"
            assert configs["github"]["headers"]["Authorization"] == "Bearer tok"

    def test_add_sse_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_remote("events", "sse", "https://example.com/sse")

            configs = pm.get_mcp_server_configs()
            assert configs["events"]["type"] == "sse"
            assert configs["events"]["url"] == "https://example.com/sse"
            assert "headers" not in configs["events"]

    def test_add_remote_invalid_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            with pytest.raises(ValueError, match="http.*sse"):
                pm.add_remote("bad", "websocket", "ws://example.com")

    def test_add_json_http_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_json("custom", {
                "type": "http",
                "url": "https://example.com/mcp",
                "headers": {"X-Key": "val"},
                "tags": ["api"],
            })

            plugins = pm.list_plugins()
            assert len(plugins) == 1
            assert plugins[0].type == "http"
            assert plugins[0].url == "https://example.com/mcp"
            assert plugins[0].tags == ["api"]

    def test_add_json_stdio_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_json("fs", {
                "command": "npx",
                "args": ["-y", "@mcp/server-filesystem"],
                "tags": ["filesystem"],
            })

            configs = pm.get_mcp_server_configs()
            assert configs["fs"]["type"] == "stdio"
            assert configs["fs"]["command"] == "npx"

    def test_add_json_missing_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            with pytest.raises(ValueError):
                pm.add_json("bad", {"type": "stdio"})

    def test_add_json_missing_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            with pytest.raises(ValueError):
                pm.add_json("bad", {"type": "http"})

    def test_backward_compat_no_type_field(self):
        """Existing plugins.json without 'type' field still loads as stdio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plugins.json"
            path.write_text(json.dumps({
                "mcpServers": {
                    "old": {"command": "npx", "args": ["-y", "server"]}
                }
            }))

            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            configs = pm.get_mcp_server_configs()
            assert configs["old"]["type"] == "stdio"
            assert configs["old"]["command"] == "npx"

    def test_persistence_all_types(self):
        """All transport types survive save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm1 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm1.add("stdio-srv", "node", ["server.js"])
            pm1.add_remote("http-srv", "http", "https://example.com")
            pm1.add_remote("sse-srv", "sse", "https://example.com/sse",
                           headers={"Auth": "Bearer x"})

            pm2 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            assert len(pm2.list_plugins()) == 3

            configs = pm2.get_mcp_server_configs()
            assert configs["stdio-srv"]["type"] == "stdio"
            assert configs["http-srv"]["type"] == "http"
            assert configs["sse-srv"]["type"] == "sse"
            assert configs["sse-srv"]["headers"]["Auth"] == "Bearer x"

    def test_tags_persisted(self):
        """Plugin tags survive save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm1 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm1.add_json("tagged", {
                "command": "node", "args": [],
                "tags": ["db", "sql"],
            })

            pm2 = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            assert pm2.list_plugins()[0].tags == ["db", "sql"]

    def test_invalid_stdio_no_command_skipped(self):
        """stdio plugin without command is silently skipped during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plugins.json"
            path.write_text(json.dumps({
                "mcpServers": {
                    "bad": {"type": "stdio", "args": []}
                }
            }))

            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            assert len(pm.list_plugins()) == 0

    def test_invalid_http_no_url_skipped(self):
        """http plugin without url is silently skipped during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plugins.json"
            path.write_text(json.dumps({
                "mcpServers": {
                    "bad": {"type": "http"}
                }
            }))

            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            assert len(pm.list_plugins()) == 0

    def test_stdio_plugin_format_unchanged(self):
        """Existing Phase 6 stdio tests still pass — format is stable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add("test", "npx", ["-y", "server"])

            configs = pm.get_mcp_server_configs()
            assert configs["test"]["type"] == "stdio"
            assert configs["test"]["command"] == "npx"
            assert configs["test"]["args"] == ["-y", "server"]


# ── Plugin Matcher / Auto-Loading Tests ──


class TestPluginMatcher:
    def test_worker_gets_all(self):
        """Workers get all plugins regardless of tags."""
        plugins = [
            PluginConfig(name="a", command="x", tags=["db"]),
            PluginConfig(name="b", command="y", tags=["code"]),
            PluginConfig(name="c", command="z"),  # untagged
        ]
        matched = PluginMatcher.match_plugins(plugins, "anything", role="worker")
        assert set(matched) == {"a", "b", "c"}

    def test_master_gets_none(self):
        """Master agents get no auto-loaded plugins."""
        plugins = [
            PluginConfig(name="a", command="x", tags=["all"]),
            PluginConfig(name="b", command="y"),  # untagged
        ]
        matched = PluginMatcher.match_plugins(plugins, "anything", role="master")
        assert matched == []

    def test_expert_tag_match(self):
        """Experts get plugins matching prompt keywords."""
        plugins = [
            PluginConfig(name="db-plugin", command="x", tags=["database", "sql"]),
            PluginConfig(name="fs-plugin", command="y", tags=["filesystem"]),
        ]
        matched = PluginMatcher.match_plugins(
            plugins, "Query the database for user records", role="expert"
        )
        assert "db-plugin" in matched
        assert "fs-plugin" not in matched

    def test_expert_no_match(self):
        """Experts don't get plugins with non-matching tags."""
        plugins = [
            PluginConfig(name="db-plugin", command="x", tags=["database"]),
        ]
        matched = PluginMatcher.match_plugins(
            plugins, "Fix the CSS styles", role="expert"
        )
        assert matched == []

    def test_untagged_included_for_non_master(self):
        """Untagged plugins go to everyone except master."""
        plugins = [PluginConfig(name="generic", command="x")]

        assert PluginMatcher.match_plugins(plugins, "anything", "worker") == ["generic"]
        assert PluginMatcher.match_plugins(plugins, "anything", "expert") == ["generic"]
        assert PluginMatcher.match_plugins(plugins, "anything", "reviewer") == ["generic"]
        assert PluginMatcher.match_plugins(plugins, "anything", "master") == []

    def test_case_insensitive_tag_matching(self):
        """Tag matching is case-insensitive."""
        plugins = [
            PluginConfig(name="gh", command="x", tags=["GitHub"]),
        ]
        matched = PluginMatcher.match_plugins(
            plugins, "check the github repository", role="expert"
        )
        assert "gh" in matched

    def test_get_plugins_for_agent(self):
        """PluginManager.get_plugins_for_agent() integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_json("db", {"command": "db-server", "args": [], "tags": ["database"]})
            pm.add_json("fs", {"command": "fs-server", "args": [], "tags": ["filesystem"]})
            pm.add("generic", "echo", ["hello"])

            # Worker gets all
            worker_configs = pm.get_plugins_for_agent("do stuff", role="worker")
            assert set(worker_configs.keys()) == {"db", "fs", "generic"}

            # Expert with database prompt gets db + generic
            expert_configs = pm.get_plugins_for_agent(
                "query the database", role="expert"
            )
            assert "db" in expert_configs
            assert "generic" in expert_configs
            assert "fs" not in expert_configs

            # Master gets none
            master_configs = pm.get_plugins_for_agent("orchestrate", role="master")
            assert master_configs == {}


# ── Per-Agent Resolution Tests ──


class TestPerAgentModels:
    """Test that models correctly carry per-agent MCP server fields."""

    def test_spawn_request_mcp_servers(self):
        req = SpawnRequest(
            parent_id="parent",
            prompt="test",
            mcp_servers={"db": {"type": "http", "url": "https://db.example.com"}},
        )
        assert req.mcp_servers is not None
        assert "db" in req.mcp_servers

    def test_spawn_request_exclude_plugins(self):
        req = SpawnRequest(
            parent_id="parent",
            prompt="test",
            exclude_plugins=["filesystem", "github"],
        )
        assert req.exclude_plugins == ["filesystem", "github"]

    def test_spawn_request_defaults_none(self):
        req = SpawnRequest(parent_id=None, prompt="test")
        assert req.mcp_servers is None
        assert req.exclude_plugins is None

    def test_agent_descriptor_mcp_servers(self):
        desc = AgentDescriptor(
            mcp_servers={"test": {"type": "stdio", "command": "echo"}},
            exclude_plugins=["removed"],
        )
        assert desc.mcp_servers is not None
        assert desc.exclude_plugins == ["removed"]

    def test_expert_profile_mcp_servers(self):
        profile = ExpertProfile(
            expert_id="test",
            name="Test Expert",
            description="testing",
            system_prompt="test",
            mcp_servers={"db": {"type": "http", "url": "https://db.test.com"}},
        )
        assert profile.mcp_servers is not None
        assert "db" in profile.mcp_servers


# ── Agent Manager Plugin Resolution Tests ──


class TestAgentPluginResolution:
    """Test _resolve_agent_plugins logic (unit tests against AgentManager)."""

    def _make_manager(self, global_plugins=None, plugin_manager=None):
        """Create a minimal AgentManager for testing plugin resolution."""
        from prsm.engine.agent_manager import AgentManager
        from prsm.engine.message_router import MessageRouter
        from prsm.engine.expert_registry import ExpertRegistry
        from prsm.engine.config import EngineConfig

        router = MessageRouter()
        registry = ExpertRegistry()
        config = EngineConfig()

        return AgentManager(
            router=router,
            expert_registry=registry,
            config=config,
            plugin_mcp_servers=global_plugins or {},
            plugin_manager=plugin_manager,
        )

    def test_resolve_global_default(self):
        """No overrides → all global plugins."""
        manager = self._make_manager(global_plugins={
            "fs": {"type": "stdio", "command": "fs-server"},
            "db": {"type": "http", "url": "https://db.example.com"},
        })
        request = SpawnRequest(parent_id=None, prompt="do stuff")
        result = manager._resolve_agent_plugins(request)
        assert set(result.keys()) == {"fs", "db"}

    def test_resolve_explicit_mcp_servers(self):
        """Per-agent mcp_servers merges on top of global."""
        manager = self._make_manager(global_plugins={
            "fs": {"type": "stdio", "command": "old-fs"},
        })
        request = SpawnRequest(
            parent_id=None,
            prompt="test",
            mcp_servers={
                "fs": {"type": "stdio", "command": "new-fs"},
                "extra": {"type": "http", "url": "https://extra.com"},
            },
        )
        result = manager._resolve_agent_plugins(request)
        assert result["fs"]["command"] == "new-fs"
        assert "extra" in result

    def test_resolve_exclude_plugins(self):
        """exclude_plugins removes from effective set."""
        manager = self._make_manager(global_plugins={
            "fs": {"type": "stdio", "command": "fs-server"},
            "db": {"type": "http", "url": "https://db.example.com"},
        })
        request = SpawnRequest(
            parent_id=None,
            prompt="test",
            exclude_plugins=["db"],
        )
        result = manager._resolve_agent_plugins(request)
        assert "fs" in result
        assert "db" not in result

    def test_resolve_explicit_plus_exclude(self):
        """Both explicit and exclude work together."""
        manager = self._make_manager(global_plugins={
            "fs": {"type": "stdio", "command": "fs-server"},
            "db": {"type": "http", "url": "https://db.example.com"},
        })
        request = SpawnRequest(
            parent_id=None,
            prompt="test",
            mcp_servers={"extra": {"type": "http", "url": "https://x.com"}},
            exclude_plugins=["db"],
        )
        result = manager._resolve_agent_plugins(request)
        assert "fs" in result
        assert "extra" in result
        assert "db" not in result

    def test_resolve_auto_match_with_plugin_manager(self):
        """When plugin_manager present and no explicit servers, auto-match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_json("db", {"command": "db-server", "args": [], "tags": ["database"]})
            pm.add_json("fs", {"command": "fs-server", "args": [], "tags": ["filesystem"]})

            manager = self._make_manager(
                global_plugins=pm.get_mcp_server_configs(),
                plugin_manager=pm,
            )

            # Expert prompt about database → only db matched
            request = SpawnRequest(
                parent_id=None,
                prompt="query the database",
                role=AgentRole.EXPERT,
            )
            result = manager._resolve_agent_plugins(request)
            assert "db" in result
            assert "fs" not in result

    def test_resolve_auto_match_worker_gets_all(self):
        """Workers get all plugins via auto-match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PluginManager(project_dir=Path(tmpdir), cwd=Path(tmpdir))
            pm.add_json("db", {"command": "db-server", "args": [], "tags": ["database"]})
            pm.add_json("fs", {"command": "fs-server", "args": [], "tags": ["filesystem"]})

            manager = self._make_manager(
                global_plugins=pm.get_mcp_server_configs(),
                plugin_manager=pm,
            )

            request = SpawnRequest(
                parent_id=None,
                prompt="do anything",
                role=AgentRole.WORKER,
            )
            result = manager._resolve_agent_plugins(request)
            assert set(result.keys()) == {"db", "fs"}


# ── Headless TUI Tests — Plugin Commands ──


def _capture_tool_log_writes(tl):
    """Spy on ToolLog.write() to capture all logged messages."""
    captured = []
    original_write = tl.write

    def spy_write(content, *args, **kwargs):
        captured.append(str(content))
        return original_write(content, *args, **kwargs)

    tl.write = spy_write
    return captured


@pytest.mark.asyncio
async def test_plugin_add_http_command():
    """Typing /plugin add NAME --type http --url URL works."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert("/plugin add myhttp --type http --url https://example.com/mcp")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "myhttp" in log_text
            assert "http" in log_text.lower()


@pytest.mark.asyncio
async def test_plugin_add_json_command():
    """Typing /plugin add-json NAME {...} works."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            editor.insert('/plugin add-json test {"command":"echo","args":["hi"]}')
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "test" in log_text
            assert "JSON" in log_text or "added" in log_text.lower()


@pytest.mark.asyncio
async def test_plugin_tag_command():
    """Typing /plugin tag NAME tag1 tag2 sets tags."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            # First add a plugin, then tag it
            editor.insert("/plugin add testplugin echo hello")
            await pilot.press("enter")
            await pilot.pause()

            editor.insert("/plugin tag testplugin database sql")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "tag" in log_text.lower() or "Tags" in log_text


@pytest.mark.asyncio
async def test_plugin_list_shows_type():
    """Typing /plugin list shows transport type."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar
    from prsm.tui.widgets.tool_log import ToolLog

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            tl = screen.query_one("#tool-log", ToolLog)
            captured = _capture_tool_log_writes(tl)

            editor = inp.query_one("#prompt-input")
            editor.focus()
            await pilot.pause()

            # Add a plugin first
            editor.insert("/plugin add testplug echo hello")
            await pilot.press("enter")
            await pilot.pause()

            editor.insert("/plugin list")
            await pilot.press("enter")
            await pilot.pause()

            log_text = " ".join(captured)
            assert "stdio" in log_text or "testplug" in log_text
