"""Plugin manager — load, register, and merge external MCP server configs.

Supports three transport types (matching Claude CLI):
- stdio: local subprocess servers (command + args + env)
- http: streamable HTTP servers (url + headers)
- sse: Server-Sent Events servers (url + headers)

Supports two config sources:
1. Project-level: ~/.prsm/projects/{ID}/plugins.json
2. Workspace-level: .mcp.json in cwd (with .prsm.json fallback)

Config format (both files):
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "tags": ["filesystem", "code"]
        },
        "github": {
            "type": "http",
            "url": "https://mcp.github.com/v1",
            "headers": {"Authorization": "Bearer ghp_..."},
            "tags": ["github", "vcs"]
        }
    }
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """A single MCP server plugin configuration.

    Supports three transport types:
    - stdio: command + args + env (local subprocess)
    - http: url + headers (streamable HTTP)
    - sse: url + headers (Server-Sent Events)
    """

    name: str
    type: str = "stdio"

    # stdio fields
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None

    # http / sse fields
    url: str | None = None
    headers: dict[str, str] | None = None

    # auto-loading metadata
    tags: list[str] = field(default_factory=list)


class PluginMatcher:
    """Matches plugins to agents based on tags and prompt keywords.

    Rules:
    1. Master agents get no auto-loaded plugins (they delegate).
    2. Workers get all plugins regardless of tags.
    3. Experts/reviewers get untagged plugins + tag-matched plugins.
    4. Tag matching: any tag found as substring in prompt → include.
    """

    @staticmethod
    def match_plugins(
        plugins: list[PluginConfig],
        prompt: str,
        role: str = "worker",
    ) -> list[str]:
        """Return names of plugins that match the agent's context."""
        if role == "master":
            return []

        prompt_lower = prompt.lower()
        matched: list[str] = []

        for plugin in plugins:
            # Untagged plugins go to everyone except master
            if not plugin.tags:
                matched.append(plugin.name)
                continue

            # Workers get everything
            if role == "worker":
                matched.append(plugin.name)
                continue

            # For experts/reviewers: match tags against prompt
            for tag in plugin.tags:
                if tag.lower() in prompt_lower:
                    matched.append(plugin.name)
                    break

        return matched


class PluginManager:
    """Manages external MCP server plugins for agent sessions."""

    def __init__(
        self,
        project_dir: Path | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._cwd = cwd or Path.cwd()
        self._plugins: dict[str, PluginConfig] = {}
        self._load_project_plugins()
        self._load_workspace_plugins()

    def _project_plugins_path(self) -> Path | None:
        if self._project_dir:
            return self._project_dir / "plugins.json"
        return None

    def _workspace_plugins_paths(self) -> list[Path]:
        """Workspace config paths in load order.

        `.mcp.json` is preferred and loaded last so it overrides `.prsm.json`
        when both define the same server name.
        """
        return [self._cwd / ".prsm.json", self._cwd / ".mcp.json"]

    def _parse_plugin_config(self, name: str, cfg: dict) -> PluginConfig | None:
        """Parse a single plugin config from JSON dict.

        Returns None if required fields are missing.
        """
        plugin_type = cfg.get("type", "stdio")

        if plugin_type == "stdio" and not cfg.get("command"):
            logger.warning("stdio plugin '%s' missing 'command', skipping", name)
            return None
        if plugin_type in ("http", "sse") and not cfg.get("url"):
            logger.warning("%s plugin '%s' missing 'url', skipping", plugin_type, name)
            return None

        return PluginConfig(
            name=name,
            type=plugin_type,
            command=cfg.get("command"),
            args=cfg.get("args", []),
            env=cfg.get("env"),
            url=cfg.get("url"),
            headers=cfg.get("headers"),
            tags=cfg.get("tags", []),
        )

    def _load_project_plugins(self) -> None:
        path = self._project_plugins_path()
        if not path:
            logger.debug("No project_dir set; skipping project plugins")
            return
        if not path.exists():
            logger.debug("Project plugins not found at %s", path)
            return
        try:
            data = json.loads(path.read_text())
            names = list(data.get("mcpServers", {}).keys())
            for name, cfg in data.get("mcpServers", {}).items():
                parsed = self._parse_plugin_config(name, cfg)
                if parsed:
                    self._plugins[name] = parsed
            logger.info(
                "Loaded project plugins from %s: [%s]",
                path, ", ".join(names) if names else "none",
            )
        except Exception:
            logger.warning("Failed to load project plugins from %s", path)

    def _load_workspace_plugins(self) -> None:
        for path in self._workspace_plugins_paths():
            if not path.exists():
                logger.debug("Workspace plugin file not found: %s", path)
                continue
            try:
                data = json.loads(path.read_text())
                names = list(data.get("mcpServers", {}).keys())
                for name, cfg in data.get("mcpServers", {}).items():
                    parsed = self._parse_plugin_config(name, cfg)
                    if parsed:
                        self._plugins[name] = parsed
                logger.info(
                    "Loaded workspace plugins from %s: [%s]",
                    path, ", ".join(names) if names else "none",
                )
            except Exception:
                logger.warning("Failed to load workspace plugins from %s", path)

    def add(self, name: str, command: str, args: list[str]) -> None:
        """Register a stdio plugin and persist to project-level config."""
        self._plugins[name] = PluginConfig(
            name=name, type="stdio", command=command, args=args,
        )
        self._save_project_plugins()

    def add_remote(
        self,
        name: str,
        transport_type: str,
        url: str,
        headers: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register an http or sse plugin and persist."""
        if transport_type not in ("http", "sse"):
            raise ValueError(
                f"Remote transport must be 'http' or 'sse', got '{transport_type}'"
            )
        self._plugins[name] = PluginConfig(
            name=name, type=transport_type, url=url,
            headers=headers, tags=tags or [],
        )
        self._save_project_plugins()

    def add_json(self, name: str, config: dict[str, Any]) -> None:
        """Register a plugin from raw JSON config dict.

        Accepts the same format as Claude CLI's .mcp.json entries.
        """
        parsed = self._parse_plugin_config(name, config)
        if parsed is None:
            raise ValueError(f"Invalid plugin config for '{name}'")
        self._plugins[name] = parsed
        self._save_project_plugins()

    def remove(self, name: str) -> bool:
        """Unregister a plugin. Returns False if not found."""
        if name not in self._plugins:
            return False
        del self._plugins[name]
        self._save_project_plugins()
        return True

    def list_plugins(self) -> list[PluginConfig]:
        """Return all registered plugins."""
        return list(self._plugins.values())

    def get_mcp_server_configs(self) -> dict[str, Any]:
        """Build MCP server config dict for merging into agent sessions.

        Returns dict in the format expected by ClaudeAgentOptions.mcp_servers.
        """
        configs: dict[str, Any] = {}
        for name, plugin in self._plugins.items():
            if plugin.type == "stdio":
                config: dict[str, Any] = {
                    "type": "stdio",
                    "command": plugin.command,
                    "args": plugin.args,
                }
                if plugin.env:
                    config["env"] = plugin.env
            elif plugin.type in ("http", "sse"):
                config = {
                    "type": plugin.type,
                    "url": plugin.url,
                }
                if plugin.headers:
                    config["headers"] = plugin.headers
            else:
                logger.warning("Unknown plugin type %r for %s", plugin.type, name)
                continue
            configs[name] = config
        return configs

    def get_plugins_for_agent(
        self,
        prompt: str,
        role: str = "worker",
    ) -> dict[str, Any]:
        """Build MCP server configs filtered by agent context.

        Uses PluginMatcher to select relevant plugins based on
        the agent's prompt and role.
        """
        matched = PluginMatcher.match_plugins(
            list(self._plugins.values()), prompt, role,
        )
        all_configs = self.get_mcp_server_configs()
        return {
            name: all_configs[name]
            for name in matched
            if name in all_configs
        }

    def _save_project_plugins(self) -> None:
        """Persist project-level plugin config to disk."""
        path = self._project_plugins_path()
        if not path:
            return
        data: dict[str, Any] = {"mcpServers": {}}
        for name, plugin in self._plugins.items():
            entry: dict[str, Any] = {"type": plugin.type}
            if plugin.type == "stdio":
                entry["command"] = plugin.command
                entry["args"] = plugin.args
                if plugin.env:
                    entry["env"] = plugin.env
            elif plugin.type in ("http", "sse"):
                entry["url"] = plugin.url
                if plugin.headers:
                    entry["headers"] = plugin.headers
            if plugin.tags:
                entry["tags"] = plugin.tags
            data["mcpServers"][name] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
