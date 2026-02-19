from __future__ import annotations

from unittest.mock import MagicMock

from prsm.engine.mcp_server.tools import OrchestrationTools


def _tools() -> OrchestrationTools:
    return OrchestrationTools(
        agent_id="agent-test",
        manager=MagicMock(),
        router=MagicMock(),
        expert_registry=MagicMock(),
        cwd=".",
    )


def test_package_manager_installs_require_permission() -> None:
    tools = _tools()
    assert tools._evaluate_bash_permission("apt install -y ripgrep") is True
    assert tools._evaluate_bash_permission("sudo apt-get install python3-pip") is True
    assert tools._evaluate_bash_permission("snap install code --classic") is True
    assert tools._evaluate_bash_permission("pip install textual") is True
    assert tools._evaluate_bash_permission("python -m pip install pytest") is True


def test_safe_command_does_not_require_permission() -> None:
    tools = _tools()
    assert tools._evaluate_bash_permission("rg TODO src") is False
