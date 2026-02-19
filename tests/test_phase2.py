"""Phase 2 headless tests — session, conversation switching, input history."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Tree

from prsm.tui.app import PrsmApp
from prsm.shared.models.agent import AgentState
from prsm.shared.models.message import MessageRole
from prsm.tui.widgets.agent_tree import AgentTree
from prsm.tui.widgets.conversation import ConversationView, MessageWidget
from prsm.tui.widgets.input_bar import InputBar
from prsm.tui.widgets.status_bar import StatusBar
from prsm.tui.widgets.tool_log import ToolLog


@pytest.fixture
def clean_home():
    """Provide a clean temporary HOME for tests to avoid loading stale sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = tmpdir
        # Also create a fake cwd so persistence doesn't use actual project dir
        fake_cwd = Path(tmpdir) / "test-project"
        fake_cwd.mkdir()
        old_cwd = Path.cwd()
        os.chdir(fake_cwd)
        try:
            yield tmpdir
        finally:
            os.chdir(old_cwd)
            if old_home:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)


@pytest.mark.asyncio
async def test_phase2(clean_home):
    """Run all Phase 2 checks inside a single headless app session."""
    app = PrsmApp()
    results = []

    # Force demo mode by hiding the claude CLI from shutil.which
    with patch("prsm.adapters.orchestrator.shutil.which", return_value=None):
      async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen

        # 1. Layout — all widgets present
        tree = screen.query_one("#agent-tree", AgentTree)
        conv = screen.query_one("#conversation", ConversationView)
        inp = screen.query_one("#input-bar", InputBar)
        sb = screen.query_one("#status-bar", StatusBar)
        tl = screen.query_one("#tool-log", ToolLog)
        results.append(("layout_widgets", True))

        # 2. Session populated with 6 agents
        session = screen.session
        agent_count = len(session.agents)
        results.append(("agent_count_6", agent_count == 6))

        # 3. Default active agent is root (Orchestrator)
        results.append(("active_root", session.active_agent_id == "root"))

        # 4. Conversation shows root messages (5 messages for orchestrator)
        root_msgs = session.get_messages("root")
        results.append(("root_msg_count_5", len(root_msgs) == 5))

        # 5. Message container has MessageWidgets rendered
        container = conv.query_one("#message-container")
        msg_widgets = container.query(MessageWidget)
        results.append(("root_widgets_rendered", len(msg_widgets) == 5))

        # 6. Switch to w1 (Code Explorer) by selecting tree node
        def find_agent_node(node, agent_id):
            if node.data and hasattr(node.data, "id") and node.data.id == agent_id:
                return node
            for child in node.children:
                result = find_agent_node(child, agent_id)
                if result:
                    return result
            return None

        w1_node = find_agent_node(tree.root, "w1")
        results.append(("w1_node_found", w1_node is not None))

        if w1_node:
            tree.select_node(w1_node)
            await pilot.pause()

            # Check conversation switched
            results.append(("conv_switched_w1", conv._current_agent_id == "w1"))
            w1_msgs = session.get_messages("w1")
            results.append(("w1_msg_count_5", len(w1_msgs) == 5))

            # Check status bar updated
            results.append(("sb_agent_w1", sb.agent_name == "Code Explorer"))
            results.append(("sb_model_opus", sb.model == "opus-4.6"))

        # 7. Switch to w3 (Code Writer) — should have 1 system message
        w3_node = find_agent_node(tree.root, "w3")
        if w3_node:
            tree.select_node(w3_node)
            await pilot.pause()
            results.append(("conv_switched_w3", conv._current_agent_id == "w3"))
            w3_msgs = session.get_messages("w3")
            results.append(("w3_msg_count_1", len(w3_msgs) == 1))
            results.append(("sb_agent_w3", sb.agent_name == "Code Writer"))

        # 8. Switch back to root
        root_node = tree.root
        tree.select_node(root_node)
        await pilot.pause()
        results.append(("conv_back_root", conv._current_agent_id == "root"))

        # 9. Message roles are correct enums
        for msg in root_msgs:
            assert isinstance(msg.role, MessageRole), f"Role {msg.role} is not MessageRole"
        results.append(("roles_are_enums", True))

        # 10. Tool call messages have tool_calls populated
        tool_msgs = [m for m in root_msgs if m.role == MessageRole.TOOL]
        results.append(("root_has_tool_msg", len(tool_msgs) >= 1))
        if tool_msgs:
            results.append(("tool_has_calls", len(tool_msgs[0].tool_calls) > 0))
        else:
            results.append(("tool_has_calls", False))

        # 11. F1 toggles tool log
        await pilot.press("f1")
        await pilot.pause()
        results.append(("tool_log_visible", tl.has_class("visible")))
        await pilot.press("f1")
        await pilot.pause()
        results.append(("tool_log_hidden", not tl.has_class("visible")))

    # Print results
    print("\n" + "=" * 60)
    print("Phase 2 Test Results")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{passed + failed} passed")
    if failed:
        print(f"  {failed} FAILED")
        return False
    return True


if __name__ == "__main__":
    success = asyncio.run(test_phase2())
    raise SystemExit(0 if success else 1)
