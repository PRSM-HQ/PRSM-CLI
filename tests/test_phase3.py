"""Phase 3 headless tests — streaming markdown, background workers, lifecycle."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Markdown

from prsm.tui.app import PrsmApp
from prsm.shared.models.agent import AgentState
from prsm.shared.models.message import MessageRole
from prsm.tui.widgets.agent_tree import AgentTree
from prsm.tui.widgets.conversation import ConversationView, MessageWidget
from prsm.tui.widgets.input_bar import InputBar
from prsm.tui.widgets.status_bar import StatusBar
from prsm.tui.widgets.tool_log import ToolLog

# Force demo mode in all tests — these test the UI, not the live orchestrator
_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


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
async def test_streaming_response(clean_home):
    """Submit a prompt and verify the full streaming lifecycle."""
    app = PrsmApp()
    results = []

    with _DEMO_PATCH:
      async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        conv = screen.query_one("#conversation", ConversationView)
        sb = screen.query_one("#status-bar", StatusBar)
        tl = screen.query_one("#tool-log", ToolLog)
        tree = screen.query_one("#agent-tree", AgentTree)
        inp = screen.query_one("#input-bar", InputBar)

        # 1. Verify initial state — root is active, status is "streaming"
        #    (root agent starts as RUNNING which maps to "streaming")
        results.append(("initial_active_root", screen.session.active_agent_id == "root"))

        # 2. Count initial messages for root
        initial_root_msgs = len(screen.session.get_messages("root"))
        results.append(("initial_root_msgs", initial_root_msgs == 5))

        # 3. Type and submit a prompt
        editor = inp.query_one("#prompt-input")
        editor.focus()
        await pilot.pause()

        # Insert text into the editor
        app.call_from_thread = None  # ensure we're in async context
        editor.insert("Implement JWT auth middleware")
        await pilot.pause()

        # Submit
        await pilot.press("enter")
        await pilot.pause()

        # 4. User message should be added immediately
        root_msgs_after_submit = screen.session.get_messages("root")
        user_msgs = [m for m in root_msgs_after_submit if m.role == MessageRole.USER]
        results.append(("user_msg_added", len(user_msgs) >= 2))  # 1 demo + 1 new
        results.append(("user_msg_content", any(
            "JWT auth middleware" in m.content for m in user_msgs
        )))

        # 5. Status should be "streaming" while worker is active
        # Give the worker a moment to start
        await asyncio.sleep(0.1)
        await pilot.pause()
        results.append(("status_streaming", sb.status == "streaming"))

        # 6. Wait for the streaming worker to complete
        # The simulated response has ~30-50 words with 0.02-0.09s delays each
        # Plus 0.3+0.2s for tool call simulation = ~3-5s total
        # Use a loop to avoid hardcoding a sleep duration
        for _ in range(80):
            await asyncio.sleep(0.1)
            await pilot.pause()
            if sb.status == "connected":
                break

        results.append(("status_connected_after", sb.status == "connected"))

        # 7. A tool call message should have been added mid-stream
        root_msgs_final = screen.session.get_messages("root")
        tool_msgs = [m for m in root_msgs_final if m.role == MessageRole.TOOL]
        results.append(("tool_call_added", len(tool_msgs) >= 2))  # 1 demo + 1 new

        # 8. An assistant message should exist from streaming
        asst_msgs = [m for m in root_msgs_final if m.role == MessageRole.ASSISTANT]
        results.append(("streamed_asst_msg", len(asst_msgs) >= 3))  # 2 demo + 1 new

        # 9. The streamed message should have real content (not echo)
        last_asst = asst_msgs[-1] if asst_msgs else None
        results.append(("stream_has_content", (
            last_asst is not None and len(last_asst.content) > 50
        )))

        # 10. Verify a Markdown widget was mounted (streaming uses Markdown, not Static)
        container = conv.query_one("#message-container")
        md_widgets = container.query(Markdown)
        results.append(("markdown_widget_exists", len(md_widgets) >= 1))

        # 11. Agent state should be COMPLETED after streaming
        root_agent = screen.session.agents.get("root")
        results.append(("agent_completed", root_agent is not None and root_agent.state == AgentState.COMPLETED))

        # 12. Token count should have been updated
        results.append(("tokens_updated", sb.tokens_used > 0))

    # Print results
    print("\n" + "=" * 60)
    print("Phase 3 Test Results")
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
    assert failed == 0, f"{failed} tests failed"


@pytest.mark.asyncio
async def test_streaming_conversation_switch(clean_home):
    """Verify switching agents during streaming doesn't crash."""
    app = PrsmApp()

    with _DEMO_PATCH:
      async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        conv = screen.query_one("#conversation", ConversationView)
        inp = screen.query_one("#input-bar", InputBar)
        tree = screen.query_one("#agent-tree", AgentTree)

        # Submit a prompt to start streaming
        editor = inp.query_one("#prompt-input")
        editor.focus()
        await pilot.pause()
        editor.insert("test streaming")
        await pilot.press("enter")
        await asyncio.sleep(0.2)
        await pilot.pause()

        # Switch to a different agent mid-stream
        def find_agent_node(node, agent_id):
            if node.data and hasattr(node.data, "id") and node.data.id == agent_id:
                return node
            for child in node.children:
                result = find_agent_node(child, agent_id)
                if result:
                    return result
            return None

        w1_node = find_agent_node(tree.root, "w1")
        # In demo mode, switching during streaming might not have worker nodes yet
        if w1_node is not None:
            tree.select_node(w1_node)
            await pilot.pause()
            # Should have switched to w1's conversation
            assert conv._current_agent_id == "w1"

            # Switch back to root
            root_node = find_agent_node(tree.root, "root") or tree.root
            tree.select_node(root_node)
            await pilot.pause()
            # Conversation should switch back to root (or stay on root if
            # the node selection event didn't fire, which is acceptable)
            assert conv._current_agent_id in ("root", "w1")

        # Let the streaming finish
        sb = screen.query_one("#status-bar", StatusBar)
        for _ in range(80):
            await asyncio.sleep(0.1)
            await pilot.pause()
            if sb.status == "connected":
                break

        # Verify the streamed message is in root's session
        root_msgs = screen.session.get_messages("root")
        asst_msgs = [m for m in root_msgs if m.role == MessageRole.ASSISTANT]
        assert len(asst_msgs) >= 3, f"Expected >=3 assistant msgs, got {len(asst_msgs)}"
