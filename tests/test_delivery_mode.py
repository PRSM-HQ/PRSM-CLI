"""Tests for delivery-mode modal behavior (interrupt/inject/queue)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from prsm.adapters.events import ToolCallCompleted
from prsm.engine.models import AgentState
from prsm.shared.models.agent import AgentNode
from textual.widgets import Button, Static
from prsm.tui.widgets.conversation import QueuedPromptWidget


_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


def test_delivery_mode_enter_submits_default_queue_on_second_enter():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                result_holder: list[str | None] = []
                screen = DeliveryModeScreen()
                app.push_screen(screen, callback=lambda r: result_holder.append(r))
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()
                assert result_holder == []

                await pilot.pause(0.35)
                await pilot.press("enter")
                await pilot.pause()

                assert result_holder == ["queue"]

    asyncio.run(_run())


def test_delivery_mode_ignores_first_enter_even_after_guard_window():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                result_holder: list[str | None] = []
                screen = DeliveryModeScreen()
                app.push_screen(screen, callback=lambda r: result_holder.append(r))
                await pilot.pause(0.35)
                await pilot.press("enter")
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)
                assert result_holder == []

    asyncio.run(_run())


def test_delivery_mode_ignores_enter_when_mount_is_delayed():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                result_holder: list[str | None] = []
                screen = DeliveryModeScreen()
                # Simulate scheduler/render delay between creation and mount.
                await pilot.pause(0.35)
                app.push_screen(screen, callback=lambda r: result_holder.append(r))
                await pilot.pause()

                # Immediate Enter after mount should still be guarded.
                await pilot.press("enter")
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)
                assert result_holder == []

    asyncio.run(_run())


def test_delivery_mode_hotkey_select_inject():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                result_holder: list[str | None] = []
                screen = DeliveryModeScreen()
                app.push_screen(screen, callback=lambda r: result_holder.append(r))
                await pilot.pause()

                await pilot.pause(0.35)
                await pilot.press("j")
                await pilot.pause()

                assert result_holder == ["inject"]

    asyncio.run(_run())


def test_delivery_mode_stays_open_when_background_chat_updates():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                screen = DeliveryModeScreen()
                app.push_screen(screen, callback=lambda _r: None)
                await pilot.pause()

                main = app.screen_stack[0]
                main.mount(Static("background update"))
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)

    asyncio.run(_run())


def test_main_screen_shows_delivery_mode_when_submitting_while_running():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen
        from prsm.tui.widgets.input_bar import InputBar

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                main._live_mode = True
                main.bridge._running = True

                main.on_input_bar_submitted(InputBar.Submitted("follow-up prompt"))
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)
                queue_btn = app.screen.query_one("#btn-queue", Button)
                assert queue_btn.has_focus

    asyncio.run(_run())


def test_main_screen_shows_delivery_mode_on_enter_keypath_while_running():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen
        from prsm.tui.widgets.input_bar import InputBar

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                main._live_mode = True
                main.bridge._running = True

                input_bar = main.query_one(InputBar)
                input_bar.set_text("follow-up prompt")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)

    asyncio.run(_run())


def test_main_screen_shows_delivery_mode_when_agent_busy_even_if_bridge_flag_false():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.screens.delivery_mode import DeliveryModeScreen
        from prsm.tui.widgets.input_bar import InputBar

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                main._live_mode = True
                main.bridge._running = False
                main.session.agents["agent-1"] = AgentNode(
                    id="agent-1",
                    name="Worker",
                    state=AgentState.RUNNING,
                )

                input_bar = main.query_one(InputBar)
                input_bar.set_text("follow-up prompt")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

                assert isinstance(app.screen, DeliveryModeScreen)

    asyncio.run(_run())


def test_delivery_mode_cancel_does_not_trigger_actions():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                calls: list[str] = []
                main._pending_inject_text = "follow-up prompt"
                main._do_interrupt = lambda prompt: calls.append(f"interrupt:{prompt}")  # type: ignore[method-assign]
                main._do_inject = lambda prompt: calls.append(f"inject:{prompt}")  # type: ignore[method-assign]
                main._do_queue = lambda prompt: calls.append(f"queue:{prompt}")  # type: ignore[method-assign]

                main._handle_delivery_mode("cancel")

                assert main._pending_inject_text is None
                assert calls == []

    asyncio.run(_run())


def test_queue_cancel_uses_plan_index_not_prompt_text():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                main._do_queue("duplicate")
                main._do_queue("duplicate")
                await pilot.pause()

                main.on_queued_prompt_widget_cancelled(
                    QueuedPromptWidget.Cancelled(plan_index=2, prompt="duplicate")
                )
                await pilot.pause()

                assert main._delivery_queued_prompts == [(1, "duplicate")]
                assert 2 not in main._queued_prompt_widgets

    asyncio.run(_run())


def test_inject_prompt_consumed_for_matching_agent_tool_completion():
    async def _run() -> None:
        from prsm.tui.app import PrsmApp
        from prsm.tui.widgets.agent_tree import AgentTree
        from prsm.tui.widgets.conversation import ConversationView
        from prsm.tui.widgets.tool_log import ToolLog

        app = PrsmApp()
        with _DEMO_PATCH:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()

                main = app.screen
                main._inject_prompts_by_agent = {"root-agent": ["inject this"]}
                injected: list[str] = []
                main._do_interrupt = lambda prompt: injected.append(prompt)  # type: ignore[method-assign]

                conv = main.query_one("#conversation", ConversationView)
                tl = main.query_one("#tool-log", ToolLog)
                tree = main.query_one("#agent-tree", AgentTree)
                event = ToolCallCompleted(
                    agent_id="root-agent",
                    tool_id="tool-1",
                    result="ok",
                    is_error=False,
                )

                await main.event_processor._handle_tool_call_completed(event, conv, tl, tree)

                assert injected == ["inject this"]
                assert "root-agent" not in main._inject_prompts_by_agent

    asyncio.run(_run())
