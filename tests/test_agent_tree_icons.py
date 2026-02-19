from prsm.adapters.agent_adapter import STATE_ICONS
from prsm.engine.models import AgentState
from prsm.tui.widgets.agent_tree import _animated_state_icon, _state_uses_animation


def test_waiting_for_child_has_distinct_icon_mapping() -> None:
    child_icon, child_color = STATE_ICONS[AgentState.WAITING_FOR_CHILD]
    parent_icon, parent_color = STATE_ICONS[AgentState.WAITING_FOR_PARENT]

    assert child_icon != parent_icon
    assert child_color == "yellow"
    assert parent_color == "yellow"


def test_animated_state_icon_for_active_work_is_green_arrow() -> None:
    icon_a, color_a = _animated_state_icon(AgentState.RUNNING, 0, True)
    icon_b, color_b = _animated_state_icon(AgentState.RUNNING, 1, True)

    assert color_a == "green"
    assert color_b == "green"
    assert icon_a != icon_b


def test_animated_state_icon_for_waiting_for_child_blinks_yellow_circle() -> None:
    blink_on_icon, blink_on_color = _animated_state_icon(
        AgentState.WAITING_FOR_CHILD,
        0,
        True,
    )
    blink_off_icon, blink_off_color = _animated_state_icon(
        AgentState.WAITING_FOR_CHILD,
        0,
        False,
    )

    assert blink_on_color == "yellow"
    assert blink_off_color == "yellow"
    assert blink_on_icon != blink_off_icon


def test_state_uses_animation_treats_waiting_for_child_as_special_active_state() -> None:
    assert _state_uses_animation(AgentState.RUNNING)
    assert _state_uses_animation(AgentState.WAITING_FOR_PARENT)
    assert _state_uses_animation(AgentState.WAITING_FOR_EXPERT)
    assert _state_uses_animation(AgentState.WAITING_FOR_CHILD)
    assert not _state_uses_animation(AgentState.COMPLETED)
