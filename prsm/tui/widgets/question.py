"""Question widget — renders structured questions with clickable options."""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static
from textual.widget import Widget


class QuestionWidget(Widget):
    """Renders a question from an agent with clickable option buttons.

    When the user clicks an option (or types a custom response via the
    input bar), the widget posts an ``Answered`` message and visually
    locks itself so the user cannot click again.
    """

    class Answered(Message):
        """Posted when the user selects an option."""

        def __init__(
            self,
            request_id: str,
            answer: str,
        ) -> None:
            super().__init__()
            self.request_id = request_id
            self.answer = answer

    DEFAULT_CSS = """
    QuestionWidget {
        layout: vertical;
        margin: 1 0 1 4;
        padding: 1 2;
        background: $surface-darken-1;
        border: round $warning;
        height: auto;
    }

    QuestionWidget .question-text {
        margin-bottom: 1;
        color: $warning;
        text-style: bold;
        height: auto;
        width: 100%;
    }

    QuestionWidget .question-option-list {
        layout: vertical;
        height: auto;
        width: 100%;
    }

    QuestionWidget .question-option-row {
        layout: vertical;
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }

    QuestionWidget .question-btn {
        width: 100%;
        height: auto;
        min-height: 3;
        content-align: left top;
        padding: 1 2;
        text-align: left;
    }

    QuestionWidget.answered {
        border: round $success-darken-1;
        background: $surface-darken-2;
    }

    QuestionWidget.answered Button {
        opacity: 50%;
    }

    QuestionWidget .selected-answer {
        color: $success;
        text-style: italic;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        request_id: str,
        agent_name: str,
        question: str,
        options: list[dict[str, str]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._request_id = request_id
        self._agent_name = agent_name
        self._question = question
        self._options = options
        self._answered = False

    def compose(self):
        safe_name = self._agent_name.replace("[", "\\[")
        safe_q = self._question.replace("[", "\\[")
        yield Static(
            f"[bold $warning]{safe_name}[/bold $warning] asks:",
            classes="question-label",
            markup=True,
        )
        yield Static(safe_q, classes="question-text", markup=True)

        if self._options:
            with Vertical(classes="question-option-list"):
                for i, opt in enumerate(self._options):
                    label = opt.get("label", f"Option {i + 1}")
                    desc = opt.get("description", "")
                    # Show the full description on the button, not just the short label
                    button_text = desc if desc else label
                    safe_text = button_text.replace("[", "\\[")
                    yield Button(
                        safe_text,
                        variant="primary",
                        id=f"qopt-{i}",
                        classes="question-btn",
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._answered:
            return
        btn_id = event.button.id or ""
        if not btn_id.startswith("qopt-"):
            return

        idx = int(btn_id.removeprefix("qopt-"))
        if idx < len(self._options):
            answer = self._options[idx].get("label", f"Option {idx + 1}")
        else:
            answer = event.button.label

        self._mark_answered(answer)

    def submit_custom(self, text: str) -> None:
        """Called externally when user types a custom answer."""
        if self._answered:
            return
        self._mark_answered(text)

    def _mark_answered(self, answer: str) -> None:
        self._answered = True
        self.add_class("answered")
        safe = answer.replace("[", "\\[")
        self.mount(Static(
            f"[green]\\> {safe}[/green]",
            classes="selected-answer",
            markup=True,
        ))
        self.post_message(self.Answered(
            request_id=self._request_id,
            answer=answer,
        ))
