"""Panel steering behavior for in-flight LLM requests.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_steer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


PROMPT = "LLM request in progress: (a)bort & close  (c)ontinue  (s)teer"


class _FakeBackendClient:
    """Records the control frames the panel sends to the backend."""

    def __init__(self, sent):
        self._sent = sent

    def cancel(self):
        self._sent.append("cancel")


def _make_ludvart():
    runner = Ludvart(["true"])
    runner._panel = AiPanel(cols=80, height=10, provider="openai")
    runner._render_split = lambda: None
    return runner


def _start_llm_request(runner):
    runner._panel.thinking = True
    runner._llm_request_in_flight = True


def _open_steer_input(runner):
    runner._request_toggle_close()
    runner._panel_input(b"s")


def test_prompt_offers_three_options():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._request_toggle_close()
    assert runner._panel.confirm_prompt == PROMPT
    print("in-flight prompt offers abort, continue, and steer: OK")


def test_abort_cancels_and_closes():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._request_toggle_close()
    runner._panel_input(b"a")
    assert runner._panel_closing is True
    assert runner._confirm_close is False
    assert runner._ask_cancel.is_set()
    assert runner._panel.thinking is False
    print("abort cancels and closes: OK")


def test_continue_keeps_request_running():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._request_toggle_close()
    runner._panel_input(b"c")
    assert runner._panel_closing is False
    assert runner._confirm_close is False
    assert not runner._ask_cancel.is_set()
    assert runner._panel.thinking is True
    print("continue leaves request untouched: OK")


def test_escape_at_prompt_continues_request():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._request_toggle_close()
    runner._panel_input(b"\x1b")
    assert runner._confirm_close is False
    assert not runner._ask_cancel.is_set()
    assert runner._panel.thinking is True
    print("escape at the choice prompt continues: OK")


def test_steer_enters_editable_input_mode():
    runner = _make_ludvart()
    _start_llm_request(runner)
    _open_steer_input(runner)
    assert runner._steer_input is True
    assert runner._confirm_close is False
    assert runner._panel.steer_prompt == "Steer request: "
    assert runner._panel.confirm_prompt == ""
    assert runner._panel.editor.text == ""
    print("steer opens editable input: OK")


def test_steer_input_collects_text():
    runner = _make_ludvart()
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"fix the ")
    runner._panel_input(b"tests")
    assert runner._panel.editor.text == "fix the tests"
    assert runner._steer_pending is None
    print("steer input collects text: OK")


def test_steer_escape_restores_draft_and_continues():
    runner = _make_ludvart()
    runner._panel.editor.insert("original draft")
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"new instruction")
    runner._panel_input(b"\x1b")
    assert runner._steer_input is False
    assert runner._panel.steer_prompt == ""
    assert runner._panel.editor.text == "original draft"
    assert not runner._ask_cancel.is_set()
    assert runner._panel.thinking is True
    assert runner._steer_pending is None
    print("escape abandons steer and restores draft: OK")


def test_empty_steer_submit_continues_request():
    runner = _make_ludvart()
    runner._panel.editor.insert("original draft")
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"\r")
    assert runner._steer_input is False
    assert runner._panel.editor.text == "original draft"
    assert runner._steer_pending is None
    assert not runner._ask_cancel.is_set()
    assert runner._panel.thinking is True
    print("empty steer submit continues request: OK")


def test_steer_submit_queues_reissue():
    runner = _make_ludvart()
    runner._ask_root_question = "list files"
    runner._panel.interim = "Reading directory"
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"only show Python files")
    runner._panel_input(b"\r")
    assert runner._ask_cancel.is_set()
    assert runner._panel.thinking is True
    assert runner._panel.interim == ""
    assert runner._panel.activity == "Steering"
    assert runner._steer_input is False
    assert runner._steer_user_echo == "only show Python files"
    assert "only show Python files" in runner._steer_pending
    # The interrupted turn keeps its own narration and tool results in the
    # backend's history, so replaying them into the message would duplicate
    # them and re-bill the tokens.
    assert "list files" not in runner._steer_pending
    assert "Reading directory" not in runner._steer_pending
    print("steer submit queues a replacement request: OK")


def test_steer_tells_the_backend_to_stop():
    """The whole point: the in-flight turn must be interrupted, not awaited.

    Without this frame the backend runs the turn to completion -- every model
    call and every tool -- and the correction only lands afterwards, which the
    user experiences as ludvart ignoring them until it has finished.
    """
    runner = _make_ludvart()
    sent = []
    runner._backend_client = _FakeBackendClient(sent)
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"stop and do this instead")
    runner._panel_input(b"\r")
    assert sent == ["cancel"], sent
    print("steer tells the backend to stop: OK")


def test_finish_ask_launches_pending_steer():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._ask_root_question = "root request"
    runner._steer_pending = "replacement request"
    runner._steer_user_echo = "change direction"
    calls = []
    runner._start_ask = lambda question, **kwargs: calls.append((question, kwargs))

    runner._finish_ask()

    assert calls == [
        (
            "replacement request",
            {"user_echo": "change direction", "root_question": "root request"},
        )
    ]
    assert runner._steer_pending is None
    assert runner._steer_user_echo is None
    print("pending steer starts only after the old worker finishes: OK")


def test_a_steer_keeps_an_answer_that_arrived_first():
    """Steering must not swallow a turn that had already answered.

    The cancel flag is only noticed between agent steps, so a turn that
    finished while the user was typing the correction still returns its answer,
    and the backend has already persisted it. Dropping it left the panel
    missing a reply the model believes it gave -- the user watched the summary
    appear and then vanish.
    """
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._ask_root_question = "root request"
    runner._steer_pending = "replacement request"
    runner._steer_user_echo = "change direction"
    runner._ask_result = "the original answer"
    runner._ask_cancel.set()  # submitting a steer cancels the in-flight turn
    calls = []

    def fake_start_ask(question, **kwargs):
        calls.append((question, kwargs, list(runner._panel.messages)))

    runner._start_ask = fake_start_ask

    runner._finish_ask()

    assert len(calls) == 1, "the steered turn must still start"
    # Delivered before the steered turn echoes the correction, so the panel
    # reads in the order the conversation actually happened.
    assert calls[0][2][-1] == ("ludvart", "the original answer"), calls[0][2]
    assert runner._ask_result == "", "the answer must not be delivered twice"
    print("a steer keeps an answer that arrived first: OK")


def test_a_steer_that_interrupted_a_silent_turn_adds_nothing():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._steer_pending = "replacement request"
    runner._steer_user_echo = "change direction"
    runner._ask_result = ""  # the worker unwound before it produced anything
    runner._ask_cancel.set()
    runner._start_ask = lambda question, **kwargs: None

    runner._finish_ask()

    assert not [m for m in runner._panel.messages if m[0] == "ludvart"]
    print("a steer that interrupted a silent turn adds nothing: OK")


def test_finish_ask_completion_exits_steer_input():
    runner = _make_ludvart()
    runner._steer_saved_draft = "keep this"
    runner._steer_saved_cursor = len(runner._steer_saved_draft)
    runner._steer_input = True
    runner._panel.steer_prompt = "Steer request: "
    runner._panel.editor.set_text("discard this")
    runner._panel.thinking = True
    runner._ask_result = "completed answer"

    runner._finish_ask()

    assert runner._steer_input is False
    assert runner._panel.steer_prompt == ""
    assert runner._panel.editor.text == "keep this"
    assert runner._panel.messages[-1] == ("ludvart", "completed answer")
    print("natural completion exits steer input and restores draft: OK")


def test_compose_steer_question_includes_required_context():
    question = Ludvart._compose_steer_question("STEER")
    assert "STEER" in question
    assert "do not repeat" in question.lower()
    print("steer request carries the correction: OK")


def test_repeated_steer_preserves_root_question():
    runner = _make_ludvart()
    _start_llm_request(runner)
    runner._ask_root_question = "original root"
    _open_steer_input(runner)
    runner._panel_input(b"first correction")
    runner._panel_input(b"\r")
    first = runner._steer_pending
    calls = []
    runner._start_ask = lambda question, **kwargs: calls.append((question, kwargs))
    runner._finish_ask()
    assert calls[0][1]["root_question"] == "original root"

    runner._ask_root_question = calls[0][1]["root_question"]
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"second correction")
    runner._panel_input(b"\r")
    assert "second correction" in runner._steer_pending
    assert "first correction" not in runner._steer_pending
    assert first is not runner._steer_pending
    print("repeated steering carries only the newest correction: OK")


def test_steer_prompt_renders_on_input_line():
    runner = _make_ludvart()
    _start_llm_request(runner)
    _open_steer_input(runner)
    runner._panel_input(b"hello")
    bottom = runner._panel.render(height=8, cols=80)[-1].decode("utf-8", "replace")
    assert "Steer request: " in bottom
    assert "hello" in bottom
    assert "ludvart>" not in bottom
    print("steer prompt renders as the editable input line: OK")


def main():
    test_prompt_offers_three_options()
    test_abort_cancels_and_closes()
    test_continue_keeps_request_running()
    test_escape_at_prompt_continues_request()
    test_steer_enters_editable_input_mode()
    test_steer_input_collects_text()
    test_steer_escape_restores_draft_and_continues()
    test_empty_steer_submit_continues_request()
    test_steer_submit_queues_reissue()
    test_steer_tells_the_backend_to_stop()
    test_finish_ask_launches_pending_steer()
    test_a_steer_keeps_an_answer_that_arrived_first()
    test_a_steer_that_interrupted_a_silent_turn_adds_nothing()
    test_finish_ask_completion_exits_steer_input()
    test_compose_steer_question_includes_required_context()
    test_repeated_steer_preserves_root_question()
    test_steer_prompt_renders_on_input_line()
    print("\nALL steering tests passed.")


if __name__ == "__main__":
    main()
