"""``display.thread_only_channels``: transient output only inside threads.

In a listed channel the turn shows just the final answer — tool progress, interim assistant
messages, thinking progress and the prepended reasoning block are suppressed. A thread under
that channel (``source.thread_id`` set) keeps the normal display settings.
"""

import importlib
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.display_config import is_thread_only_display_channel
from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner, _run_with_agent

PARENT = "1509077323490791515"
THREAD = "1533727118754840576"

DISPLAY = {
    "tool_progress": "all",
    "interim_assistant_messages": True,
    "thinking_progress": True,
    "show_reasoning": True,
    "thread_only_channels": [PARENT],
}


def _source(*, chat_id=PARENT, thread_id=None, chat_type="group"):
    return SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type=chat_type, thread_id=thread_id)


# ── matcher ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("configured", "source", "expected"),
    [
        ([PARENT], _source(), True),
        ([int(PARENT)], _source(), True),
        (f"{PARENT}, 42", _source(), True),
        (int(PARENT), _source(), True),
        ([PARENT], _source(chat_id="other"), False),
        # Discord threads carry thread_id (chat_id may be the thread or the parent).
        ([PARENT], _source(chat_id=THREAD, thread_id=THREAD, chat_type="thread"), False),
        ([PARENT], _source(chat_id=PARENT, thread_id=THREAD, chat_type="thread"), False),
        ([], _source(), False),
        (None, _source(), False),
    ],
)
def test_is_thread_only_display_channel(configured, source, expected):
    config = {"display": {"thread_only_channels": configured}}
    assert is_thread_only_display_channel(config, source) is expected


def test_is_thread_only_display_channel_tolerates_missing_display():
    assert is_thread_only_display_channel({}, _source()) is False
    assert is_thread_only_display_channel({"display": "bad"}, _source()) is False
    assert is_thread_only_display_channel(None, _source()) is False


# ── per-turn display settings ────────────────────────────────────────


def _display_settings(monkeypatch, tmp_path, source, display):
    import hermes_yaml as yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": display}), encoding="utf-8")
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = _make_runner(ProgressCaptureAdapter(platform=Platform.DISCORD))
    return runner._run_agent_display_settings(source)


def test_parent_channel_suppresses_progress_interim_and_thinking(monkeypatch, tmp_path):
    disp = _display_settings(monkeypatch, tmp_path, _source(), DISPLAY)

    assert disp.progress_mode == "off"
    assert disp.tool_progress_enabled is False
    assert disp.interim_assistant_messages_enabled is False
    assert disp._thinking_enabled is False


def test_thread_under_parent_keeps_display(monkeypatch, tmp_path):
    disp = _display_settings(
        monkeypatch, tmp_path, _source(chat_id=THREAD, thread_id=THREAD, chat_type="thread"), DISPLAY,
    )

    assert disp.progress_mode == "all"
    assert disp.tool_progress_enabled is True
    assert disp.interim_assistant_messages_enabled is True
    assert disp._thinking_enabled is True


def test_unlisted_channel_keeps_display(monkeypatch, tmp_path):
    disp = _display_settings(monkeypatch, tmp_path, _source(chat_id="other"), DISPLAY)

    assert disp.tool_progress_enabled is True
    assert disp.interim_assistant_messages_enabled is True


def test_log_mode_is_not_chat_output_and_is_kept(monkeypatch, tmp_path):
    disp = _display_settings(monkeypatch, tmp_path, _source(), {**DISPLAY, "tool_progress": "log"})

    assert disp.progress_mode == "log"
    assert disp.log_mode_enabled is True
    assert disp.tool_progress_enabled is False


# ── end to end through _run_agent ────────────────────────────────────


class ChattyAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("Let me look around first.", already_streamed=False)
        if self.tool_progress_callback:
            self.tool_progress_callback("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _run(monkeypatch, tmp_path, *, chat_id, thread_id, chat_type):
    return await _run_with_agent(
        monkeypatch, tmp_path, ChattyAgent,
        session_id=f"sess-thread-only-{thread_id or 'parent'}",
        config_data={"display": DISPLAY, "streaming": {"enabled": False}},
        platform=Platform.DISCORD, chat_id=chat_id, chat_type=chat_type, thread_id=thread_id,
    )


@pytest.mark.asyncio
async def test_parent_channel_turn_sends_only_the_final_answer(monkeypatch, tmp_path):
    adapter, result = await _run(monkeypatch, tmp_path, chat_id=PARENT, thread_id=None, chat_type="group")

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_thread_turn_still_shows_progress_and_interim(monkeypatch, tmp_path):
    adapter, _ = await _run(monkeypatch, tmp_path, chat_id=THREAD, thread_id=THREAD, chat_type="thread")

    sent = [call["content"] for call in adapter.sent]
    assert "Let me look around first." in sent
    assert any("pwd" in text for text in sent + [edit["content"] for edit in adapter.edits])


# ── final-response reasoning ─────────────────────────────────────────


def test_prepended_reasoning_is_thread_only(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": DISPLAY})
    result = {"last_reasoning": "private chain"}
    runner = GatewayTurnPersistenceMixin()

    parent = runner._hmwa_prepend_reasoning(result, "final answer", _source(), False)
    thread = runner._hmwa_prepend_reasoning(
        result, "final answer", _source(chat_id=THREAD, thread_id=THREAD, chat_type="thread"), False,
    )
    other = runner._hmwa_prepend_reasoning(result, "final answer", _source(chat_id="other"), False)

    assert parent == "final answer"
    assert "private chain" in thread and thread.endswith("final answer")
    assert "private chain" in other


def test_source_without_thread_attributes_is_not_suppressed():
    assert is_thread_only_display_channel(
        {"display": {"thread_only_channels": [PARENT]}}, SimpleNamespace(platform=Platform.DISCORD),
    ) is False
