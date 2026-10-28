"""``channel_overrides.<id>.allow_silence``: a human turn may end on a bare silence marker.

The adapter stamps every non-internal event from an opted-in channel (or a thread under it)
``reply_expected=False`` at ``handle_message`` entry, so the runner's existing unaddressed-silence
path (no ⚠️ fallback, persisted ``reply_expected`` for crash recovery) applies unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import ChannelOverride, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.response_filters import is_intentional_silence_response
from gateway.session import SessionSource
from tests.gateway.test_gateway_silence_tokens import _runner


class _DiscordLikeAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="x")


def _adapter(overrides):
    adapter = _DiscordLikeAdapter(
        PlatformConfig(enabled=True, token="***", channel_overrides=overrides), Platform.DISCORD)
    adapter.set_message_handler(AsyncMock(return_value=None))
    started = []
    adapter._start_session_processing = lambda event, key: started.append(event) or True
    return adapter, started


def _event(chat_id="quiet", *, thread_id=None, parent_chat_id=None, internal=False, reply_expected=None):
    source = SessionSource(
        platform=Platform.DISCORD, chat_id=chat_id, chat_type="thread" if thread_id else "group",
        user_id="u1", thread_id=thread_id, parent_chat_id=parent_chat_id,
    )
    return MessageEvent(
        text="side chatter", message_type=MessageType.TEXT, source=source, message_id="m1",
        internal=internal, reply_expected=reply_expected,
    )


async def _admit(overrides, event):
    adapter, started = _adapter(overrides)
    await adapter.handle_message(event)
    assert started == [event]
    return event


@pytest.mark.asyncio
async def test_human_turn_in_allow_silence_channel_is_unaddressed():
    event = await _admit({"quiet": ChannelOverride(allow_silence=True)}, _event(reply_expected=True))
    assert event.reply_expected is False


@pytest.mark.asyncio
async def test_thread_inherits_parent_allow_silence():
    event = await _admit(
        {"quiet": ChannelOverride(allow_silence=True)},
        _event("thread_9", thread_id="thread_9", parent_chat_id="quiet"),
    )
    assert event.reply_expected is False


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {},
    {"quiet": ChannelOverride(allow_silence=True)},
    {"loud": ChannelOverride(allow_silence=False)},
    {"loud": ChannelOverride(model="some/model")},
], ids=["no-overrides", "other-channel", "explicit-false", "override-without-key"])
async def test_other_channels_keep_reply_expected(overrides):
    event = await _admit(overrides, _event("loud"))
    assert event.reply_expected is None


@pytest.mark.asyncio
async def test_internal_turn_keeps_its_own_verdict():
    event = await _admit({"quiet": ChannelOverride(allow_silence=True)}, _event(internal=True))
    assert event.reply_expected is None


@pytest.mark.asyncio
async def test_busy_follow_up_is_stamped_before_it_is_queued():
    """A message arriving mid-turn lands in the pending slot already carrying the verdict, so a
    queued chain's terminal turn (``queued_terminal_reply_expected``) honors it too."""
    adapter, _started = _adapter({"quiet": ChannelOverride(allow_silence=True)})
    event = _event()
    session_key = adapter._event_session_key(event)
    adapter._active_sessions[session_key] = object()
    adapter._heal_stale_session_lock = lambda _key: None
    queued = []
    adapter._handle_message_while_active = AsyncMock(side_effect=lambda ev, key: queued.append(ev))

    await adapter.handle_message(event)

    assert queued == [event] and event.reply_expected is False


def _silent_agent_result():
    return {
        "final_response": "[SILENT]",
        "messages": [{"role": "user", "content": "side chatter"},
                     {"role": "assistant", "content": "[SILENT]"}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id,silent", [("quiet", True), ("loud", False)])
async def test_discord_human_turn_silence_end_to_end(monkeypatch, tmp_path, chat_id, silent):
    """Admitted through the adapter, then run: the opted-in channel stays quiet (no ⚠️ fallback),
    and the persisted user row records ``reply_expected: false`` for crash recovery."""
    event = await _admit({"quiet": ChannelOverride(allow_silence=True)}, _event(chat_id))
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_silent_agent_result())

    response = await runner._handle_message_with_agent(
        event, event.source, f"agent:main:discord:group:{chat_id}", 1)

    if silent:
        assert response == ""
        metadata = runner._run_agent.await_args.kwargs["persist_user_display_metadata"]
        assert metadata.get("reply_expected") is False
    else:
        assert response and not is_intentional_silence_response(response)
