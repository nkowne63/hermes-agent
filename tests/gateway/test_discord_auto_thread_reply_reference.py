"""Reply references for Discord auto-threads.

A thread started from a text-channel message shares that message's id; the message lives in the
parent channel (the thread only holds a THREAD_STARTER_MESSAGE system stub), so a reply to it from
inside the thread must name the parent channel. Discord-refused reply anchors (10008, or a 50035
form error on ``message_reference``) retry as a plain send; any other failure does not.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.discord.adapter as discord_adapter_module
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD_ID = 900
PARENT_ID = 111
THREAD_ID = 555  # == the starter message id in the parent channel


@pytest.fixture(autouse=True)
def _capture_message_reference(monkeypatch):
    """Record MessageReference kwargs whether discord.py is real or mocked."""
    def fake_reference(**kwargs):
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(discord_adapter_module.discord, "MessageReference", fake_reference, raising=False)


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _text_parent():
    return SimpleNamespace(id=PARENT_ID, type=0)


def _thread(*, thread_id=THREAD_ID, parent=None, parent_id=PARENT_ID, send=None):
    return SimpleNamespace(
        id=thread_id, parent_id=parent_id, parent=parent, guild=SimpleNamespace(id=GUILD_ID),
        send=send or AsyncMock(return_value=SimpleNamespace(id=7001)),
    )


def _adapter(channel, *, request=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=request or AsyncMock(return_value={"id": "8001"})),
    )
    return adapter


def _discord_http_error(code, detail=""):
    """The string discord.py's HTTPException renders (``In <field>: <reason>`` for form errors)."""
    reason = "Not Found" if code == 10008 else "Bad Request"
    return RuntimeError(f"{404 if code == 10008 else 400} {reason} (error code: {code}): {detail}")


class TestAutoThreadStarterReferencesParent:
    def test_reference_to_starter_uses_parent_channel(self):
        thread = _thread(parent=_text_parent())
        ref = _adapter(thread)._message_reference_from_ids(str(THREAD_ID), thread)
        assert (ref.message_id, ref.channel_id, ref.guild_id, ref.fail_if_not_exists) == (
            THREAD_ID, PARENT_ID, GUILD_ID, False)

    @pytest.mark.asyncio
    async def test_send_into_auto_thread_replies_to_starter_in_parent(self):
        thread = _thread(parent=_text_parent())
        adapter = _adapter(thread)

        result = await adapter.send(
            str(THREAD_ID), "answer", reply_to=str(THREAD_ID), metadata={"thread_id": str(THREAD_ID)})

        assert result.success is True
        reference = thread.send.await_args.kwargs["reference"]
        assert (reference.message_id, reference.channel_id) == (THREAD_ID, PARENT_ID)

    @pytest.mark.asyncio
    async def test_native_voice_payload_names_parent_channel(self, tmp_path):
        thread = _thread(parent=_text_parent())
        request = AsyncMock(return_value={"id": "8001"})
        adapter = _adapter(thread, request=request)
        audio = tmp_path / "clip.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 64)

        result = await adapter.send_voice(str(THREAD_ID), str(audio), reply_to=str(THREAD_ID))

        assert result.success is True
        form = request.await_args.kwargs["form"]
        payload = json.loads(next(part["value"] for part in form if part["name"] == "payload_json"))
        assert payload["message_reference"] == {
            "message_id": str(THREAD_ID), "channel_id": str(PARENT_ID), "fail_if_not_exists": False}


class TestOrdinaryReferencesUnchanged:
    def test_in_thread_message_keeps_thread_channel(self):
        thread = _thread(parent=_text_parent())
        ref = _adapter(thread)._message_reference_from_ids("777", thread)
        assert (ref.message_id, ref.channel_id) == (777, THREAD_ID)

    def test_plain_channel_reply_keeps_channel(self):
        channel = SimpleNamespace(id=PARENT_ID, guild=SimpleNamespace(id=GUILD_ID))
        ref = _adapter(channel)._message_reference_from_ids("777", channel)
        assert (ref.message_id, ref.channel_id, ref.guild_id) == (777, PARENT_ID, GUILD_ID)

    def test_forum_post_starter_stays_in_thread(self):
        # Forum/media posts keep their real starter message inside the thread.
        thread = _thread(parent=SimpleNamespace(id=PARENT_ID, type=15))
        ref = _adapter(thread)._message_reference_from_ids(str(THREAD_ID), thread)
        assert ref.channel_id == THREAD_ID

    def test_uncached_parent_stays_in_thread(self):
        thread = _thread(parent=None)
        ref = _adapter(thread)._message_reference_from_ids(str(THREAD_ID), thread)
        assert ref.channel_id == THREAD_ID


REJECTED_REFERENCE_ERRORS = [
    _discord_http_error(50035, "Invalid Form Body\nIn message_reference: Cannot reply to a system message"),
    _discord_http_error(50035, "Invalid Form Body\nIn message_reference: Unknown message"),
    _discord_http_error(50035, "Invalid Form Body\nIn message_reference.channel_id: Invalid channel"),
    _discord_http_error(10008, "Unknown Message"),
]
UNRELATED_ERRORS = [
    _discord_http_error(50035, "Invalid Form Body\nIn content: Must be 2000 or fewer in length."),
    _discord_http_error(50035, "Invalid Form Body\nIn embeds.0.description: This field is required"),
    _discord_http_error(50013, "Missing Permissions"),
    RuntimeError("boom"),
]


class TestReplyReferenceRejectionRetry:
    @pytest.mark.parametrize("err", REJECTED_REFERENCE_ERRORS, ids=str)
    def test_classified_as_rejected(self, err):
        assert DiscordAdapter._is_reply_reference_rejected(err) is True

    @pytest.mark.parametrize("err", REJECTED_REFERENCE_ERRORS, ids=str)
    @pytest.mark.asyncio
    async def test_send_retries_without_reference(self, err):
        send = AsyncMock(side_effect=[err, SimpleNamespace(id=7002)])
        thread = _thread(parent=_text_parent(), send=send)

        result = await _adapter(thread).send(
            str(THREAD_ID), "answer", reply_to=str(THREAD_ID), metadata={"thread_id": str(THREAD_ID)})

        assert result.success is True and result.message_id == "7002"
        assert send.await_count == 2
        assert send.await_args_list[0].kwargs["reference"].channel_id == PARENT_ID
        assert send.await_args_list[1].kwargs["reference"] is None


class TestOtherErrorsDoNotRetry:
    @pytest.mark.parametrize("err", UNRELATED_ERRORS, ids=str)
    def test_not_classified_as_rejected(self, err):
        assert DiscordAdapter._is_reply_reference_rejected(err) is False

    @pytest.mark.parametrize("err", UNRELATED_ERRORS, ids=str)
    @pytest.mark.asyncio
    async def test_send_fails_without_plain_retry(self, err):
        send = AsyncMock(side_effect=err)
        thread = _thread(parent=_text_parent(), send=send)

        result = await _adapter(thread).send(
            str(THREAD_ID), "answer", reply_to=str(THREAD_ID), metadata={"thread_id": str(THREAD_ID)})

        assert result.success is False
        assert send.await_count == 1
