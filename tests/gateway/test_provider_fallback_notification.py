"""Tests for provider fallback notifications."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_runner(*, enabled: bool):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="tok",
                home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="1501915980434112593", name="fallback"),
                gateway_fallback_notification=enabled,
            )
        }
    )
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._thread_metadata_for_target = (
        lambda *args, **kwargs: {"thread_id": str(args[2])} if len(args) > 2 and args[2] else None
    )
    return runner, adapter


@pytest.mark.asyncio
async def test_provider_fallback_notification_sends_to_home_channel():
    runner, adapter = _make_runner(enabled=True)
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="channel")

    sent = await runner._send_provider_fallback_notification(
        source=source,
        from_provider="qwen-oauth",
        to_provider="claude-acp",
        from_model="qwen3:8b",
        to_model="claude-sonnet-4.6",
    )

    assert sent is True
    adapter.send.assert_awaited_once()
    chat_id = adapter.send.await_args.args[0]
    message = adapter.send.await_args.args[1]
    assert chat_id == "1501915980434112593"
    assert "qwen-oauth -> claude-acp" in message
    assert "qwen3:8b -> claude-sonnet-4.6" in message


@pytest.mark.asyncio
async def test_provider_fallback_notification_can_be_disabled():
    runner, adapter = _make_runner(enabled=False)
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="channel")

    sent = await runner._send_provider_fallback_notification(
        source=source,
        from_provider="qwen-oauth",
        to_provider="claude-acp",
    )

    assert sent is False
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_fallback_session_error_sends_to_discord_session():
    runner, adapter = _make_runner(enabled=False)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1501181227359539401",
        chat_type="thread",
        thread_id="1501181227359539401",
        message_id="222",
    )

    sent = await runner._send_provider_fallback_session_error(
        source=source,
        from_provider="devin-acp",
        to_provider="openrouter",
        from_model="swe-1.6-fast",
        to_model="qwen3:8b",
    )

    assert sent is True
    adapter.send.assert_awaited_once()
    chat_id = adapter.send.await_args.args[0]
    message = adapter.send.await_args.args[1]
    reply_to = adapter.send.await_args.kwargs["reply_to"]
    metadata = adapter.send.await_args.kwargs["metadata"]
    assert chat_id == "1501181227359539401"
    assert reply_to == "222"
    assert metadata["thread_id"] == "1501181227359539401"
    assert metadata["non_conversational"] is True
    assert "Provider fallback in this session" in message
    assert "devin-acp -> openrouter" in message
