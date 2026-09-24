"""Tests for steering cron output into an active gateway session."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key


def _source(*, thread_id="thread-1", user_id="user-1"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        thread_id=thread_id,
        user_id=user_id,
        chat_type="thread" if thread_id else "channel",
        scope_id="guild-1",
    )


def _runner_for(source, agent):
    runner = object.__new__(GatewayRunner)
    key = build_session_key(source)
    runner._running_agents = {key: agent}
    runner._session_sources = OrderedDict({key: source})
    return runner


def test_steers_matching_active_origin(monkeypatch):
    from gateway import run as gateway_run

    source = _source()
    agent = MagicMock()
    agent.steer.return_value = True
    runner = _runner_for(source, agent)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    accepted = gateway_run.steer_active_agent_for_origin(
        {
            "platform": "discord",
            "chat_id": "channel-1",
            "thread_id": "thread-1",
            "user_id": "user-1",
            "scope_id": "guild-1",
        },
        "remember the acceptance criteria",
    )

    assert accepted is True
    agent.steer.assert_called_once_with("remember the acceptance criteria")


def test_does_not_steer_pending_sentinel_or_other_origin(monkeypatch):
    from gateway import run as gateway_run

    source = _source()
    agent = MagicMock()
    runner = _runner_for(source, _AGENT_PENDING_SENTINEL)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    assert gateway_run.steer_active_agent_for_origin(
        {"platform": "discord", "chat_id": "other-channel", "thread_id": "thread-1"},
        "do not inject here",
    ) is False
    assert gateway_run.steer_active_agent_for_origin(
        {"platform": "discord", "chat_id": "channel-1", "thread_id": "thread-1"},
        "do not inject while booting",
    ) is False
    agent.steer.assert_not_called()


def test_does_not_steer_when_gateway_is_not_running(monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)
    assert gateway_run.steer_active_agent_for_origin(
        {"platform": "discord", "chat_id": "channel-1"}, "brief"
    ) is False
