"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from unittest.mock import patch

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override, GatewayRunner
from gateway.session import SessionSource


class TestGetChannelOverride:


    def test_no_override_when_channel_not_in_overrides(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "999": ChannelOverride(model="openrouter/healer-alpha"),
                    },
                ),
            },
        )
        assert _get_channel_override(config, Platform.DISCORD, "123") is None

    def test_returns_override_when_channel_matches(self):
        ov = ChannelOverride(
            model="openrouter/healer-alpha",
            provider="openrouter",
            system_prompt="You are a summarizer.",
        )
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"1234567890": ov},
                ),
            },
        )
        result = _get_channel_override(config, Platform.DISCORD, "1234567890")
        assert result is not None
        assert result.model == "openrouter/healer-alpha"
        assert result.provider == "openrouter"
        assert result.system_prompt == "You are a summarizer."


    def test_thread_id_lookup_when_chat_id_misses(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "thread_99": ChannelOverride(model="topic-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config, Platform.DISCORD, "parent_chan", thread_id="thread_99"
        )
        assert result is not None
        assert result.model == "topic-model"


class TestResolveModelForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(model="anthropic/claude-opus-4.6"),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        model = runner._resolve_model_for_channel(Platform.DISCORD, "chan_1")
        assert model == "anthropic/claude-opus-4.6"


class TestGetSystemPromptForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(system_prompt="You are a coding assistant."),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner._ephemeral_system_prompt = "Global prompt"
        prompt = runner._get_system_prompt_for_channel(Platform.DISCORD, "chan_1")
        assert prompt == "You are a coding assistant."


class TestResolveSessionAgentRuntimePriority:
    """Model/runtime priority: session /model → channel_overrides → global."""

    def test_channel_override_beats_global(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(
                            model="channel/model",
                            provider="openrouter",
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan_1",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "k",
                 "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "openrouter",
                     "api_key": "k2",
                     "base_url": "https://openrouter.ai/api/v1",
                     "api_mode": "chat_completions",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "channel/model"
        assert runtime["provider"] == "openrouter"



class TestResolveSessionReasoningConfigPriority:
    """Reasoning priority: session /reasoning → channel_overrides → per-model/global."""

    @staticmethod
    def _runner(monkeypatch, overrides, *, session_reasoning=None):
        from types import SimpleNamespace

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides),
            },
        )
        runner._peek_session_state = lambda _key: (
            None if session_reasoning is None
            else SimpleNamespace(conversation=SimpleNamespace(reasoning_override=session_reasoning))
        )
        monkeypatch.setattr(
            GatewayRunner, "_load_reasoning_config",
            staticmethod(lambda _model="": {"enabled": True, "effort": "xhigh"}),
        )
        return runner

    @staticmethod
    def _resolve(runner, source):
        return runner._resolve_session_reasoning_config(
            source=source, session_key="agent:main:discord:group:food", model="some/model",
        )

    def test_channel_override_beats_per_model_and_global(self, monkeypatch):
        runner = self._runner(monkeypatch, {"food": ChannelOverride(reasoning_effort="medium")})
        source = SessionSource(platform=Platform.DISCORD, chat_id="food", user_id="u1")
        assert self._resolve(runner, source) == {"enabled": True, "effort": "medium"}

    def test_session_override_beats_channel_override(self, monkeypatch):
        runner = self._runner(
            monkeypatch, {"food": ChannelOverride(reasoning_effort="medium")},
            session_reasoning={"enabled": True, "effort": "low"},
        )
        source = SessionSource(platform=Platform.DISCORD, chat_id="food", user_id="u1")
        assert self._resolve(runner, source) == {"enabled": True, "effort": "low"}

    def test_thread_inherits_parent_channel_override(self, monkeypatch):
        runner = self._runner(monkeypatch, {"food": ChannelOverride(reasoning_effort="low")})
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="thread_7", thread_id="thread_7",
            parent_chat_id="food", user_id="u1",
        )
        assert self._resolve(runner, source) == {"enabled": True, "effort": "low"}

    @pytest.mark.parametrize("value", ["none", False, "disabled"])
    def test_channel_override_can_disable_reasoning(self, monkeypatch, value):
        runner = self._runner(monkeypatch, {"food": ChannelOverride(reasoning_effort=value)})
        source = SessionSource(platform=Platform.DISCORD, chat_id="food", user_id="u1")
        assert self._resolve(runner, source) == {"enabled": False}

    def test_other_channel_and_unparseable_value_fall_through(self, monkeypatch):
        runner = self._runner(monkeypatch, {
            "food": ChannelOverride(reasoning_effort="medium"),
            "typo": ChannelOverride(reasoning_effort="hgih"),
        })
        other = SessionSource(platform=Platform.DISCORD, chat_id="elsewhere", user_id="u1")
        typo = SessionSource(platform=Platform.DISCORD, chat_id="typo", user_id="u1")
        assert self._resolve(runner, other) == {"enabled": True, "effort": "xhigh"}
        assert self._resolve(runner, typo) == {"enabled": True, "effort": "xhigh"}
