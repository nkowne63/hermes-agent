"""Tests for Discord channel_prompts resolution and injection."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = types.ModuleType("discord")
    discord_mod.Intents = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_adapter():
    _ensure_discord_mock()
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._client = None
    return adapter


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = "Global prompt"
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="12345",
        chat_type="thread",
        user_id="user-1",
    )


def _channel(channel_id, topic, category_id=None, parent=None):
    return SimpleNamespace(
        id=int(channel_id), topic=topic, category_id=category_id, parent=parent, guild=SimpleNamespace(id=1),
    )


class TestResolveChannelPrompts:
    def test_no_prompt_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_prompt("123") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_prompts": {"100": "Research mode"}}
        assert adapter._resolve_channel_prompt("100") == "Research mode"

    def test_description_off_by_default(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_prompt("100", None, _channel("100", topic="Be terse")) is None

    def test_global_flag_prepends_description_before_configured_prompt(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_description_as_prompt": True, "channel_prompts": {"100": "Research mode"},
        }
        prompt = adapter._resolve_channel_prompt("100", None, _channel("100", topic="Be terse"))
        assert "Be terse" in prompt
        assert prompt.index("Be terse") < prompt.index("Research mode")

    def test_category_on_channel_override_off(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "category_defaults": {"9": {"description_as_prompt": True}},
            "channel_defaults": {"101": {"description_as_prompt": False}},
        }
        assert "Alpha" in adapter._resolve_channel_prompt("100", None, _channel("100", "Alpha", category_id=9))
        assert adapter._resolve_channel_prompt("101", None, _channel("101", "Beta", category_id=9)) is None
        assert adapter._resolve_channel_prompt("102", None, _channel("102", "Gamma", category_id=8)) is None

    def test_channel_on_overrides_global_off_and_category_off(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "category_defaults": {"9": {"description_as_prompt": False}},
            "channel_defaults": {"100": {"description_as_prompt": True}},
        }
        assert "Alpha" in adapter._resolve_channel_prompt("100", None, _channel("100", "Alpha", category_id=9))

    def test_thread_uses_parent_topic_and_parent_override(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_defaults": {"100": {"description_as_prompt": True}}}
        parent = _channel("100", "Parent topic")
        thread = _channel("555", None, parent=parent)
        assert "Parent topic" in adapter._resolve_channel_prompt("555", "100", thread)

    def test_blank_topic_yields_nothing(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_description_as_prompt": True}
        assert adapter._resolve_channel_prompt("100", None, _channel("100", "   ")) is None

    def test_channel_looked_up_from_client_when_not_given(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_description_as_prompt": True}
        adapter._client = SimpleNamespace(get_channel=lambda cid: _channel(str(cid), "From cache"))
        assert "From cache" in adapter._resolve_channel_prompt("100")



@pytest.mark.asyncio
async def test_retry_preserves_channel_prompt(monkeypatch):
    runner = _make_runner()
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1", last_prompt_tokens=10),
        load_transcript=lambda session_id: [
            {"role": "user", "content": "original message"},
            {"role": "assistant", "content": "old reply"},
        ],
        rewrite_transcript=MagicMock(),
    )
    runner._handle_message = AsyncMock(return_value="ok")

    event = MessageEvent(
        text="/retry",
        message_type=gateway_run.MessageType.COMMAND,
        source=_make_source(),
        raw_message=SimpleNamespace(),
        channel_prompt="Channel prompt",
    )

    result = await runner._handle_retry_command(event)

    assert result == "ok"
    retried_event = runner._handle_message.await_args.args[0]
    assert retried_event.channel_prompt == "Channel prompt"


