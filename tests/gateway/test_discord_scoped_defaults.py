"""Discord ``channel_defaults`` / ``category_defaults`` / ``guild_defaults``.

Per-scope ``require_mention`` and ``thread_response`` resolve channel > category > guild >
global flag. A thread inherits its parent channel's rule and the parent's category. Explicit
gates keep winning: ``free_response_channels`` (and bot threads) never need a mention, and a
scoped ``thread_response: true`` never overrides ``no_thread_channels`` or a free-response
channel's inline reply.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD = 777
CATEGORY = 1501180412385558690
CHANNEL = 1501181227359539401
OTHER_CHANNEL = 1501181227359539402


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = CHANNEL, *, category_id: int | None = None, guild_id: int = GUILD):
        self.id = channel_id
        self.name = f"chan-{channel_id}"
        self.guild = SimpleNamespace(id=guild_id, name="Hermes Server")
        self.category_id = category_id
        self.topic = None


class FakeThread:
    """Like discord.Thread: no own category; it lives on the parent channel."""

    def __init__(self, channel_id: int = 999, *, parent=None):
        self.id = channel_id
        self.name = "thread"
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(id=GUILD, name="Hermes Server")
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    for key in (
        "DISCORD_AUTO_THREAD", "DISCORD_IGNORED_CHANNELS", "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_FREE_RESPONSE_CHANNELS", "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_FREE_RESPONSE_AUTO_THREAD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    # Thread messages would otherwise scan channel history through the stubbed client.
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=4242))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))
    return adapter


def make_message(channel, content: str = "hello", mentions=None):
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(UTC),
        channel=channel,
        author=SimpleNamespace(id=42, display_name="TestUser", name="TestUser"),
        guild=getattr(channel, "guild", None),
    )


def _dispatched_chat_type(adapter) -> str:
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0].source.chat_type


# ── require_mention hierarchy ────────────────────────────────────────


@pytest.mark.asyncio
async def test_category_default_lifts_mention_and_keeps_auto_thread(adapter):
    adapter.config.extra["category_defaults"] = {
        str(CATEGORY): {"require_mention": False, "thread_response": True},
    }

    await adapter._handle_message(make_message(FakeTextChannel(category_id=CATEGORY)))

    adapter._auto_create_thread.assert_awaited_once()
    assert _dispatched_chat_type(adapter) == "thread"


@pytest.mark.asyncio
async def test_channel_default_beats_category_default(adapter):
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": False}}
    adapter.config.extra["channel_defaults"] = {str(CHANNEL): {"require_mention": True}}

    await adapter._handle_message(make_message(FakeTextChannel(category_id=CATEGORY)))
    adapter.handle_message.assert_not_awaited()

    # A sibling channel in the same category still follows the category rule.
    await adapter._handle_message(make_message(FakeTextChannel(OTHER_CHANNEL, category_id=CATEGORY)))
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_category_default_beats_guild_default(adapter):
    adapter.config.extra["guild_defaults"] = {str(GUILD): {"require_mention": False}}
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": True}}

    await adapter._handle_message(make_message(FakeTextChannel(category_id=CATEGORY)))
    adapter.handle_message.assert_not_awaited()

    # Uncategorised channel in the same guild falls through to the guild rule.
    await adapter._handle_message(make_message(FakeTextChannel(OTHER_CHANNEL)))
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_scoped_rule_beats_global_flag_in_both_directions(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    adapter.config.extra["guild_defaults"] = {str(GUILD): {"require_mention": True}}

    await adapter._handle_message(make_message(FakeTextChannel()))
    adapter.handle_message.assert_not_awaited()

    # Another guild has no rule: the global ``require_mention: false`` applies.
    await adapter._handle_message(make_message(FakeTextChannel(OTHER_CHANNEL, guild_id=888)))
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_unquoted_yaml_integer_ids_and_string_values(adapter):
    adapter.config.extra["category_defaults"] = {CATEGORY: {"require_mention": "false"}}

    await adapter._handle_message(make_message(FakeTextChannel(category_id=CATEGORY)))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_free_response_channel_beats_scoped_require_mention(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", str(CHANNEL))
    adapter.config.extra["channel_defaults"] = {str(CHANNEL): {"require_mention": True}}

    await adapter._handle_message(make_message(FakeTextChannel()))

    # Free-response replies inline (no free_response_auto_thread opt-in).
    adapter._auto_create_thread.assert_not_awaited()
    assert _dispatched_chat_type(adapter) == "group"


@pytest.mark.asyncio
async def test_explicit_mention_still_works_where_mention_is_required(adapter):
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": True}}
    bot_user = adapter._client.user

    await adapter._handle_message(make_message(
        FakeTextChannel(category_id=CATEGORY), content=f"<@{bot_user.id}> hi", mentions=[bot_user],
    ))

    adapter.handle_message.assert_awaited_once()


# ── threads under a categorised channel ──────────────────────────────


@pytest.mark.asyncio
async def test_thread_inherits_category_of_its_parent_channel(adapter):
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": False}}
    parent = FakeTextChannel(category_id=CATEGORY)

    await adapter._handle_message(make_message(FakeThread(555, parent=parent)))

    # Already in a thread: dispatched there, no new thread.
    adapter._auto_create_thread.assert_not_awaited()
    assert _dispatched_chat_type(adapter) == "thread"


@pytest.mark.asyncio
async def test_thread_matches_channel_defaults_through_its_parent(adapter):
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": True}}
    adapter.config.extra["channel_defaults"] = {str(CHANNEL): {"require_mention": False}}
    parent = FakeTextChannel(category_id=CATEGORY)

    await adapter._handle_message(make_message(FakeThread(555, parent=parent)))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_thread_own_channel_rule_beats_parent_rule(adapter):
    adapter.config.extra["channel_defaults"] = {
        "555": {"require_mention": True},
        str(CHANNEL): {"require_mention": False},
    }

    await adapter._handle_message(make_message(FakeThread(555, parent=FakeTextChannel())))

    adapter.handle_message.assert_not_awaited()


def test_category_id_lookup_shapes(adapter):
    assert adapter._get_category_id(FakeTextChannel(category_id=CATEGORY)) == str(CATEGORY)
    assert adapter._get_category_id(FakeThread(parent=FakeTextChannel(category_id=CATEGORY))) == str(CATEGORY)
    # ``category`` object instead of ``category_id`` (e.g. a partially cached channel).
    channel = SimpleNamespace(id=1, category=SimpleNamespace(id=CATEGORY))
    assert adapter._get_category_id(channel) == str(CATEGORY)
    assert adapter._get_category_id(FakeTextChannel()) is None
    assert adapter._get_category_id(FakeThread(parent=None)) is None


# ── thread_response ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guild_thread_response_false_replies_inline(adapter):
    adapter.config.extra["guild_defaults"] = {
        str(GUILD): {"require_mention": False, "thread_response": False},
    }

    await adapter._handle_message(make_message(FakeTextChannel()))

    adapter._auto_create_thread.assert_not_awaited()
    assert _dispatched_chat_type(adapter) == "group"


@pytest.mark.asyncio
async def test_channel_thread_response_true_beats_guild_false(adapter):
    adapter.config.extra["guild_defaults"] = {
        str(GUILD): {"require_mention": False, "thread_response": False},
    }
    adapter.config.extra["channel_defaults"] = {str(CHANNEL): {"thread_response": True}}

    await adapter._handle_message(make_message(FakeTextChannel()))

    adapter._auto_create_thread.assert_awaited_once()
    assert _dispatched_chat_type(adapter) == "thread"


@pytest.mark.asyncio
async def test_no_thread_channels_beat_scoped_thread_response(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", str(CHANNEL))
    adapter.config.extra["channel_defaults"] = {
        str(CHANNEL): {"require_mention": False, "thread_response": True},
    }

    await adapter._handle_message(make_message(FakeTextChannel()))

    adapter._auto_create_thread.assert_not_awaited()
    assert _dispatched_chat_type(adapter) == "group"


@pytest.mark.asyncio
async def test_no_thread_channels_accepts_a_category_id(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", str(CATEGORY))
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    await adapter._handle_message(make_message(FakeTextChannel(category_id=CATEGORY)))

    adapter._auto_create_thread.assert_not_awaited()
    assert _dispatched_chat_type(adapter) == "group"


# ── other ingress paths ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recovered_message_uses_scoped_require_mention(adapter):
    adapter._discord_message_admission = MagicMock(return_value=(True, False))
    adapter._handle_message = AsyncMock(return_value=True)
    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": False}}

    assert await adapter._dispatch_recovered_message(
        make_message(FakeTextChannel(category_id=CATEGORY))) is True
    assert await adapter._dispatch_recovered_message(
        make_message(FakeTextChannel(OTHER_CHANNEL))) is False
    adapter._handle_message.assert_awaited_once()


def test_admission_treats_scoped_free_channel_like_free_response(adapter, monkeypatch):
    """A message @-mentioning someone else is dropped unless the channel needs no mention."""
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    adapter._is_allowed_user = MagicMock(return_value=True)
    other = SimpleNamespace(id=820, bot=False)

    def third_party(channel):
        msg = make_message(channel, content="<@820> look", mentions=[other])
        msg.author = SimpleNamespace(id=42, bot=False, display_name="TestUser", name="TestUser")
        msg.type = discord_platform.discord.MessageType.default
        return msg

    admitted, _ = adapter._discord_message_admission(
        third_party(FakeTextChannel(category_id=CATEGORY)), claim=False)
    assert admitted is False

    adapter.config.extra["category_defaults"] = {str(CATEGORY): {"require_mention": False}}
    admitted, _ = adapter._discord_message_admission(
        third_party(FakeTextChannel(OTHER_CHANNEL, category_id=CATEGORY)), claim=False)
    assert admitted is True


# ── YAML bridge ──────────────────────────────────────────────────────


def test_yaml_defaults_reach_platform_extra(monkeypatch, tmp_path):
    import hermes_yaml as yaml
    from gateway.config import Platform, load_gateway_config

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "discord": {
            "guild_defaults": {str(GUILD): {"require_mention": True, "thread_response": True}},
            "category_defaults": {str(CATEGORY): {"require_mention": False}},
            "channel_defaults": {str(CHANNEL): {"thread_response": False}},
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token")

    extra = load_gateway_config().platforms[Platform.DISCORD].extra

    assert extra["guild_defaults"] == {str(GUILD): {"require_mention": True, "thread_response": True}}
    assert extra["category_defaults"] == {str(CATEGORY): {"require_mention": False}}
    assert extra["channel_defaults"] == {str(CHANNEL): {"thread_response": False}}
