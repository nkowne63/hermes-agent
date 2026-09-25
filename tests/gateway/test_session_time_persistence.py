"""Reset policy rotates overdue conversations; ``mode: none`` never does."""
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize("mode", ["idle", "daily", "both"])
def test_reset_config_rotates_overdue_conversation(tmp_path, mode):
    """A session whose last activity predates the policy boundary resets on the next message —
    and a missing routing index promotes the durable row to a reset boundary instead of resuming."""
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": mode, "idle_minutes": 1},
        "reset_by_type": {"dm": {"mode": mode, "idle_minutes": 1}},
        "reset_by_platform": {"telegram": {"mode": mode, "idle_minutes": 1}},
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="time-invariant", user_id="test")
    old = store.get_or_create_session(source)
    messages = [{"role": "user", "content": "keep my conversation"},
                {"role": "assistant", "content": "including after a restart"}]
    for message in messages:
        store.append_to_transcript(old.session_id, message)
    old.updated_at = datetime.now() - timedelta(days=3)
    store._save()

    routed = store.get_or_create_session(source)
    assert routed.session_id != old.session_id
    assert routed.was_auto_reset is True
    assert routed.auto_reset_reason in {"idle", "daily"}
    assert store._db.get_session(old.session_id)["end_reason"] == routed.auto_reset_reason
    # The old transcript survives the boundary for /resume.
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(old.session_id)] == messages

    # Missing routing indexes recover the fresh successor row (it is not overdue); the ended
    # predecessor stays behind its reset boundary.
    store._db.replace_gateway_routing_entries({}, scope=store._routing_scope())
    store._entries.clear()
    recovered = store.get_or_create_session(source)
    assert recovered.session_id == routed.session_id
    store._db.close()


def test_none_mode_never_rotates_durable_conversation(tmp_path):
    """``session_reset.mode: none`` opts out of ALL automatic resets: elapsed time cannot replace
    a durable conversation; explicit boundaries still can."""
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": "none", "idle_minutes": 1},
        "reset_by_type": {"dm": {"mode": "none", "idle_minutes": 1}},
        "reset_by_platform": {"telegram": {"mode": "none", "idle_minutes": 1}},
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="time-invariant", user_id="test")
    old = store.get_or_create_session(source)
    messages = [{"role": "user", "content": "keep my conversation"},
                {"role": "assistant", "content": "including after a restart"}]
    for message in messages:
        store.append_to_transcript(old.session_id, message)
    old.updated_at = datetime.now() - timedelta(days=3)
    store._save()
    routed = store.get_or_create_session(source)
    assert routed.session_id == old.session_id
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(routed.session_id)] == messages
    assert store._db.get_session(old.session_id)["end_reason"] is None
    # Missing routing indexes must recover the same durable transcript too.
    store._db.replace_gateway_routing_entries({}, scope=store._routing_scope())
    store._entries.clear()
    recovered = store.get_or_create_session(source)
    assert recovered.session_id == old.session_id
    explicit = store.reset_session(recovered.session_key)
    assert explicit.session_id != old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == "session_reset"
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(old.session_id)] == messages
    store._db.close()
