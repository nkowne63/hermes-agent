"""session_reset config parsing + idle/daily reset policy (suspend sweep and routing-time reset)."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, SessionResetPolicy, load_gateway_config
from gateway.session import SessionSource, SessionStore


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _store(tmp_path, policy: dict) -> SessionStore:
    config = GatewayConfig.from_dict({"default_reset_policy": policy})
    return SessionStore(tmp_path / "sessions", config)


class TestSessionResetConfig:
    def test_session_reset_yaml_bridges_to_default_policy(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.dump({"session_reset": {"mode": "both", "idle_minutes": 1440, "at_hour": 4}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config = load_gateway_config()

        assert config.default_reset_policy.mode == "both"
        assert config.default_reset_policy.idle_minutes == 1440
        assert config.default_reset_policy.at_hour == 4

    def test_nested_gateway_session_reset_fallback(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.dump({"gateway": {"session_reset": {"mode": "daily", "at_hour": 6}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config = load_gateway_config()

        assert config.default_reset_policy.mode == "daily"
        assert config.default_reset_policy.at_hour == 6

    def test_missing_session_reset_defaults_to_none(self):
        config = GatewayConfig.from_dict({})
        assert config.default_reset_policy.mode == "none"

    def test_get_reset_policy_precedence(self):
        config = GatewayConfig.from_dict({
            "default_reset_policy": {"mode": "none"},
            "reset_by_type": {"group": {"mode": "daily", "at_hour": 3}},
            "reset_by_platform": {"telegram": {"mode": "idle", "idle_minutes": 60}},
        })
        assert config.get_reset_policy(platform=Platform.TELEGRAM).mode == "idle"
        assert config.get_reset_policy(session_type="group").mode == "daily"
        assert config.get_reset_policy(platform=Platform.DISCORD, session_type="dm").mode == "none"

    def test_invalid_policy_values_sanitized(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.dump({"session_reset": {"mode": "daily", "at_hour": 99, "idle_minutes": -5}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config = load_gateway_config()

        assert config.default_reset_policy.at_hour == 4
        assert config.default_reset_policy.idle_minutes == 1440


class TestPolicyResetReason:
    """Boundary math for _policy_reset_reason (local wall clock, same as the removed upstream code)."""

    def test_idle_overdue(self):
        policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        old = datetime.now() - timedelta(minutes=61)
        assert SessionStore._policy_reset_reason(policy, old) == "idle"
        fresh = datetime.now() - timedelta(minutes=30)
        assert SessionStore._policy_reset_reason(policy, fresh) is None

    def test_daily_boundary(self):
        policy = SessionResetPolicy(mode="daily", at_hour=4)
        now = datetime.now()
        last_reset = now.replace(hour=4, minute=0, second=0, microsecond=0)
        if now.hour < 4:
            last_reset -= timedelta(days=1)
        # Activity before the last 04:00 boundary is due; activity after it is not.
        assert SessionStore._policy_reset_reason(policy, last_reset - timedelta(minutes=1)) == "daily"
        assert SessionStore._policy_reset_reason(policy, last_reset + timedelta(minutes=1)) is None

    def test_both_mode_whichever_first(self):
        policy = SessionResetPolicy(mode="both", at_hour=4, idle_minutes=60)
        stale = datetime.now() - timedelta(days=2)
        assert SessionStore._policy_reset_reason(policy, stale) in {"idle", "daily"}

    def test_none_mode_never_resets(self):
        policy = SessionResetPolicy(mode="none")
        ancient = datetime.now() - timedelta(days=365)
        assert SessionStore._policy_reset_reason(policy, ancient) is None


class TestSuspendDueSessions:
    def test_suspends_only_overdue_sessions(self, tmp_path):
        store = _store(tmp_path, {"mode": "daily", "at_hour": 4})
        old = store.get_or_create_session(_source())
        old.updated_at = datetime.now() - timedelta(days=2)
        fresh_source = SessionSource(platform=Platform.TELEGRAM, chat_id="456", user_id="u2")
        fresh = store.get_or_create_session(fresh_source)
        store._save()

        assert store.suspend_due_sessions() == 1
        assert store._entries[old.session_key].suspended is True
        assert store._entries[fresh.session_key].suspended is False
        # Idempotent: already-suspended entries are not counted again.
        assert store.suspend_due_sessions() == 0

    def test_none_mode_suspends_nothing(self, tmp_path):
        store = _store(tmp_path, {"mode": "none"})
        entry = store.get_or_create_session(_source())
        entry.updated_at = datetime.now() - timedelta(days=30)
        store._save()
        assert store.suspend_due_sessions() == 0
        assert store._entries[entry.session_key].suspended is False

    def test_active_processes_block_suspension(self, tmp_path):
        config = GatewayConfig.from_dict({"default_reset_policy": {"mode": "daily", "at_hour": 4}})
        store = SessionStore(
            tmp_path / "sessions", config, has_active_processes_fn=lambda key: True)
        entry = store.get_or_create_session(_source())
        entry.updated_at = datetime.now() - timedelta(days=2)
        store._save()
        assert store.suspend_due_sessions() == 0
        assert store._entries[entry.session_key].suspended is False


class TestRoutingTimeReset:
    """The next inbound message after the boundary starts a fresh session — whether or not the
    housekeeping sweep already suspended the entry."""

    def test_suspended_session_resets_on_next_message(self, tmp_path):
        store = _store(tmp_path, {"mode": "daily", "at_hour": 4})
        old = store.get_or_create_session(_source())
        store.append_to_transcript(old.session_id, {"role": "user", "content": "before reset"})
        old.updated_at = datetime.now() - timedelta(days=2)
        store._save()
        store.suspend_due_sessions()

        routed = store.get_or_create_session(_source())

        assert routed.session_id != old.session_id
        assert routed.was_auto_reset is True
        assert routed.auto_reset_reason == "suspended"
        assert routed.prev_session_id == old.session_id
        assert store._db.get_session(old.session_id)["end_reason"] == "suspended"
        store._db.close()

    def test_unswept_overdue_session_resets_at_routing(self, tmp_path):
        """A message landing between the boundary and the sweep still resets (reason = policy)."""
        store = _store(tmp_path, {"mode": "idle", "idle_minutes": 60})
        old = store.get_or_create_session(_source())
        old.updated_at = datetime.now() - timedelta(minutes=90)
        store._save()

        routed = store.get_or_create_session(_source())

        assert routed.session_id != old.session_id
        assert routed.auto_reset_reason == "idle"
        assert store._db.get_session(old.session_id)["end_reason"] == "idle"
        store._db.close()

    def test_recovered_overdue_row_resets_instead_of_resuming(self, tmp_path):
        store = _store(tmp_path, {"mode": "daily", "at_hour": 4})
        old = store.get_or_create_session(_source())
        store.append_to_transcript(old.session_id, {"role": "user", "content": "durable"})
        # Backdate the durable row itself: recovery derives updated_at from last_activity_at.
        stale_ts = (datetime.now() - timedelta(days=2)).timestamp()
        store._db._write_sql(
            "UPDATE sessions SET last_activity_at = ?, started_at = ? WHERE id = ?",
            (stale_ts, stale_ts, old.session_id))
        # Drop the routing index so the next access must recover from state.db.
        store._db.replace_gateway_routing_entries({}, scope=store._routing_scope())
        store._entries.clear()

        routed = store.get_or_create_session(_source())

        assert routed.session_id != old.session_id
        assert routed.auto_reset_reason == "daily"
        assert store._db.get_session(old.session_id)["end_reason"] == "daily"
        store._db.close()

    def test_fresh_session_unaffected(self, tmp_path):
        store = _store(tmp_path, {"mode": "both", "at_hour": 4, "idle_minutes": 1440})
        old = store.get_or_create_session(_source())
        routed = store.get_or_create_session(_source())
        assert routed.session_id == old.session_id
        store._db.close()
