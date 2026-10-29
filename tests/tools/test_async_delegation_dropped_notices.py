"""Every way the ledger drops an undelivered completion for good is published once, after its transaction
commits, so a delivery loop can tell the completion's chat (``drain_dropped_completions``)."""

import queue
import time

import pytest

from tools import async_delegation as ad

_ROUTE = "agent:main:discord:thread:111:222"


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _pending(delegation_id: str) -> None:
    now = time.time()
    event = {"type": "async_delegation", "delegation_id": delegation_id, "session_key": _ROUTE,
             "dispatched_at": now, "completed_at": now, "status": "completed", "summary": "RESULT TEXT"}
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "RESULT TEXT"})


def _set(delegation_id: str, **cols) -> None:
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(f"UPDATE async_delegations SET {', '.join(f'{k}=?' for k in cols)} WHERE delegation_id=?",
                     (*cols.values(), delegation_id))


def _drop(path: str, delegation_id: str) -> None:
    q: queue.Queue = queue.Queue()
    if path == "replay_age":
        _set(delegation_id, completed_at=time.time() - ad._MAX_COMPLETION_REPLAY_AGE_S - 60)
        assert ad.restore_undelivered_completions(q) == 0
    elif path == "release_budget":
        assert ad.claim_completion_delivery(delegation_id, "claim")
        _set(delegation_id, delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
        assert ad.release_completion_delivery(delegation_id, "claim")
    elif path == "sweep_budget":
        abandoned = time.time() - 3600
        _set(delegation_id, delivery_claim="gateway:stale", delivery_claimed_at=abandoned, updated_at=abandoned,
             delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
        assert ad.sweep_orphaned_completions(q) == 0
    elif path == "requeue_budget":
        assert ad.claim_completion_delivery(delegation_id, "claim")
        assert ad.complete_completion_delivery(delegation_id, "claim")
        _set(delegation_id, delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
        assert ad.requeue_admitted_completion(delegation_id, q) is False
    else:
        assert ad.claim_completion_delivery(delegation_id, "claim")
        assert ad.drop_completion_delivery(delegation_id, "claim")
    assert q.empty()


@pytest.mark.parametrize(("path", "reason"), [
    ("replay_age", "replay_age"),
    ("release_budget", "attempts_exhausted"),
    ("sweep_budget", "attempts_exhausted"),
    ("requeue_budget", "attempts_exhausted"),
    ("target_gone", "target_gone"),
])
def test_each_terminal_drop_is_published_once_with_its_route(path, reason):
    _pending("deleg_dropped")
    assert ad.drain_dropped_completions() == []

    _drop(path, "deleg_dropped")

    assert ad.get_durable_delegation("deleg_dropped")["delivery_state"] == "dropped"
    (dropped,) = ad.drain_dropped_completions()
    assert (dropped["delegation_id"], dropped["reason"]) == ("deleg_dropped", reason)
    assert dropped["event"]["session_key"] == _ROUTE and dropped["event"]["delegation_id"] == "deleg_dropped"
    assert "summary" not in dropped["event"]  # only the route travels, never the result
    assert ad.drain_dropped_completions() == []


def test_a_drop_whose_transaction_rolls_back_is_not_published():
    _pending("deleg_rolled_back")
    with pytest.raises(RuntimeError):
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET delivery_state='dropped' WHERE delegation_id=?",
                         ("deleg_rolled_back",))
            ad._stage_dropped(conn, "deleg_rolled_back", "target_gone")
            raise RuntimeError("commit never happens")
    assert ad.get_durable_delegation("deleg_rolled_back")["delivery_state"] == "pending"
    assert ad.drain_dropped_completions() == []


def test_settles_that_drop_nothing_publish_nothing():
    _pending("deleg_kept")
    assert ad.claim_completion_delivery("deleg_kept", "claim")
    assert ad.release_completion_delivery("deleg_kept", "claim")  # within budget: pending again
    assert not ad.drop_completion_delivery("deleg_kept", "not-the-claim")
    assert ad.get_durable_delegation("deleg_kept")["delivery_state"] == "pending"
    assert ad.drain_dropped_completions() == []
    # The owner (this process) is alive and the row is unclaimed: the sweep leaves it alone.
    assert ad.sweep_orphaned_completions(queue.Queue(), now=time.time() + 3600) == 0
    assert ad.drain_dropped_completions() == []
