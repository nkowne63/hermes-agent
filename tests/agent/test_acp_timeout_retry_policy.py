"""Regression tests for subprocess ACP timeout retry policy."""

from __future__ import annotations

from agent.conversation_loop import _is_acp_session_prompt_timeout
from agent.copilot_acp_client import ACPProviderTimeoutError


def test_acp_session_prompt_timeout_is_detected_from_structured_error():
    err = ACPProviderTimeoutError("Copilot ACP", "session/prompt")

    assert _is_acp_session_prompt_timeout(err, "copilot-acp") is True


def test_acp_session_prompt_timeout_detects_legacy_message_shape():
    err = TimeoutError("Timed out waiting for Copilot ACP response to session/prompt.")

    assert _is_acp_session_prompt_timeout(err, "copilot-acp") is True


def test_non_acp_timeout_keeps_default_retry_policy():
    err = TimeoutError("request timed out")

    assert _is_acp_session_prompt_timeout(err, "openrouter") is False

