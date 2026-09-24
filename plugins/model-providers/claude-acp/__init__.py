"""Claude ACP provider profile.

claude-acp does not speak OpenAI-over-HTTP: it drives an external ACP subprocess
(``npx -y @agentclientprotocol/claude-agent-acp``) over stdio, so the profile
supplies its own client via :meth:`ProviderProfile.create_client` — the same
registration seam copilot-acp uses. Launch details come from this profile (env
vars win over the static defaults); ``acp.claude`` config settings are read at
client construction.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _normalize_anthropic_model(model: str) -> str:
    """Anthropic-wire model id (``claude-opus-4.6`` → ``claude-opus-4-6``); the raw
    string when the adapter is unavailable."""
    try:
        from agent.anthropic_message_convert import normalize_model_name

        return normalize_model_name(model)
    except Exception:
        return model


class ClaudeACPProfile(ProviderProfile):
    """Claude ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client.

        The selected model reaches the adapter as ``ANTHROPIC_MODEL`` (normalized
        to Anthropic's hyphenated id form). ``acp.claude`` tool-restriction keys
        (``hermes_tools_only``, ``allowed_tools``, ``deny_tools``,
        ``agent_config``, ``tool_platform``, ``mcp_bridge``) are read but not
        applied — they belong to the unported ACPProviderAdapter framework.
        """
        from agent.acp_settings import load_acp_settings
        from agent.copilot_acp_client import CopilotACPClient

        settings = load_acp_settings("claude", "claude_acp")
        return CopilotACPClient(
            model_env_var="ANTHROPIC_MODEL", model_env_skip=("claude", "claude-acp"),
            model_env_normalize=_normalize_anthropic_model, acp_settings=settings, **client_kwargs)


claude_acp = ClaudeACPProfile(
    name="claude-acp", aliases=("claude-agent-acp", "anthropic-acp"),
    display_name="Claude ACP",
    description="Claude ACP (Spawns claude-agent-acp)",
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by ACP subprocess
    base_url="acp://claude",  # ACP internal scheme
    auth_type="external_process",
    process_command="npx",
    process_args=("-y", "@agentclientprotocol/claude-agent-acp"),
    process_command_env_vars=("HERMES_CLAUDE_ACP_COMMAND", "CLAUDE_AGENT_ACP_PATH"),
    process_args_env_var="HERMES_CLAUDE_ACP_ARGS",
    fallback_models=(
        "claude-fable-5", "claude-sonnet-4.6", "claude-opus-4.6", "claude-opus-4.8",
        "claude-haiku-4.5"),
)

register_provider(claude_acp)
