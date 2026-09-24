"""Devin ACP provider profile.

devin-acp does not speak OpenAI-over-HTTP: it drives an external ACP subprocess
(``devin acp``) over stdio, so the profile supplies its own client via
:meth:`ProviderProfile.create_client` — the same registration seam copilot-acp
uses. Launch details come from this profile (env vars win over the static
defaults); ``acp.devin`` config settings are read at client construction.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class DevinACPProfile(ProviderProfile):
    """Devin ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client.

        ``acp.devin.reasoning_effort`` (or legacy ``thinking_level``) is passed to
        the Devin CLI as ``DEVIN_REASONING_EFFORT``; the selected model reaches it
        as ``DEVIN_MODEL`` (see ``CopilotACPClient.model_env_var``). Tool-restriction
        keys in ``acp.devin`` (``hermes_tools_only``, ``allowed_tools``,
        ``deny_tools``, ``agent_config``, ``tool_platform``, ``mcp_bridge``) are
        read but not applied — they belong to the unported ACPProviderAdapter
        framework.
        """
        from agent.acp_settings import load_acp_settings
        from agent.copilot_acp_client import CopilotACPClient

        settings = load_acp_settings("devin", "devin_acp")
        extra_env = {}
        effort = str(settings.get("reasoning_effort") or settings.get("thinking_level") or "").strip()
        if effort:
            extra_env["DEVIN_REASONING_EFFORT"] = effort
        return CopilotACPClient(
            extra_env=extra_env, model_env_var="DEVIN_MODEL",
            model_env_skip=("devin", "devin-acp"), acp_settings=settings, **client_kwargs)


devin_acp = DevinACPProfile(
    name="devin-acp", aliases=("devin", "devin-acp-agent"),
    display_name="Devin ACP",
    description="Devin ACP (Spawns devin acp)",
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by ACP subprocess
    base_url="acp://devin",  # ACP internal scheme
    auth_type="external_process",
    process_command="devin",
    process_args=("acp",),
    process_command_env_vars=("HERMES_DEVIN_ACP_COMMAND", "DEVIN_CLI_PATH"),
    process_args_env_var="HERMES_DEVIN_ACP_ARGS",
    fallback_models=(
        "swe-1.6", "glm-5.2", "gpt-5.4-mini", "gpt-5.4", "claude-sonnet-4.5",
        "adaptive", "devin-acp"),
)

register_provider(devin_acp)
