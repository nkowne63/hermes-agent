"""Devin ACP provider profile.

Devin ACP uses an external ACP subprocess — not the standard REST/OpenAI
transport. The profile captures the registry metadata so Hermes can treat it as
an external-process backend.
"""

from providers import register_provider
from providers.base import ProviderProfile


class DevinACPProfile(ProviderProfile):
    """Devin ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


devin_acp = DevinACPProfile(
    name="devin-acp",
    aliases=("devin", "devin-acp-agent"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://devin",
    auth_type="external_process",
)

register_provider(devin_acp)
