"""Per-provider ``acp:`` config section reader for external-process (ACP) providers.

``config.yaml`` may carry per-provider ACP settings::

    acp:
      devin:
        hermes_tools_only: true
        reasoning_effort: medium
        tool_platform: discord
      claude:
        hermes_tools_only: true
        tool_platform: discord

:func:`load_acp_settings` merges ``acp.<provider_key>`` over a legacy top-level
section (``devin_acp`` / ``claude_acp`` / ``copilot_acp``), matching the shape the
pre-upstream fork consumed. Only launch-adjacent keys are honored by the minimal
registration port — see the provider profiles under ``plugins/model-providers/``
for which keys each one applies; tool-restriction keys (``hermes_tools_only``,
``allowed_tools``, ``deny_tools``, ``agent_config``, ``tool_platform``,
``mcp_bridge``) belong to the ACPProviderAdapter framework, which is not ported.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_acp_settings(provider_key: str, legacy_key: str = "") -> dict[str, Any]:
    """Merged ``acp.<provider_key>`` over legacy ``<legacy_key>`` config section.

    Both sections are optional; a missing or malformed section contributes
    nothing. ``acp.<provider_key>`` wins on key conflicts.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    acp_cfg = cfg.get("acp") if isinstance(cfg, dict) else {}
    provider_cfg = acp_cfg.get(provider_key) if isinstance(acp_cfg, dict) else {}
    legacy_cfg = cfg.get(legacy_key) if legacy_key and isinstance(cfg, dict) else {}
    merged: dict[str, Any] = {}
    if isinstance(legacy_cfg, dict):
        merged.update(legacy_cfg)
    if isinstance(provider_cfg, dict):
        merged.update(provider_cfg)
    return merged
