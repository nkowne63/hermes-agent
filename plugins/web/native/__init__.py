"""Native local web extraction plugin."""

from __future__ import annotations

from plugins.web.native.provider import NativeWebExtractProvider


def register(ctx) -> None:
    """Register the local, extract-only provider."""
    ctx.register_web_search_provider(NativeWebExtractProvider())
