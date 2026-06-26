"""Native lightweight web extract plugin."""

from __future__ import annotations

from plugins.web.native.provider import NativeWebExtractProvider


def register(ctx) -> None:
    """Register the native lightweight extract provider."""
    ctx.register_web_search_provider(NativeWebExtractProvider())
