"""Tests for the native lightweight web extract provider."""

from __future__ import annotations

import sys
from types import SimpleNamespace


class _FakeResponse:
    def __init__(self, *, url: str, text: str, content_type: str = "text/html"):
        self.url = url
        self.text = text
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        return _FakeResponse(
            url=url,
            text=(
                "<html><head><title>Example</title></head>"
                "<body><nav>skip</nav><article><h1>Example</h1>"
                "<p>Hello from native extract.</p></article></body></html>"
            ),
        )


def test_native_provider_extracts_locally(monkeypatch):
    from plugins.web.native import provider as native_provider
    from plugins.web.native.provider import NativeWebExtractProvider

    monkeypatch.setattr(native_provider, "_dependency_status", lambda: "rs_trafilatura")
    monkeypatch.setattr(native_provider, "_extract_content", lambda html, url: ("Example", "Hello markdown"))
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=_FakeClient))

    result = NativeWebExtractProvider().extract(["https://example.com"])

    assert result == [
        {
            "url": "https://example.com",
            "title": "Example",
            "content": "Hello markdown",
            "raw_content": "Hello markdown",
            "metadata": {
                "sourceURL": "https://example.com",
                "title": "Example",
                "backend": "native",
                "extractor": "rs_trafilatura",
            },
        }
    ]


def test_native_provider_is_extract_only():
    from plugins.web.native.provider import NativeWebExtractProvider

    provider = NativeWebExtractProvider()
    assert provider.name == "native"
    assert provider.supports_search() is False
    assert provider.supports_extract() is True
