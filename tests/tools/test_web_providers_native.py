"""Tests for the native local web extract provider."""

from __future__ import annotations

import httpx
import pytest
from types import SimpleNamespace


class _FakeResponse:
    def __init__(self, url: str, text: str, content_type: str = "text/html; charset=utf-8"):
        self.url = url
        self.text = text
        self.headers = {"content-type": content_type}
        self.encoding = "utf-8"

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

    def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(
            url,
            """
            <html>
              <head><title>Native example</title></head>
              <body>
                <nav>Navigation should be omitted.</nav>
                <article>
                  <h1>Main headline</h1>
                  <p>Body text from the local extractor.</p>
                  <ul><li>First item</li></ul>
                </article>
              </body>
            </html>
            """,
        )


def test_native_provider_extracts_html_without_external_api(monkeypatch):
    from plugins.web.native.provider import NativeWebExtractProvider

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    result = NativeWebExtractProvider().extract(["https://example.com/article"])

    assert len(result) == 1
    item = result[0]
    assert item["title"] == "Native example"
    assert "Main headline" in item["content"]
    assert "Body text from the local extractor." in item["content"]
    assert "First item" in item["content"]
    assert "Navigation should be omitted." not in item["content"]
    assert item["metadata"]["backend"] == "native"


def test_native_provider_is_extract_only_and_available_without_api_key():
    from plugins.web.native.provider import NativeWebExtractProvider

    provider = NativeWebExtractProvider()
    assert provider.name == "native"
    assert provider.is_available() is True
    assert provider.supports_search() is False
    assert provider.supports_extract() is True


def test_native_extract_backend_is_selected_without_falling_back_to_brave(monkeypatch):
    from tools import web_tools

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(
        web_tools,
        "_load_web_config",
        lambda: {
            "backend": "firecrawl",
            "search_backend": "brave-free",
            "extract_backend": "native",
        },
    )

    assert web_tools._get_search_backend() == "brave-free"
    assert web_tools._get_extract_backend() == "native"


def test_native_redirect_guard_rejects_private_target(monkeypatch):
    from plugins.web.native import provider
    from tools import url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: False)
    response = SimpleNamespace(
        is_redirect=True,
        headers={"location": "http://127.0.0.1:8080/secret"},
        url="https://public.example/article",
        next_request=None,
    )

    with pytest.raises(ValueError, match="Blocked redirect"):
        provider._ssrf_redirect_guard(response)
