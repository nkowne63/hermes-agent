"""Local native URL extraction without a hosted extraction API.

The provider intentionally supports extraction only. It uses Hermes' existing
``httpx`` dependency and Python's standard-library HTML parser, so selecting
``web.extract_backend: native`` does not require an API key or an extra package.
Optional third-party extractors are deliberately not required for availability:
the standard-library fallback must always remain usable.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; HermesNativeExtract/1.0; "
    "+https://github.com/NousResearch/hermes-agent)"
)
_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
_SUPPORTED_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPES = ("text/plain", "application/json", "application/xml", "text/xml")
_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "nav", "footer", "aside", "form"})
_BLOCK_TAGS = frozenset({"address", "article", "blockquote", "div", "dl", "fieldset", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul"})


def _ssrf_redirect_guard(response: Any) -> None:
    """Re-check every redirect target before httpx follows it."""
    from tools.url_safety import is_safe_url, redirect_target_from_response

    redirect_url = redirect_target_from_response(response)
    if redirect_url and not is_safe_url(redirect_url):
        raise ValueError(f"Blocked redirect to private/internal address: {redirect_url}")


class _ReadableHTMLParser(HTMLParser):
    """Small dependency-free HTML-to-markdown-ish text parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: List[str] = []
        self._meta_title = ""
        self._heading_level: Optional[int] = None
        self._list_item = False
        self._line_parts: List[str] = []
        self._lines: List[str] = []

    @staticmethod
    def _attrs(attrs: List[tuple[str, Optional[str]]]) -> Dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def _flush(self) -> None:
        text = " ".join(self._line_parts)
        text = re.sub(r"\s+", " ", text).strip()
        self._line_parts.clear()
        if not text:
            return
        if self._heading_level is not None:
            line = f"{'#' * self._heading_level} {text}"
        elif self._list_item:
            line = f"- {text}"
        else:
            line = text
        if not self._lines or self._lines[-1] != line:
            self._lines.append(line)

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in _SKIP_TAGS:
                self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_TAGS:
            self._flush()
            self._skip_depth = 1
            return
        attributes = self._attrs(attrs)
        if tag == "meta":
            name = attributes.get("property", attributes.get("name", "")).lower()
            if name in {"og:title", "twitter:title"} and not self._meta_title:
                self._meta_title = attributes.get("content", "").strip()
            return
        if tag in _BLOCK_TAGS:
            self._flush()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
        elif tag == "li":
            self._list_item = True
        elif tag == "br":
            self._flush()

    def handle_startendtag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag in _SKIP_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self._flush()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = None
        elif tag == "li":
            self._list_item = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._line_parts.append(data)

    def finish(self) -> tuple[str, str]:
        self._flush()
        title = " ".join(" ".join(self._title_parts).split()) or self._meta_title
        return title, "\n\n".join(self._lines).strip()


def _extract_html(html: str) -> tuple[str, str]:
    parser = _ReadableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
        title, content = parser.finish()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Native HTML parser failed: %s", exc)
        title, content = "", ""

    if content:
        return title, content

    # Keep a conservative fallback for malformed pages that the parser cannot
    # turn into block text. It still removes executable/style content.
    fallback = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", html)
    fallback = re.sub(r"(?is)<[^>]+>", " ", fallback)
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return title, fallback


def _content_kind(content_type: str, text: str) -> str:
    content_type = content_type.lower().split(";", 1)[0].strip()
    if content_type in _SUPPORTED_HTML_TYPES or text.lstrip().startswith("<"):
        return "html"
    if content_type in _TEXT_TYPES or content_type.startswith("text/"):
        return "text"
    return "unsupported"


class NativeWebExtractProvider(WebSearchProvider):
    """Extract public web pages locally without a hosted API."""

    @property
    def name(self) -> str:
        return "native"

    @property
    def display_name(self) -> str:
        return "Native Extract"

    def is_available(self) -> bool:
        # httpx is already a required Hermes web-tool dependency. No vendor
        # credential or optional parser package is needed for the stdlib path.
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        import httpx

        timeout = float(kwargs.get("timeout") or 20.0)
        results: List[Dict[str, Any]] = []
        try:
            from tools.interrupt import is_interrupted
        except ImportError:
            is_interrupted = lambda: False  # type: ignore[assignment]

        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.8,*/*;q=0.5",
                },
            ) as client:
                for url in urls:
                    if is_interrupted():
                        results.append({"url": url, "title": "", "content": "", "error": "Interrupted"})
                        continue
                    if not isinstance(url, str) or not url.strip():
                        results.append({"url": str(url), "title": "", "content": "", "error": "URL must be a non-empty string"})
                        continue
                    try:
                        response = client.get(url)
                        response.raise_for_status()
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > _MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"Response exceeds {_MAX_DOWNLOAD_BYTES} byte limit")
                        body = response.text
                        if len(body.encode(response.encoding or "utf-8", errors="replace")) > _MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"Response exceeds {_MAX_DOWNLOAD_BYTES} byte limit")
                        kind = _content_kind(response.headers.get("content-type", ""), body)
                        final_url = str(response.url)
                        if kind == "html":
                            title, content = _extract_html(body)
                        elif kind == "text":
                            title, content = "", body.strip()
                        else:
                            raise ValueError(
                                "Native Extract supports HTML/text pages only "
                                f"(received {response.headers.get('content-type', 'unknown')})"
                            )
                        results.append(
                            {
                                "url": final_url,
                                "title": title,
                                "content": content,
                                "raw_content": content,
                                "metadata": {"backend": "native", "content_type": response.headers.get("content-type", "")},
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Native extract failed for %s: %s", url, exc)
                        results.append(
                            {
                                "url": url,
                                "title": "",
                                "content": "",
                                "raw_content": "",
                                "error": f"Native extract failed: {exc}",
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Native extract client failed: %s", exc)
            return [
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Native extract failed: {exc}",
                }
                for url in urls
            ]
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Native Extract",
            "badge": "free · local · extract only",
            "tag": "Local HTML/text extraction using Hermes' built-in HTTP client; no API key required.",
            "env_vars": [],
        }
