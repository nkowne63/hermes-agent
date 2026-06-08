"""Native lightweight web content extraction.

This provider is intentionally extract-only. It fetches safe public URLs via
``httpx`` and extracts the main page content locally. ``rs-trafilatura`` is
preferred for speed and low overhead; ``trafilatura`` and then readability
fallbacks are used when available.

Config keys this provider responds to::

    web:
      extract_backend: "native"

No API key is required.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; HermesNativeExtract/1.0; "
    "+https://github.com/nkowne63/hermes-agent)"
)


def _dependency_status() -> str | None:
    for module_name in ("rs_trafilatura", "trafilatura", "readability"):
        try:
            __import__(module_name)
            return module_name
        except ImportError:
            continue
    return None


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _title_from_html(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    return " ".join(unescape(re.sub(r"(?is)<[^>]+>", " ", match.group(1))).split())


def _extract_with_rs_trafilatura(html: str, url: str) -> str | None:
    try:
        import rs_trafilatura  # type: ignore
    except ImportError:
        return None

    extract = getattr(rs_trafilatura, "extract", None)
    if not callable(extract):
        return None

    for kwargs in (
        {"url": url, "output_markdown": True, "include_tables": True},
        {"url": url, "output_format": "markdown"},
        {"url": url, "format": "markdown"},
        {"url": url},
    ):
        try:
            result = extract(html, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("rs-trafilatura extract failed for %s: %s", url, exc)
            return None
        if isinstance(result, str) and result.strip():
            return result.strip()
    return None


def _extract_with_trafilatura(html: str, url: str) -> str | None:
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None

    try:
        result = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("trafilatura extract failed for %s: %s", url, exc)
        return None
    return result.strip() if isinstance(result, str) and result.strip() else None


def _extract_with_readability(html: str) -> str | None:
    try:
        from readability import Document  # type: ignore
    except ImportError:
        return None

    try:
        summary = Document(html).summary()
    except Exception as exc:  # noqa: BLE001
        logger.debug("readability extract failed: %s", exc)
        return None
    text = _html_to_text(summary)
    return text or None


def _extract_content(html: str, url: str) -> tuple[str, str]:
    title = _title_from_html(html)
    content = (
        _extract_with_rs_trafilatura(html, url)
        or _extract_with_trafilatura(html, url)
        or _extract_with_readability(html)
        or _html_to_text(html)
    )
    return title, content


class NativeWebExtractProvider(WebSearchProvider):
    """Extract public web pages locally without a hosted extract API."""

    @property
    def name(self) -> str:
        return "native"

    @property
    def display_name(self) -> str:
        return "Native Extract"

    def is_available(self) -> bool:
        return _dependency_status() is not None

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": url, "title": "", "content": "", "error": "Interrupted"} for url in urls]

            if _dependency_status() is None:
                try:
                    from tools.lazy_deps import ensure

                    ensure("web.native_extract", prompt=False)
                except Exception as exc:  # noqa: BLE001
                    return [
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Native extract dependencies are not installed. "
                                "Install rs-trafilatura or trafilatura. "
                                f"Details: {exc}"
                            ),
                        }
                        for url in urls
                    ]

            import httpx

            max_chars = int(kwargs.get("max_chars") or 300_000)
            timeout = float(kwargs.get("timeout") or 20.0)
            results: List[Dict[str, Any]] = []
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                for url in urls:
                    try:
                        response = client.get(url)
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "")
                        if "html" not in content_type and "xml" not in content_type and response.text.lstrip().startswith("<") is False:
                            text = response.text.strip()
                            title = ""
                        else:
                            title, text = _extract_content(response.text, str(response.url))
                        if max_chars > 0:
                            text = text[:max_chars]
                        results.append(
                            {
                                "url": str(response.url),
                                "title": title,
                                "content": text,
                                "raw_content": text,
                                "metadata": {
                                    "sourceURL": str(response.url),
                                    "title": title,
                                    "backend": "native",
                                    "extractor": _dependency_status() or "html",
                                },
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
            return results
        except Exception as exc:  # noqa: BLE001
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

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Native Extract",
            "badge": "free · local · extract only",
            "tag": "Lightweight local HTML-to-markdown extraction via rs-trafilatura/trafilatura.",
            "env_vars": [],
        }
