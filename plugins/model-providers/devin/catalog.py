"""Devin model catalog: the account's live ``GetCliModelConfigs`` roster, normalized for Hermes.

The roster is credential-scoped (plan/team gated), so it is fetched per credential and cached in-process;
the chat transport reads the cache to learn which uids are server-side routers and each model's limits.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import protocol as wire

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_MAX_OUTPUT_TOKENS = 64_000

# Server-side routers resolved through AssignModel before chatting. Discovery marks them; ``adaptive``
# is also known statically so a turn on it works before (or without) a catalog fetch.
STATIC_ROUTER_UIDS = frozenset({"adaptive"})

# Display slots requested for parity with the native client but never surfaced (quick review, internal).
_INTERNAL_DISPLAYS = frozenset({wire.DISPLAY_QUICK_REVIEW, wire.DISPLAY_INTERNAL_DEFAULT})
# Fusion pairings (``fusion-<lead>-sidekick-<x>``) are orchestrated by the native client, which runs the
# lead and pairs the sidekick locally; the composite uid itself has no server provider.
_FUSION = re.compile(r"^fusion-.+-sidekick-")


@dataclass(frozen=True)
class DevinModel:
    uid: str
    label: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_images: bool
    supports_thinking: bool
    supports_parallel_tool_calls: bool
    is_router: bool
    is_default: bool
    cost_per_million: dict[str, float]


def _cost(config: dict[str, Any]) -> dict[str, float]:
    """Per-million-token rates from the config's cost dimensions; reading stops at the ``Sidekick``
    marker that separates a composite's own card from its components'."""
    cost = {"input": 0.0, "output": 0.0, "cache_read": 0.0}
    labels = {"input": "input", "cached input": "cache_read", "output": "output"}
    for dimension in config.get("model_dimensions") or ():
        label = str(dimension.get("label") or "").strip().lower()
        if label == "sidekick":
            break
        if dimension.get("kind") not in wire.COST_KINDS or label not in labels:
            continue
        match = re.match(r"\s*(\d+(?:\.\d+)?)\s*([kmb])?", str(dimension.get("denominator") or ""), re.IGNORECASE)
        tokens = 1_000_000.0
        if match:
            scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get((match.group(2) or "").lower(), 1.0)
            tokens = float(match.group(1)) * scale or 1_000_000.0
        cost[labels[label]] = round(float(dimension.get("value") or 0.0) * 1_000_000 / tokens, 6)
    return cost


def normalize(response: dict[str, Any]) -> list[DevinModel]:
    """Usable models from a decoded ``GetCliModelConfigsResponse``: the account default first, then by uid."""
    default_uid = str((response.get("default_override_model_config") or {}).get("model_uid") or "").strip()
    models: dict[str, DevinModel] = {}
    for config in response.get("client_model_configs") or ():
        uid = str(config.get("model_uid") or "").strip()
        info = config.get("model_info") or {}
        display = info.get("display_option", 0)
        if not uid or uid in models or config.get("disabled") or display in _INTERNAL_DISPLAYS or _FUSION.match(uid):
            continue
        features = info.get("model_features")
        # ``is_model_router`` also marks harness-backed composites, which are ordinary chat uids; only
        # harness-less routing slots take the AssignModel path.
        is_router = (display == wire.DISPLAY_MODEL_ROUTER or bool(info.get("is_model_router"))) and not info.get(
            "harness_uids")
        context = config.get("max_tokens") or info.get("max_tokens") or 0
        max_output = info.get("max_output_tokens") or 0
        models[uid] = DevinModel(
            uid=uid, label=str(config.get("label") or "").strip() or uid,
            context_window=context if context > 0 else DEFAULT_CONTEXT_WINDOW,
            max_output_tokens=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_TOKENS,
            # Router configs ship no features: the model they route to decides, and Cascade only serves
            # tool-calling models.
            supports_tools=bool(features.get("supports_tool_calls")) if features is not None else True,
            supports_images=bool(features.get("supports_images") if features is not None
                                 else config.get("supports_images")),
            supports_thinking=bool(features.get("supports_thinking")) if features is not None else False,
            supports_parallel_tool_calls=bool((features or {}).get("supports_parallel_tool_calls")),
            is_router=is_router, is_default=uid == default_uid, cost_per_million=_cost(config),
        )
    return sorted(models.values(), key=lambda m: (not m.is_default, m.uid))


# ── process cache ────────────────────────────────────────────────────────────────────────────────

_CACHE_TTL_SECONDS = 3600.0
_lock = threading.Lock()
_cache: dict[str, tuple[float, list[DevinModel]]] = {}  # keyed by base URL + credential fingerprint


def _cache_key(api_key: str, base_url: str) -> str:
    import hashlib

    return f"{base_url}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"


def remember(api_key: str, base_url: str, models: list[DevinModel]) -> None:
    with _lock:
        _cache[_cache_key(api_key, base_url)] = (time.monotonic(), models)


def cached_models() -> list[DevinModel]:
    """Every fresh cached model across credentials (newest fetch wins per uid)."""
    now = time.monotonic()
    merged: dict[str, DevinModel] = {}
    with _lock:
        for stamp, models in sorted(_cache.values(), key=lambda item: item[0]):
            if now - stamp <= _CACHE_TTL_SECONDS:
                merged.update((m.uid, m) for m in models)
    return list(merged.values())


def cached_model(uid: str) -> Optional[DevinModel]:
    return next((m for m in cached_models() if m.uid == uid), None)


def is_router(uid: str) -> bool:
    model = cached_model(uid)
    return model.is_router if model is not None else uid in STATIC_ROUTER_UIDS


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def fetch(api_key: str, base_url: str, *, timeout: float = 15.0, http_client: Any = None) -> Optional[list[DevinModel]]:
    """The account's models via ``GetCliModelConfigs``; None on any request/decode failure or an empty
    roster (the backend answers an empty 200 when the pinned client identity is stale)."""
    import httpx

    base = (base_url or wire.DEFAULT_BASE_URL).rstrip("/")
    body = wire.encode(wire.GET_CLI_MODEL_CONFIGS_REQUEST, {"metadata": wire.discovery_metadata(api_key)})
    try:
        if http_client is not None:
            response = http_client.post(base + wire.GET_CLI_MODEL_CONFIGS_PATH, content=body,
                                        headers=wire.unary_headers(), timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(base + wire.GET_CLI_MODEL_CONFIGS_PATH, content=body,
                                       headers=wire.unary_headers())
    except Exception as exc:
        logger.debug("devin: GetCliModelConfigs failed: %s", type(exc).__name__)
        return None
    if response.status_code != 200:
        logger.debug("devin: GetCliModelConfigs returned HTTP %s", response.status_code)
        return None
    decoded = wire.decode_unary(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, response.content)
    models = normalize(decoded) if decoded is not None else []
    if not models:
        logger.warning("devin: empty model catalog; the pinned Devin CLI identity may be stale")
        return None
    remember(api_key, base, models)
    return models
