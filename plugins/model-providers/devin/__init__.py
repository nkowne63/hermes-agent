"""Devin (Cognition) provider profile: the models a Devin/Windsurf account serves, over Codeium Cascade.

Inference is not OpenAI-over-HTTP: it is Connect + protobuf against ``server.codeium.com`` (the API the
Devin CLI and Windsurf use), so the profile supplies its own client (``transport.py``). Credentials:

* ``hermes auth add devin`` — the Devin CLI browser login (PKCE, ``auth.py``); the session JWT is pooled.
* ``DEVIN_API_KEY`` / ``hermes auth add devin --type api-key`` — a Devin or Windsurf API key.

The model roster is account-scoped and comes live from ``GetCliModelConfigs`` (``catalog.py``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig, pkce_auth_handler
from providers import register_provider
from providers.base import ProviderProfile

from . import catalog
from . import protocol as wire
from .auth import AUTHORIZE_URL, REDIRECT_PORT, TOKEN_URL, token_request

# Curated floor merged into the live picker, so only a model Devin still serves (live 2026-10: SWE-1.6 Fast
# answers permission_denied "no longer available for your account").
FALLBACK_MODELS = ("swe-1-6",)
# Seed context window (GetCliModelConfigs reports 200k); the fetched catalog is authoritative when present.
_SEED_CONTEXT = {"swe-1-6": 200_000}

_LOGIN = OAuthPKCEConfig(
    client_id="", authorize_url=AUTHORIZE_URL, token_url=TOKEN_URL, redirect_port=REDIRECT_PORT,
    redirect_path="/callback", label="Devin", token_request=token_request,
    extra_authorize_params={"prompt": "select_account"},
)
_pkce_handler = pkce_auth_handler(_LOGIN)


def _auth_handler(action: str, args: Any) -> bool:
    """Own ``hermes auth add devin`` (browser login) unless an API key was asked for; every other action
    and ``--type api-key`` take the built-in api-key/pool path."""
    if action != "add" or str(getattr(args, "auth_type", "") or "").strip().lower() in ("api-key", "api_key"):
        return False
    return _pkce_handler(action, args)


def _classify_api_error(error: Any, **_: Any) -> Optional[Mapping[str, Any]]:
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None
    if details.get("large_history"):
        # Opaque invalid_argument on a large history: compress instead of failing as a bad request.
        return {"reason": "context_overflow"}
    if str(details.get("connect_code") or "").lower() == "permission_denied" and "model" in str(
            details.get("message") or "").lower():
        # A model the account may not use (retired, plan-gated) is not a dead credential.
        return {"reason": "model_entitlement"}
    return None


class DevinProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from .transport import DevinClient

        allowed = {"api_key", "base_url", "timeout", "http_client"}
        return DevinClient(**{k: v for k, v in client_kwargs.items() if k in allowed})

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        # Read by the client (never sent as JSON): the Cascade thread id that ties this session's turns.
        return {"devin_cascade_id": session_id} if session_id else {}

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 15.0) -> list[str] | None:
        if not (api_key or "").strip():
            return None
        # GetCliModelConfigs returns the full per-account catalog and often takes 7-15s to answer.
        models = catalog.fetch(api_key.strip(), base_url or self.base_url, timeout=max(timeout, 15.0))
        return [m.uid for m in models] if models else None

    def get_model_context_length(self, model: str) -> int | None:
        known = catalog.cached_model(model)
        return known.context_window if known else _SEED_CONTEXT.get(model)

    def get_max_tokens(self, model: str | None) -> int | None:
        return None  # per-model output caps come from the live catalog inside the client


register_provider(DevinProfile(
    name="devin", aliases=("cognition", "devin-cli"), display_name="Devin",
    description="Devin (Cognition) — SWE models and frontier models via your Devin/Windsurf account",
    signup_url="https://app.devin.ai/", env_vars=("DEVIN_API_KEY", "DEVIN_BASE_URL"),
    base_url=wire.DEFAULT_BASE_URL, auth_type="api_key", supports_health_check=False,
    fallback_models=FALLBACK_MODELS, default_aux_model="swe-1-6",
    native_reasoning_details_type="devin.native_assistant",
    auth_handler=_auth_handler, classify_api_error=_classify_api_error,
))
