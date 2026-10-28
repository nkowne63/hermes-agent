"""Devin sign-in: the Devin CLI's browser login (Authorization Code + PKCE on a loopback redirect).

``app.devin.ai/auth/cli/continue`` authorizes, and ``api.devin.ai/auth/cli/token`` exchanges the code and
verifier (JSON, no client secret) for a long-lived Devin session JWT. There is no refresh grant: an
expired token means signing in again. The generic ``hermes_cli.auth_oauth_pkce_plugin`` flow runs the
browser leg and stores the pool row; this module supplies only the non-standard token request.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Mapping, Optional

AUTHORIZE_URL = "https://app.devin.ai/auth/cli/continue"
TOKEN_URL = "https://api.devin.ai/auth/cli/token"
# The CLI's registered loopback redirect: http://127.0.0.1:59653/callback.
REDIRECT_PORT = 59653
_FALLBACK_TTL_SECONDS = 365 * 24 * 3600
_TIMEOUT_SECONDS = 30.0


def _err(message: str, code: str, *, relogin: bool = False):
    from hermes_cli.auth_constants import AuthError

    return AuthError(f"devin: {message}", provider="devin", code=code, relogin_required=relogin)


def _jwt_ttl_seconds(token: str) -> Optional[int]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, TypeError):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return int(exp - time.time()) if isinstance(exp, (int, float)) else None


def token_request(grant: Mapping[str, str]) -> Mapping[str, Any]:
    """``OAuthPKCEConfig.token_request``: exchange the authorization code at the Devin CLI token endpoint.

    Returns token-endpoint fields the generic flow understands (``access_token``, ``expires_in`` taken
    from the JWT ``exp``, else a year). A refresh grant cannot be served: Devin issues no refresh token.
    """
    if grant.get("grant_type") != "authorization_code":
        raise _err("the sign-in has expired; run `hermes auth add devin` again.", "invalid_grant", relogin=True)
    from hermes_cli.auth import _default_verify
    from hermes_cli.auth_constants import httpx

    try:
        response = httpx.post(TOKEN_URL, json={"code": grant["code"], "code_verifier": grant["code_verifier"]},
                              headers={"Accept": "application/json", "Content-Type": "application/json"},
                              timeout=_TIMEOUT_SECONDS, verify=_default_verify())
    except Exception as exc:
        raise _err(f"token request failed: {type(exc).__name__}", "oauth_token_exchange_failed") from exc
    if response.status_code >= 400:
        raise _err(f"token request failed with HTTP {response.status_code}.", "oauth_token_exchange_failed")
    try:
        payload = response.json()
    except ValueError as exc:
        raise _err("token response was not JSON.", "oauth_token_exchange_failed") from exc
    token = str((payload or {}).get("token") or "").strip() if isinstance(payload, dict) else ""
    if not token:
        raise _err("token response carried no token.", "oauth_token_exchange_failed")
    ttl = _jwt_ttl_seconds(token)
    return {"access_token": token, "expires_in": ttl if ttl and ttl > 0 else _FALLBACK_TTL_SECONDS}
