"""Display-safe model/account identity helpers for gateway surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _claim_email(token: Any) -> str:
    if not isinstance(token, str) or not token.strip():
        return ""
    try:
        from hermes_cli.auth import _decode_jwt_claims

        claims = _decode_jwt_claims(token)
    except Exception:
        return ""
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def credential_identity(runtime: Optional[Mapping[str, Any]]) -> str:
    """Return a display-safe account label for the selected credential."""

    if not runtime:
        return ""

    email = str(runtime.get("credential_email") or "").strip()
    if not email:
        email = _claim_email(runtime.get("api_key"))
    if email:
        return email

    label = str(runtime.get("credential_label") or "").strip()
    if label:
        return label

    source = str(runtime.get("source") or runtime.get("credential_source") or "").strip()
    if source and source not in {"explicit", "config", "env"}:
        return source
    return ""


def credential_account_line(runtime: Optional[Mapping[str, Any]]) -> str:
    """Return a human-readable account line, or an empty string if unknown."""

    identity = credential_identity(runtime)
    if not identity:
        return ""
    provider = str((runtime or {}).get("provider") or "").strip()
    if provider:
        return f"Account: `{identity}`"
    return f"Account: `{identity}`"
