from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError

from app.config import Settings, get_settings


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    roles: list[str]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _identity_from_claims(claims: dict[str, object]) -> Identity:
    tenant_id = claims.get("tenant_id")
    roles = claims.get("roles")
    if (
        not isinstance(tenant_id, str)
        or not tenant_id.strip()
        or not isinstance(roles, list)
        or not roles
        or any(not isinstance(role, str) or not role.strip() for role in roles)
    ):
        raise _unauthorized()
    return Identity(tenant_id=tenant_id, roles=roles)


def _jwt_identity(token: str, settings: Settings) -> Identity:
    if (
        not settings.jwt_issuer
        or not settings.jwt_audience
        or not settings.jwt_public_key
    ):
        raise _unauthorized()
    public_key = settings.jwt_public_key.get_secret_value().replace("\\n", "\n")
    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["exp", "tenant_id", "roles"]},
        )
    except InvalidTokenError as exc:
        raise _unauthorized() from exc
    return _identity_from_claims(claims)


async def get_identity(request: Request) -> Identity:
    """Resolve a tenant identity from dev-only headers or a verified JWT."""
    settings = get_settings()
    if settings.env.casefold() == "dev":
        tenant_id = request.headers.get("X-Tenant-ID", "")
        roles_header = request.headers.get("X-Roles", "")
        roles = [role.strip() for role in roles_header.split(",") if role.strip()]
        if tenant_id.strip() and roles:
            return Identity(tenant_id=tenant_id.strip(), roles=roles)
        raise _unauthorized()

    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise _unauthorized()
    return _jwt_identity(token.strip(), settings)
