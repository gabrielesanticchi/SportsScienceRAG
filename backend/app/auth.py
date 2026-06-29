"""JWT authentication and tenant context."""

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

security = HTTPBearer()


@dataclass(frozen=True)
class TenantContext:
    user_id: UUID
    tenant_id: UUID
    email: str
    role: str


def create_access_token(
    *,
    user_id: UUID,
    tenant_id: UUID,
    email: str,
    role: str = "member",
    settings: Settings | None = None,
) -> str:
    """Issue a signed JWT for local testing."""
    cfg = settings or get_settings()
    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "email": email,
        "role": role,
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> TenantContext:
    cfg = settings or get_settings()
    try:
        payload = jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    try:
        return TenantContext(
            user_id=UUID(payload["sub"]),
            tenant_id=UUID(payload["tenant_id"]),
            email=payload.get("email", ""),
            role=payload.get("role", "member"),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token claims",
        ) from exc


async def get_current_tenant(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TenantContext:
    return decode_token(credentials.credentials, settings)
