"""Dev auth token endpoint for frontend integration."""

from uuid import UUID

from fastapi import APIRouter

from app.auth import create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

DEMO_USER_ID = UUID("22222222-2222-2222-2222-222222222222")
DEMO_TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


@router.get("/dev-token")
async def dev_token() -> dict[str, str]:
    token = create_access_token(
        user_id=DEMO_USER_ID,
        tenant_id=DEMO_TENANT_ID,
        email="demo@example.com",
        role="admin",
    )
    return {
        "token": token,
        "tenant_id": str(DEMO_TENANT_ID),
    }
