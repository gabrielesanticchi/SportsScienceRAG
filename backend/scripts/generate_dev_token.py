"""Generate a development JWT for the seeded demo tenant."""

from uuid import UUID

from app.auth import create_access_token

DEMO_USER_ID = UUID("22222222-2222-2222-2222-222222222222")
DEMO_TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def main() -> None:
    token = create_access_token(
        user_id=DEMO_USER_ID,
        tenant_id=DEMO_TENANT_ID,
        email="demo@example.com",
        role="admin",
    )
    print(token)


if __name__ == "__main__":
    main()
