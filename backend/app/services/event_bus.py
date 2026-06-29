"""Redis-backed parsing event bus for SSE subscribers."""

import json
from typing import Any

import redis

from app.config import get_settings


def _channel(tenant_id: str, workspace_id: str) -> str:
    return f"parsing:{tenant_id}:{workspace_id}"


def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.celery_result_backend, decode_responses=True)


def publish_parsing_event(
    tenant_id: str,
    workspace_id: str,
    event: dict[str, Any],
) -> None:
    client = get_redis_client()
    client.publish(_channel(tenant_id, workspace_id), json.dumps(event))


def subscribe_parsing_events(tenant_id: str, workspace_id: str):
    client = get_redis_client()
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(_channel(tenant_id, workspace_id))
    return pubsub
