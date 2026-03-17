import os
from redis.asyncio import Redis

_redis_instance: Redis | None = None


def get_redis() -> Redis:
    """Return a shared async Redis client for the process lifetime."""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=False,
        )
    return _redis_instance
