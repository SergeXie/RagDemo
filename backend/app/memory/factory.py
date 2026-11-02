"""对话记忆工厂：按配置选择后端，Redis 失败自动降级。"""

from __future__ import annotations

from redis.asyncio import Redis

from app.core.config import Settings
from app.core.logging import get_logger

from .base import ConversationMemory
from .fallback import InMemoryConversationMemory
from .redis_store import RedisConversationMemory

logger = get_logger(__name__)


async def create_memory(settings: Settings) -> ConversationMemory:
    """构建对话记忆实例。

    - memory_backend=redis：强制 Redis，连接失败直接抛错（严格模式）；
    - memory_backend=memory：纯内存；
    - memory_backend=auto（默认）：优先 Redis，ping 不通则降级为内存并告警。
    """
    backend = settings.memory_backend

    if backend == "memory":
        logger.info("对话记忆使用内存后端（降级模式，数据不持久）")
        return InMemoryConversationMemory()

    redis = Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
        max_connections=20,
    )

    if backend == "redis":
        if not await RedisConversationMemory(redis).ping():
            raise ConnectionError(
                f"无法连接 Redis（{settings.redis_url}），请检查服务是否启动。"
            )
        logger.info("对话记忆使用 Redis 后端: %s", settings.redis_url)
        return RedisConversationMemory(redis, prefix=settings.redis_key_prefix)

    # auto：探测后决定
    probe = RedisConversationMemory(redis, prefix=settings.redis_key_prefix)
    if await probe.ping():
        logger.info("对话记忆使用 Redis 后端: %s", settings.redis_url)
        return probe

    logger.warning(
        "Redis 不可达（%s），自动降级为内存记忆后端 —— 仅适用于开发/单实例场景！",
        settings.redis_url,
    )
    try:
        await redis.aclose()
    except Exception:  # noqa: BLE001
        pass
    return InMemoryConversationMemory()
