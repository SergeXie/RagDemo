"""基于 Redis 的对话记忆实现。

Key 设计（prefix 默认 "chat"）：
  {prefix}:conv:{user_id}:{conversation_id}:messages   LIST[json]  消息流水
  {prefix}:conv:{user_id}:{conversation_id}:summary    STRING                   早期对话摘要
  {prefix}:user:{user_id}:convs                        ZSET(score=创建时间戳)  会话索引
  {prefix}:conv:{user_id}:{conversation_id}:owner      STRING                   归属用户（越权校验）

user_id 直接编入 key，天然实现多用户数据物理隔离；
owner 字段用于二次校验，防止 conversation_id 被跨用户猜测访问。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from redis.asyncio import Redis

from app.core.logging import get_logger

from .base import ConversationMemory

logger = get_logger(__name__)


class RedisConversationMemory(ConversationMemory):
    """Redis 实现，线程/协程安全，支持集群部署。"""

    def __init__(self, redis: Redis, prefix: str = "chat") -> None:
        self._redis = redis
        self._p = prefix

    # ---------------- key 构造 ----------------

    def _msgs_key(self, user_id: str, cid: str) -> str:
        return f"{self._p}:conv:{user_id}:{cid}:messages"

    def _summary_key(self, user_id: str, cid: str) -> str:
        return f"{self._p}:conv:{user_id}:{cid}:summary"

    def _owner_key(self, user_id: str, cid: str) -> str:
        return f"{self._p}:conv:{user_id}:{cid}:owner"

    def _index_key(self, user_id: str) -> str:
        return f"{self._p}:user:{user_id}:convs"

    # ---------------- 会话生命周期 ----------------

    async def create_conversation(self, user_id: str, conversation_id: str) -> None:
        await self._redis.set(self._owner_key(user_id, conversation_id), user_id)
        await self._redis.zadd(self._index_key(user_id), {conversation_id: time.time()})

    async def belongs_to(self, user_id: str, conversation_id: str) -> bool:
        owner = await self._redis.get(self._owner_key(user_id, conversation_id))
        return owner == user_id

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        if not await self.belongs_to(user_id, conversation_id):
            return False
        await self._redis.delete(self._msgs_key(user_id, conversation_id))
        await self._redis.delete(self._summary_key(user_id, conversation_id))
        await self._redis.delete(self._owner_key(user_id, conversation_id))
        await self._redis.zrem(self._index_key(user_id), conversation_id)
        return True

    async def list_conversations(self, user_id: str) -> list[str]:
        # ZREVRANGE：最新会话在前
        ids = await self._redis.zrevrange(self._index_key(user_id), 0, -1)
        return [i.decode() if isinstance(i, bytes) else i for i in ids]

    # ---------------- 消息读写 ----------------

    async def append_message(self, user_id: str, conversation_id: str, message: dict[str, Any]) -> None:
        await self._redis.rpush(
            self._msgs_key(user_id, conversation_id),
            json.dumps(message, ensure_ascii=False, default=str),
        )

    async def get_messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(self._msgs_key(user_id, conversation_id), 0, -1)
        return [json.loads(m) if isinstance(m, bytes) else json.loads(m) for m in raw]

    # ---------------- 摘要 ----------------

    async def get_summary(self, user_id: str, conversation_id: str) -> str | None:
        val = await self._redis.get(self._summary_key(user_id, conversation_id))
        if val is None:
            return None
        return val.decode() if isinstance(val, bytes) else val

    async def set_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        await self._redis.set(self._summary_key(user_id, conversation_id), summary)

    async def trim_messages(self, user_id: str, conversation_id: str, keep_recent: int) -> None:
        key = self._msgs_key(user_id, conversation_id)
        total = await self._redis.llen(key)
        if total > keep_recent:
            await self._redis.ltrim(key, total - keep_recent, -1)

    # ---------------- 运维 ----------------

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis ping 失败: %s", exc)
            return False

    async def close(self) -> None:
        await self._redis.aclose()


def new_conversation_id() -> str:
    """生成会话 ID。"""
    return uuid.uuid4().hex
