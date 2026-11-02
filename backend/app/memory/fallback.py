"""进程内对话记忆实现。

用途：
  1. Redis 不可用时的自动降级，保证服务可用性（AP 倾向）；
  2. 单元测试无需外部依赖。

语义与 Redis 版本完全一致，数据仅存于本进程，重启即失。
"""

from __future__ import annotations

import time
from typing import Any

from .base import ConversationMemory


class InMemoryConversationMemory(ConversationMemory):
    """dict + list 实现，接口与 Redis 版对齐。"""

    def __init__(self) -> None:
        # {user_id: {cid: {"messages": [...], "summary": str|None, "created_at": float}}}
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    def _conv(self, user_id: str, cid: str) -> dict[str, Any] | None:
        return self._store.get(user_id, {}).get(cid)

    # ---------------- 会话生命周期 ----------------

    async def create_conversation(self, user_id: str, conversation_id: str) -> None:
        self._store.setdefault(user_id, {})[conversation_id] = {
            "messages": [],
            "summary": None,
            "created_at": time.time(),
        }

    async def belongs_to(self, user_id: str, conversation_id: str) -> bool:
        return self._conv(user_id, conversation_id) is not None

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        conv = self._conv(user_id, conversation_id)
        if conv is None:
            return False
        del self._store[user_id][conversation_id]
        return True

    async def list_conversations(self, user_id: str) -> list[str]:
        convs = self._store.get(user_id, {})
        return sorted(convs, key=lambda c: convs[c]["created_at"], reverse=True)

    # ---------------- 消息读写 ----------------

    async def append_message(self, user_id: str, conversation_id: str, message: dict[str, Any]) -> None:
        conv = self._conv(user_id, conversation_id)
        if conv is None:
            await self.create_conversation(user_id, conversation_id)
            conv = self._conv(user_id, conversation_id)
        conv["messages"].append(dict(message))

    async def get_messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        conv = self._conv(user_id, conversation_id)
        return [dict(m) for m in conv["messages"]] if conv else []

    # ---------------- 摘要 ----------------

    async def get_summary(self, user_id: str, conversation_id: str) -> str | None:
        conv = self._conv(user_id, conversation_id)
        return conv["summary"] if conv else None

    async def set_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        conv = self._conv(user_id, conversation_id)
        if conv is not None:
            conv["summary"] = summary

    async def trim_messages(self, user_id: str, conversation_id: str, keep_recent: int) -> None:
        conv = self._conv(user_id, conversation_id)
        if conv is not None and len(conv["messages"]) > keep_recent:
            conv["messages"] = conv["messages"][-keep_recent:]
