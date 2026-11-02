"""对话记忆抽象接口。

所有实现必须保证：同一 user_id 下的 conversation 数据互不可见，
即 user_id 是数据隔离的最外层边界。
"""

from __future__ import annotations

import abc
from typing import Any


class ConversationMemory(abc.ABC):
    """多轮对话存储抽象。消息以 dict 形式存储：{"role", "content", "timestamp"}。"""

    # ---------------- 会话生命周期 ----------------

    @abc.abstractmethod
    async def create_conversation(self, user_id: str, conversation_id: str) -> None:
        """注册一个新会话（记录归属用户与创建时间）。"""

    @abc.abstractmethod
    async def belongs_to(self, user_id: str, conversation_id: str) -> bool:
        """判断会话是否属于该用户（防止越权访问他人会话）。"""

    @abc.abstractmethod
    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """删除会话及其全部历史，返回是否存在。"""

    @abc.abstractmethod
    async def list_conversations(self, user_id: str) -> list[str]:
        """列出用户的全部会话 ID（按创建时间倒序）。"""

    # ---------------- 消息读写 ----------------

    @abc.abstractmethod
    async def append_message(self, user_id: str, conversation_id: str, message: dict[str, Any]) -> None:
        """追加一条消息。"""

    @abc.abstractmethod
    async def get_messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        """读取会话全部原始消息（不含摘要）。"""

    # ---------------- 摘要 ----------------

    @abc.abstractmethod
    async def get_summary(self, user_id: str, conversation_id: str) -> str | None:
        """获取早期对话摘要，无则返回 None。"""

    @abc.abstractmethod
    async def set_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        """写入 / 更新早期对话摘要。"""

    @abc.abstractmethod
    async def trim_messages(self, user_id: str, conversation_id: str, keep_recent: int) -> None:
        """裁剪历史，仅保留最近 keep_recent 条消息（被裁剪部分已压缩为摘要）。"""

    # ---------------- 连接管理 ----------------

    async def ping(self) -> bool:
        """健康检查。默认视为可用。"""
        return True

    async def close(self) -> None:
        """释放底层连接资源。"""
