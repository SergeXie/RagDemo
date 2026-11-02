"""对话记忆模块。

提供多轮对话历史的存储、按用户隔离、以及长对话摘要压缩能力。
"""

from .base import ConversationMemory
from .fallback import InMemoryConversationMemory
from .factory import create_memory
from .redis_store import RedisConversationMemory
from .summarizer import Summarizer

__all__ = [
    "ConversationMemory",
    "InMemoryConversationMemory",
    "RedisConversationMemory",
    "Summarizer",
    "create_memory",
]
