"""对话相关数据模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """流式对话请求。"""

    query: str = Field(..., min_length=1, max_length=8000, description="用户问题")
    user_id: str = Field(..., min_length=1, max_length=128, description="用户 ID（数据隔离维度）")
    conversation_id: str | None = Field(
        default=None, description="会话 ID；不传则由服务端创建新会话"
    )


class ConversationCreated(BaseModel):
    """新会话创建响应。"""

    conversation_id: str
    user_id: str
    created_at: datetime


class ConversationHistory(BaseModel):
    """会话历史响应。"""

    conversation_id: str
    user_id: str
    messages: list[ChatMessage]
    summary: str | None = Field(default=None, description="早期对话摘要（如有）")


class SourceChunk(BaseModel):
    """检索命中的知识片段。"""

    doc_id: str
    doc_name: str
    content: str
    score: float
    page: int | None = None


class AgentEvent(BaseModel):
    """SSE 流式事件的统一信封。

    type 取值：
      - thinking   Agent 思考中（等待首个 token）
      - tool_start 开始调用工具
      - tool_end   工具返回结果
      - token      LLM 生成的增量 token
      - sources    检索命中的知识片段
      - done       本次回复完成
      - error      出错
    """

    type: Literal["thinking", "tool_start", "tool_end", "token", "sources", "done", "error"]
    data: dict = Field(default_factory=dict)
