"""对话接口：SSE 流式对话 + 会话管理。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.agent.service import AgentService
from app.api.deps import get_agent_service, get_memory
from app.core.exceptions import ConversationNotFoundError
from app.memory.base import ConversationMemory
from app.memory.redis_store import new_conversation_id
from app.schemas.chat import ChatMessage, ChatRequest, ConversationHistory

router = APIRouter()


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    service: AgentService = Depends(get_agent_service),
):
    """SSE 流式对话。

    事件格式：`data: {"type": "...", "data": {...}}\\n\\n`，type 取值见
    AgentEvent：thinking / tool_start / tool_end / token / sources / done / error。
    """
    generator = service.chat_stream(
        query=req.query,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
    )

    async def _sse():
        async for event in generator:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 反向代理（nginx）默认缓冲会破坏流式，显式关闭
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations")
async def list_conversations(
    user_id: str = Query(..., min_length=1, description="用户 ID"),
    memory: ConversationMemory = Depends(get_memory),
):
    """列出用户的全部会话 ID（最新在前）。"""
    ids = await memory.list_conversations(user_id)
    return {"user_id": user_id, "conversation_ids": ids, "total": len(ids)}


@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistory)
async def get_history(
    conversation_id: str,
    user_id: str = Query(..., min_length=1),
    memory: ConversationMemory = Depends(get_memory),
):
    """读取会话历史（含早期摘要）。跨用户访问返回 404，不泄露存在性。"""
    if not await memory.belongs_to(user_id, conversation_id):
        raise ConversationNotFoundError("会话不存在或无权访问")

    raw = await memory.get_messages(user_id, conversation_id)
    summary = await memory.get_summary(user_id, conversation_id)
    messages = [
        ChatMessage(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=_parse_ts(m.get("timestamp")),
        )
        for m in raw
    ]
    return ConversationHistory(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=messages,
        summary=summary,
    )


@router.post("/conversations")
async def create_conversation(
    user_id: str = Query(..., min_length=1),
    memory: ConversationMemory = Depends(get_memory),
):
    """显式创建一个空会话（前端「新对话」按钮使用）。"""
    cid = new_conversation_id()
    await memory.create_conversation(user_id, cid)
    return {
        "conversation_id": cid,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Query(..., min_length=1),
    memory: ConversationMemory = Depends(get_memory),
):
    """删除会话及其全部历史。"""
    deleted = await memory.delete_conversation(user_id, conversation_id)
    if not deleted:
        raise ConversationNotFoundError("会话不存在或无权访问")
    return {"conversation_id": conversation_id, "deleted": True}


def _parse_ts(value) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
