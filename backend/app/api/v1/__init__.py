"""API v1 路由。"""

from fastapi import APIRouter

from app.api.v1 import chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["健康检查"])
api_router.include_router(chat.router, prefix="/chat", tags=["对话"])
api_router.include_router(documents.router, prefix="/documents", tags=["知识库文档"])
