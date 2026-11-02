"""应用级依赖：从 app.state 获取单例服务，避免全局变量。"""

from __future__ import annotations

from fastapi import Request

from app.agent.service import AgentService
from app.core.config import Settings
from app.memory.base import ConversationMemory
from app.rag.pipeline import IngestionPipeline
from app.rag.vector_store import FaissVectorStore


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_memory(request: Request) -> ConversationMemory:
    return request.app.state.memory


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline


def get_vector_store(request: Request) -> FaissVectorStore:
    return request.app.state.vector_store


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
