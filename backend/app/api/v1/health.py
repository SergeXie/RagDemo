"""健康检查接口。"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response, status

from app import __version__
from app.api.deps import get_memory, get_vector_store
from app.memory.base import ConversationMemory
from app.rag.vector_store import FaissVectorStore

router = APIRouter()


@router.get("/health")
async def health(
    response: Response,
    memory: ConversationMemory = Depends(get_memory),
    store: FaissVectorStore = Depends(get_vector_store),
):
    """存活 + 就绪检查。Redis 不可用时返回 degraded（503），便于负载均衡摘流。"""
    redis_ok = await memory.ping()
    backend = type(memory).__name__

    status_code = status.HTTP_200_OK if redis_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    response.status_code = status_code

    return {
        "status": "healthy" if redis_ok else "degraded",
        "version": __version__,
        "time": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "memory_backend": backend,
        "redis_ok": redis_ok,
        "vector_store": store.stats(),
    }
