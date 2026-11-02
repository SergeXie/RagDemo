"""FastAPI 应用入口。

lifespan 负责依序构建：配置 -> 日志 -> 对话记忆(Redis) -> Embedding ->
向量库 -> 入库管道 -> Agent 服务；关闭时按反序释放资源。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.agent.service import AgentService
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.memory.base import ConversationMemory
from app.memory.factory import create_memory
from app.rag.embeddings import EmbeddingClient
from app.rag.pipeline import IngestionPipeline
from app.rag.vector_store import FaissVectorStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。优先使用 create_app 注入的配置（测试场景）。"""
    settings = getattr(app.state, "settings", None) or get_settings()
    setup_logging(settings.log_level)
    logger.info("启动智能客服系统 v%s", __version__)

    # 1. 对话记忆（Redis，失败自动降级内存）
    memory: ConversationMemory = await create_memory(settings)

    # 2. 向量库（Embedding + FAISS）
    embedder = EmbeddingClient(settings)
    store = FaissVectorStore(settings.vector_store_path, dim=embedder.dim)
    pipeline = IngestionPipeline(settings, embedder, store)

    # 3. Agent 服务
    agent_service = AgentService(settings, memory, embedder, store)

    # 挂载到 app.state，供依赖注入获取
    app.state.settings = settings
    app.state.memory = memory
    app.state.embedder = embedder
    app.state.vector_store = store
    app.state.pipeline = pipeline
    app.state.agent_service = agent_service

    logger.info("系统就绪：向量库 %s", store.stats())
    try:
        yield
    finally:
        logger.info("关闭中：释放对话记忆连接…")
        await memory.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂（便于测试注入配置）。"""
    settings = settings or get_settings()

    app = FastAPI(
        title="智能客服咨询系统",
        description=(
            "基于 LangChain + FastAPI 的企业级智能客服："
            "多工具 Agent、Redis 多轮记忆与摘要、RAG 知识库检索、SSE 流式响应。"
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS：前后端分离部署
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.embedder = None
    app.state.vector_store = None
    app.state.pipeline = None
    app.state.agent_service = None
    app.state.memory = None

    # ---------------- 全局异常处理 ----------------

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("业务异常 %s %s: %s", request.url.path, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "请求参数校验失败",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常 %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "服务内部错误", "detail": None},
        )

    # ---------------- 路由 ----------------

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "智能客服咨询系统",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
