"""应用配置。

使用 pydantic-settings 从环境变量 / .env 文件加载配置，
所有配置项均有安全默认值，密钥类配置必须通过环境变量注入。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置模型。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ===================== LLM =====================
    llm_api_key: str = Field(default="", description="LLM API Key")
    llm_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="OpenAI 兼容 API 地址",
    )
    llm_model: str = Field(default="doubao-seed-2-1-pro-260915", description="对话模型名")
    llm_temperature: float = Field(default=0.3, description="采样温度")
    llm_timeout: float = Field(default=60.0, description="LLM 请求超时（秒）")

    # ===================== Embedding =====================
    embedding_api_key: str = Field(default="", description="Embedding API Key")
    embedding_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="Embedding OpenAI 兼容 API 地址",
    )
    embedding_model: str = Field(default="doubao-embedding-text-240715", description="Embedding 模型名")
    embedding_dim: int = Field(default=2560, description="向量维度")
    embedding_batch_size: int = Field(default=64, description="批量嵌入大小")

    # ===================== 联网搜索 =====================
    web_search_model: str = Field(
        default="",
        description="Responses API 联网搜索模型，留空复用 LLM 配置",
    )

    # ===================== Redis / 记忆 =====================
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis 连接串")
    redis_key_prefix: str = Field(default="chat", description="Redis key 前缀")
    memory_backend: str = Field(
        default="auto",
        description="记忆后端：auto（Redis 不可用时降级为内存）/ redis / memory",
    )
    memory_max_messages: int = Field(default=20, description="触发摘要压缩的消息数阈值")
    memory_keep_recent: int = Field(default=6, description="压缩后保留的最近消息数")
    memory_summary_max_tokens: int = Field(default=500, description="摘要长度上限（近似 token）")

    # ===================== 知识库 / RAG =====================
    vector_store_path: Path = Field(default=Path("./data/vector_store"), description="FAISS 持久化目录")
    upload_dir: Path = Field(default=Path("./data/uploads"), description="原始文件暂存目录")
    chunk_size: int = Field(default=500, description="分块大小（字符）")
    chunk_overlap: int = Field(default=50, description="分块重叠（字符）")
    top_k: int = Field(default=4, description="检索返回条数")
    similarity_threshold: float = Field(default=0.3, description="余弦相似度过滤阈值")
    max_upload_mb: int = Field(default=20, description="单文件上传大小上限（MB）")

    # ===================== 服务 =====================
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:8080",
        description="允许的跨域来源，逗号分隔",
    )

    # ===================== 校验 =====================
    @field_validator("memory_backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        allowed = {"auto", "redis", "memory"}
        if v.lower() not in allowed:
            raise ValueError(f"memory_backend 必须是 {allowed} 之一")
        return v.lower()

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_web_search_model(self) -> str:
        return self.web_search_model or self.llm_model

    @property
    def embedding_api_key_effective(self) -> str:
        """Embedding Key 为空时回退到 LLM Key（多数网关两者通用）。"""
        return self.embedding_api_key or self.llm_api_key

    @property
    def embedding_base_url_effective(self) -> str:
        return self.embedding_base_url or self.llm_base_url


@lru_cache
def get_settings() -> Settings:
    """带缓存的全局配置单例。"""
    return Settings()
