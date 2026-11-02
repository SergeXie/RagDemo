"""Embedding 客户端：OpenAI 兼容接口，批量向量化 + 归一化。"""

from __future__ import annotations

import asyncio

import numpy as np
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """对 /embeddings 接口的轻量封装。

    支持两种网关形态：
      - OpenAI 标准：base_url 指向 API 根（如 .../api/v3），SDK 拼接 /embeddings，
        input 为字符串列表，响应 data 为 [{embedding}] 列表；
      - 火山方舟多模态：base_url 以 /embeddings/multimodal 结尾（doubao-embedding-vision-*），
        该模型不支持标准 /embeddings 接口，需逐条调用，input 为
        [{"type": "text", "text": ...}]，响应 data 为 {embedding} 对象。
    """

    def __init__(self, settings: Settings) -> None:
        api_key = settings.embedding_api_key_effective
        if not api_key:
            raise LLMServiceError(
                "未配置 Embedding API Key",
                detail="请设置 EMBEDDING_API_KEY（或 LLM_API_KEY 作为回退）",
            )
        base = settings.embedding_base_url_effective.rstrip("/")
        self._multimodal_path = ""
        if base.endswith("/embeddings/multimodal"):
            self._multimodal_path = "/embeddings/multimodal"
            base = base[: -len("/embeddings/multimodal")]
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base,
            timeout=settings.llm_timeout,
        )
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._dim = settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量嵌入，返回 L2 归一化后的 (N, dim) float32 矩阵。"""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
            # 温和限速，避免网关限流
            if i + self._batch_size < len(texts):
                await asyncio.sleep(0.05)
        arr = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(arr)

    async def embed_query(self, text: str) -> np.ndarray:
        """单条查询嵌入，返回 (dim,) 归一化向量。"""
        vecs = await self._embed_batch([text])
        arr = np.asarray(vecs, dtype=np.float32)
        return _l2_normalize(arr)[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._multimodal_path:
            return await self._embed_multimodal(batch)
        try:
            resp = await self._client.embeddings.create(model=self._model, input=batch)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Embedding 调用失败: {exc}") from exc
        # 按 index 排序保证顺序稳定
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    async def _embed_multimodal(self, batch: list[str]) -> list[list[float]]:
        """多模态端点：一次请求对应一条输入，受限并发逐条调用。"""
        sem = asyncio.Semaphore(8)

        async def one(text: str) -> list[float]:
            async with sem:
                try:
                    resp = await self._client.post(
                        self._multimodal_path,
                        cast_to=object,
                        body={
                            "model": self._model,
                            "input": [{"type": "text", "text": text}],
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    raise LLMServiceError(f"Embedding 调用失败: {exc}") from exc
                data = resp.get("data")
                # 标准返回 data 为对象；批量形态兼容列表
                if isinstance(data, list):
                    return [d["embedding"] for d in data][0]
                if isinstance(data, dict) and "embedding" in data:
                    return data["embedding"]
                raise LLMServiceError(f"Embedding 响应格式异常: {list(resp.keys())}")

        return list(await asyncio.gather(*(one(t) for t in batch)))


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2 归一化，使内积等价于余弦相似度。零向量原样返回防止 NaN。"""
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)
