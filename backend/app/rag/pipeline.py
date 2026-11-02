"""文档入库流水线：解析 -> 分块 -> 向量化 -> 写入向量库 -> 原文件归档。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.document_parser import parse_document
from app.rag.embeddings import EmbeddingClient
from app.rag.splitter import split_pages
from app.rag.vector_store import FaissVectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """知识库入库管道，对外仅暴露 ingest / delete 两个操作。"""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingClient,
        store: FaissVectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    @property
    def store(self) -> FaissVectorStore:
        """暴露底层向量库，供接口层查询元数据。"""
        return self._store

    async def ingest(self, filename: str, data: bytes) -> tuple[str, int]:
        """解析并入库。返回 (doc_id, chunk_count)。"""
        # 1. 解析
        parsed = parse_document(filename, data)

        # 2. 分块
        chunks = split_pages(
            parsed.pages,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        # 3. 向量化
        texts = [text for text, _ in chunks]
        vectors = await self._embedder.embed_documents(texts)

        # 4. 写入向量库（内部持久化）
        doc_id, count = await self._store.add_document(
            doc_name=filename,
            file_type=parsed.file_type,
            chunks=chunks,
            vectors=vectors,
        )

        # 5. 归档原始文件，便于审计与重建
        self._archive(filename, doc_id, data)

        logger.info("入库完成 doc_id=%s file=%s chunks=%d", doc_id, filename, count)
        return doc_id, count

    async def delete(self, doc_id: str) -> int:
        """从知识库删除文档，返回移除块数。"""
        return await self._store.delete_document(doc_id)

    def _archive(self, filename: str, doc_id: str, data: bytes) -> None:
        try:
            base = Path(self._settings.upload_dir)
            day_dir = base / datetime.now(timezone.utc).strftime("%Y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / f"{doc_id}__{filename}").write_bytes(data)
        except Exception as exc:  # noqa: BLE001 - 归档失败不影响主流程
            logger.warning("原始文件归档失败（不影响入库）: %s", exc)
