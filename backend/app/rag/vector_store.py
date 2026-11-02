"""FAISS 向量库封装。

设计要点：
  - 使用 IndexFlatIP（内积）存储 L2 归一化向量，内积 == 余弦相似度；
  - 每个向量携带 string id（uuid），元数据（文档名、页码等）存 JSON 侧文件；
  - 支持按 doc_id 整库删除（FAISS 无原生删除，采用「筛选重建」策略，
    中小规模知识库场景下可接受；亿级请迁移 Milvus）；
  - 增量 add 与 search 并发安全（asyncio.Lock 保护索引重建窗口）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "chunks.json"
_DOCS_FILE = "documents.json"


@dataclass
class ChunkMeta:
    """单个向量块的元数据。"""

    chunk_id: str
    doc_id: str
    doc_name: str
    file_type: str
    text: str
    page: int | None


@dataclass
class DocMeta:
    """知识库文档注册信息。"""

    doc_id: str
    doc_name: str
    file_type: str
    chunk_count: int
    uploaded_at: str


@dataclass
class SearchResult:
    """检索命中结果。"""

    text: str
    doc_id: str
    doc_name: str
    page: int | None
    score: float


class FaissVectorStore:
    """线程安全的 FAISS 本地向量库。"""

    def __init__(self, path: Path, dim: int) -> None:
        self._path = Path(path)
        self._dim = dim
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        # 空库：IndexIDMap2 包裹 FlatIP，id 为自增 int64
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._id_to_chunk: dict[int, ChunkMeta] = {}
        self._documents: dict[str, DocMeta] = {}
        self._load()

    # ---------------- 持久化 ----------------

    def _load(self) -> None:
        idx_file = self._path / _INDEX_FILE
        meta_file = self._path / _META_FILE
        docs_file = self._path / _DOCS_FILE
        if not (idx_file.exists() and meta_file.exists()):
            return
        try:
            self._index = faiss.read_index(str(idx_file))
            chunks_raw = json.loads(meta_file.read_text(encoding="utf-8"))
            self._id_to_chunk = {int(k): ChunkMeta(**v) for k, v in chunks_raw.items()}
            if docs_file.exists():
                docs_raw = json.loads(docs_file.read_text(encoding="utf-8"))
                self._documents = {k: DocMeta(**v) for k, v in docs_raw.items()}
            logger.info(
                "加载向量库: %d 向量 / %d 文档 (%s)",
                self._index.ntotal, len(self._documents), self._path,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"向量库加载失败: {exc}") from exc

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._path / _INDEX_FILE))
        chunks = {str(k): asdict(v) for k, v in self._id_to_chunk.items()}
        (self._path / _META_FILE).write_text(
            json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        docs = {k: asdict(v) for k, v in self._documents.items()}
        (self._path / _DOCS_FILE).write_text(
            json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # ---------------- 写入 ----------------

    async def add_document(
        self,
        doc_name: str,
        file_type: str,
        chunks: list[tuple[str, int | None]],
        vectors: np.ndarray,
    ) -> tuple[str, int]:
        """写入一个文档的全部块。返回 (doc_id, chunk_count)。"""
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"块数({len(chunks)})与向量数({vectors.shape[0]})不一致"
            )
        if vectors.shape[0] == 0:
            raise VectorStoreError("没有可写入的文本块")
        if vectors.shape[1] != self._dim:
            raise VectorStoreError(
                f"向量维度 {vectors.shape[1]} 与库维度 {self._dim} 不符（检查 EMBEDDING_MODEL 配置）"
            )

        doc_id = uuid.uuid4().hex[:12]
        ids = np.arange(self._index.ntotal, self._index.ntotal + len(chunks), dtype=np.int64)
        metas = [
            ChunkMeta(
                chunk_id=uuid.uuid4().hex[:12],
                doc_id=doc_id,
                doc_name=doc_name,
                file_type=file_type,
                text=text,
                page=page,
            )
            for (text, page) in chunks
        ]

        async with self._lock:
            self._index.add_with_ids(vectors.astype(np.float32), ids)
            for int_id, meta in zip(ids.tolist(), metas):
                self._id_to_chunk[int_id] = meta
            self._documents[doc_id] = DocMeta(
                doc_id=doc_id,
                doc_name=doc_name,
                file_type=file_type,
                chunk_count=len(chunks),
                uploaded_at=datetime.now(timezone.utc).isoformat(),
            )
            self._save()

        logger.info("文档入库 doc_id=%s name=%s chunks=%d", doc_id, doc_name, len(chunks))
        return doc_id, len(chunks)

    async def delete_document(self, doc_id: str) -> int:
        """删除文档及其全部向量。返回删除的块数。

        FAISS 不支持原生删除，这里通过「保留其余向量重建索引」实现。
        """
        async with self._lock:
            if doc_id not in self._documents:
                raise DocumentNotFoundError(f"文档不存在: {doc_id}")

            keep_ids = [
                i for i, m in self._id_to_chunk.items() if m.doc_id != doc_id
            ]
            removed = len(self._id_to_chunk) - len(keep_ids)

            # 提取保留向量并重建
            old_index = self._index
            if keep_ids:
                kept_vectors = old_index.reconstruct_n(0, old_index.ntotal)
                keep_arr = np.asarray(keep_ids, dtype=np.int64)
                new_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))
                new_index.add_with_ids(kept_vectors[keep_arr], keep_arr)
                self._index = new_index
            else:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

            self._id_to_chunk = {i: self._id_to_chunk[i] for i in keep_ids}
            del self._documents[doc_id]
            self._save()

        logger.info("文档删除 doc_id=%s 移除块数=%d", doc_id, removed)
        return removed

    # ---------------- 检索 ----------------

    async def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        """相似度检索。空库返回空列表。"""
        if self._index.ntotal == 0:
            return []

        def _do_search() -> list[SearchResult]:
            k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(
                query_vector.reshape(1, -1).astype(np.float32), k
            )
            results: list[SearchResult] = []
            for score, int_id in zip(scores[0], ids[0]):
                if int_id == -1:  # FAISS 用 -1 表示未命中
                    continue
                meta = self._id_to_chunk.get(int(int_id))
                if meta is None:
                    continue
                results.append(
                    SearchResult(
                        text=meta.text,
                        doc_id=meta.doc_id,
                        doc_name=meta.doc_name,
                        page=meta.page,
                        score=float(score),
                    )
                )
            return results

        # reconstruct/search 为 CPU 密集，扔到线程池避免阻塞事件循环
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_search)

    # ---------------- 元数据查询 ----------------

    def list_documents(self) -> list[DocMeta]:
        docs = sorted(self._documents.values(), key=lambda d: d.uploaded_at, reverse=True)
        return docs

    def stats(self) -> dict:
        return {
            "total_vectors": int(self._index.ntotal),
            "total_documents": len(self._documents),
            "dim": self._dim,
            "path": str(self._path),
        }
