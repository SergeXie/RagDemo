"""项目根目录演示脚本：向量化文档并做语义搜索。
流程：加载 backend/.env 配置 -> 文本向量化 -> 写入 FAISS 向量库 -> 查询 -> 打印结果。
用法：
    python test_vector_search.py
说明：
- 向量库写入 backend/data/test_vs（独立演示目录，每次运行前清空，不污染正式知识库）。
- 【无降级】仅调用真实 Embedding API，API调用失败直接抛出异常退出，不使用本地模拟向量。
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 与服务端保持一致：.env 与 ./data/* 相对路径均按 backend/ 解析
os.chdir(BACKEND)

from app.core.config import get_settings  # noqa: E402
from app.rag.vector_store import FaissVectorStore  # noqa: E402

DOCUMENTS = [
    "Python 是一种面向对象的解释型编程语言，广泛用于数据科学和 AI 开发",
    "机器学习是人工智能的子领域，让计算机从数据中学习规律",
    "深度学习使用多层神经网络，在图像识别和 NLP 任务中表现优异",
    "向量数据库专门存储高维向量，支持语义相似度搜索",
    "PostgreSQL 是功能强大的开源关系型数据库",
    "Redis 是基于内存的高性能键值数据库，常用于缓存",
    "Docker 容器化技术让应用可以在任何环境中一致运行",
    "Git 是分布式版本控制系统，是现代软件开发的基础工具",
]

QUERIES = [
    "应该选关系型数据库还是内存数据库",
    "什么技术适合做图像识别",
    "推荐一个 Python 数据科学的工具",
]

STORE_DIR = Path("data/test_vs")


async def _embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """仅调用Embedding API，失败直接抛出异常，无本地降级兜底"""
    settings = get_settings()
    key = settings.embedding_api_key_effective
    if not key or key.startswith("your-"):
        raise RuntimeError("EMBEDDING_API_KEY 未配置或为占位符，无法调用Embedding API")

    from app.rag.embeddings import EmbeddingClient
    last_exc: Exception | None = None
    # 保留最多2次重试，重试后仍然失败直接抛错退出
    for attempt in range(2):
        try:
            print(f"[Embedding] 尝试调用API，第{attempt+1}次")
            vectors = await EmbeddingClient(settings).embed_documents(texts)
            return vectors, f"Embedding API（{settings.embedding_model}）"
        except Exception as exc:
            last_exc = exc
            print(f"[Embedding API 异常] attempt={attempt+1}, err={exc}")
            if attempt == 0:
                await asyncio.sleep(1.0)

    # 两次重试全部失败，抛出原始异常，不再降级
    raise RuntimeError(f"Embedding API 两次调用全部失败: {last_exc}") from last_exc


async def main() -> None:
    settings = get_settings()
    # 演示目录每次清空，保证重复运行结果可预期
    store_path = BACKEND / STORE_DIR
    if store_path.exists():
        for child in store_path.iterdir():
            child.unlink()
    store = FaissVectorStore(store_path, dim=settings.embedding_dim)

    # 1. 向量化并入库
    vectors, mode = await _embed(DOCUMENTS)
    doc_id, count = await store.add_document(
        doc_name="demo_sentences.txt",
        file_type="txt",
        chunks=[(text, None) for text in DOCUMENTS],
        vectors=vectors,
    )
    print(f"向量化模式: {mode}")
    print(f"入库完成: doc_id={doc_id}, {count} 个分块 -> {store_path}")

    # 2. 搜索
    print("\n===== 语义搜索 =====")
    for query in QUERIES:
        qvec, _ = await _embed([query])
        results = await store.search(qvec[0], top_k=3)
        print(f"\n查询: {query}")
        for rank, hit in enumerate(results, 1):
            print(f"  {rank}. [{hit.score:.4f}] {hit.text}")

    # 3. 状态与落盘文件
    stats = store.stats()
    print("\n===== 向量库状态 =====")
    print(
        f"向量总数: {stats['total_vectors']}, 文档数: {stats['total_documents']}, "
        f"维度: {stats['dim']}, 路径: {stats['path']}"
    )
    for file in sorted(store_path.iterdir()):
        print(f"  - {file.name} ({file.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
