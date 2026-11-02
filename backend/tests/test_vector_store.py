"""FAISS 向量库测试（使用确定性假向量，不依赖外部 Embedding API）。"""

import asyncio

import numpy as np
import pytest

from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.rag.vector_store import FaissVectorStore

DIM = 32


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray([vec], dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _make_vector(seed: int) -> np.ndarray:
    """生成第 seed 个标准基向量（相互正交 => 相似度可预期）。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    return FaissVectorStore(tmp_path / "vs", dim=DIM)


@pytest.mark.asyncio
async def test_empty_search(store: FaissVectorStore):
    assert await store.search(np.zeros(DIM, dtype=np.float32)) == []


@pytest.mark.asyncio
async def test_add_and_search(store: FaissVectorStore):
    v1 = _make_vector(1)
    v2 = _make_vector(2)
    await store.add_document(
        "docA.pdf", "pdf",
        chunks=[("A的内容", 1), ("A的第二页", 2)],
        vectors=np.vstack([v1, v2]),
    )

    results = await store.search(v1, top_k=2)
    assert results
    assert results[0].doc_name == "docA.pdf"
    assert results[0].text == "A的内容"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)  # 自身检索=1
    assert results[0].page == 1


@pytest.mark.asyncio
async def test_multi_document_ranking(store: FaissVectorStore):
    va, vb = _make_vector(10), _make_vector(20)
    await store.add_document("A.txt", "text", [("alpha", None)], va.reshape(1, -1))
    await store.add_document("B.txt", "text", [("beta", None)], vb.reshape(1, -1))

    results = await store.search(va, top_k=2)
    assert results[0].doc_name == "A.txt"
    assert results[1].doc_name == "B.txt"


@pytest.mark.asyncio
async def test_dimension_mismatch_rejected(store: FaissVectorStore):
    with pytest.raises(VectorStoreError):
        await store.add_document(
            "x.txt", "text", [("t", None)], np.zeros((1, DIM + 5), dtype=np.float32)
        )


@pytest.mark.asyncio
async def test_chunk_vector_count_mismatch(store: FaissVectorStore):
    with pytest.raises(VectorStoreError):
        await store.add_document(
            "x.txt", "text", [("t1", None), ("t2", None)],
            np.zeros((1, DIM), dtype=np.float32),
        )


@pytest.mark.asyncio
async def test_delete_document(store: FaissVectorStore):
    va, vb = _make_vector(30), _make_vector(40)
    id_a, _ = await store.add_document("A.txt", "text", [("a", None)], va.reshape(1, -1))
    await store.add_document("B.txt", "text", [("b", None)], vb.reshape(1, -1))

    removed = await store.delete_document(id_a)
    assert removed == 1
    docs = store.list_documents()
    assert [d.doc_name for d in docs] == ["B.txt"]
    # A 的向量已不可检索
    results = await store.search(va, top_k=2)
    assert all(r.doc_id != id_a for r in results)


@pytest.mark.asyncio
async def test_delete_nonexistent(store: FaissVectorStore):
    with pytest.raises(DocumentNotFoundError):
        await store.delete_document("nope")


@pytest.mark.asyncio
async def test_persistence_reload(tmp_path):
    path = tmp_path / "vs"
    v = _make_vector(99)

    store1 = FaissVectorStore(path, dim=DIM)
    doc_id, _ = await store1.add_document(
        "持久化.txt", "text", [("内容XYZ", 3)], v.reshape(1, -1)
    )

    # 重新加载
    store2 = FaissVectorStore(path, dim=DIM)
    assert store2.stats()["total_vectors"] == 1
    results = await store2.search(v, top_k=1)
    assert results[0].text == "内容XYZ"
    assert results[0].doc_id == doc_id
    assert results[0].doc_name == "持久化.txt"


@pytest.mark.asyncio
async def test_concurrent_adds(store: FaissVectorStore):
    async def add(i: int):
        await store.add_document(
            f"doc{i}.txt", "text", [(f"t{i}", None)], _make_vector(100 + i).reshape(1, -1)
        )

    await asyncio.gather(*[add(i) for i in range(5)])
    assert store.stats()["total_vectors"] == 5
    assert store.stats()["total_documents"] == 5
