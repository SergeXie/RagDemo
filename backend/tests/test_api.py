"""API 层集成测试：健康检查、参数校验、会话隔离（不依赖 LLM/Redis）。

Agent 服务依赖外部 LLM API，这里仅验证路由、异常处理与记忆隔离链路。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.memory.fallback import InMemoryConversationMemory


@pytest.fixture
def client() -> TestClient:
    """构建测试用 App：跳过 lifespan（不连 Redis），手动注入内存记忆。"""
    settings = Settings(
        _env_file=None,
        memory_backend="memory",
        llm_api_key="test-key",
        vector_store_path="data/test_vs",
    )
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        # 手动注入 lifespan 中间件未执行时的依赖
        memory = InMemoryConversationMemory()
        app.state.settings = settings
        app.state.memory = memory
        app.state.vector_store = _FakeStore()
        yield c


class _FakeStore:
    def list_documents(self):
        return []

    def stats(self):
        return {"total_vectors": 0, "total_documents": 0, "dim": 0, "path": "test"}


def test_root(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_openapi_schema(client: TestClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/v1/chat/stream" in paths
    assert "/api/v1/documents/upload" in paths
    assert "/api/v1/health" in paths


def test_chat_stream_rejects_empty_query(client: TestClient):
    resp = client.post("/api/v1/chat/stream", json={"query": "", "user_id": "u1"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "VALIDATION_ERROR"


def test_chat_stream_rejects_missing_user(client: TestClient):
    resp = client.post("/api/v1/chat/stream", json={"query": "你好"})
    assert resp.status_code == 422


def test_conversation_isolation_via_api(client: TestClient):
    """用户 A 创建的会话，用户 B 无法读取历史。"""
    # A 创建会话
    resp = client.post("/api/v1/chat/conversations", params={"user_id": "user_a"})
    assert resp.status_code == 200
    cid = resp.json()["conversation_id"]

    # A 可读（空历史）
    resp = client.get(
        f"/api/v1/chat/conversations/{cid}/history", params={"user_id": "user_a"}
    )
    assert resp.status_code == 200
    assert resp.json()["messages"] == []

    # B 不可读
    resp = client.get(
        f"/api/v1/chat/conversations/{cid}/history", params={"user_id": "user_b"}
    )
    assert resp.status_code == 404

    # B 删除也会失败
    resp = client.delete(
        f"/api/v1/chat/conversations/{cid}", params={"user_id": "user_b"}
    )
    assert resp.status_code == 404


def test_conversation_list_scoped(client: TestClient):
    client.post("/api/v1/chat/conversations", params={"user_id": "u1"})
    client.post("/api/v1/chat/conversations", params={"user_id": "u1"})

    resp = client.get("/api/v1/chat/conversations", params={"user_id": "u1"})
    assert resp.json()["total"] == 2

    resp = client.get("/api/v1/chat/conversations", params={"user_id": "u2"})
    assert resp.json()["total"] == 0


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    # 注入的内存记忆 ping 为 True
    assert resp.status_code == 200
    body = resp.json()
    assert body["memory_backend"] == "InMemoryConversationMemory"
    assert body["status"] == "healthy"


def test_upload_rejects_oversized_file(client: TestClient):
    # 将应用注入的配置上限临时置 0（client.app 即 fixture 创建的 app）
    settings = client.app.state.settings
    original = settings.max_upload_mb
    settings.max_upload_mb = 0
    try:
        big = b"x" * (1024 * 1024)  # 1MB > 0MB
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("big.txt", big, "text/plain")},
        )
        assert resp.status_code == 413
        assert resp.json()["code"] == "FILE_TOO_LARGE"
    finally:
        settings.max_upload_mb = original


def test_delete_nonexistent_document(client: TestClient):
    resp = client.delete("/api/v1/documents/not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "DOCUMENT_NOT_FOUND"
