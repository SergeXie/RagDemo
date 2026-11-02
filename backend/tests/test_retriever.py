"""文档检索工具测试。"""

import pytest

from app.agent.tools.retriever import create_retriever_tool
from app.core.config import Settings


class _FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.0]


class _FakeStore:
    async def search(self, query_vec: list[float], top_k: int) -> list:
        return []


@pytest.mark.asyncio
async def test_retriever_tool_awaits_search_function():
    settings = Settings(_env_file=None, embedding_api_key="test", embedding_dim=1)
    tool = create_retriever_tool(settings, _FakeEmbedder(), _FakeStore())

    result = await tool.ainvoke({"query": "谢希杰"})

    assert result == "知识库中未找到与该问题相关的内容。"
