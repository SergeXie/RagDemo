"""文档检索工具：从 FAISS 知识库检索企业制度文档片段。

检索结果同时通过返回值（给模型）和模块级 sink（给 SSE sources 事件）
两条通道传递 —— 前者供模型引用，后者供前端展示引用来源。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import FaissVectorStore, SearchResult


class _RetrieverInput(BaseModel):
    query: str = Field(..., description="检索关键词或问题，例如：报销标准、请假流程")


def create_retriever_tool(
    settings: Settings,
    embedder: EmbeddingClient,
    store: FaissVectorStore,
    sink: dict | None = None,
) -> StructuredTool:
    """构建知识库检索工具。

    sink: 可选的共享字典，调用后最新检索结果写入 sink["sources"]，
    供流式接口在生成 sources 事件时读取。
    """

    async def search_documents(query: str) -> str:
        """检索企业内部制度文档（规章制度、流程、标准等）。

        当用户询问公司政策、制度、流程、标准、管理办法等问题时使用本工具。
        输入应为精炼的检索词。返回最相关的若干片段及其出处。
        """
        try:
            query_vec = await embedder.embed_query(query)
            results = await store.search(query_vec, top_k=settings.top_k)
        except Exception as exc:  # noqa: BLE001
            return f"知识库检索失败：{exc}"

        # 过滤低相似度噪声
        results = [
            r for r in results if r.score >= settings.similarity_threshold
        ]
        if sink is not None:
            sink["sources"] = [_to_dict(r) for r in results]

        if not results:
            return "知识库中未找到与该问题相关的内容。"

        lines: list[str] = []
        for i, r in enumerate(results, start=1):
            page_info = f" 第{r.page}页" if r.page else ""
            lines.append(
                f"[{i}] 来源：{r.doc_name}{page_info}（相似度 {r.score:.2f}）\n{r.text}"
            )
        return "\n\n".join(lines)

    return StructuredTool.from_function(
        coroutine=search_documents,
        name="document_search",
        description=(
            "检索企业内部制度文档知识库。"
            "当用户询问公司规章制度、流程、报销/请假/考勤标准、管理办法等内部政策问题时，"
            "必须先调用本工具获取准确依据再回答。"
        ),
        args_schema=_RetrieverInput,
    )


def _to_dict(r: SearchResult) -> dict:
    return {
        "doc_id": r.doc_id,
        "doc_name": r.doc_name,
        "content": r.text,
        "score": round(r.score, 4),
        "page": r.page,
    }
