"""联网搜索工具：基于 OpenAI Responses API 的原生 web_search 能力。

对接火山方舟等兼容网关时，通过 responses.create 携带
tools=[{"type": "web_search"}] 让模型在服务端完成检索并返回带来源的答案。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class _WebSearchInput(BaseModel):
    query: str = Field(..., description="联网搜索的问题，例如：2026 年最新社保政策")


def _extract_text(response: Any) -> str:
    """从 Responses 对象中提取文本与引用来源。"""
    parts: list[str] = []

    # 1) output_text 便捷属性（SDK 有则直接用）
    output_text = getattr(response, "output_text", "")
    if output_text:
        parts.append(output_text)

    # 2) 遍历 output 项兜底提取（不同网关实现略有差异）
    citations: list[str] = []
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text and text not in parts:
                parts.append(text)
            # 收集 URL 类引用
            annotations = getattr(block, "annotations", None) or []
            for ann in annotations:
                url = getattr(ann, "url", None) or getattr(ann, "cite", None)
                if url and url not in citations:
                    citations.append(str(url))

    if citations:
        parts.append("参考来源：" + "；".join(citations[:5]))
    return "\n".join(parts).strip()


def create_web_search_tool(settings: Settings) -> StructuredTool | None:
    """构建联网搜索工具；未配置 Key 时返回 None（Agent 降级为无此工具）。"""
    api_key = settings.llm_api_key
    if not api_key:
        logger.warning("未配置 LLM_API_KEY，联网搜索工具不可用")
        return None

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )
    model = settings.effective_web_search_model

    async def web_search(query: str) -> str:
        """搜索互联网获取实时信息。

        当用户询问时效性强的问题（新闻、天气、股价、最新政策等）
        或知识库中没有答案的公开信息时使用。
        """
        try:
            response = await client.responses.create(
                model=model,
                input=[{"role": "user", "content": query}],
                tools=[{"type": "web_search"}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网搜索失败: %s", exc)
            return f"联网搜索暂时不可用：{exc}"

        text = _extract_text(response)
        return text or "搜索未返回有效结果，请尝试换一种问法。"

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "联网搜索实时信息。当用户询问新闻、天气、股价、"
            "最新政策变动等时效性问题，或内部知识库无法回答的公开信息时使用。"
        ),
        args_schema=_WebSearchInput,
    )
