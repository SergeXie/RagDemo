"""Agent 服务：组装工具、构建 LangGraph ReAct Agent、编排流式对话。

职责边界：
  - 从对话记忆加载历史（摘要 + 近期消息）并注入上下文；
  - 运行 LangGraph ReAct Agent，将工具调用 / token 转换为统一事件流；
  - 回复结束后持久化消息到 Redis，并按需触发摘要压缩。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agent.prompts import SUMMARY_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from app.agent.tools import calculator_tool, create_retriever_tool, create_web_search_tool
from app.core.config import Settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger
from app.memory.base import ConversationMemory
from app.memory.redis_store import new_conversation_id
from app.memory.summarizer import Summarizer, maybe_compress_history
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import FaissVectorStore

logger = get_logger(__name__)

# 事件类型常量
EVT_THINKING = "thinking"
EVT_TOOL_START = "tool_start"
EVT_TOOL_END = "tool_end"
EVT_TOKEN = "token"
EVT_SOURCES = "sources"
EVT_DONE = "done"
EVT_ERROR = "error"


class AgentService:
    """智能客服 Agent 编排服务（单例，由应用 lifespan 构建）。"""

    def __init__(
        self,
        settings: Settings,
        memory: ConversationMemory,
        embedder: EmbeddingClient,
        store: FaissVectorStore,
    ) -> None:
        self._settings = settings
        self._memory = memory

        # 每次请求独立的检索结果收集器
        self._sink: dict[str, Any] = {}

        # 工具集：检索器写入 sink，便于向前端推送 sources 事件
        tools = [create_retriever_tool(settings, embedder, store, sink=self._sink), calculator_tool]
        web_search = create_web_search_tool(settings)
        if web_search is not None:
            tools.append(web_search)

        if not settings.llm_api_key:
            raise LLMServiceError(
                "未配置 LLM_API_KEY",
                detail="请复制 backend/.env.example 为 backend/.env 并填入 API Key",
            )

        self._llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            streaming=True,
        )
        self._agent = create_react_agent(self._llm, tools, prompt=SYSTEM_PROMPT)
        self._summarizer = Summarizer(self._llm, max_chars=settings.memory_summary_max_tokens)
        logger.info(
            "Agent 初始化完成 model=%s tools=%s",
            settings.llm_model, [t.name for t in tools],
        )

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式对话。产出统一事件 dict（对应 schemas.chat.AgentEvent）。"""
        # 会话准备：新建或校验归属（防止跨用户访问）
        if not conversation_id:
            conversation_id = new_conversation_id()
            await self._memory.create_conversation(user_id, conversation_id)
        elif not await self._memory.belongs_to(user_id, conversation_id):
            yield _evt(EVT_ERROR, {"message": "会话不存在或无权访问", "code": "FORBIDDEN"})
            return

        yield _evt(EVT_THINKING, {"conversation_id": conversation_id})

        try:
            # 1. 落库用户消息
            await self._memory.append_message(
                user_id, conversation_id, _msg("user", query)
            )

            # 2. 构建上下文（系统提示 + 摘要 + 历史 + 本轮）
            history = await self._memory.get_messages(user_id, conversation_id)
            summary = await self._memory.get_summary(user_id, conversation_id)
            messages = self._build_messages(query, history, summary)

            # 3. 每次请求重置检索收集器（单进程内请求串行访问 sink 的窗口极小；
            #    生产多 worker 部署时 worker 进程隔离，天然安全）
            self._sink.clear()

            # 4. 流式运行 Agent，答案累积到 answer_parts
            answer_parts: list[str] = []
            async for event in self._run_agent(messages, answer_parts):
                yield event
            final_answer = "".join(answer_parts).strip()

            # 5. 持久化助手回复
            if final_answer:
                await self._memory.append_message(
                    user_id, conversation_id, _msg("assistant", final_answer)
                )

            yield _evt(
                EVT_DONE,
                {
                    "conversation_id": conversation_id,
                    "answer_length": len(final_answer or ""),
                },
            )

            # 6. 异步触发历史压缩（不阻塞响应结束）
            await maybe_compress_history(
                self._memory,
                self._summarizer,
                user_id,
                conversation_id,
                max_messages=self._settings.memory_max_messages,
                keep_recent=self._settings.memory_keep_recent,
            )

        except Exception as exc:  # noqa: BLE001 - 流内错误必须转事件
            logger.exception("对话处理失败 conv=%s", conversation_id)
            yield _evt(EVT_ERROR, {"message": f"服务内部错误：{exc}", "code": "INTERNAL"})

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        history: list[dict[str, Any]],
        summary: str | None,
    ) -> list:
        """组装 Agent 输入：系统提示（含摘要）+ 用户可感知历史 + 本轮问题。"""
        system_content = SYSTEM_PROMPT
        if summary:
            system_content += "\n\n" + SUMMARY_CONTEXT_TEMPLATE.format(summary=summary)

        messages: list = [SystemMessage(content=system_content)]
        # 历史中最后一条就是本轮 user 消息，故从倒数第二条开始回放
        for m in history[:-1]:
            if m.get("role") == "user":
                messages.append(HumanMessage(content=m.get("content", "")))
            elif m.get("role") == "assistant":
                messages.append(AIMessage(content=m.get("content", "")))
        messages.append(HumanMessage(content=query))
        return messages

    async def _run_agent(self, messages: list, answer_parts: list[str]):
        """运行 Agent 并产出事件流。

        使用 stream_mode=["updates", "messages"] 双通道：
          - messages：LLM 逐 token 增量，驱动前端打字机效果；
          - updates：agent / tools 节点状态变更，识别工具调用起止。

        完整回答文本累积到调用方传入的 answer_parts 列表中
        （异步生成器无法 return 值，故通过可变参数回传）。
        """
        try:
            stream = self._agent.astream(
                {"messages": messages},
                config={"recursion_limit": 25},
                stream_mode=["updates", "messages"],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Agent 启动失败: {exc}") from exc

        async for mode, chunk in stream:
            if mode == "messages":
                msg_chunk, _metadata = chunk
                text = _chunk_text(msg_chunk)
                if text:
                    answer_parts.append(text)
                    yield _evt(EVT_TOKEN, {"text": text})

            elif mode == "updates":
                for node_name, state in chunk.items():
                    if not isinstance(state, dict):
                        continue
                    for msg in state.get("messages", []):
                        # agent 节点：携带 tool_calls 的 AIMessage => 工具开始
                        if node_name == "agent" and isinstance(msg, AIMessage):
                            for tc in getattr(msg, "tool_calls", None) or []:
                                yield _evt(
                                    EVT_TOOL_START,
                                    {
                                        "tool": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                        "id": tc.get("id", ""),
                                    },
                                )
                        # tools 节点：ToolMessage => 工具结束
                        elif node_name == "tools" and isinstance(msg, ToolMessage):
                            content = _tool_text(msg.content)
                            yield _evt(
                                EVT_TOOL_END,
                                {
                                    "tool": msg.name or "",
                                    "result": content[:500],
                                },
                            )
                            # 检索工具执行完毕，推送来源事件
                            if (msg.name or "") == "document_search" and self._sink.get("sources"):
                                yield _evt(EVT_SOURCES, {"sources": self._sink["sources"]})


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def _evt(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "data": data}


def _msg(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _chunk_text(msg_chunk: Any) -> str:
    """从消息增量块中提取文本。兼容 str / list[dict] 两种 content 形态。"""
    content = getattr(msg_chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _tool_text(content: Any) -> str:
    """工具返回内容统一转文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)[:500]
    return str(content)
