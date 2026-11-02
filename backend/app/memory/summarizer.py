"""长对话摘要压缩算法。

当会话消息数超过阈值时，将较早的消息批量交给 LLM 生成摘要，
与既有摘要合并后写回，再裁剪原始历史 —— 在有限上下文窗口内
最大程度保留多轮对话的关键信息，避免上下文信息丢失。
"""

from __future__ import annotations

import asyncio

from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_SUMMARY_PROMPT = """\
你是一个对话历史压缩器。以下是用户与客服之间更早的对话记录（可能已有一份摘要）。

【已有摘要】
{existing_summary}

【较早的对话记录】
{history}

请将以上内容合并压缩成一段简洁的中文摘要，要求：
1. 保留用户的核心诉求、已确认的关键事实（如订单号、产品型号、退款金额等实体信息）；
2. 保留客服已给出的重要承诺或结论；
3. 遗忘寒暄、重复表述等低信息量内容；
4. 摘要长度不超过 {max_chars} 字，直接输出摘要正文，不要任何前缀说明。
"""


class Summarizer:
    """基于 LLM 的对话摘要生成器。llm 需为 langchain BaseChatModel（支持 ainvoke）。"""

    def __init__(self, llm, max_chars: int = 500) -> None:
        self._llm = llm
        self._max_chars = max_chars

    async def summarize(
        self,
        existing_summary: str | None,
        older_messages: list[dict],
    ) -> str:
        """将旧摘要 + 较早消息压缩为新摘要。

        LLM 失败时降级为「截断拼接」，保证主流程不因摘要失败而中断。
        """
        if not older_messages and not existing_summary:
            return ""

        history_text = "\n".join(
            f"[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in older_messages
        )
        prompt = _SUMMARY_PROMPT.format(
            existing_summary=existing_summary or "（无）",
            history=history_text or "（无）",
            max_chars=self._max_chars,
        )

        try:
            response = await self._llm.ainvoke([{"role": "user", "content": prompt}])
            summary = (response.content or "").strip()[: self._max_chars * 2]
            if not summary:
                raise LLMServiceError("摘要模型返回空内容")
            return summary
        except Exception as exc:  # noqa: BLE001 - 摘要失败必须降级而非抛出
            logger.warning("摘要生成失败，降级为截断拼接: %s", exc)
            return self._fallback_summary(existing_summary, older_messages)

    @staticmethod
    def _fallback_summary(existing_summary: str | None, older_messages: list[dict]) -> str:
        """无 LLM 时的降级策略：保留最近若干条旧消息的关键片段。"""
        parts: list[str] = [existing_summary] if existing_summary else []
        for msg in older_messages[-4:]:
            role = "用户" if msg.get("role") == "user" else "客服"
            content = (msg.get("content") or "").strip()
            if content:
                parts.append(f"{role}曾说：{content[:100]}")
        return "；".join(p for p in parts if p)


async def maybe_compress_history(
    memory,
    summarizer: Summarizer,
    user_id: str,
    conversation_id: str,
    *,
    max_messages: int,
    keep_recent: int,
) -> bool:
    """检查并执行一次摘要压缩。返回是否发生了压缩。

    算法：
      1. 消息数未超过 max_messages -> 不处理；
      2. 取出全部消息，计算需压缩的数量 to_compress = total - keep_recent；
      3. 用「旧摘要 + 待压缩消息」生成新摘要并写回；
      4. 裁剪原始历史，仅保留最近 keep_recent 条。

    并发安全：整个流程加会话级 asyncio.Lock，避免并发请求触发重复摘要。
    """
    lock = _get_lock(user_id, conversation_id)
    async with lock:
        messages = await memory.get_messages(user_id, conversation_id)
        total = len(messages)
        if total <= max_messages:
            return False

        to_compress = total - keep_recent
        older = messages[:to_compress]
        existing_summary = await memory.get_summary(user_id, conversation_id)

        new_summary = await summarizer.summarize(existing_summary, older)
        if new_summary:
            await memory.set_summary(user_id, conversation_id, new_summary)
        await memory.trim_messages(user_id, conversation_id, keep_recent)

        logger.info(
            "会话摘要压缩完成 conv=%s 压缩消息数=%d 保留=%d",
            conversation_id, to_compress, keep_recent,
        )
        return True


# 进程级会话锁表（生产多实例时可换分布式锁，单实例 asyncio 锁已足够）
_locks: dict[str, asyncio.Lock] = {}


def _get_lock(user_id: str, conversation_id: str) -> asyncio.Lock:
    key = f"{user_id}::{conversation_id}"
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock
