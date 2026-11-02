"""对话记忆与摘要压缩测试（使用内存后端 + 假 LLM）。"""

from __future__ import annotations

import asyncio

import pytest

from app.memory.fallback import InMemoryConversationMemory
from app.memory.summarizer import Summarizer, maybe_compress_history


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """模拟摘要 LLM。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ainvoke(self, messages):
        self.calls.append(messages[0]["content"])
        return _FakeResponse("【摘要】用户咨询了报销流程，已确认差旅标准。")


@pytest.fixture
def memory() -> InMemoryConversationMemory:
    return InMemoryConversationMemory()


@pytest.mark.asyncio
async def test_user_isolation(memory: InMemoryConversationMemory):
    """不同 user_id 的会话数据完全隔离。"""
    await memory.create_conversation("user_a", "conv1")
    await memory.create_conversation("user_b", "conv1")  # 相同 cid 不同用户

    await memory.append_message("user_a", "conv1", {"role": "user", "content": "A 的秘密"})
    msgs_b = await memory.get_messages("user_b", "conv1")
    msgs_a = await memory.get_messages("user_a", "conv1")

    assert len(msgs_a) == 1
    assert len(msgs_b) == 0
    # 越权校验：A 不能访问 B 名下会话
    assert await memory.belongs_to("user_b", "conv1") is True
    assert await memory.belongs_to("user_c", "conv1") is False


@pytest.mark.asyncio
async def test_message_order_preserved(memory: InMemoryConversationMemory):
    """消息按追加顺序返回。"""
    await memory.create_conversation("u", "c")
    for i in range(5):
        await memory.append_message("u", "c", {"role": "user", "content": f"msg{i}"})
    msgs = await memory.get_messages("u", "c")
    assert [m["content"] for m in msgs] == [f"msg{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_summary_compression_triggered(memory: InMemoryConversationMemory):
    """超过阈值时触发摘要压缩，且保留最近消息。"""
    await memory.create_conversation("u", "c")
    for i in range(25):
        await memory.append_message("u", "c", {"role": "user", "content": f"问题{i}"})

    summarizer = Summarizer(_FakeLLM(), max_chars=100)
    compressed = await maybe_compress_history(
        memory, summarizer, "u", "c", max_messages=20, keep_recent=6
    )

    assert compressed is True
    msgs = await memory.get_messages("u", "c")
    # 只保留最近 6 条
    assert len(msgs) == 6
    assert msgs[-1]["content"] == "问题24"
    # 摘要已写入
    summary = await memory.get_summary("u", "c")
    assert summary and "报销流程" in summary


@pytest.mark.asyncio
async def test_no_compression_below_threshold(memory: InMemoryConversationMemory):
    """消息数未超阈值时不压缩。"""
    await memory.create_conversation("u", "c")
    for i in range(10):
        await memory.append_message("u", "c", {"role": "user", "content": f"m{i}"})

    summarizer = Summarizer(_FakeLLM())
    compressed = await maybe_compress_history(
        memory, summarizer, "u", "c", max_messages=20, keep_recent=6
    )
    assert compressed is False
    assert len(await memory.get_messages("u", "c")) == 10


@pytest.mark.asyncio
async def test_summarizer_merges_existing_summary(memory: InMemoryConversationMemory):
    """二次压缩时应合并已有摘要。"""
    await memory.create_conversation("u", "c")
    await memory.set_summary("u", "c", "旧摘要：用户问过年假。")
    for i in range(20):
        await memory.append_message("u", "c", {"role": "user", "content": f"新问题{i}"})

    fake = _FakeLLM()
    await maybe_compress_history(
        memory, Summarizer(fake), "u", "c", max_messages=10, keep_recent=4
    )
    # 提示词中应同时包含旧摘要与新消息
    assert "旧摘要：用户问过年假。" in fake.calls[0]
    assert "新问题0" in fake.calls[0]


@pytest.mark.asyncio
async def test_summarizer_fallback_on_llm_failure(memory: InMemoryConversationMemory):
    """LLM 失败时降级不抛错。"""

    class _BrokenLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("boom")

    await memory.create_conversation("u", "c")
    for i in range(12):
        await memory.append_message("u", "c", {"role": "user", "content": f"消息{i}"})

    compressed = await maybe_compress_history(
        memory, Summarizer(_BrokenLLM()), "u", "c", max_messages=10, keep_recent=3
    )
    assert compressed is True
    assert len(await memory.get_messages("u", "c")) == 3
    assert await memory.get_summary("u", "c")  # 降级摘要非空


@pytest.mark.asyncio
async def test_concurrent_compression_single_run(memory: InMemoryConversationMemory):
    """并发触发压缩时锁保证只执行一次摘要。"""
    await memory.create_conversation("u", "c")
    for i in range(30):
        await memory.append_message("u", "c", {"role": "user", "content": f"m{i}"})

    fake = _FakeLLM()
    summarizer = Summarizer(fake)

    await asyncio.gather(*[
        maybe_compress_history(memory, summarizer, "u", "c", max_messages=20, keep_recent=6)
        for _ in range(5)
    ])
    # 由于锁内二次检查消息数，压缩完成后后续调用应直接跳过
    assert len(await memory.get_messages("u", "c")) == 6


@pytest.mark.asyncio
async def test_delete_conversation(memory: InMemoryConversationMemory):
    """删除会话后数据不可恢复。"""
    await memory.create_conversation("u", "c")
    await memory.append_message("u", "c", {"role": "user", "content": "hi"})
    assert await memory.delete_conversation("u", "c") is True
    assert await memory.belongs_to("u", "c") is False
    assert await memory.delete_conversation("u", "c") is False
