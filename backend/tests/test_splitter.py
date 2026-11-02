"""文本分块器测试。"""

import pytest

from app.rag.document_parser import ParsedPage
from app.rag.splitter import split_pages, split_text


class TestSplitText:
    def test_short_text_unchanged(self):
        assert split_text("你好世界", chunk_size=100) == ["你好世界"]

    def test_empty_text(self):
        assert split_text("", chunk_size=100) == []
        assert split_text("   ", chunk_size=100) == []

    def test_long_text_chunked(self):
        text = "这是一个测试句子。" * 100  # 900 字
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            # 含重叠时单块可能略超，但不应超过 chunk_size + overlap
            assert len(c) <= 200 + 20

    def test_overlap_present(self):
        text = "AAAA。BBBB。CCCC。DDDD。EEEE。FFFF。"
        chunks = split_text(text, chunk_size=10, chunk_overlap=3)
        assert len(chunks) >= 2

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            split_text("x" * 100, chunk_size=10, chunk_overlap=10)

    def test_hard_split_without_separator(self):
        # 无分隔符的超长串应被硬切
        text = "a" * 250
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2
        assert all(len(c) <= 110 for c in chunks)

    def test_newline_priority(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = split_text(text, chunk_size=15, chunk_overlap=0)
        # 按段落边界切分，不应在段落中间硬切
        assert all("第" in c or "段" in c for c in chunks)


class TestSplitPages:
    def test_page_number_propagated(self):
        pages = [
            ParsedPage(text="页一内容。" * 30, page=1),
            ParsedPage(text="页二内容。" * 30, page=2),
        ]
        chunks = split_pages(pages, chunk_size=50, chunk_overlap=5)
        assert chunks
        pages_seen = {p for _, p in chunks}
        assert pages_seen == {1, 2}

    def test_pageless_chunks(self):
        pages = [ParsedPage(text="无页码内容", page=None)]
        chunks = split_pages(pages, chunk_size=100, chunk_overlap=0)
        assert chunks == [("无页码内容", None)]
