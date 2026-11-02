"""文本分块器。

递归按分隔符优先级切分，保持语义边界，块间保留重叠窗口。
不依赖外部库，逻辑与 LangChain RecursiveCharacterTextSplitter 对齐。
"""

from __future__ import annotations

from app.rag.document_parser import ParsedPage


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: tuple[str, ...] = ("\n\n", "\n", "。", ". ", "；", "; ", "，", ", ", " ", ""),
) -> list[str]:
    """递归切分文本为不超过 chunk_size 的块，相邻块重叠 overlap 字符。"""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # 找到第一个能切开的分隔符
    for sep in separators:
        if sep == "":
            # 硬切：无可用分隔符时按长度切
            chunks = [
                text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)
            ]
            return [c for c in chunks if c.strip()]
        if sep in text:
            pieces = text.split(sep)
            merged: list[str] = []
            buf = ""
            for piece in pieces:
                candidate = buf + sep + piece if buf else piece
                if len(candidate) <= chunk_size:
                    buf = candidate
                else:
                    if buf:
                        merged.append(buf)
                    # 单个 piece 本身就超长：递归深层切分
                    if len(piece) > chunk_size:
                        sub = split_text(piece, chunk_size, chunk_overlap, separators)
                        merged.extend(sub)
                        buf = ""
                    else:
                        buf = piece
            if buf:
                merged.append(buf)
            # 施加重叠窗口
            return _add_overlap([m for m in merged if m.strip()], chunk_overlap)
    return [text]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        result.append(tail + cur)
    return result


def split_pages(
    pages: list[ParsedPage],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[tuple[str, int | None]]:
    """将解析页列表切块，返回 (块文本, 页码) 列表，页码信息随块传递。"""
    out: list[tuple[str, int | None]] = []
    for page in pages:
        for chunk in split_text(page.text, chunk_size, chunk_overlap):
            out.append((chunk, page.page))
    return out
