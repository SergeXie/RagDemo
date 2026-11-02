"""多格式文档解析器。

支持：PDF / DOCX / TXT / Markdown / CSV / Excel。
统一输出为分页文本段落列表，供下游分块器处理。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """一页 / 一段可定位的文本。"""

    text: str
    page: int | None = None  # 仅 PDF 有真实页码


@dataclass
class ParsedDocument:
    """解析结果。"""

    pages: list[ParsedPage] = field(default_factory=list)
    file_type: str = ""


# 扩展名 -> 内部类型
SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
    ".md": "markdown",
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
}


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """解析上传文件字节流。

    Raises:
        UnsupportedFileTypeError: 扩展名不支持。
    """
    ext = Path(filename).suffix.lower()
    file_type = SUPPORTED_TYPES.get(ext)
    if file_type is None:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型：{ext or '(无扩展名)'}",
            detail=f"仅支持 {', '.join(sorted(SUPPORTED_TYPES))}",
        )

    parsers = {
        "pdf": _parse_pdf,
        "docx": _parse_docx,
        "text": _parse_text,
        "markdown": _parse_text,
        "csv": _parse_csv,
        "excel": _parse_excel,
    }
    pages = parsers[file_type](data)
    # 过滤空白段
    pages = [p for p in pages if p.text and p.text.strip()]
    if not pages:
        raise ValueError(f"文档 {filename} 未解析出任何文本内容")
    logger.info("解析文档 %s 类型=%s 段落数=%d", filename, file_type, len(pages))
    return ParsedDocument(pages=pages, file_type=file_type)


# ---------------- 各格式实现 ----------------

def _parse_pdf(data: bytes) -> list[ParsedPage]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[ParsedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(ParsedPage(text=text, page=i))
    return pages


def _parse_docx(data: bytes) -> list[ParsedPage]:
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(data))
    # 按段落切分，每 5 段聚合为一块，避免碎片化
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    pages: list[ParsedPage] = []
    for i in range(0, len(paragraphs), 5):
        pages.append(ParsedPage(text="\n".join(paragraphs[i : i + 5])))
    # 表格内容也纳入
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                pages.append(ParsedPage(text=" | ".join(cells)))
    return pages


def _parse_text(data: bytes) -> list[ParsedPage]:
    """TXT / Markdown：优先 UTF-8，失败回退 GBK（中文 Windows 场景）。"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("gbk", errors="replace")
    # 按空行分段
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [text] if text.strip() else []
    return [ParsedPage(text=b) for b in blocks]


def _parse_csv(data: bytes) -> list[ParsedPage]:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data))
    return _dataframe_to_pages(df)


def _parse_excel(data: bytes) -> list[ParsedPage]:
    import pandas as pd

    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)  # 全部 sheet
    pages: list[ParsedPage] = []
    for sheet_name, df in sheets.items():
        sheet_pages = _dataframe_to_pages(df)
        # 为每个 sheet 的内容加上表名前缀，保留结构信息
        for p in sheet_pages:
            p.text = f"[{sheet_name}] {p.text}"
        pages.extend(sheet_pages)
    return pages


def _dataframe_to_pages(df) -> list[ParsedPage]:
    """将 DataFrame 转为「表头: 值」的可读文本行，每行一页。"""
    columns = [str(c) for c in df.columns]
    rows: list[ParsedPage] = []
    for _, row in df.iterrows():
        pairs = [f"{col}={val}" for col, val in zip(columns, row) if str(val) != "nan"]
        if pairs:
            rows.append(ParsedPage(text="，".join(pairs)))
    return rows
