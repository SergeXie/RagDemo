"""知识库文档相关数据模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """文档上传并入库后的响应。"""

    doc_id: str
    doc_name: str
    file_type: str
    chunk_count: int
    uploaded_at: datetime


class DocumentInfo(BaseModel):
    """知识库中的文档元信息。"""

    doc_id: str
    doc_name: str
    file_type: str
    chunk_count: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """知识库文档列表。"""

    total: int
    documents: list[DocumentInfo]
