"""知识库文档接口：上传入库、列表查询、删除。"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_app_settings, get_pipeline, get_vector_store
from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundError, FileTooLargeError, ValidationError
from app.rag.pipeline import IngestionPipeline
from app.rag.vector_store import FaissVectorStore
from app.schemas.document import DocumentInfo, DocumentListResponse, DocumentUploadResponse

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="PDF/DOCX/TXT/MD/CSV/XLSX 文件"),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
):
    """上传文档并自动解析、分块、向量化后入库。"""
    if not file.filename:
        raise ValidationError("缺少文件名")

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"文件超过 {settings.max_upload_mb}MB 上限",
            detail=f"实际大小 {len(data) / 1024 / 1024:.1f}MB",
        )
    if not data:
        raise ValidationError("上传文件为空")

    doc_id, chunk_count = await pipeline.ingest(file.filename, data)

    # 从向量库元数据取准确入库时间
    docs = {d.doc_id: d for d in pipeline.store.list_documents()}
    meta = docs.get(doc_id)
    return DocumentUploadResponse(
        doc_id=doc_id,
        doc_name=file.filename,
        file_type=meta.file_type if meta else "",
        chunk_count=chunk_count,
        uploaded_at=(
            datetime.fromisoformat(meta.uploaded_at)
            if meta
            else datetime.now(timezone.utc)
        ),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    store: FaissVectorStore = Depends(get_vector_store),
):
    """列出知识库中的全部文档。"""
    docs = store.list_documents()
    return DocumentListResponse(
        total=len(docs),
        documents=[
            DocumentInfo(
                doc_id=d.doc_id,
                doc_name=d.doc_name,
                file_type=d.file_type,
                chunk_count=d.chunk_count,
                uploaded_at=datetime.fromisoformat(d.uploaded_at),
            )
            for d in docs
        ],
    )


@router.get("/stats")
async def stats(store: FaissVectorStore = Depends(get_vector_store)):
    """向量库统计信息（运维监控用）。"""
    return store.stats()


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    """从知识库删除文档（含全部向量块）。"""
    try:
        removed = await pipeline.delete(doc_id)
    except DocumentNotFoundError:
        raise DocumentNotFoundError(f"文档不存在: {doc_id}")
    return {"doc_id": doc_id, "removed_chunks": removed}
