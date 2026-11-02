"""业务异常定义与统一错误码。

所有业务异常继承 AppError，由 FastAPI 全局异常处理器转为
结构化 JSON 响应：{"code": ..., "message": ..., "detail": ...}
"""

from __future__ import annotations


class AppError(Exception):
    """业务异常基类。"""

    code: str = "APP_ERROR"
    status_code: int = 500

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigError(AppError):
    """配置缺失或非法。"""

    code = "CONFIG_ERROR"
    status_code = 500


class LLMServiceError(AppError):
    """LLM 调用失败。"""

    code = "LLM_SERVICE_ERROR"
    status_code = 502


class DocumentProcessingError(AppError):
    """文档解析 / 入库失败。"""

    code = "DOCUMENT_PROCESSING_ERROR"
    status_code = 422


class UnsupportedFileTypeError(DocumentProcessingError):
    """不支持的文件类型。"""

    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 415


class FileTooLargeError(AppError):
    """上传文件超出大小限制。"""

    code = "FILE_TOO_LARGE"
    status_code = 413


class VectorStoreError(AppError):
    """向量库操作失败。"""

    code = "VECTOR_STORE_ERROR"
    status_code = 500


class ConversationNotFoundError(AppError):
    """会话不存在。"""

    code = "CONVERSATION_NOT_FOUND"
    status_code = 404


class DocumentNotFoundError(AppError):
    """知识库文档不存在。"""

    code = "DOCUMENT_NOT_FOUND"
    status_code = 404


class ValidationError(AppError):
    """入参校验失败。"""

    code = "VALIDATION_ERROR"
    status_code = 422
