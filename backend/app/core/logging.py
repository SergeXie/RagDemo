"""统一日志配置。"""

from __future__ import annotations

import logging
import sys


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)


def setup_logging(level: str = "INFO") -> None:
    """初始化根日志器，避免 uvicorn / libraries 重复输出。"""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # 清理已有 handler，防止重复打印
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)

    # 降噪：只保留重要库的 WARNING 及以上
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """按模块名获取 logger。"""
    return logging.getLogger(name)
