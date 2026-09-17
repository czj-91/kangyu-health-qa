"""loguru 结构化日志配置"""
import sys
from loguru import logger
from config import LOG_LEVEL, LOG_FORMAT

logger.remove()

if LOG_FORMAT == "structured":
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
else:
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=fmt,
    colorize=LOG_FORMAT == "pretty",
)

# 文件日志
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format=fmt,
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

