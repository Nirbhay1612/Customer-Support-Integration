"""
Application logging utilities.

Provides centralized logging with:
- Configurable log levels
- Consistent console and file formatting
- Rotating file logs
- Thread-safe, idempotent logger configuration
- Graceful fallback when file logging is unavailable
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Any

from config import settings


# =========================================================
# Constants
# =========================================================

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = Path(settings.BASE_DIR) / "logs"
LOG_FILE = LOG_DIR / "app.log"

MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

_CONFIGURED_ATTRIBUTE = "_customer_support_logger_configured"
_CONFIGURATION_LOCK = RLock()

VALID_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# =========================================================
# Log Level Helper
# =========================================================

def _get_log_level(level: str | None = None) -> int:
    """Return a validated numeric logging level."""

    configured_level = level if level is not None else settings.LOG_LEVEL
    level_name = (
        configured_level.strip().upper()
        if isinstance(configured_level, str) and configured_level.strip()
        else "INFO"
    )

    if level_name not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level_name}. "
            f"Allowed levels: {', '.join(VALID_LOG_LEVELS)}."
        )

    return VALID_LOG_LEVELS[level_name]


# =========================================================
# Formatter
# =========================================================

def _create_formatter() -> logging.Formatter:
    """Create the standard application log formatter."""

    return logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )


# =========================================================
# Logger Configuration
# =========================================================

def _configure_logger(
    logger: logging.Logger,
) -> logging.Logger:
    """Configure a logger once with console and file handlers."""

    with _CONFIGURATION_LOCK:
        if getattr(logger, _CONFIGURED_ATTRIBUTE, False):
            return logger

        level = _get_log_level()
        formatter = _create_formatter()

        logger.setLevel(level)
        logger.propagate = False

        # Console output remains available even when file
        # logging cannot be initialized.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=LOG_FILE,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except OSError:
            # File logging is optional. Keep console logging
            # operational if the filesystem is unavailable.
            warning_record = logging.LogRecord(
                name=logger.name,
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg=(
                    "File logging is unavailable; "
                    "console logging remains enabled."
                ),
                args=(),
                exc_info=None,
            )
            console_handler.handle(warning_record)

        setattr(logger, _CONFIGURED_ATTRIBUTE, True)
        return logger


# =========================================================
# Public Logger Factory
# =========================================================

def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger for an application module."""

    logger_name = (
        name.strip()
        if isinstance(name, str) and name.strip()
        else str(settings.APP_NAME or "ai_customer_support").strip()
        or "ai_customer_support"
    )

    return _configure_logger(logging.getLogger(logger_name))


# =========================================================
# Default Application Logger
# =========================================================

logger = get_logger(settings.APP_NAME)


# =========================================================
# Convenience Logging Functions
# =========================================================

def log_debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Write a DEBUG-level message."""

    logger.debug(message, *args, **kwargs)


def log_info(message: str, *args: Any, **kwargs: Any) -> None:
    """Write an INFO-level message."""

    logger.info(message, *args, **kwargs)


def log_warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Write a WARNING-level message."""

    logger.warning(message, *args, **kwargs)


def log_error(message: str, *args: Any, **kwargs: Any) -> None:
    """Write an ERROR-level message."""

    logger.error(message, *args, **kwargs)


def log_exception(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an exception with traceback information."""

    logger.exception(message, *args, **kwargs)
