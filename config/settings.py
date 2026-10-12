
"""
Centralized application configuration.

Loads environment variables, resolves project paths,
and validates application settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from dotenv import load_dotenv


# =========================================================
# Project Paths
# =========================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_FILE: Final[Path] = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)


# =========================================================
# Environment Variable Helpers
# =========================================================

def get_env(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str:
    """Read and normalize a string environment variable."""
    value = os.getenv(key)

    if value is None or not value.strip():
        value = default

    if value is None or not value.strip():
        if required:
            raise ValueError(
                f"Required environment variable '{key}' is missing."
            )
        return ""

    return value.strip()


def get_int(key: str, default: int) -> int:
    """Read an integer environment variable."""
    value = os.getenv(key)

    if value is None or not value.strip():
        return default

    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be an integer."
        ) from exc


def get_bool(key: str, default: bool = False) -> bool:
    """Read a strictly validated boolean environment variable."""
    value = os.getenv(key)

    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "on"}:
        return True

    if normalized in {"false", "0", "no", "off"}:
        return False

    raise ValueError(
        f"Environment variable '{key}' must be a boolean."
    )


def resolve_path(value: str) -> Path:
    """Resolve relative paths against the project root."""
    if not value or not value.strip():
        raise ValueError("Path cannot be empty.")

    path = Path(value.strip()).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def validate_optional_url(name: str, value: str) -> None:
    """Validate an optional HTTP or HTTPS URL."""
    if not value:
        return

    try:
        parsed = urlparse(value)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any(char.isspace() for char in value)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                f"{name} must be a valid HTTP or HTTPS URL "
                "without embedded credentials."
            )

        # Accessing .port validates its syntax and range.
        _ = parsed.port

    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid HTTP or HTTPS URL "
            "with a valid port and hostname."
        ) from exc


# =========================================================
# Application Settings
# =========================================================

APP_NAME = get_env("APP_NAME", "AI Customer Support")
APP_ENV = get_env("APP_ENV", "development").lower()
DEBUG = get_bool("DEBUG", False)


# =========================================================
# LLM Settings
# =========================================================

LLM_PROVIDER = get_env("LLM_PROVIDER", "groq").lower()

LLM_MODEL = get_env(
    "LLM_MODEL",
    "llama-3.3-70b-versatile",
)

GROQ_API_KEY = get_env("GROQ_API_KEY")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4o-mini")


# =========================================================
# RAG Settings
# =========================================================

KNOWLEDGE_BASE_PATH = resolve_path(
    get_env("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
)

VECTOR_STORE_PATH = resolve_path(
    get_env("VECTOR_STORE_PATH", "data/vector_store")
)

EMBEDDING_MODEL = get_env(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 150)
TOP_K = get_int("TOP_K", 4)


# =========================================================
# Database Settings
# =========================================================

DATABASE_URL = get_env(
    "DATABASE_URL",
    "sqlite:///./customer_support.db",
)


# =========================================================
# Customer API
# =========================================================

CUSTOMER_API_URL = get_env("CUSTOMER_API_URL")
CUSTOMER_API_KEY = get_env("CUSTOMER_API_KEY")


# =========================================================
# Order API
# =========================================================

ORDER_API_URL = get_env("ORDER_API_URL")
ORDER_API_KEY = get_env("ORDER_API_KEY")


# =========================================================
# Email Settings
# =========================================================

SMTP_HOST = get_env("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = get_int("SMTP_PORT", 587)
SMTP_USERNAME = get_env("SMTP_USERNAME")
SMTP_PASSWORD = get_env("SMTP_PASSWORD")


# =========================================================
# Ticket System
# =========================================================

TICKET_API_URL = get_env("TICKET_API_URL")
TICKET_API_KEY = get_env("TICKET_API_KEY")


# =========================================================
# Webhook and Application Security
# =========================================================

WEBHOOK_SECRET = get_env("WEBHOOK_SECRET")
SECRET_KEY = get_env("SECRET_KEY")


# =========================================================
# Logging
# =========================================================

LOG_LEVEL = get_env("LOG_LEVEL", "INFO").upper()


# =========================================================
# Configuration Validation
# =========================================================

def validate_settings() -> None:
    """Validate application settings before startup."""

    if not APP_NAME:
        raise ValueError("APP_NAME cannot be empty.")

    if APP_ENV not in {
        "development",
        "testing",
        "production",
    }:
        raise ValueError(
            "APP_ENV must be development, testing, or production."
        )

    if LLM_PROVIDER not in {"groq", "openai"}:
        raise ValueError(
            "LLM_PROVIDER must be either 'groq' or 'openai'."
        )

    if not LLM_MODEL:
        raise ValueError("LLM_MODEL cannot be empty.")

    if not OPENAI_MODEL:
        raise ValueError("OPENAI_MODEL cannot be empty.")

    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL cannot be empty.")

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative and "
            "smaller than CHUNK_SIZE."
        )

    if TOP_K <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    if not SMTP_HOST:
        raise ValueError("SMTP_HOST cannot be empty.")

    if not 1 <= SMTP_PORT <= 65535:
        raise ValueError(
            "SMTP_PORT must be between 1 and 65535."
        )

    allowed_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if LOG_LEVEL not in allowed_log_levels:
        raise ValueError(
            f"Invalid LOG_LEVEL: {LOG_LEVEL}. "
            f"Allowed values: {', '.join(sorted(allowed_log_levels))}."
        )

    validate_optional_url("CUSTOMER_API_URL", CUSTOMER_API_URL)
    validate_optional_url("ORDER_API_URL", ORDER_API_URL)
    validate_optional_url("TICKET_API_URL", TICKET_API_URL)

    if APP_ENV == "production":
        if DEBUG:
            raise ValueError(
                "DEBUG must be disabled in production."
            )

        if (
            len(SECRET_KEY) < 32
            or SECRET_KEY == "change-this-secret-in-production"
        ):
            raise ValueError(
                "SECRET_KEY must contain at least 32 characters "
                "in production."
            )

        if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required for production with Groq."
            )

        if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required for production with OpenAI."
            )


validate_settings()


# =========================================================
# Settings Namespace
# =========================================================

class Settings:
    """Centralized settings namespace."""

    BASE_DIR = BASE_DIR
    ENV_FILE = ENV_FILE

    APP_NAME = APP_NAME
    APP_ENV = APP_ENV
    DEBUG = DEBUG

    LLM_PROVIDER = LLM_PROVIDER
    LLM_MODEL = LLM_MODEL
    GROQ_API_KEY = GROQ_API_KEY
    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_MODEL = OPENAI_MODEL

    KNOWLEDGE_BASE_PATH = KNOWLEDGE_BASE_PATH
    VECTOR_STORE_PATH = VECTOR_STORE_PATH
    EMBEDDING_MODEL = EMBEDDING_MODEL
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    TOP_K = TOP_K

    DATABASE_URL = DATABASE_URL

    CUSTOMER_API_URL = CUSTOMER_API_URL
    CUSTOMER_API_KEY = CUSTOMER_API_KEY

    ORDER_API_URL = ORDER_API_URL
    ORDER_API_KEY = ORDER_API_KEY

    SMTP_HOST = SMTP_HOST
    SMTP_PORT = SMTP_PORT
    SMTP_USERNAME = SMTP_USERNAME
    SMTP_PASSWORD = SMTP_PASSWORD

    TICKET_API_URL = TICKET_API_URL
    TICKET_API_KEY = TICKET_API_KEY

    WEBHOOK_SECRET = WEBHOOK_SECRET
    SECRET_KEY = SECRET_KEY

    LOG_LEVEL = LOG_LEVEL


settings = Settings()
