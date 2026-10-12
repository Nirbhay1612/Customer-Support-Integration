"""
Database connection and session management.

Provides:
- SQLAlchemy engine configuration
- Session factory
- Database initialization
- Context-managed database sessions
- Database health checks
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def _create_engine() -> Engine:
    """Create and configure the SQLAlchemy database engine."""

    database_url = settings.DATABASE_URL

    if not isinstance(database_url, str) or not database_url.strip():
        raise ValueError("DATABASE_URL must be a non-empty string.")

    engine_kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
    }

    # SQLite requires this option when connections may be
    # accessed from different threads.
    if database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
        }

        # In-memory SQLite databases generally work best with
        # a single shared connection when used across sessions.
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            from sqlalchemy.pool import StaticPool

            engine_kwargs["poolclass"] = StaticPool

    try:
        return create_engine(database_url, **engine_kwargs)
    except Exception:
        # Avoid logging the database URL, which may contain credentials.
        logger.exception("Failed to create database engine.")
        raise


engine = _create_engine()


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Provide a transaction-managed database session.

    Commits when the operation succeeds, rolls back if an
    exception occurs, and always closes the session.

    Example:
        with get_db() as db:
            customer = db.query(Customer).first()
    """

    db = SessionLocal()

    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Database transaction failed.")
        raise
    finally:
        db.close()


def get_session() -> Session:
    """
    Create and return a database session.

    The caller is responsible for closing the session.

    Example:
        db = get_session()
        try:
            # Perform database operations.
            ...
        finally:
            db.close()
    """

    return SessionLocal()


def init_db() -> None:
    """
    Create tables registered in Base.metadata.

    Import ORM models before creating tables so SQLAlchemy
    has registered their table definitions.
    """

    try:
        # Import models to register their table definitions.
        from src.database import models  # noqa: F401

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception:
        logger.exception("Database initialization failed.")
        raise


def check_database_connection() -> bool:
    """
    Check whether the database is reachable.

    Returns:
        True if the health check succeeds; otherwise False.
    """

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning(
            "Database health check failed.",
            exc_info=True,
        )
        return False


def get_database_url() -> str:
    """
    Return the configured database URL.

    Warning:
        This returns the full URL, which may contain credentials.
        Avoid logging or exposing its value in API responses.
    """

    return settings.DATABASE_URL
