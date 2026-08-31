"""Engine, session handling and bootstrap for the phone database."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import config

from .models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            pool_pre_ping=True,
            future=True,
            echo=False,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a read-oriented session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def create_database_if_missing() -> None:
    """Issue `CREATE DATABASE` when the target does not exist yet.

    SQLite needs nothing (the file is created on connect). MySQL and PostgreSQL
    both require connecting to the server rather than the target database, which
    is why `build_database_url(include_database=False)` exists.
    """
    if config.DB_BACKEND == "sqlite":
        return

    server_url = config.build_database_url(include_database=False)
    server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT", future=True)

    try:
        with server_engine.connect() as conn:
            if config.DB_BACKEND == "mysql":
                conn.execute(
                    text(
                        f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
                logger.info("MySQL database '%s' is ready", config.DB_NAME)
            else:
                exists = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": config.DB_NAME},
                ).scalar()
                if not exists:
                    conn.execute(text(f'CREATE DATABASE "{config.DB_NAME}"'))
                logger.info("PostgreSQL database '%s' is ready", config.DB_NAME)
    finally:
        server_engine.dispose()


def init_db(drop_existing: bool = False) -> None:
    """Create the database and all tables. Safe to run repeatedly."""
    create_database_if_missing()
    engine = get_engine()
    if drop_existing:
        logger.warning("Dropping existing tables")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Schema ready at %s", config.DATABASE_URL.split("@")[-1])


def healthcheck() -> tuple[bool, str]:
    """Return (ok, message) describing database reachability."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as exc:  # surfaced to the API's /health endpoint
        return False, str(exc)
