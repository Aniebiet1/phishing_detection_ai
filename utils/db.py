"""Database configuration and session helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = BASE_DIR / "data" / "app_state.db"


def _resolve_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

    # Render may provide postgres:// URLs; SQLAlchemy expects postgresql+psycopg://
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+psycopg://", 1)

    if raw.startswith("postgresql://") and "+" not in raw.split("://", 1)[0]:
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)

    return raw


DATABASE_URL = _resolve_database_url()

_ENGINE_KWARGS: dict = {
    "pool_pre_ping": True,
    "future": True,
}
if DATABASE_URL.startswith("sqlite"):
    _ENGINE_KWARGS["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_ENGINE_KWARGS)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables for local bootstrap; production should use Alembic migrations."""

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
