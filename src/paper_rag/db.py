from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from paper_rag.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def create_engine_from_settings(settings: Settings):
    return create_engine(settings.postgres_dsn, pool_pre_ping=True)


_engine = create_engine_from_settings(get_settings())
SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
