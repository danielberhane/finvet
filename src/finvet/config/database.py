"""Database connection and session management."""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from .settings import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
# Use NullPool for better compatibility with async/threading
engine = create_engine(
    settings.postgres_url,
    poolclass=NullPool,
    echo=False,  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for declarative models
Base = declarative_base()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            session.execute(...)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    Dependency for FastAPI endpoints.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""

    Base.metadata.create_all(bind=engine)

    # Migration: add data_sources column to audit_executions if missing
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'audit_executions' AND column_name = 'data_sources'"
        ))
        if not result.fetchone():
            conn.execute(text("ALTER TABLE audit_executions ADD COLUMN data_sources JSONB"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_data_sources "
                "ON audit_executions USING gin(data_sources)"
            ))
            conn.commit()
            logger.info("Migrated audit_executions: added data_sources column")


def check_connection():
    """Check if database connection is healthy."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
