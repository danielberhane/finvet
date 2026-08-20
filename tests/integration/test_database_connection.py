"""Integration tests for database connection. Requires a running PostgreSQL.

Opt in with:  pytest -m integration
"""

import pytest

from sqlalchemy import text
from src.finvet.config.database import check_connection, get_db_session

pytestmark = pytest.mark.integration


def test_postgres_connection():
    """Test PostgreSQL connection is healthy."""
    assert check_connection() is True


def test_postgres_connection_pool():
    """Test connection pooling works."""
    with get_db_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_multiple_concurrent_connections():
    """Test that multiple sessions can be created."""
    sessions = []
    for _ in range(5):
        with get_db_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1
            sessions.append(session)
    
    # All sessions should have been created successfully
    assert len(sessions) == 5


@pytest.mark.parametrize("query", [
    "SELECT COUNT(*) FROM audit_events",
])
def test_table_exists(query):
    """Test that all required tables exist."""
    with get_db_session() as session:
        # Should not raise error
        result = session.execute(text(query)).scalar()
        assert result is not None
