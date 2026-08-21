"""Tests for schema initialization on application startup.

init_db() creates the audit tables (Base.metadata.create_all) plus the
data_sources migration. It was never called at startup, so a fresh Postgres —
the exact case for a `docker compose up` clone — had no audit_events table and
the first verification's audit write failed. Startup now calls it, wrapped so a
Postgres-down boot logs and continues rather than crash-looping the container
(preserving the lazy-engine resilience the app already had).
"""

import asyncio
from unittest.mock import patch

import finvet.main as main


def test_startup_initializes_the_schema():
    with patch.object(main, "init_db") as init_db:
        asyncio.run(main.startup_event())
    init_db.assert_called_once()


def test_startup_survives_a_database_that_is_down():
    """A DB unreachable at boot must not crash the app — audit writes degrade
    on their own, and the schema is created on the next boot once DB is up."""
    with patch.object(main, "init_db", side_effect=OSError("connection refused")):
        asyncio.run(main.startup_event())  # must not raise
