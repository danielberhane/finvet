#!/usr/bin/env python3
"""Initialize PostgreSQL database with all tables."""

from src.finvet.config.database import Base, engine, check_connection

# Imported for the side effect of registering the tables on Base.metadata;
# create_all() below only sees models that have been imported.
from src.finvet.audit.models import AuditEvent, AuditExecution  # noqa: F401

def init_database():
    """Create all database tables."""
    print("Checking database connection...")
    if not check_connection():
        print("❌ Database connection failed!")
        print("Make sure PostgreSQL is running: docker-compose up -d")
        return False

    print("✅ Database connection successful")
    print("\nCreating database tables...")

    try:
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        print("\nTables created:")
        print("  - audit_events (audit trail)")
        print("  - audit_executions (execution summaries)")
        return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("FinVet Database Initialization")
    print("=" * 60)
    success = init_database()
    if success:
        print("\n" + "=" * 60)
        print("Database ready! Next steps:")
        print("  Start the API: uv run uvicorn src.finvet.main:app --reload")
        print("=" * 60)
    else:
        print("\n⚠️  Database initialization failed")
