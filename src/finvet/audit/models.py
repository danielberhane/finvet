"""SQLAlchemy models for audit trail."""

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
from ..config.database import Base


class AuditEvent(Base):
    """Audit event model (immutable append-only)."""

    __tablename__ = "audit_events"

    event_id = Column(String(50), primary_key=True)
    request_id = Column(String(50), nullable=False, index=True)
    parent_event_id = Column(String(50), ForeignKey("audit_events.event_id"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(String(50), nullable=False, index=True)
    agent = Column(String(50), nullable=True)
    data = Column(JSONB, nullable=False)
    created_at = Column(String(50), nullable=False, default=lambda: datetime.utcnow().isoformat())

    def __repr__(self):
        return f"<AuditEvent {self.event_id} type={self.event_type} request={self.request_id}>"


class AuditExecution(Base):
    """Audit execution summary model."""

    __tablename__ = "audit_executions"

    execution_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(String(50), nullable=False)
    claim_text = Column(Text, nullable=False)
    claim_hash = Column(String(64), nullable=False, index=True)
    verdict = Column(String(50), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    agents_run = Column(JSONB, nullable=True)
    total_events = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    execution_hash = Column(String(64), nullable=False)
    full_trace = Column(JSONB, nullable=False)
    data_sources = Column(JSONB, nullable=True)  # {"xbrl": {...}, "rag": {...}, "a2a": {...}}
    created_at = Column(String(50), nullable=False, default=lambda: datetime.utcnow().isoformat())

    def __repr__(self):
        return f"<AuditExecution {self.request_id} verdict={self.verdict} confidence={self.confidence}>"


# Create indexes
Index("idx_request_id", AuditEvent.request_id)
Index("idx_event_type", AuditEvent.event_type)
Index("idx_timestamp", AuditEvent.timestamp)
Index("idx_claim_hash", AuditExecution.claim_hash)
Index("idx_verdict", AuditExecution.verdict)
Index("idx_data_sources", AuditExecution.data_sources, postgresql_using="gin")
