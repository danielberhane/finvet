"""Tests for AuditDatabase.list_executions and GET /audit endpoint."""

from unittest.mock import MagicMock, patch
import pytest


class TestListExecutions:
    """Tests for AuditDatabase.list_executions."""

    def _make_execution(self, request_id="req_abc", verdict="SUPPORTS",
                        agents_run=None, timestamp="2026-01-15T10:00:00"):
        """Return a mock AuditExecution ORM object."""
        e = MagicMock()
        e.execution_id = 1
        e.request_id = request_id
        e.timestamp = timestamp
        e.claim_text = "Apple revenue was $391B in FY2024"
        e.claim_hash = "abc123"
        e.verdict = verdict
        e.confidence = 0.92
        e.agents_run = agents_run or ["sec"]
        e.total_events = 8
        e.execution_time_ms = 4200
        e.execution_hash = "def456"
        e.full_trace = {}
        e.data_sources = {"xbrl": {}}
        e.created_at = timestamp
        return e

    @patch("src.finvet.audit.database.get_db_session")
    def test_list_executions_no_filters(self, mock_session_ctx):
        """list_executions with no filters returns all executions."""
        from src.finvet.audit.database import AuditDatabase

        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [self._make_execution()]

        db = AuditDatabase()
        results = db.list_executions()

        assert len(results) == 1
        assert results[0]["verdict"] == "SUPPORTS"
        assert results[0]["request_id"] == "req_abc"

    @patch("src.finvet.audit.database.get_db_session")
    def test_list_executions_verdict_filter(self, mock_session_ctx):
        """list_executions with verdict filter passes it to the query."""
        from src.finvet.audit.database import AuditDatabase

        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        db = AuditDatabase()
        results = db.list_executions(verdict="REFUTES")

        assert results == []
        mock_query.filter.assert_called()

    @patch("src.finvet.audit.database.get_db_session")
    def test_list_executions_returns_empty_on_exception(self, mock_session_ctx):
        """list_executions returns [] when the database raises."""
        from src.finvet.audit.database import AuditDatabase

        mock_session_ctx.return_value.__enter__.side_effect = Exception("db down")
        db = AuditDatabase()
        results = db.list_executions()
        assert results == []


class TestAuditLoggerListExecutions:
    """AuditLogger.list_executions delegates to AuditDatabase."""

    def test_delegates_to_database(self):
        """list_executions passes all args through to db.list_executions."""
        from src.finvet.audit.logger import AuditLogger

        logger_inst = AuditLogger()
        logger_inst.db = MagicMock()
        logger_inst.db.list_executions.return_value = [{"request_id": "req_x"}]

        result = logger_inst.list_executions(
            verdict="SUPPORTS",
            agent="sec",
            date_from="2026-01-01T00:00:00",
            date_to="2026-12-31T23:59:59",
            data_source="xbrl",
            limit=25,
            offset=10,
        )

        logger_inst.db.list_executions.assert_called_once_with(
            verdict="SUPPORTS",
            agent="sec",
            date_from="2026-01-01T00:00:00",
            date_to="2026-12-31T23:59:59",
            data_source="xbrl",
            limit=25,
            offset=10,
        )
        assert result == [{"request_id": "req_x"}]


class TestAuditListEndpoint:
    """Tests for GET /audit endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.finvet.api.routes.audit import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @patch("src.finvet.api.routes.audit.get_audit_logger")
    def test_returns_items_and_total(self, mock_get_logger, client):
        """GET /audit returns items list and total count."""
        mock_logger = MagicMock()
        mock_logger.list_executions.return_value = [
            {"request_id": "req_1", "verdict": "SUPPORTS", "confidence": 0.9,
             "claim_text": "Apple revenue $391B", "timestamp": "2026-01-15T10:00:00",
             "agents_run": ["sec"], "execution_time_ms": 4200,
             "data_sources": {}, "total_events": 5, "execution_hash": "abc"}
        ]
        mock_get_logger.return_value = mock_logger

        resp = client.get("/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert body["total"] == 1
        assert len(body["items"]) == 1

    @patch("src.finvet.api.routes.audit.get_audit_logger")
    def test_passes_verdict_filter(self, mock_get_logger, client):
        """GET /audit?verdict=REFUTES passes verdict to list_executions."""
        mock_logger = MagicMock()
        mock_logger.list_executions.return_value = []
        mock_get_logger.return_value = mock_logger

        client.get("/audit?verdict=REFUTES")
        mock_logger.list_executions.assert_called_once_with(
            verdict="REFUTES", agent=None, date_from=None,
            date_to=None, data_source=None, limit=50, offset=0,
        )

    @patch("src.finvet.api.routes.audit.get_audit_logger")
    def test_passes_all_filters(self, mock_get_logger, client):
        """GET /audit passes all query params to list_executions."""
        mock_logger = MagicMock()
        mock_logger.list_executions.return_value = []
        mock_get_logger.return_value = mock_logger

        client.get(
            "/audit?verdict=SUPPORTS&agent=sec"
            "&date_from=2026-01-01T00%3A00%3A00&date_to=2026-12-31T23%3A59%3A59"
            "&data_source=xbrl&limit=25&offset=10"
        )
        mock_logger.list_executions.assert_called_once_with(
            verdict="SUPPORTS", agent="sec",
            date_from="2026-01-01T00:00:00", date_to="2026-12-31T23:59:59",
            data_source="xbrl", limit=25, offset=10,
        )
