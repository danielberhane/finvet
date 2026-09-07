"""Tests for the SEC MCP call timeout.

The server processes each requested XBRL concept independently against the
filing, with no caching, so latency scales linearly with concept count:
~2.0s for one, 7.8s for the five cashflow concepts, 15-17s for the twelve income
concepts. The old 15s default sat exactly on that last figure, so income
statements timed out systematically rather than occasionally — 23 of 117 cases
in a full retrieval sweep produced nothing at all, 19 of them income.
"""

from unittest.mock import patch

from finvet.config.settings import settings
from finvet.mcp.mcp_client import MCPClient
from finvet.mcp.sec_edgar import CONCEPTS_BY_TYPE, SECEdgarClient


class TestTimeoutDefaults:

    def test_default_comes_from_settings(self):
        assert MCPClient("http://localhost:9870").timeout == settings.sec_mcp_timeout_s

    def test_explicit_timeout_wins(self):
        assert MCPClient("http://localhost:9870", timeout=5).timeout == 5

    def test_sec_client_uses_the_configured_timeout(self):
        assert SECEdgarClient()._mcp.timeout == settings.sec_mcp_timeout_s


class TestTimeoutIsSufficientForTheLargestRequest:

    def test_covers_the_slowest_observed_statement(self):
        """Income is the widest request at twelve concepts, measured at 15-17s.
        The budget must clear that with room for a slow filing, or the widest
        statement is the one that reliably fails."""
        widest = max(len(c) for c in CONCEPTS_BY_TYPE.values())
        seconds_per_concept = 1.25  # measured against the live MCP server
        assert settings.sec_mcp_timeout_s >= widest * seconds_per_concept * 2

    def test_income_is_the_widest_request(self):
        """Pins the assumption behind the budget above."""
        assert len(CONCEPTS_BY_TYPE["income"]) == max(
            len(c) for c in CONCEPTS_BY_TYPE.values()
        )


class TestTimeoutIsApplied:

    def test_httpx_client_is_built_with_the_timeout(self):
        with patch("finvet.mcp.mcp_client.httpx.Client") as fake:
            MCPClient("http://localhost:9870", timeout=42)
        assert fake.call_args.kwargs["timeout"].connect == 42

    def test_default_timeout_reaches_httpx(self):
        """Every production caller is `MCPClient(base_url)` with no timeout.

        The constructor once handed its raw argument -- None -- to httpx, which
        httpx reads as "wait forever": sec_mcp_timeout_s was dead code, the
        TimeoutException handler could never fire, and a stalled server hung a
        request for hours. The explicit-timeout test above passed throughout,
        because it never exercised the default path.
        """
        with patch("finvet.mcp.mcp_client.httpx.Client") as fake:
            MCPClient("http://localhost:9870")
        assert fake.call_args.kwargs["timeout"].connect == settings.sec_mcp_timeout_s
