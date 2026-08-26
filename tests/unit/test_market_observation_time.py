"""A quote's observation time comes from the source, or it does not exist.

Stop condition 7: a market observation may not carry an observation time the
source did not supply.

`FinnhubClient.get_quote` did exactly that. `datetime.fromtimestamp(data.get(
"t", time.time()))` substituted the API server's wall clock whenever Finnhub
omitted its quote timestamp, and converted a UTC epoch in the server's local
timezone. The result was persisted as `latest_trading_day` and read downstream
as provenance — a number that looks source-provided, is not, and moves
depending on where the process happens to run.

Mock mode stamped `today` the same way, so demo data was indistinguishable from
a real observation.

The other half is the trust boundary. `_MARKET_FIELD_FOR_METRIC` mixes fields
from two different producers: `get_stock_quote` (price, open, high, volume),
which has an observation time, and `get_company_overview` (market cap, P/E,
dividend yield, 52-week range), which has none. The resolver keyed on payload
shape, not on which tool answered, so a market-cap figure with no observation
time could still become the number a verdict rests on.
"""

import pytest


class TestTheProducerSuppliesTheTimeOrNothing:
    """Driven through the real client. The substitution lives in `get_quote`,
    so testing a wrapper would pass whether or not it was fixed."""

    def _quote(self, monkeypatch, payload):
        from finvet.mcp.finnhub import FinnhubClient

        client = FinnhubClient(api_key="x", mock_mode=False)
        monkeypatch.setattr(client, "_request",
                            lambda endpoint, params: payload)
        return client.get_quote("AAPL")

    @pytest.mark.parametrize("payload,expected", [
        ({"c": 150.0, "pc": 148.0, "t": 1756000000}, "2025-08-24"),  # UTC
        ({"c": 150.0, "pc": 148.0}, None),             # source omitted it
        ({"c": 150.0, "pc": 148.0, "t": 0}, None),     # malformed
        ({"c": 150.0, "pc": 148.0, "t": -1}, None),    # malformed
        ({"c": 150.0, "pc": 148.0, "t": True}, None),  # bool is an int here
        ({"c": 150.0, "pc": 148.0, "t": "yesterday"}, None),
    ], ids=["utc", "omitted", "zero", "negative", "bool", "string"])
    def test_observation_time_is_source_supplied_or_absent(self, monkeypatch,
                                                           payload, expected):
        assert self._quote(monkeypatch, payload).latest_trading_day == expected

    def test_the_timestamp_is_read_in_utc_not_local_time(self, monkeypatch):
        """`t` is a UTC epoch. A local conversion attributes a quote near
        midnight to whichever day the server happens to be in, and the same
        quote then carries different provenance in two deployments."""
        from datetime import datetime, timezone

        epoch = 1756000000
        expected = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")

        observed = self._quote(
            monkeypatch, {"c": 1.0, "pc": 1.0, "t": epoch}).latest_trading_day

        assert observed == expected

    @pytest.mark.parametrize("tz", ["Pacific/Kiritimati", "UTC",
                                    "America/Los_Angeles"])
    def test_the_day_does_not_move_with_the_server_timezone(self, monkeypatch,
                                                            tz):
        """The guarantee, made independent of where the test runs.

        1755993720 is 2025-08-24T00:02:00Z -- two minutes past midnight UTC.
        Converted locally it lands on the 24th in UTC, the 24th in Kiritimati
        (+14) and the *23rd* in Los Angeles (-7). Only a UTC conversion gives
        the same answer everywhere, and asserting a literal date on a
        developer machine that happens to sit west of UTC would have hidden
        that.
        """
        import time

        monkeypatch.setenv("TZ", tz)
        time.tzset()
        try:
            observed = self._quote(
                monkeypatch,
                {"c": 1.0, "pc": 1.0, "t": 1755993720}).latest_trading_day
        finally:
            monkeypatch.delenv("TZ", raising=False)
            time.tzset()

        assert observed == "2025-08-24", (
            f"the trading day moved with the server timezone ({tz}), so the "
            f"same quote carries different provenance in two deployments")

    def test_a_quote_still_returns_its_price_without_a_timestamp(self):
        """Losing the observation time costs the comparison, not the quote:
        the value is still shown, it just cannot be trusted for a verdict."""
        from unittest.mock import patch

        from finvet.mcp.finnhub import FinnhubClient

        client = FinnhubClient(api_key="x", mock_mode=False)
        with patch.object(client, "_request",
                          lambda endpoint, params: {"c": 150.0, "pc": 148.0}):
            quote = client.get_quote("AAPL")

        assert quote.price == 150.0
        assert quote.latest_trading_day is None


class TestMockDataIsNotAnObservation:

    def test_mock_mode_never_supplies_an_observation_time(self):
        from finvet.mcp.finnhub import FinnhubClient

        assert FinnhubClient(mock_mode=True).get_quote("AAPL").latest_trading_day is None

    def test_mock_mode_says_it_is_mock(self):
        from finvet.mcp.finnhub import FinnhubClient

        assert FinnhubClient(mock_mode=True).get_quote("AAPL").source_mode == "mock"

    def test_a_real_quote_says_it_is_live(self, monkeypatch):
        from finvet.mcp.finnhub import FinnhubClient

        client = FinnhubClient(api_key="x", mock_mode=False)
        monkeypatch.setattr(
            client, "_request",
            lambda endpoint, params: {"c": 1.0, "pc": 1.0, "t": 1756000000})

        assert client.get_quote("AAPL").source_mode == "live"


def _claim(metric="closing_price", value=150.0):
    from finvet.models.claim import ParsedClaim

    return ParsedClaim(claim_type="market", ticker="AAPL", metric=metric,
                       operator="eq", value=value)


def _record(tool, payload):
    from finvet.models.evidence import tool_record_from_result

    return tool_record_from_result(tool, {"ticker": "AAPL"}, payload)


class TestTheBoundaryRequiresASourceTime:
    """Validated again here, even though the producer validates it.

    A trust boundary that assumes its producer is correct is not a boundary.
    """

    def _resolve(self, tool, payload, metric="closing_price"):
        from finvet.models.evidence import resolve_trusted_observation

        return resolve_trusted_observation(_claim(metric), [_record(tool, payload)])

    QUOTE = {"success": True, "symbol": "AAPL", "price": 150.0,
             "latest_trading_day": "2026-08-24"}

    def test_a_quote_records_an_observation_time_not_a_period_end(self):
        """A current quote has an observation time, not a reporting period.
        Putting the trading day in `period_end` filed a quote as though it
        closed a fiscal period."""
        observation = self._resolve("get_stock_quote", self.QUOTE)

        assert observation is not None
        assert observation.observed_at == "2026-08-24"
        assert observation.period_end is None

    @pytest.mark.parametrize("day", [None, "", "24-08-2026", "2026-8-4",
                                     "yesterday", "2026-13-45"])
    def test_a_missing_or_malformed_day_yields_no_observation(self, day):
        payload = {**self.QUOTE, "latest_trading_day": day}
        assert self._resolve("get_stock_quote", payload) is None

    def test_a_valid_day_is_accepted(self):
        assert self._resolve("get_stock_quote", self.QUOTE) is not None


class TestCompanyOverviewIsNotAnObservation:
    """Market cap and P/E have no observation time in the producer contract,
    so they cannot support a deterministic comparison — D10."""

    OVERVIEW = {"success": True, "symbol": "AAPL", "market_cap": 3.4e12,
                "pe_ratio": 31.2, "dividend_yield": 0.0044,
                "fifty_two_week_high": 260.1, "fifty_two_week_low": 164.0}

    @pytest.mark.parametrize("metric", ["market_cap", "pe_ratio",
                                        "dividend_yield", "52_week_high",
                                        "52_week_low"])
    def test_overview_metrics_yield_no_trusted_observation(self, metric):
        from finvet.models.evidence import resolve_trusted_observation

        record = _record("get_company_overview", self.OVERVIEW)
        assert resolve_trusted_observation(_claim(metric, 1.0), [record]) is None

    def test_not_even_if_the_payload_carries_a_trading_day(self):
        """The tool decides, not the payload shape: an overview response that
        happened to include a date would otherwise slip through."""
        from finvet.models.evidence import resolve_trusted_observation

        record = _record("get_company_overview",
                         {**self.OVERVIEW, "latest_trading_day": "2026-08-24"})
        assert resolve_trusted_observation(
            _claim("market_cap", 3.4e12), [record]) is None

    def test_the_control_a_quote_metric_still_resolves(self):
        """Without this the assertions above pass on a resolver that rejects
        everything."""
        from finvet.models.evidence import resolve_trusted_observation

        record = _record("get_stock_quote",
                         {"success": True, "symbol": "AAPL", "price": 150.0,
                          "latest_trading_day": "2026-08-24"})
        assert resolve_trusted_observation(_claim(), [record]) is not None


class TestTheObservationTimeReachesTheAuditRecord:
    """Step 8: absence is a test failure, not a display-only omission.

    Provenance that stops at the agent is not provenance. If `observed_at` does
    not reach `data_sources`, the audit trail records a market verdict with no
    statement of when the price it rests on was true.
    """

    def _metadata(self):
        from unittest.mock import MagicMock, patch

        from finvet.agents.base import VerdictOutput
        from finvet.agents.market_agent.react_agent import MarketAgent
        from finvet.graph.nodes.response_generator import response_generator
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import TrustedObservation

        agent = MarketAgent.__new__(MarketAgent)
        agent.agent_type = "market"
        agent.max_iterations = 3
        agent.react_agent = MagicMock()
        agent.react_agent.invoke.return_value = {"messages": []}

        claim = ParsedClaim(claim_type="market", ticker="AAPL",
                            metric="closing_price", operator="eq", value=150.0)
        observation = TrustedObservation(
            tool="get_stock_quote", metric="closing_price", value=150.0,
            observed_at="2026-08-24", source_id="AAPL")

        with patch.object(MarketAgent, "_extract_verdict",
                          return_value=VerdictOutput(
                              verdict="SUPPORTS", confidence=0.9,
                              retrieved_value=150.0, reasoning="matches",
                              source_description="quote")), \
             patch("finvet.agents.base.resolve_trusted_observation",
                   return_value=observation):
            evidence = agent.execute({"parsed_claim": claim})

        evidence["tools_called"] = ["get_stock_quote"]
        result = response_generator({
            "request_id": "req_000000000020",
            "claim_raw": "AAPL closed at $150",
            "parsed_claim": claim,
            "agent_evidence": evidence,
            "final_verdict": "SUPPORTS", "final_confidence": 0.9,
            "execution_start_time": "2026-08-26T00:00:00",
        })
        return evidence, result["final_response"]["metadata"]

    def test_the_agent_carries_the_observation_time(self):
        evidence, _ = self._metadata()
        assert evidence["trusted_observation"]["observed_at"] == "2026-08-24"

    def test_it_reaches_the_response_metadata(self):
        _, metadata = self._metadata()
        assert metadata["trusted_observation"]["observed_at"] == "2026-08-24"

    def test_a_quote_reports_no_period_end_in_the_audit_record(self):
        """The distinction has to survive to the record, or a reader cannot
        tell an observation from a closed reporting period."""
        _, metadata = self._metadata()
        assert metadata["trusted_observation"]["period_end"] is None


class TestTheMockFlagSurvivesTheToolBoundary:
    """`QuoteResult(success=True, **quote.model_dump())` drops any field the
    result model does not declare, without error. A flag that is set on the
    client and dropped one layer up looks implemented and is not."""

    def _quote_payload(self, mock_mode):
        from unittest.mock import patch

        from finvet.mcp.finnhub import FinnhubClient
        from finvet.tools import market_tools

        client = FinnhubClient(api_key="x", mock_mode=mock_mode)
        if not mock_mode:
            patcher = patch.object(
                client, "_request",
                lambda endpoint, params: {"c": 150.0, "pc": 148.0,
                                          "t": 1756000000})
            patcher.start()
        with patch.object(market_tools, "_get_client", lambda: client):
            payload = market_tools.get_stock_quote.invoke({"ticker": "AAPL"})
        if not mock_mode:
            patcher.stop()
        return payload

    def test_a_mock_quote_reaches_the_tool_marked_mock(self):
        payload = self._quote_payload(mock_mode=True)
        assert payload["source_mode"] == "mock"

    def test_a_mock_quote_carries_no_trading_day(self):
        """The safety property, independent of the display flag."""
        assert self._quote_payload(mock_mode=True)["latest_trading_day"] is None

    def test_a_live_quote_reaches_the_tool_marked_live(self):
        payload = self._quote_payload(mock_mode=False)
        assert payload["source_mode"] == "live"
        assert payload["latest_trading_day"] == "2025-08-24"

    def test_a_mock_quote_yields_no_trusted_observation(self):
        """End to end: mock data cannot become the number a verdict rests on."""
        from finvet.models.evidence import (
            resolve_trusted_observation, tool_record_from_result)

        record = tool_record_from_result(
            "get_stock_quote", {"ticker": "AAPL"},
            self._quote_payload(mock_mode=True))

        assert resolve_trusted_observation(_claim(), [record]) is None
